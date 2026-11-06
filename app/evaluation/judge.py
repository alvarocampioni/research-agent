from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.llm import LLMClient, LLMError

MAX_JUDGE_SOURCES = 8
MAX_SOURCE_CHARS = 1000
MAX_TOTAL_SOURCE_CHARS = 6000


class JudgeScores(BaseModel):
    justification: str
    overall: Literal["poor", "fair", "good"]
    answer_relevance: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    citation_accuracy: int = Field(ge=1, le=5)

    def mean(self) -> float:
        return (self.answer_relevance + self.faithfulness + self.completeness + self.citation_accuracy) / 4

    def is_inconsistent(self) -> bool:
        """Notas que contradizem o veredito geral do proprio judge."""
        m = self.mean()
        if self.overall == "poor":
            return m > 2.5
        if self.overall == "good":
            return m < 3.5
        return m < 2 or m > 4


JUDGE_SYSTEM = (
    "Você é um avaliador rigoroso. Só dê notas altas com evidência explícita nas fontes recuperadas. "
    "Se não houver suporte, reduza a nota e explique por quê.\n\n"
    "ESCALA: notas INTEIRAS de 1 a 5, em que 1 = muito ruim e 5 = excelente. NUNCA use escala de 0 a 10 "
    "(nada de '8/10'). Nota 1 é permitida, mas só para um relatório realmente ruim.\n"
    "- answer_relevance: o relatório responde à pergunta e cobre os critérios esperados?\n"
    "- faithfulness: as afirmações são sustentadas pelas fontes citadas?\n"
    "- completeness: cobre os aspectos pedidos, com as limitações relevantes?\n"
    "- citation_accuracy: as citações apontam para fontes que de fato dizem aquilo?\n"
    "Ordem da resposta: PRIMEIRO a justificativa (no máximo 3 frases); depois overall = 'poor', 'fair' ou 'good' "
    "(sua avaliação geral); só então as quatro notas, coerentes com o overall: good = 4 ou 5, fair = 2 a 4, poor = 1 ou 2."
)
JUDGE_MAX_ATTEMPTS = 3


def _retry_reminder(scores: "JudgeScores") -> str:
    return (
        f"\n\nATENÇÃO: sua resposta anterior foi incoerente: overall='{scores.overall}' com notas "
        f"{scores.answer_relevance}/{scores.faithfulness}/{scores.completeness}/{scores.citation_accuracy}. "
        "Use inteiros de 1 a 5 (5 = melhor), nunca escala de 0 a 10, e mantenha as notas coerentes com o overall "
        "e com a justificativa (good = 4 ou 5, fair = 2 a 4, poor = 1 ou 2)."
    )


class LLMJudge:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(self, question: str, expected_criteria: list[str], report, sources: list) -> JudgeScores:
        system = JUDGE_SYSTEM
        criteria = "\n- ".join(expected_criteria)
        report_text = (
            f"Resumo: {report.summary}\n"
            f"Achados: {report.findings}\n"
            f"Inferências: {report.inferences}\n"
            f"Limitações: {report.limitations}"
        )
        cited_ids = {
            source_id
            for finding in report.findings
            for source_id in finding.source_ids
        }
        selected_sources = [s for s in sources if s.id in cited_ids][:MAX_JUDGE_SOURCES]
        if not selected_sources:
            selected_sources = sources[:MAX_JUDGE_SOURCES]

        source_lines = []
        remaining_chars = MAX_TOTAL_SOURCE_CHARS
        for source in selected_sources:
            excerpt = source.content[:min(MAX_SOURCE_CHARS, remaining_chars)]
            if len(source.content) > len(excerpt):
                excerpt += " [trecho truncado]"
            source_lines.append(
                f"[{source.id}] {source.title} — {source.url}\n{excerpt}"
            )
            remaining_chars -= min(len(source.content), MAX_SOURCE_CHARS)
            if remaining_chars <= 0:
                break
        source_text = "\n\n".join(source_lines)
        user = (
            f"Pergunta: {question}\n\n"
            f"Critérios esperados:\n- {criteria}\n\n"
            f"Relatório:\n{report_text}\n\n"
            "Fontes citadas (trechos limitados; considere a possibilidade de contexto omitido):\n"
            f"{source_text}"
        )

        reminder = ""
        for _ in range(JUDGE_MAX_ATTEMPTS):
            scores = self.llm.generate(JudgeScores, system, user + reminder, step="judge_report")
            if not scores.is_inconsistent():
                return scores
            reminder = _retry_reminder(scores)
        raise LLMError("judge devolveu notas incoerentes com o proprio veredito apos varias tentativas")


MAX_CLAIM_SOURCES = 3
MAX_CLAIM_EVIDENCE_CHARS = 1200


class ClaimVerdict(BaseModel):
    index: int
    supported: bool
    reason: str = ""


class ClaimVerdicts(BaseModel):
    verdicts: list[ClaimVerdict]


class ClaimSupportJudge:
    """Para cada achado do relatorio: o trecho da fonte citada realmente sustenta a afirmacao?

    E a metrica de fidelidade que a validacao estrutural de citacoes nao consegue dar.
    Se o judge nao devolver exatamente um veredito por afirmacao, a medicao e marcada `incomplete` e fica
    FORA das medias (antes, veredito ausente virava 'nao sustentado' e derrubava a taxa sem motivo)."""

    SYSTEM = (
        "Voce verifica afirmacoes contra evidencias. Para cada afirmacao, responda supported=true SOMENTE se a "
        "evidencia citada afirma isso de forma explicita. Nao use conhecimento proprio e nao seja generoso: "
        "evidencia relacionada, mas que nao diz o que a afirmacao diz, e supported=false. "
        "A evidencia pode estar em outro idioma (ex.: ingles): compare o SENTIDO, nao as palavras. "
        "Devolva EXATAMENTE um veredito para cada afirmacao, usando o mesmo index (de 0 ate N-1). "
        "No campo reason, no maximo 15 palavras."
    )
    MAX_ATTEMPTS = 2

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(self, report, sources: list) -> dict:
        findings = report.findings
        if not findings:
            return {"total": 0, "supported": 0, "rate": None, "unsupported": [], "incomplete": False, "verdicts": []}
        by_id = {s.id: s for s in sources}
        blocks = []
        for i, f in enumerate(findings):
            cited = [by_id[sid] for sid in f.source_ids if sid in by_id][:MAX_CLAIM_SOURCES]
            evidence = "\n".join(f"[{s.id}] {s.content[:MAX_CLAIM_EVIDENCE_CHARS]}" for s in cited) or "(nenhuma)"
            blocks.append(f"index {i}\nAfirmacao: {f.text}\nEvidencia citada:\n{evidence}")
        n = len(findings)
        user = f"Sao {n} afirmacoes (index 0 ate {n - 1}). Devolva {n} veredictos.\n\n" + "\n\n".join(blocks)

        by_index: dict[int, ClaimVerdict] = {}
        for attempt in range(self.MAX_ATTEMPTS):
            out = self.llm.generate(ClaimVerdicts, self.SYSTEM, user, step="judge_claims")
            by_index = {v.index: v for v in out.verdicts}
            if set(by_index) == set(range(n)):
                break
            user += f"\n\nATENCAO: voce devolveu {len(out.verdicts)} veredictos; sao necessarios exatamente {n}, com index 0 a {n - 1}."
        else:
            return {"total": n, "supported": None, "rate": None, "unsupported": [], "incomplete": True,
                    "verdicts": [{"index": i, "supported": v.supported, "reason": v.reason} for i, v in sorted(by_index.items())]}

        unsupported = [f.text[:120] for i, f in enumerate(findings) if not by_index[i].supported]
        return {"total": n, "supported": n - len(unsupported), "rate": round((n - len(unsupported)) / n, 4),
                "unsupported": unsupported, "incomplete": False,
                "verdicts": [{"index": i, "supported": by_index[i].supported, "reason": by_index[i].reason} for i in range(n)]}
