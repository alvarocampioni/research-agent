from app.evaluation.judge import JudgeScores, LLMJudge
from app.models.report import Finding, Report, Source
from app.models.run import RetrievedSource


class CapturingLLM:
    def generate(self, schema, system, user, step):
        self.user = user
        return JudgeScores(
            answer_relevance=4,
            faithfulness=4,
            completeness=4,
            citation_accuracy=4,
            justification="ok",
            overall="good",
        )


def test_judge_limits_source_context_to_cited_excerpts():
    llm = CapturingLLM()
    report = Report(
        question="pergunta",
        summary="resumo",
        findings=[Finding(text="achado", source_ids=[1])],
        sources=[Source(id=1, title="Citada", url="https://example.com/1")],
    )
    sources = [
        RetrievedSource(
            id=1,
            title="Citada",
            url="https://example.com/1",
            content="evidência " * 2000,
        ),
        RetrievedSource(
            id=2,
            title="Não citada",
            url="https://example.com/2",
            content="conteúdo irrelevante " * 2000,
        ),
    ]

    LLMJudge(llm).evaluate("pergunta", [], report, sources)

    assert "Citada" in llm.user
    assert "Não citada" not in llm.user
    assert "[trecho truncado]" in llm.user
    assert len(llm.user) < 8000
