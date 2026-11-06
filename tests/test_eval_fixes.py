"""Regressoes das correcoes na camada de avaliacao (cada teste falharia no codigo antigo)."""
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.evaluation.comparison import compare_agents
from app.evaluation.deterministic import check_expected_criteria
from app.evaluation.judge import ClaimSupportJudge, ClaimVerdict, ClaimVerdicts, JudgeScores, LLMJudge
from app.evaluation.metrics import usage_from_calls
from app.evaluation.runner import run_single_question
from app.models.report import Finding, Report, Source
from app.models.run import FindingDraft, LLMCall, QueryPlan, ReportDraft, RetrievedSource
from tests.fakes import FakeLLM, FakeSearch

DATASET = json.loads((Path(__file__).parent.parent / "app" / "evaluation" / "dataset.json").read_text(encoding="utf-8"))
Q1 = DATASET[0]


def make_report(*texts, text_summary="resumo"):
    return Report(
        question="q", summary=text_summary,
        findings=[Finding(text=t, source_ids=[1]) for t in texts],
        sources=[Source(id=1, title="T", url="https://example.com/1")],
    )


# --- 1. cobertura de criterios -------------------------------------------------------------
def test_criteria_coverage_rejects_text_with_only_generic_words():
    """Antes: qualquer palavra ('de', 'e', ...) bastava, entao este texto cobria 3 de 3 criterios."""
    empty = make_report("Os bancos diferem bastante, de acordo com as fontes, e ambos sao muito usados.")
    r = check_expected_criteria(empty, Q1["expected_criteria"], Q1["important_facts"])
    assert r["criteria_coverage"] == 0.0 and r["valid"] is False
    assert r["matched_criteria"] == []


def test_criteria_coverage_accepts_relevant_text_and_fuzzy_facts():
    rep = make_report(
        "O PostgreSQL e um banco relacional que usa SQL e prioriza consistencia forte com transacoes.",
        "O MongoDB e orientado a documentos, com flexibilidade e escalabilidade horizontal.",
        text_summary="Comparacao entre modelo de dados relacional e documental, consistencia e escalabilidade.",
    )
    r = check_expected_criteria(rep, Q1["expected_criteria"], Q1["important_facts"])
    assert r["criteria_coverage"] >= 0.66
    assert "PostgreSQL é relacional e usa SQL" in r["matched_facts"]  # antes exigia a frase exata


def test_criteria_coverage_without_report_is_invalid():
    assert check_expected_criteria(None, ["x"], [])["valid"] is False


# --- 2. custo por execucao -----------------------------------------------------------------
def test_usage_from_calls():
    calls = [LLMCall(step="a", input_tokens=1_000_000, output_tokens=0), LLMCall(step="b", input_tokens=0, output_tokens=1_000_000)]
    assert usage_from_calls(calls, 0.15, 0.60) == (2_000_000, 0.75)


def _settings():
    return Settings(**{**Settings.from_env().__dict__, "llm_price_input_per_m": 1.0, "llm_price_output_per_m": 1.0})


def _scripted(n_runs):
    draft = ReportDraft(summary="resumo", findings=[FindingDraft(text="afirmacao", source_ids=[1])])
    scores = JudgeScores(answer_relevance=4, faithfulness=4, completeness=4, citation_accuracy=4, justification="ok", overall="good")
    return FakeLLM({
        QueryPlan: [QueryPlan(queries=["alpha", "beta"]) for _ in range(n_runs)],
        ReportDraft: [draft for _ in range(n_runs)],
        JudgeScores: [scores for _ in range(n_runs)],
        ClaimVerdicts: [ClaimVerdicts(verdicts=[ClaimVerdict(index=0, supported=True)]) for _ in range(n_runs)],
    })


def test_cost_is_per_run_not_cumulative_across_the_shared_client():
    """Antes: cost_usd/tokens_used vinham do cliente compartilhado e cresciam a cada pergunta."""
    llm, search, settings = _scripted(3), FakeSearch(), _settings()
    rows = [run_single_question(settings, llm, search, "a", {**Q1, "id": f"q{i}"}) for i in range(3)]
    assert [r["tokens_used"] for r in rows] == [300, 300, 300]
    assert rows[0]["cost_usd"] == rows[1]["cost_usd"] == rows[2]["cost_usd"] > 0
    assert rows[0]["judge_tokens"] > 0          # custo do judge fica separado do custo do agente
    assert rows[0]["claim_support"]["rate"] == 1.0


def test_a_crashing_run_is_recorded_not_fatal():
    llm = FakeLLM({})  # sem respostas: generate() levanta KeyError no meio da execucao
    row = run_single_question(_settings(), llm, FakeSearch(), "a", Q1)
    assert row["crashed"] is True and row["report"] is None and row["errors"]


# --- 3. afirmacoes sustentadas -------------------------------------------------------------
def _verdicts(*flags, start=0):
    return ClaimVerdicts(verdicts=[ClaimVerdict(index=start + i, supported=f) for i, f in enumerate(flags)])


SOURCES = [RetrievedSource(id=1, title="T", url="https://example.com/1", content="evidencia")]


def test_claim_support_rate_with_complete_verdicts():
    rep = make_report("afirmacao 0", "afirmacao 1", "afirmacao 2")
    out = ClaimSupportJudge(FakeLLM({ClaimVerdicts: [_verdicts(True, False, True)]})).evaluate(rep, SOURCES)
    assert out["total"] == 3 and out["supported"] == 2 and out["rate"] == round(2 / 3, 4)
    assert out["incomplete"] is False and len(out["verdicts"]) == 3 and len(out["unsupported"]) == 1


def test_claim_support_retries_when_verdicts_are_missing():
    rep = make_report("a0", "a1", "a2")
    llm = FakeLLM({ClaimVerdicts: [_verdicts(True), _verdicts(True, True, False)]})  # 1o: so 1 de 3
    out = ClaimSupportJudge(llm).evaluate(rep, SOURCES)
    assert out["incomplete"] is False and out["rate"] == round(2 / 3, 4)
    assert len(llm.calls) == 2


def test_claim_support_incomplete_is_not_scored_as_unsupported():
    """Antes: veredito ausente virava 'nao sustentado' e derrubava a taxa (ex.: 1/7 num relatorio bom)."""
    rep = make_report("a0", "a1", "a2")
    llm = FakeLLM({ClaimVerdicts: [_verdicts(True), _verdicts(True)]})  # nas duas tentativas, incompleto
    out = ClaimSupportJudge(llm).evaluate(rep, SOURCES)
    assert out["incomplete"] is True and out["rate"] is None and out["supported"] is None


def test_claim_verdicts_with_wrong_indices_are_incomplete():
    rep = make_report("a0", "a1")
    llm = FakeLLM({ClaimVerdicts: [_verdicts(True, True, start=1), _verdicts(True, True, start=1)]})  # 1..2, nao 0..1
    assert ClaimSupportJudge(llm).evaluate(rep, SOURCES)["incomplete"] is True


# --- 5. judge de notas ----------------------------------------------------------------------
def _scores(v, text="ok", overall=None):
    overall = overall or ("good" if v >= 4 else "poor" if v <= 2 else "fair")
    return JudgeScores(justification=text, overall=overall, answer_relevance=v, faithfulness=v,
                       completeness=v, citation_accuracy=v)


def _judge_args():
    return ("pergunta", ["criterio"], make_report("afirmacao"), SOURCES)


def test_judge_schema_orders_justification_then_overall_then_scores_and_prompt_defines_the_scale():
    from app.evaluation.judge import JUDGE_SYSTEM
    assert list(JudgeScores.model_json_schema()["properties"])[:3] == ["justification", "overall", "answer_relevance"]
    assert "1 a 5" in JUDGE_SYSTEM and "5 = excelente" in JUDGE_SYSTEM and "0 a 10" in JUDGE_SYSTEM


def test_a_genuinely_terrible_report_can_get_all_ones():
    """O caso 'e se ele quiser mesmo dar 1 em tudo?': tudo 1 COM veredito poor e coerente e e mantido."""
    llm = FakeLLM({JudgeScores: [_scores(1, "relatorio vazio, sem fontes", overall="poor")]})
    out = LLMJudge(llm).evaluate(*_judge_args())
    assert out.faithfulness == 1 and len(llm.calls) == 1          # aceito de primeira, sem retry


def test_judge_retries_when_scores_contradict_its_own_verdict():
    """Antes: 8 de 10 execucoes vieram 1/1/1/1 com justificativa positiva ('9/10') e entraram na media."""
    llm = FakeLLM({JudgeScores: [_scores(1, "excelente, 9/10", overall="good"), _scores(5)]})
    out = LLMJudge(llm).evaluate(*_judge_args())
    assert out.faithfulness == 5 and len(llm.calls) == 2


def test_judge_gives_up_on_persistent_incoherence_instead_of_returning_it():
    from app.llm import LLMError
    llm = FakeLLM({JudgeScores: [_scores(1, "otimo", overall="good")] * 3})
    with pytest.raises(LLMError):
        LLMJudge(llm).evaluate(*_judge_args())


def test_inconsistency_rules():
    ok = [("poor", 1), ("poor", 2), ("fair", 3), ("fair", 2), ("good", 4), ("good", 5)]
    bad = [("poor", 4), ("good", 1), ("good", 3), ("fair", 1), ("fair", 5)]
    assert not any(_scores(v, overall=o).is_inconsistent() for o, v in ok)
    assert all(_scores(v, overall=o).is_inconsistent() for o, v in bad)


def test_a_low_but_coherent_mixed_score_is_kept():
    s = JudgeScores(justification="fraco", overall="poor", answer_relevance=2, faithfulness=1, completeness=2, citation_accuracy=1)
    assert LLMJudge(FakeLLM({JudgeScores: [s]})).evaluate(*_judge_args()).answer_relevance == 2


def test_judge_failure_does_not_skip_claim_support_and_leaves_run_unscored():
    draft = ReportDraft(summary="resumo", findings=[FindingDraft(text="afirmacao", source_ids=[1])])
    llm = FakeLLM({QueryPlan: [QueryPlan(queries=["alpha", "beta"])], ReportDraft: [draft],
                   JudgeScores: [_scores(1, "otimo", overall="good")] * 3, ClaimVerdicts: [_verdicts(True)]})
    row = run_single_question(_settings(), llm, FakeSearch(), "a", Q1)
    assert row["judge"] is None and any(e.startswith("judge:") for e in row["errors"])
    assert row["claim_support"]["rate"] == 1.0
    s = compare_agents({"x": [row]})["x"]
    assert s["success_rate"] == 0.0 and s["answer_relevance"] is None and s["n_judge_failed"] == 1


def test_incomplete_claim_measurements_are_counted_and_excluded_from_the_rate():
    base = {"judge": {"answer_relevance": 4, "faithfulness": 4, "completeness": 4, "citation_accuracy": 4},
            "valid_reference_rate": 1.0, "runtime_seconds": 1, "cost_usd": 0.0, "tokens_used": 1, "searches": 1, "iterations": 1}
    ok = {**base, "claim_support": {"rate": 0.8, "incomplete": False}}
    bad = {**base, "claim_support": {"rate": None, "incomplete": True}}
    s = compare_agents({"x": [ok, bad]})["x"]
    assert s["supported_claim_rate"] == 0.8 and s["n_claim_incomplete"] == 1


def test_claim_support_with_no_findings_has_no_rate():
    rep = Report(question="q", summary="s", findings=[], sources=[])
    assert ClaimSupportJudge(FakeLLM({})).evaluate(rep, [])["rate"] is None


# --- 4. falha nao vira nota 0 --------------------------------------------------------------
def test_failed_runs_do_not_drag_quality_scores():
    good = {"judge": {"answer_relevance": 4, "faithfulness": 4, "completeness": 4, "citation_accuracy": 4},
            "valid_reference_rate": 1.0, "runtime_seconds": 10, "cost_usd": 0.02, "tokens_used": 1000,
            "searches": 5, "iterations": 2, "claim_support": {"rate": 0.5}, "criteria_coverage": 1.0}
    failed = {"judge": None, "valid_reference_rate": 0.0, "runtime_seconds": 20, "cost_usd": 0.04,
              "tokens_used": 2000, "searches": 0, "iterations": 1, "crashed": True}
    s = compare_agents({"agent_x": [good, failed]})["agent_x"]
    assert s["answer_relevance"] == 4.0            # antes: 2.0 (a falha entrava como 0)
    assert s["success_rate"] == 0.5 and s["n_crashed"] == 1 and s["n_scored"] == 1
    assert s["avg_cost_usd"] == 0.03               # custo: media de TODAS (falhar tambem custa)
    assert s["supported_claim_rate"] == 0.5


def test_compare_agents_with_nothing_scored_reports_none_not_zero():
    s = compare_agents({"agent_x": [{"judge": None, "crashed": True}]})["agent_x"]
    assert s["answer_relevance"] is None and s["success_rate"] == 0.0
