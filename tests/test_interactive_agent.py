import pytest

from app.agents.graph import route_after_evaluator, route_after_researcher
from app.agents.interactive_agent import InteractiveAgent
from app.agents.state import initial_state
from app.config import Settings
from app.models.run import Analysis, Evaluation, FindingDraft, ReportDraft, ResearchPlan
from tests.fakes import FakeLLM, FakeSearch


def make_settings(**kw) -> Settings:
    base = Settings.from_env()
    return Settings(**{**base.__dict__, **kw})


def state_with(sufficient, iteration, pending):
    s = initial_state("q")
    s.update(evaluation=Evaluation(sufficient=sufficient, reason="r"), iteration=iteration, pending_queries=pending)
    return s


def test_route_sufficient_goes_to_report():
    assert route_after_evaluator(state_with(True, 1, ["x"]), 3) == "report"


def test_route_insufficient_with_budget_researches_again():
    assert route_after_evaluator(state_with(False, 1, ["x"]), 3) == "research"


def test_route_stops_at_max_iterations_even_if_llm_wants_more():
    assert route_after_evaluator(state_with(False, 3, ["x"]), 3) == "report"


def test_route_without_new_queries_goes_to_report():
    assert route_after_evaluator(state_with(False, 1, []), 3) == "report"


def test_route_without_sources_ends():
    assert route_after_researcher(initial_state("q")) == "end"


def test_evaluator_filters_duplicates_and_caps_to_budget():
    ev = Evaluation(sufficient=False, reason="r", new_queries=["AAA", "bbb", "bbb ", "ccc", "ddd"])
    agent = InteractiveAgent(FakeLLM({Evaluation: [ev]}), FakeSearch(), make_settings(max_searches=4))
    s = initial_state("q")
    s.update(attempted=["aaa"], analysis=Analysis(), queries_run=["aaa"])
    out = agent.evaluator(s)
    assert out["pending_queries"] == ["bbb", "ccc", "ddd"]


def report_draft():
    return ReportDraft(summary="s", findings=[FindingDraft(text="f", source_ids=[1])])


def build(evals, searches=None, **kw):
    pytest.importorskip("langgraph")
    llm = FakeLLM({
        ResearchPlan: [ResearchPlan(info_needed=["x", "y"], queries=["primeira busca"])],
        Analysis: [Analysis(gaps=["g"]) for _ in evals],
        Evaluation: evals,
        ReportDraft: [report_draft()],
    })
    return InteractiveAgent(llm, searches or FakeSearch(), make_settings(**kw)), llm


def test_graph_researches_again_then_stops_when_sufficient():
    agent, llm = build([Evaluation(sufficient=False, reason="falta y", new_queries=["segunda busca"]),
                        Evaluation(sufficient=True, reason="ok")])
    res = agent.run("pergunta")
    assert res.iterations == 2
    assert res.queries == ["primeira busca", "segunda busca"]
    assert res.report is not None and res.errors == []
    assert [c.step for c in res.llm_calls] == ["plan", "analyze", "evaluate", "analyze", "evaluate", "write_report"]


def test_graph_respects_max_iterations():
    evals = [Evaluation(sufficient=False, reason="mais", new_queries=[f"busca nova {i}"]) for i in range(5)]
    agent, _ = build(evals[:2], max_iterations=2, max_searches=10)
    res = agent.run("pergunta")
    assert res.iterations == 2
    assert len(res.queries) == 2
    assert res.report is not None


def test_graph_stops_when_llm_only_repeats_queries():
    agent, _ = build([Evaluation(sufficient=False, reason="mais", new_queries=["Primeira  busca"])])
    res = agent.run("pergunta")
    assert res.iterations == 1 and res.report is not None


def test_graph_without_sources_returns_error():
    agent, _ = build([], searches=FakeSearch(fail_on={"primeira busca"}))
    res = agent.run("pergunta")
    assert res.report is None
    assert any("nenhuma fonte" in e for e in res.errors)
