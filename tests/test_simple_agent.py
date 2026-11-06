from app.agents.simple_agent import SimpleAgent
from app.config import Settings
from app.llm import LLMError
from app.models.run import FindingDraft, QueryPlan, ReportDraft
from tests.fakes import FakeLLM, FakeSearch


def make_settings(**kw) -> Settings:
    base = Settings.from_env()
    return Settings(**{**base.__dict__, **kw})


def draft(*findings):
    return ReportDraft(summary="resumo", findings=list(findings), inferences=["minha inferência"])


def test_happy_path_builds_report_with_real_urls():
    llm = FakeLLM({QueryPlan: [QueryPlan(queries=["langchain", "haystack"])],
                   ReportDraft: [draft(FindingDraft(text="A", source_ids=[1]), FindingDraft(text="B", source_ids=[2]))]})
    agent = SimpleAgent(llm, FakeSearch(), make_settings())
    res = agent.run("compare langchain e haystack")
    assert res.errors == []
    assert [s.url for s in res.report.sources] == ["https://example.com/langchain", "https://example.com/haystack"]
    assert res.report.inferences == ["minha inferência"]
    assert len(res.llm_calls) == 2


def test_invented_source_ids_are_dropped_and_demoted():
    llm = FakeLLM({QueryPlan: [QueryPlan(queries=["langchain"])],
                   ReportDraft: [draft(FindingDraft(text="ok", source_ids=[1, 99]),
                                       FindingDraft(text="sem fonte", source_ids=[42]))]})
    res = SimpleAgent(llm, FakeSearch(), make_settings()).run("pergunta")
    assert [f.source_ids for f in res.report.findings] == [[1]]
    assert "sem fonte" in res.report.inferences
    assert any("reclassificadas" in l for l in res.report.limitations)


def test_respects_max_searches_and_skips_duplicates():
    search = FakeSearch()
    plan = QueryPlan(queries=["aaa", "AAA  ", "bbb", "ccc", "ddd"])
    llm = FakeLLM({QueryPlan: [plan], ReportDraft: [draft(FindingDraft(text="x", source_ids=[1]))]})
    res = SimpleAgent(llm, search, make_settings(max_searches=3)).run("pergunta")
    assert search.queries == ["aaa", "bbb", "ccc"]
    assert res.queries == ["aaa", "bbb", "ccc"]


def test_search_failure_is_recorded_and_run_continues():
    llm = FakeLLM({QueryPlan: [QueryPlan(queries=["bad query", "good query"])],
                   ReportDraft: [draft(FindingDraft(text="x", source_ids=[1]))]})
    res = SimpleAgent(llm, FakeSearch(fail_on={"bad query"}), make_settings()).run("pergunta")
    assert res.report is not None
    assert res.errors == ["falha simulada"]


def test_no_sources_means_no_report():
    llm = FakeLLM({QueryPlan: [QueryPlan(queries=["bad query"])]})
    res = SimpleAgent(llm, FakeSearch(fail_on={"bad query"}), make_settings()).run("pergunta")
    assert res.report is None
    assert any("nenhuma fonte" in e for e in res.errors)


def test_llm_error_is_captured():
    class Boom(FakeLLM):
        def generate(self, *a, **k):
            raise LLMError("quebrou")
    res = SimpleAgent(Boom({}), FakeSearch(), make_settings()).run("pergunta")
    assert res.report is None and res.errors == ["quebrou"]
