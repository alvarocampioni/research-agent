from starlette.testclient import TestClient

from app.api import app
from app.config import Settings
from app.evaluation.deterministic import run_deterministic_checks
from app.harness.budget import BudgetExceeded, BudgetManager
from app.harness.tool_guard import ToolGuard
from app.models.report import Report, Source, Finding


def make_settings(**kw) -> Settings:
    base = Settings.from_env()
    return Settings(**{**base.__dict__, **kw})


def test_tool_guard_deduplicates_and_caps_queries():
    guard = ToolGuard(max_results_per_query=2, max_queries_per_run=3)
    assert guard.dedupe_queries(["Alpha", "alpha ", "beta", "gamma", "delta"]) == ["Alpha", "beta", "gamma"]


def test_budget_rejects_excess_cost():
    mgr = BudgetManager(max_cost_usd=0.01)
    mgr.snapshot.input_tokens = 1_000_000
    mgr.snapshot.output_tokens = 1_000_000
    try:
        mgr.enforce(0.02, 0.01)
        assert False, "expected BudgetExceeded"
    except BudgetExceeded:
        pass


def test_deterministic_checks_accept_valid_report():
    report = Report(
        question="q",
        summary="sumário",
        findings=[Finding(text="A", source_ids=[1])],
        inferences=["inferência"],
        limitations=["nenhuma"],
        sources=[Source(id=1, title="Fonte", url="https://example.com/a")],
    )
    checks = run_deterministic_checks({
        "report": report,
        "iterations": 1,
        "max_iterations": 3,
        "max_cost_usd": 0.10,
        "estimated_cost_usd": 0.01,
        "duplicate_queries": False,
    }, ["modelo de dados", "escalabilidade"], ["PostgreSQL usa SQL"])
    assert checks["report_structure"]["valid"] is True
    assert checks["citation_validation"]["valid"] is True


def test_fastapi_health_route_is_available():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
