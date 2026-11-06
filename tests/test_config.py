from app.config import Settings


def test_defaults(monkeypatch):
    for var in ("MAX_SEARCHES", "MAX_COST_USD", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.max_searches == 6
    assert s.max_cost_usd == 0.10
    assert s.llm_model == "gpt-4o-mini"


def test_env_override(monkeypatch):
    monkeypatch.setenv("MAX_ITERATIONS", "5")
    monkeypatch.setenv("MAX_COST_USD", "0.5")
    s = Settings.from_env()
    assert s.max_iterations == 5
    assert s.max_cost_usd == 0.5
