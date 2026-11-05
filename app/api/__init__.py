from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import get_settings
from app.database import ResearchDatabase
from app.evaluation.comparison import compare_agents
from app.evaluation.runner import run_benchmark
from app.llm import LLMClient
from app.tools.search import WebSearchTool
from app.agents.interactive_agent import InteractiveAgent
from app.agents.simple_agent import SimpleAgent


class ResearchRequest(BaseModel):
    question: str
    agent: str = "a"


class BenchmarkRequest(BaseModel):
    limit: int | None = None


def create_app() -> FastAPI:
    settings = get_settings()
    db = ResearchDatabase(settings.database_path)
    llm = LLMClient(settings)
    search_tool = WebSearchTool(settings.search_api_key)
    app = FastAPI(title="AI Research Agent")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/research")
    def start_research(payload: ResearchRequest) -> dict[str, Any]:

        agent_cls = {"a": SimpleAgent, "agent_a": SimpleAgent, "b": InteractiveAgent, "agent_b": InteractiveAgent}[payload.agent.lower()]
        result = agent_cls(llm, search_tool, settings).run(payload.question)
        run_id = db.save_run(
            agent_cls.name,
            payload.question,
            {"report": result.report.model_dump() if result.report else None, "queries": result.queries, "errors": result.errors, "iterations": result.iterations},
        )
        return {"id": run_id, "agent": agent_cls.name, "status": "done", "result": result.model_dump() if hasattr(result, "model_dump") else result}

    @app.get("/research")
    def list_research() -> list[dict[str, Any]]:
        return db.list_runs()

    @app.get("/research/{run_id}")
    def get_research(run_id: int) -> dict[str, Any]:
        entry = db.get_run(run_id)
        if not entry:
            return {"error": "not found"}
        return entry

    @app.post("/evaluation/run")
    def run_evaluation(payload: BenchmarkRequest) -> dict[str, Any]:
        dataset = run_benchmark(settings)
        summary = compare_agents(dataset)
        payload_result = {"dataset": dataset, "comparison": summary}
        db.save_benchmark(payload_result)
        return payload_result

    @app.get("/evaluation/results")
    def get_results() -> dict[str, Any]:
        rows = db.list_benchmarks()
        return rows[0] if rows else {"results": None}

    return app


app = create_app()
