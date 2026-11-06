from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from app.agents.interactive_agent import InteractiveAgent
from app.agents.simple_agent import SimpleAgent
from app.config import Settings
from app.evaluation.deterministic import run_deterministic_checks
from app.evaluation.judge import ClaimSupportJudge, LLMJudge
from app.evaluation.metrics import usage_from_calls
from app.harness.controller import ExecutionController
from app.llm import LLMClient, LLMError
from app.tools.search import WebSearchTool


DATASET_PATH = Path(__file__).with_name("dataset.json")


def load_dataset(path: str | Path = DATASET_PATH) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_judge_llm(settings: Settings, llm: LLMClient) -> LLMClient:
    if settings.judge_model and settings.judge_model != settings.llm_model:
        return LLMClient(replace(settings, llm_model=settings.judge_model))
    return llm


def run_single_question(settings: Settings, llm: LLMClient, search_tool: WebSearchTool, agent_name: str,
                        question: dict, judge_llm: LLMClient | None = None) -> dict:
    judge_llm = judge_llm or llm
    agent_cls = SimpleAgent if agent_name.lower() in {"a", "agent_a"} else InteractiveAgent
    judge_model = settings.judge_model or settings.llm_model
    base = {
        "question_id": question["id"], "question": question["question"], "agent": agent_cls.name,
        "judge_model": judge_model, "self_judged": judge_model == settings.llm_model,
    }
    controller = ExecutionController(settings, llm=llm, search_tool=search_tool)

    calls_before = len(llm.calls)
    start = time.monotonic()
    crash = None
    result = None
    try:
        result = controller.run(agent_cls, question["question"])
    except Exception as exc:
        crash = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - start
    tokens, cost = usage_from_calls(list(llm.calls[calls_before:]),
                                    settings.llm_price_input_per_m, settings.llm_price_output_per_m)

    if result is None:
        return {**base, "report": None, "queries": [], "retrieved": [], "iterations": 0, "searches": 0,
                "runtime_seconds": elapsed, "cost_usd": cost, "tokens_used": tokens, "errors": [crash],
                "crashed": True, "valid_reference_rate": 0.0, "criteria_coverage": None,
                "deterministic": None, "judge": None, "claim_support": None, "judge_tokens": 0}

    deterministic = run_deterministic_checks(
        {
            "report": result.report,
            "iterations": result.iterations,
            "max_iterations": settings.max_iterations,
            "max_cost_usd": settings.max_cost_usd,
            "estimated_cost_usd": cost,
            "duplicate_queries": len(result.queries) != len(set(result.queries)),
        },
        question.get("expected_criteria", []),
        question.get("important_facts", []),
    )
    cit = deterministic["citation_validation"]
    valid_reference_rate = cit["valid_urls"] / max(1, cit["total_urls"])

    errors = list(result.errors)
    judge_result, claim_support = None, None
    judge_before = len(judge_llm.calls)
    if result.report:
        try:
            judge_result = LLMJudge(judge_llm).evaluate(
                question["question"], question.get("expected_criteria", []), result.report, result.retrieved)
        except LLMError as exc:
            errors.append(f"judge: {exc}")
        try:
            claim_support = ClaimSupportJudge(judge_llm).evaluate(result.report, result.retrieved)
        except LLMError as exc:
            errors.append(f"claim_support: {exc}")
    judge_tokens, _ = usage_from_calls(list(judge_llm.calls[judge_before:]), 0.0, 0.0)

    return {
        **base,
        "report": result.report,
        "queries": result.queries,
        "retrieved": result.retrieved,
        "iterations": result.iterations,
        "searches": len(result.retrieved),
        "runtime_seconds": elapsed,
        "cost_usd": cost,
        "tokens_used": tokens,
        "errors": errors,
        "crashed": False,
        "valid_reference_rate": valid_reference_rate,
        "criteria_coverage": deterministic["expected_criteria_coverage"]["criteria_coverage"],
        "deterministic": deterministic,
        "judge": judge_result.model_dump() if judge_result else None,
        "claim_support": claim_support,
        "judge_tokens": judge_tokens,
    }


def run_benchmark(settings: Settings, questions: list[dict] | None = None) -> dict[str, list[dict]]:
    questions = questions or load_dataset()
    llm = LLMClient(settings)
    judge_llm = build_judge_llm(settings, llm)
    search_tool = WebSearchTool(settings.search_api_key)
    results = {"agent_a": [], "agent_b": []}
    for question in questions:
        for agent_name in ("a", "b"):
            results[f"agent_{agent_name}"].append(
                run_single_question(settings, llm, search_tool, agent_name, question, judge_llm)
            )
    return results
