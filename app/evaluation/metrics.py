from __future__ import annotations

from statistics import mean


def usage_from_calls(calls: list, price_in_per_m: float, price_out_per_m: float) -> tuple[int, float]:
    tin = sum(c.input_tokens for c in calls)
    tout = sum(c.output_tokens for c in calls)
    return tin + tout, (tin * price_in_per_m + tout * price_out_per_m) / 1_000_000


def average(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def summarize_results(results: list[dict]) -> dict[str, float | int]:
    if not results:
        return {
            "answer_relevance": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0,
            "citation_accuracy": 0.0,
            "valid_reference_rate": 0.0,
            "avg_runtime_seconds": 0.0,
            "avg_cost_usd": 0.0,
            "avg_tokens": 0.0,
            "avg_searches": 0.0,
            "avg_iterations": 0.0,
        }

    avg_relevance = average([r.get("answer_relevance", 0.0) for r in results])
    avg_faithfulness = average([r.get("faithfulness", 0.0) for r in results])
    avg_completeness = average([r.get("completeness", 0.0) for r in results])
    avg_citation = average([r.get("citation_accuracy", 0.0) for r in results])
    valid_reference_rate = average([r.get("valid_reference_rate", 0.0) for r in results])
    avg_runtime = average([r.get("runtime_seconds", 0.0) for r in results])
    avg_cost = average([r.get("cost_usd", 0.0) for r in results])
    avg_tokens = average([r.get("tokens_used", 0.0) for r in results])
    avg_searches = average([r.get("searches", 0.0) for r in results])
    avg_iterations = average([r.get("iterations", 0.0) for r in results])

    return {
        "answer_relevance": avg_relevance,
        "faithfulness": avg_faithfulness,
        "completeness": avg_completeness,
        "citation_accuracy": avg_citation,
        "valid_reference_rate": valid_reference_rate,
        "avg_runtime_seconds": avg_runtime,
        "avg_cost_usd": avg_cost,
        "avg_tokens": avg_tokens,
        "avg_searches": avg_searches,
        "avg_iterations": avg_iterations,
    }
