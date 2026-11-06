from __future__ import annotations

from app.evaluation.metrics import average, summarize_results

QUALITY_KEYS = ("answer_relevance", "faithfulness", "completeness", "citation_accuracy", "valid_reference_rate")


def compare_agents(results_by_agent: dict[str, list[dict]]) -> dict[str, dict]:
    """Resume por agente.

    - success_rate: fracao das execucoes que produziram relatorio avaliado.
    - Notas de qualidade: media SOMENTE das execucoes avaliadas (falha nao vira nota 0).
    - Custo, tokens e tempo: media de TODAS as execucoes (falhar tambem custa).
    """
    comparison: dict[str, dict] = {}
    for agent_name, rows in results_by_agent.items():
        scored = [r for r in rows if r.get("judge")]
        ops = summarize_results([{
            "runtime_seconds": r.get("runtime_seconds", 0.0), "cost_usd": r.get("cost_usd", 0.0),
            "tokens_used": r.get("tokens_used", 0), "searches": r.get("searches", 0),
            "iterations": r.get("iterations", 1),
        } for r in rows])
        quality = summarize_results([{
            **{k: r["judge"][k] for k in QUALITY_KEYS if k in r["judge"]},
            "valid_reference_rate": r.get("valid_reference_rate", 0.0),
        } for r in scored])

        summary = dict(ops)
        for k in QUALITY_KEYS:
            summary[k] = quality[k] if scored else None
        summary.update({
            "n_runs": len(rows),
            "n_scored": len(scored),
            "success_rate": round(len(scored) / len(rows), 4) if rows else 0.0,
            "n_crashed": sum(1 for r in rows if r.get("crashed")),
            "n_judge_failed": sum(1 for r in rows if r.get("report") is not None and not r.get("judge")),
            "n_claim_incomplete": sum(1 for r in rows if (r.get("claim_support") or {}).get("incomplete")),
            "supported_claim_rate": average([r["claim_support"]["rate"] for r in scored
                                             if r.get("claim_support") and r["claim_support"]["rate"] is not None]) if scored else None,
            "criteria_coverage": average([r["criteria_coverage"] for r in scored
                                          if r.get("criteria_coverage") is not None]) if scored else None,
        })
        comparison[agent_name] = summary
    return comparison
