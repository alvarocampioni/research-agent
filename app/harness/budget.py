from __future__ import annotations

from dataclasses import dataclass

from app.models.run import LLMCall


class BudgetExceeded(Exception):
    pass

@dataclass
class BudgetSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class BudgetManager:
    def __init__(self, max_cost_usd: float):
        self.max_cost_usd = float(max_cost_usd)
        self.snapshot = BudgetSnapshot()

    def record_llm_calls(self, calls: list[LLMCall]) -> None:
        for call in calls:
            self.snapshot.input_tokens += call.input_tokens
            self.snapshot.output_tokens += call.output_tokens

    def estimate_cost(self, llm_price_input_per_m: float, llm_price_output_per_m: float) -> float:
        return (
            self.snapshot.input_tokens * llm_price_input_per_m
            + self.snapshot.output_tokens * llm_price_output_per_m
        ) / 1_000_000

    def enforce(self, llm_price_input_per_m: float, llm_price_output_per_m: float) -> None:
        self.snapshot.estimated_cost_usd = self.estimate_cost(
            llm_price_input_per_m, llm_price_output_per_m
        )
        if self.snapshot.estimated_cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"orçamento excedido: US${self.snapshot.estimated_cost_usd:.4f} > US${self.max_cost_usd:.4f}"
            )
