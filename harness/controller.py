from __future__ import annotations

import time
from typing import Any

from app.config import Settings
from app.harness.budget import BudgetExceeded, BudgetManager
from app.harness.telemetry import Telemetry
from app.llm import LLMClient
from app.tools.search import WebSearchTool
from app.agents.interactive_agent import InteractiveAgent
from app.agents.simple_agent import SimpleAgent


class ExecutionController:
    def __init__(self, settings: Settings, llm: LLMClient | None = None, search_tool: WebSearchTool | None = None):
        self.settings = settings
        self.llm = llm or LLMClient(settings)
        self.search_tool = search_tool or WebSearchTool(settings.search_api_key)
        self.telemetry = Telemetry()
        self.budget = BudgetManager(settings.max_cost_usd)

    def run(self, agent: Any, question: str) -> Any:
        started = time.monotonic()
        self.telemetry.record("start", agent=getattr(agent, "name", agent.__class__.__name__), question=question)
        try:
            result = agent(self.llm, self.search_tool, self.settings).run(question)
            self.telemetry.record("agent_result", agent=result.agent, queries=len(result.queries), iterations=result.iterations)
            self.budget.record_llm_calls(result.llm_calls)
            self.budget.enforce(
                self.settings.llm_price_input_per_m,
                self.settings.llm_price_output_per_m,
            )
            return result
        except BudgetExceeded as exc:
            self.telemetry.record("budget_exceeded", error=str(exc))
            raise
        except Exception as exc:
            self.telemetry.record("error", error=str(exc))
            raise
        finally:
            elapsed = time.monotonic() - started
            self.telemetry.record("finished", elapsed_seconds=elapsed)

    def run_question(self, agent_name: str, question: str) -> Any:
        agent_cls = {"a": SimpleAgent, "agent_a": SimpleAgent, "b": InteractiveAgent, "agent_b": InteractiveAgent}[agent_name.lower()]
        return self.run(agent_cls, question)
