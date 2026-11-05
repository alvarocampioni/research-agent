from app.agents.base import SAFETY_RULES, BaseAgent
from app.llm import LLMError
from app.models.run import AgentResult, QueryPlan, RetrievedSource

PLAN_SYSTEM = f"""Você planeja buscas na web para responder uma pergunta de pesquisa.
Gere consultas curtas, diferentes entre si e cobrindo os aspectos pedidos (em inglês costuma render mais resultados).
{SAFETY_RULES}"""


class SimpleAgent(BaseAgent):
    name = "agent_a"

    def run(self, question: str) -> AgentResult:
        start = len(self.llm.calls)
        errors: list[str] = []
        retrieved: dict[str, RetrievedSource] = {}
        result = AgentResult(agent=self.name, question=question, iterations=1)
        try:
            plan = self.llm.generate(
                QueryPlan,
                PLAN_SYSTEM + f"\nGere no máximo {self.settings.max_searches} consultas.",
                question, step="plan_queries")
            result.queries = self.search_queries(plan.queries, retrieved, errors)
            if not retrieved:
                errors.append("nenhuma fonte recuperada; relatório não gerado")
            else:
                result.report = self.write_report(question, retrieved)
        except LLMError as exc:
            errors.append(str(exc))
        result.retrieved = list(retrieved.values())
        result.llm_calls = list(self.llm.calls[start:])
        result.errors = errors
        return result
