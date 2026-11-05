from app.agents.base import SAFETY_RULES, BaseAgent, format_sources, normalize_query
from app.agents.graph import build_graph
from app.agents.state import ResearchState, initial_state
from app.llm import LLMError
from app.models.run import Analysis, AgentResult, Evaluation, ResearchPlan

PLAN_SYSTEM = f"""Você planeja uma pesquisa na web. Liste em 'info_needed' as informações que precisam ser encontradas
para responder bem (um item por aspecto pedido) e em 'queries' as consultas curtas e diferentes entre si para começar.
{SAFETY_RULES}"""

ANALYZE_SYSTEM = f"""Você analisa fontes recuperadas para uma pergunta de pesquisa.
Para cada item de info_needed, diga o que as fontes realmente sustentam. Liste em 'gaps' o que falta ou está
fraco, e em 'contradictions' onde fontes discordam. Seja específico e não preencha lacunas com conhecimento próprio.
{SAFETY_RULES}"""

EVAL_SYSTEM = """Você decide se a informação coletada basta para responder à pergunta.
sufficient=true SOMENTE se todos os itens de info_needed têm evidência nas fontes.
Se for false, proponha em 'new_queries' consultas NOVAS e específicas para as lacunas
(diferentes das já executadas). Explique em 'reason'."""


def format_analysis(a: Analysis) -> str:
    def block(title: str, items: list[str]) -> str:
        return f"{title}:\n" + ("\n".join(f"- {i}" for i in items) if items else "- (nenhum)")
    return "\n".join([block("Achados", a.relevant_findings), block("Lacunas", a.gaps),
                      block("Contradições", a.contradictions)])


class InteractiveAgent(BaseAgent):
    name = "agent_b"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = build_graph(self, self.settings.max_iterations)

    def planner(self, state: ResearchState) -> dict:
        # reserva parte do limite para consultas adicionais depois da analise
        first_round = max(1, self.settings.max_searches // 2)
        plan = self.llm.generate(
            ResearchPlan, PLAN_SYSTEM + f"\nGere no máximo {first_round} consultas.",
            state["question"], step="plan")
        return {"info_needed": plan.info_needed, "pending_queries": plan.queries[:first_round]}

    def researcher(self, state: ResearchState) -> dict:
        retrieved = dict(state["retrieved"])
        errors = list(state["errors"])
        seen = set(state["attempted"])
        executed = self.search_queries(state["pending_queries"], retrieved, errors, already_run=seen)
        if not retrieved:
            errors.append("nenhuma fonte recuperada; relatório não gerado")
        return {"retrieved": retrieved, "errors": errors, "attempted": sorted(seen),
                "queries_run": state["queries_run"] + executed,
                "pending_queries": [], "iteration": state["iteration"] + 1}

    def analyzer(self, state: ResearchState) -> dict:
        user = (f"Pergunta: {state['question']}\n\ninfo_needed:\n" +
                "\n".join(f"- {i}" for i in state["info_needed"]) +
                f"\n\nFontes:\n{format_sources(list(state['retrieved'].values()))}")
        return {"analysis": self.llm.generate(Analysis, ANALYZE_SYSTEM, user, step="analyze")}

    def evaluator(self, state: ResearchState) -> dict:
        user = (f"Pergunta: {state['question']}\n\ninfo_needed:\n" +
                "\n".join(f"- {i}" for i in state["info_needed"]) +
                f"\n\nAnálise:\n{format_analysis(state['analysis'])}"
                f"\n\nConsultas já executadas:\n" + "\n".join(f"- {q}" for q in state["queries_run"]))
        ev = self.llm.generate(Evaluation, EVAL_SYSTEM, user, step="evaluate")

        seen = set(state["attempted"])
        fresh: list[str] = []
        for q in ev.new_queries:
            key = normalize_query(q)
            if key not in seen and key not in {normalize_query(f) for f in fresh}:
                fresh.append(q)
        remaining = max(self.settings.max_searches - len(seen), 0)
        return {"evaluation": ev, "pending_queries": fresh[:remaining]}

    def reporter(self, state: ResearchState) -> dict:
        notes = format_analysis(state["analysis"]) if state["analysis"] else ""
        return {"report": self.write_report(state["question"], state["retrieved"], notes)}

    def run(self, question: str) -> AgentResult:
        start = len(self.llm.calls)
        state = initial_state(question)
        error = None
        try:
            # mantem o ultimo estado para identificar trabalho parcial em caso de falha
            for state in self.graph.stream(state, config={"recursion_limit": 30}, stream_mode="values"):
                pass
        except LLMError as exc:
            error = str(exc)
        errors = state["errors"] + ([error] if error else [])
        return AgentResult(
            agent=self.name, question=question, report=state["report"],
            queries=state["queries_run"], retrieved=list(state["retrieved"].values()),
            llm_calls=list(self.llm.calls[start:]), iterations=max(state["iteration"], 1), errors=errors)
