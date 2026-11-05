from langgraph.graph import END, StateGraph

from app.agents.state import ResearchState


def route_after_researcher(state: ResearchState) -> str:
    return "analyze" if state["retrieved"] else "end"


def route_after_evaluator(state: ResearchState, max_iterations: int) -> str:
    if state["evaluation"].sufficient:
        return "report"
    if state["iteration"] >= max_iterations:
        return "report"
    if not state["pending_queries"]:
        return "report"
    return "research"


def build_graph(agent, max_iterations: int):
    g = StateGraph(ResearchState)
    g.add_node("planner", agent.planner)
    g.add_node("researcher", agent.researcher)
    g.add_node("analyzer", agent.analyzer)
    g.add_node("evaluator", agent.evaluator)
    g.add_node("reporter", agent.reporter)

    g.set_entry_point("planner")
    g.add_edge("planner", "researcher")
    g.add_conditional_edges("researcher", route_after_researcher, {"analyze": "analyzer", "end": END})
    g.add_edge("analyzer", "evaluator")
    g.add_conditional_edges(
        "evaluator",
        lambda s: route_after_evaluator(s, max_iterations),
        {"research": "researcher", "report": "reporter"},
    )
    g.add_edge("reporter", END)
    return g.compile()
