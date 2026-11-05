from typing import TypedDict

from app.models.report import Report
from app.models.run import Analysis, Evaluation, RetrievedSource


class ResearchState(TypedDict):
    question: str
    info_needed: list[str]
    pending_queries: list[str]
    attempted: list[str]
    queries_run: list[str]
    retrieved: dict[str, RetrievedSource]
    analysis: Analysis | None
    evaluation: Evaluation | None
    iteration: int
    errors: list[str]
    report: Report | None


def initial_state(question: str) -> ResearchState:
    return ResearchState(
        question=question, info_needed=[], pending_queries=[], attempted=[], queries_run=[],
        retrieved={}, analysis=None, evaluation=None, iteration=0, errors=[], report=None)
