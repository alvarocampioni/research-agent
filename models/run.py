from pydantic import BaseModel, Field

from app.models.report import Report


class QueryPlan(BaseModel):
    queries: list[str] = Field(min_length=1)


class FindingDraft(BaseModel):
    text: str
    source_ids: list[int] = Field(default_factory=list)


class ReportDraft(BaseModel):
    summary: str
    findings: list[FindingDraft]
    inferences: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class RetrievedSource(BaseModel):
    id: int
    title: str
    url: str
    content: str


class LLMCall(BaseModel):
    step: str
    input_tokens: int = 0
    output_tokens: int = 0


class AgentResult(BaseModel):
    agent: str
    question: str
    report: Report | None = None
    queries: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedSource] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    iterations: int = 1
    errors: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    info_needed: list[str] = Field(min_length=1)
    queries: list[str] = Field(min_length=1)


class Analysis(BaseModel):
    relevant_findings: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    sufficient: bool
    reason: str
    new_queries: list[str] = Field(default_factory=list)
