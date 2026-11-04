"""Separa achados citados de inferências para permitir validação de citações."""
from pydantic import BaseModel, Field


class Source(BaseModel):
    id: int
    title: str
    url: str


class Finding(BaseModel):
    """Afirmação sustentada por fontes recuperadas."""
    text: str
    source_ids: list[int] = Field(min_length=1)


class Report(BaseModel):
    question: str
    summary: str
    findings: list[Finding]
    inferences: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[Source]
