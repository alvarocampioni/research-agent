from pydantic import BaseModel, Field, field_validator


class SearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=200)
    max_results: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, v: str) -> str:
        v = " ".join(v.split()).lower()
        if len(v) < 3:
            raise ValueError("query muito curta")
        return v


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float = 0.0
