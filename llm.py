"""Cliente de LLM: uma chamada com saída estruturada (Pydantic) + registro de tokens.

É a única classe que fala com o LangChain. O resto do código só conhece
`generate(schema, system, user, step)`, o que facilita o teste com um fake.
"""
from typing import TypeVar

from pydantic import BaseModel

from app.config import Settings
from app.models.run import LLMCall
from langchain_openai import ChatOpenAI

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Falha na chamada ao LLM ou saida que nao respeita o schema."""


class LLMClient:
    def __init__(self, settings: Settings, chat_model=None):
        self._settings = settings
        if chat_model is None:
            chat_model = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=settings.llm_temperature,
                timeout=settings.timeout_seconds,
                max_retries=2,
            )
        self._model = chat_model
        self.calls: list[LLMCall] = []

    def generate(self, schema: type[T], system: str, user: str, step: str) -> T:
        structured = self._model.with_structured_output(schema, include_raw=True)
        try:
            out = structured.invoke([("system", system), ("human", user)])
        except Exception as exc:
            raise LLMError(f"falha no LLM em '{step}': {exc}") from exc

        usage = getattr(out["raw"], "usage_metadata", None) or {}
        self.calls.append(LLMCall(
            step=step,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        ))
        if out.get("parsed") is None:
            raise LLMError(f"saída inválida em '{step}': {out.get('parsing_error')}")
        return out["parsed"]

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.calls)

    @property
    def estimated_cost_usd(self) -> float:
        s = self._settings
        tin = sum(c.input_tokens for c in self.calls)
        tout = sum(c.output_tokens for c in self.calls)
        return (tin * s.llm_price_input_per_m + tout * s.llm_price_output_per_m) / 1_000_000
