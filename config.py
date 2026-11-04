import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    llm_price_input_per_m: float
    llm_price_output_per_m: float
    search_api_key: str
    search_results_per_query: int
    max_searches: int
    max_iterations: int
    max_cost_usd: float
    timeout_seconds: float
    database_path: str
    judge_model: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_api_key=_env("LLM_API_KEY", ""),
            llm_base_url=_env("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=_env("LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=float(_env("LLM_TEMPERATURE", "0")),
            llm_price_input_per_m=float(_env("LLM_PRICE_INPUT_PER_M", "0.15")),
            llm_price_output_per_m=float(_env("LLM_PRICE_OUTPUT_PER_M", "0.60")),
            search_api_key=_env("SEARCH_API_KEY", ""),
            search_results_per_query=int(_env("SEARCH_RESULTS_PER_QUERY", "5")),
            max_searches=int(_env("MAX_SEARCHES", "6")),
            max_iterations=int(_env("MAX_ITERATIONS", "3")),
            max_cost_usd=float(_env("MAX_COST_USD", "0.10")),
            timeout_seconds=float(_env("TIMEOUT_SECONDS", "120")),
            database_path=_env("DATABASE_PATH", "data/research.db"),
            judge_model=_env("JUDGE_MODEL", ""),
        )


def get_settings() -> Settings:
    return Settings.from_env()
