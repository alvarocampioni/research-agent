from __future__ import annotations

from app.models.search import SearchArgs, SearchResult


class ToolGuard:
    def __init__(self, max_results_per_query: int = 5, max_queries_per_run: int | None = None):
        self.max_results_per_query = max_results_per_query
        self.max_queries_per_run = max_queries_per_run

    def validate_search_args(self, query: str, max_results: int | None = None) -> SearchArgs:
        resolved = max_results if max_results is not None else self.max_results_per_query
        return SearchArgs(query=query, max_results=min(resolved, self.max_results_per_query))

    def dedupe_queries(self, queries: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for q in queries:
            key = " ".join(q.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(q)
        return out[: self.max_queries_per_run] if self.max_queries_per_run is not None else out

    def limit_results(self, results: list[SearchResult]) -> list[SearchResult]:
        return results[: self.max_results_per_query]
