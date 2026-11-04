import httpx

from app.models.search import SearchArgs, SearchResult

TAVILY_URL = "https://api.tavily.com/search"
MAX_CONTENT_CHARS = 1500


class SearchError(Exception):
    pass

class WebSearchTool:
    def __init__(self, api_key: str, client: httpx.Client | None = None, timeout: float = 20.0):
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def search(self, args: SearchArgs) -> list[SearchResult]:
        if not self._api_key:
            raise SearchError("SEARCH_API_KEY não configurada")
        try:
            resp = self._client.post(
                TAVILY_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"query": args.query, "max_results": args.max_results},
            )
            resp.raise_for_status()
            raw = resp.json().get("results", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"falha na busca: {exc}") from exc

        results = []
        for item in raw[: args.max_results]:
            if not item.get("url"):
                continue
            results.append(SearchResult(
                title=item.get("title") or item["url"],
                url=item["url"],
                content=(item.get("content") or "")[:MAX_CONTENT_CHARS],
                score=item.get("score") or 0.0,
            ))
        return results
