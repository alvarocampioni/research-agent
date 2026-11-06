import httpx
import pytest

from app.models.search import SearchArgs
from app.tools.search import MAX_CONTENT_CHARS, SearchError, WebSearchTool


def make_tool(handler, key="k"):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return WebSearchTool(api_key=key, client=client)


def test_parses_results_and_truncates_content():
    def handler(request):
        return httpx.Response(200, json={"results": [
            {"title": "A", "url": "https://a.com", "content": "x" * 5000, "score": 0.9},
            {"title": "sem url", "content": "ignorado"},
        ]})
    out = make_tool(handler).search(SearchArgs(query="langchain"))
    assert len(out) == 1
    assert out[0].url == "https://a.com"
    assert len(out[0].content) == MAX_CONTENT_CHARS


def test_respects_max_results():
    def handler(request):
        items = [{"title": str(i), "url": f"https://s{i}.com", "content": "c"} for i in range(10)]
        return httpx.Response(200, json={"results": items})
    assert len(make_tool(handler).search(SearchArgs(query="langchain", max_results=3))) == 3


def test_http_error_becomes_search_error():
    with pytest.raises(SearchError):
        make_tool(lambda r: httpx.Response(500)).search(SearchArgs(query="langchain"))


def test_missing_api_key():
    with pytest.raises(SearchError):
        make_tool(lambda r: httpx.Response(200, json={}), key="").search(SearchArgs(query="langchain"))
