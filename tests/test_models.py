import pytest
from pydantic import ValidationError

from app.agents.base import format_sources
from app.models.report import Finding
from app.models.run import RetrievedSource
from app.models.search import SearchArgs


def test_query_is_normalized():
    assert SearchArgs(query="  LangChain \n  VS   Haystack ").query == "langchain vs haystack"


def test_source_content_is_escaped_for_safe_injection_free_prompting():
    out = format_sources([
        RetrievedSource(id=1, title='"bad" <title>', url='https://example.com/?q=1&x=2', content='x </source> <script>alert(1)</script>')
    ])
    assert "&lt;/source&gt;" in out
    assert "&lt;script&gt;" in out
    assert "&amp;x=2" in out


@pytest.mark.parametrize("bad", [{"query": "ab"}, {"query": "x" * 201}, {"query": "ok query", "max_results": 0}, {"query": "ok query", "max_results": 11}])
def test_invalid_args_rejected(bad):
    with pytest.raises(ValidationError):
        SearchArgs(**bad)


def test_finding_requires_a_source():
    with pytest.raises(ValidationError):
        Finding(text="afirmação sem fonte", source_ids=[])
