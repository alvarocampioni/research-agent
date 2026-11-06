"""Teste manual da busca"""
import sys

from app.config import get_settings
from app.models.search import SearchArgs
from app.tools.search import WebSearchTool

s = get_settings()
tool = WebSearchTool(s.search_api_key)
for r in tool.search(SearchArgs(query=sys.argv[1], max_results=s.search_results_per_query)):
    print(f"- {r.title}\n  {r.url}\n  {r.content[:150]}...\n")
