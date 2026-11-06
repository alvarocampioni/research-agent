import sys

from app.agents.interactive_agent import InteractiveAgent
from app.agents.simple_agent import SimpleAgent
from app.config import get_settings
from app.llm import LLMClient
from app.tools.search import WebSearchTool

AGENTS = {"a": SimpleAgent, "b": InteractiveAgent}


def main() -> None:
    which, question = sys.argv[1].lower(), sys.argv[2]
    s = get_settings()
    llm = LLMClient(s)
    result = AGENTS[which](llm, WebSearchTool(s.search_api_key), s).run(question)

    r = result.report
    if r:
        print(f"# {r.question}\n\n{r.summary}\n\n## Achados (das fontes)")
        for f in r.findings:
            print(f"- {f.text} {[*f.source_ids]}")
        print("\n## Inferências do modelo")
        for i in r.inferences:
            print(f"- {i}")
        print("\n## Limitações")
        for l in r.limitations:
            print(f"- {l}")
        print("\n## Fontes")
        for src in r.sources:
            print(f"[{src.id}] {src.title} — {src.url}")
    print(f"\n--- consultas={len(result.queries)} fontes={len(result.retrieved)} iterações={result.iterations} "
          f"tokens={llm.total_tokens} custo~US${llm.estimated_cost_usd:.4f} erros={result.errors}")


if __name__ == "__main__":
    main()
