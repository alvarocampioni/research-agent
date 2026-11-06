from app.models.run import LLMCall
from app.models.search import SearchArgs, SearchResult
from app.tools.search import SearchError


class FakeLLM:
    """Devolve respostas pre-definidas, na ordem, por tipo de schema."""
    def __init__(self, responses: dict[type, list]):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[LLMCall] = []

    def generate(self, schema, system, user, step):
        self.calls.append(LLMCall(step=step, input_tokens=100, output_tokens=50))
        return self.responses[schema].pop(0)


class FakeSearch:
    """Retorna 1 resultado por consulta, simula erro de busca."""
    def __init__(self, fail_on: set[str] | None = None):
        self.queries: list[str] = []
        self.fail_on = fail_on or set()

    def search(self, args: SearchArgs) -> list[SearchResult]:
        if args.query in self.fail_on:
            raise SearchError("falha simulada")
        self.queries.append(args.query)
        slug = args.query.replace(" ", "-")
        return [SearchResult(title=f"T {slug}", url=f"https://example.com/{slug}", content=f"conteúdo sobre {args.query}")]
