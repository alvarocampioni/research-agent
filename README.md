# ai-research-agent

A small web research agent, built twice. One version searches once and writes the report. The other plans, searches, checks whether it found enough, and goes back for more if it didn't. Both run on the same harness with the same model, search tool and limits, and a benchmark compares them.

I wanted to know if the extra planning and self-checking pays for itself. On the 20-question benchmark below it didn't: the iterative agent used about three times the tokens and wrote reports that were no better.

The questions and the prompts are in Portuguese, so the reports come out in Portuguese.

## How it works

You ask a question and get back a report: a summary, findings that each point to numbered sources, a separate list of the model's own inferences, and a list of limitations.

**Agent A** (`app/agents/simple_agent.py`) asks the model for a few search queries, runs them, and writes the report from whatever comes back. Two model calls in total.

**Agent B** (`app/agents/interactive_agent.py`) is a LangGraph graph: planner, researcher, analyzer, evaluator, reporter. After the analysis the evaluator decides if the evidence is enough. If it isn't, it proposes new queries and the graph goes back to the researcher. The loop ends when the evaluator is satisfied, when there are no new queries, or at `MAX_ITERATIONS`.

Both agents sit on the same harness (`app/harness/`): a cap on searches per run, deduplication of repeated queries, a limit on results per query, a cost ceiling, a timeout, and per-call telemetry. The cost ceiling is an estimate from token counts and the prices you put in `.env`.

### Citations

The model only cites source numbers. `finalize_report` in `app/agents/base.py` drops numbers that don't exist, moves any claim left without a valid source into the inferences list, and fills in URLs from the search results, never from the model. A report can't contain a link the search tool didn't return. That says nothing about whether the page actually supports the claim, which is what the claim check in the evaluation section is for.

Search results go into the prompt as data, and the system prompt tells the model to ignore any instructions found inside them.

### API

There's a small FastAPI app that stores runs in SQLite: `POST /research` (body: `question`, `agent` = `a` or `b`), `GET /research`, `GET /research/{id}`, `POST /evaluation/run`, `GET /evaluation/results` and `GET /health`.

## Running it

Python 3.12 or newer.

```
pip install -e ".[dev]"
cp .env.example .env
```

Fill in `.env`. The LLM can be anything with an OpenAI-compatible API (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`), and `SEARCH_API_KEY` is a Tavily key. Set the two price variables if you want the cost estimate to mean anything. For a local model through Ollama, point `LLM_BASE_URL` at `http://localhost:11434/v1`. Ollama's default context is 4096 tokens and it cuts prompts off without telling you, so raise it. I used a 16k variant.

```
python -m scripts.run_agent a "your question"
python -m scripts.run_agent b "your question"
python -m scripts.benchmark --limit 3 --no-judge
uvicorn app.api:app --reload
pytest
```

`scripts.benchmark` prints one progress line per run and appends each result to `benchmark_rows.jsonl` as it goes. Ctrl+C is safe: run it again and it picks up where it stopped. `--no-judge` skips the LLM judging, which is the slow part with a local model. `pytest` runs offline against a fake LLM and fake search.

## Evaluation

The dataset is `app/evaluation/dataset.json`: 20 questions I wrote about common technology comparisons (PostgreSQL vs MongoDB, REST vs GraphQL, and so on), each with a few expected criteria and key facts. The weak spot is that the model already knows most of these answers, so search isn't strictly needed.

Every report goes through three checks.

1. **Deterministic checks.** Report structure, URL format, whether the run stayed inside its limits, and a keyword overlap against the expected criteria. The overlap strips accents and stopwords, compares word stems, and counts a criterion as covered when enough of them show up. It's a rough signal and nothing more.
2. **LLM judge.** Scores 1 to 5 on relevance, faithfulness, completeness and citation accuracy. It writes a short justification first, then an overall verdict (poor, fair or good), then the four numbers, and the numbers have to agree with the verdict.
3. **Claim check.** A second LLM call reads each finding next to the text of the sources it cites and says whether that text supports it.

The judge and the claim check use the same model that writes the reports, unless `JUDGE_MODEL` is set.

## Results

gpt-oss 20B running locally in Ollama with a 16k context, Tavily for search, the same model as judge, one run per question. A different model will give different numbers.

| | Agent A (linear) | Agent B (iterative) |
|---|---|---|
| Reports produced | 20 / 20 | 19 / 20 |
| Tokens per question | 5.7k | 17.1k |
| Time per question | 41 s | 126 s |
| Judge score (mean of the four, 1-5) | 4.05 | 4.06 |
| Findings supported by the cited text | 100 / 126 (79%) | 86 / 131 (66%) |

Token counts leave out the judging calls.

B used about 3x the tokens and 3x the time. On the 19 questions both agents answered, the judge scored them the same on average (difference 0.00). The plausible range runs from B being 0.65 points worse to 0.63 better.
A's findings were backed by the cited text more often (79% against 66%, about 15 points per question on average). The range goes from B being 33 points worse to 2 points better.

For this kind of question and this model, searching once was the better deal.

## Things that broke along the way

- Cost and token counts were read from an LLM client shared by the whole benchmark, so each question inherited everything before it. They're now counted per run.
- The first judge put the justification last in its JSON and never said what the 1-5 scale meant. On 8 of 10 reports it gave 1/1/1/1 next to positive justifications ("9/10"). Reordering the fields and adding the verdict check fixed it.

## Layout

```
app/agents/      simple_agent.py, interactive_agent.py, graph.py, base.py
app/harness/     controller, budget, tool guard, telemetry
app/tools/       Tavily search
app/evaluation/  dataset, deterministic checks, judge, claim check, comparison, runner
app/api/         FastAPI app
scripts/         run_agent, search_once, benchmark
tests/
```
