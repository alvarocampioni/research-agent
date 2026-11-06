import json
from app.config import get_settings
from app.evaluation.comparison import compare_agents
from app.evaluation.runner import run_benchmark

res = run_benchmark(get_settings())
with open("benchmark_raw.json", "w", encoding="utf-8") as f:
    json.dump(res, f, default=str, ensure_ascii=False, indent=1)
print(json.dumps(compare_agents(res), indent=2))