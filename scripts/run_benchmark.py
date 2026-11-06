"""Executa o benchmark dos dois agentes sobre o dataset compartilhado."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.evaluation.runner import load_dataset, run_benchmark


def main() -> None:
    settings = get_settings()
    dataset = load_dataset(Path(__file__).resolve().parent.parent / "app" / "evaluation" / "dataset.json")
    results = run_benchmark(settings, dataset)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
