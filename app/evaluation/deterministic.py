from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

from app.models.report import Report


REQUIRED_SECTIONS = ["summary", "findings", "inferences", "limitations", "sources"]


def validate_report_structure(report: Report | None) -> dict[str, object]:
    if report is None:
        return {"valid": False, "issues": ["relatório ausente"]}
    issues: list[str] = []
    if not report.summary.strip():
        issues.append("summary vazia")
    if not isinstance(report.findings, list):
        issues.append("findings inválido")
    if not isinstance(report.inferences, list):
        issues.append("inferences inválido")
    if not isinstance(report.limitations, list):
        issues.append("limitations inválido")
    if not isinstance(report.sources, list):
        issues.append("sources inválido")
    return {"valid": not issues, "issues": issues}


def validate_citations(report: Report | None) -> dict[str, object]:
    """Checagem ESTRUTURAL (URLs bem formadas, ids existentes)."""
    if report is None:
        return {"valid": False, "valid_urls": 0, "total_urls": 0, "issues": ["relatório ausente"]}

    source_urls = [s.url for s in report.sources]
    valid_urls = 0
    issues: list[str] = []
    for idx, url in enumerate(source_urls, start=1):
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            issues.append(f"URL {idx} inválida: {url}")
            continue
        valid_urls += 1

    valid_source_ids = {s.id for s in report.sources}

    for finding in report.findings:
        if not finding.source_ids:
            issues.append(f"finding sem source_ids: {finding.text[:80]}")
            continue
        if not all(sid in valid_source_ids for sid in finding.source_ids):
            issues.append(f"finding referencia ids inexistentes: {finding.text[:80]}")

    return {
        "valid": not issues,
        "valid_urls": valid_urls,
        "total_urls": len(source_urls),
        "issues": issues,
    }


def check_execution_constraints(run_result: dict) -> dict[str, object]:
    issues: list[str] = []
    iterations = run_result.get("iterations", 0)
    max_iterations = run_result.get("max_iterations")
    if max_iterations is not None and iterations > max_iterations:
        issues.append("ultrapassou max_iterations")
    max_cost = run_result.get("max_cost_usd")
    cost = run_result.get("estimated_cost_usd")
    if max_cost is not None and cost is not None and cost > max_cost:
        issues.append("ultrapassou orçamento")
    if run_result.get("duplicate_queries"):
        issues.append("consultas duplicadas")
    return {"valid": not issues, "issues": issues}


COVERAGE_THRESHOLD = 0.6
_STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sobre", "entre", "e", "ou", "que", "se", "ao", "aos", "como", "mais", "menos",
    "muito", "versus", "vs", "sao", "ser", "foi", "the", "of", "and", "or", "in", "on", "for", "to", "with", "between",
}


def _stems(text: str) -> set[str]:
    """Palavras-chave normalizadas: sem acento, sem stopwords, truncadas em 5 letras (relacional ~ relacionais)."""
    norm = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return {w[:5] for w in re.findall(r"[a-z0-9]+", norm) if w not in _STOPWORDS and len(w) > 2}


def _coverage(phrase: str, text_stems: set[str]) -> float:
    keys = _stems(phrase)
    return len(keys & text_stems) / len(keys) if keys else 0.0


def check_expected_criteria(report: Report | None, expected_criteria: list[str], important_facts: list[str]) -> dict[str, object]:
    if report is None:
        return {"valid": False, "criteria_coverage": 0.0, "facts_coverage": 0.0,
                "matched_criteria": [], "missed_criteria": list(expected_criteria), "matched_facts": []}
    text = "\n".join([report.summary, *(f.text for f in report.findings), *report.inferences])
    stems = _stems(text)
    matched = [c for c in expected_criteria if _coverage(c, stems) >= COVERAGE_THRESHOLD]
    matched_facts = [f for f in important_facts if _coverage(f, stems) >= COVERAGE_THRESHOLD]
    criteria_rate = len(matched) / len(expected_criteria) if expected_criteria else 1.0
    facts_rate = len(matched_facts) / len(important_facts) if important_facts else 1.0
    return {
        "valid": criteria_rate >= 0.5,
        "criteria_coverage": round(criteria_rate, 4),
        "facts_coverage": round(facts_rate, 4),
        "matched_criteria": matched,
        "missed_criteria": [c for c in expected_criteria if c not in matched],
        "matched_facts": matched_facts,
    }


def run_deterministic_checks(run_result: dict, expected_criteria: list[str], important_facts: list[str]) -> dict[str, object]:
    report = run_result.get("report")
    return {
        "citation_validation": validate_citations(report),
        "report_structure": validate_report_structure(report),
        "execution_constraints": check_execution_constraints(run_result),
        "expected_criteria_coverage": check_expected_criteria(report, expected_criteria, important_facts),
    }
