"""Small repeatable evaluation runner for retrieval and grounded answers."""

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

from .pipeline import HackathonRAG


def load_cases(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Evaluation file must contain a JSON list")
    return value


def evaluate(rag: HackathonRAG, cases_path: Path) -> Dict[str, Any]:
    cases = load_cases(cases_path)
    details = []
    passed = 0
    retrieval_recall_at_3 = []
    retrieval_recall_at_5 = []
    reciprocal_ranks = []
    abstention_checks = []
    citation_checks = []
    citation_quote_checks = []
    latencies = []
    for case in cases:
        result = rag.ask(case["question"]).to_dict()
        latencies.append(result["latency_ms"])
        problems = []
        expected_supported = case.get("supported")
        if expected_supported is not None and result["supported"] != expected_supported:
            problems.append(f"supported={result['supported']} expected={expected_supported}")
        if expected_supported is not None:
            abstention_checks.append(result["supported"] == expected_supported)
        answer_folded = result["answer"].casefold()
        missing = [term for term in case.get("must_include", []) if term.casefold() not in answer_folded]
        if missing:
            problems.append(f"missing={missing}")
        forbidden = [term for term in case.get("must_not_include", []) if term.casefold() in answer_folded]
        if forbidden:
            problems.append(f"forbidden={forbidden}")
        expected_source = case.get("expected_source")
        expected_sources = set(case.get("expected_sources", []))
        if expected_source:
            expected_sources.add(expected_source)
        sources = {citation["source"] for citation in result["citations"]}
        if expected_source and expected_source not in sources:
            problems.append(f"expected_source={expected_source!r}, got={sorted(sources)}")
        if case.get("expected_sources"):
            missing_sources = expected_sources - sources
            if missing_sources:
                problems.append(f"missing_expected_sources={sorted(missing_sources)}")
        if expected_sources:
            retrieved_sources = [item["source"] for item in result["retrieved"]]
            retrieval_recall_at_3.append(len(expected_sources & set(retrieved_sources[:3])) / len(expected_sources))
            retrieval_recall_at_5.append(len(expected_sources & set(retrieved_sources[:5])) / len(expected_sources))
            first_rank = next(
                (rank for rank, source in enumerate(retrieved_sources, 1) if source in expected_sources),
                None,
            )
            reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
            if result["supported"]:
                citation_checks.append(bool(sources) and sources.issubset(expected_sources))
        citation_quotes_valid = all(
            isinstance(citation.get("quote"), str) and bool(citation["quote"].strip())
            for citation in result["citations"]
        )
        if result["citations"] and not citation_quotes_valid:
            problems.append("one or more citations lack a verified quote")
        if result["citations"]:
            citation_quote_checks.append(citation_quotes_valid)
        if not problems:
            passed += 1
        details.append({"question": case["question"], "problems": problems, "result": result})
    sorted_latencies = sorted(latencies)

    def percentile(values: List[int], fraction: float) -> int:
        if not values:
            return 0
        index = max(0, math.ceil(fraction * len(values)) - 1)
        return values[index]

    return {
        "passed": passed,
        "total": len(cases),
        "metrics": {
            "recall_at_3": round(statistics.mean(retrieval_recall_at_3), 4) if retrieval_recall_at_3 else None,
            "recall_at_5": round(statistics.mean(retrieval_recall_at_5), 4) if retrieval_recall_at_5 else None,
            "mrr": round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else None,
            "abstention_accuracy": round(statistics.mean(abstention_checks), 4) if abstention_checks else None,
            "citation_source_accuracy": round(statistics.mean(citation_checks), 4) if citation_checks else None,
            "citation_quote_validity": (
                round(statistics.mean(citation_quote_checks), 4) if citation_quote_checks else None
            ),
            "latency_ms": {
                "p50": percentile(sorted_latencies, 0.50),
                "p95": percentile(sorted_latencies, 0.95),
                "maximum": max(sorted_latencies) if sorted_latencies else 0,
            },
        },
        "details": details,
    }
