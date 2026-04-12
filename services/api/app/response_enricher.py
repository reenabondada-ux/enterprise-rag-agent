import re
from typing import Any, Dict, List


def compute_confidence(sources: List[Dict[str, Any]]) -> float:
    if not sources:
        return 0.0
    return round(sources[0].get("similarity", 0.0), 2)


def infer_recommended_actions(
    query: str,
    answer_text: str,
    sources: List[Dict[str, Any]],
) -> List[str]:
    """
    Temporary rule-based recommendations.
    Later replace with runbook metadata or LLM-generated action list.
    """
    text = f"{query} {answer_text}".lower()
    actions: List[str] = []
    if "latency" in text:
        actions.extend(
            [
                "Check cache hit rate",
                "Inspect DB connection pool",
                "Review recent deployment changes",
            ]
        )
    if "timeout" in text:
        actions.extend(
            [
                "Check downstream service health",
                "Inspect retry storms",
            ]
        )
    if "error" in text:
        actions.extend(
            [
                "Review error logs",
                "Check recent configuration changes",
            ]
        )
    if "file" in text and "failed" in text and "download" in text:
        service: str = None
        file_name: str = None
        location: str = None

        possible_service_names = re.search(
            r"\b([a-z0-9_]+_service)\b", text, flags=re.IGNORECASE
        )
        if possible_service_names:
            service = possible_service_names.group(1).lower()

        possible_file_names = re.search(
            r"([A-Za-z0-9._-]+\.(?:csv|json|txt|pdf|parquet))", text
        )
        if possible_file_names:
            file_name = possible_file_names.group(1)

        possible_locations = re.search(
            r"location:\s*([^\s]+)", answer_text, flags=re.IGNORECASE
        )
        if possible_locations:
            location = possible_locations.group(1)

        actions.extend(
            [
                f"Check {service} logs for errors related to {file_name}",
                f"Confirm {file_name} exists in the expected {location}",
                f"Reprocess {service} to download {file_name}."
                f"Use payload recommended in response, if available",
            ]
        )

    if not actions:
        actions.append("Review top retrieved sources and identify next action")

    # Remove duplicates while preserving order
    deduped = []
    for item in actions:
        if item not in deduped:
            deduped.append(item)
    return deduped[:3]


def enrich_answer(
    query: str, answer_text: str, sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "query": query,
        "summary": answer_text,
        "sources": sources,
        "confidence": compute_confidence(sources),
        "recommended_actions": infer_recommended_actions(query, answer_text, sources),
    }
