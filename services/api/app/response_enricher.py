import re
from typing import Any, Dict, List


ACTIONABLE_VERBS = (
    "check",
    "confirm",
    "retry",
    "increase",
    "validate",
    "monitor",
    "restart",
    "review",
    "inspect",
    "reprocess",
)


def _extract_service_name(text: str) -> str | None:
    match = re.search(
        r"\b([a-z0-9]+(?:[-_][a-z0-9]+)*(?:[-_]service))\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1)


def _extract_answer_steps(answer_text: str) -> List[str]:
    numbered_steps = [
        step.strip()
        for step in re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+(.*)$", answer_text)
        if step.strip()
    ]
    if numbered_steps:
        return numbered_steps

    sentence_candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer_text)
        if sentence.strip()
    ]
    return [
        sentence
        for sentence in sentence_candidates
        if sentence.lower().startswith(ACTIONABLE_VERBS)
    ]


def _score_timeout_action(step: str, service: str | None) -> int:
    lowered = step.lower()
    score = 0

    if lowered.startswith("check"):
        score += 5
    if lowered.startswith("retry"):
        score += 5
    if lowered.startswith("temporarily increase") or lowered.startswith("increase"):
        score += 4
    if any(
        term in lowered for term in ("504", "timeout", "readtimeout", "connecttimeout")
    ):
        score += 3
    if "exponential backoff" in lowered:
        score += 3
    if "downstream" in lowered:
        score += 2
    if any(term in lowered for term in ("health", "latency", "restart", "queue")):
        score += 1
    if service and service.lower() in lowered:
        score += 2
    if lowered.startswith("confirm the affected service"):
        score -= 3
    if lowered.startswith("ensure that"):
        score -= 2
    if lowered.startswith("validate"):
        score -= 1
    if lowered.startswith(ACTIONABLE_VERBS):
        score += 1

    return score


def compute_confidence(sources: List[Dict[str, Any]]) -> float:
    if not sources:
        return 0.0
    return round(sources[0].get("similarity", 0.0), 2)


def infer_recommended_actions(
    query: str,
    answer_text: str,
) -> List[str]:
    text = f"{query} {answer_text}".lower()
    actions: List[str] = []

    service = _extract_service_name(f"{query} {answer_text}")
    timeout_detected = any(
        term in text for term in ("timeout", "504", "readtimeout", "connecttimeout")
    )

    if timeout_detected:
        answer_steps = _extract_answer_steps(answer_text)
        scored_steps = [
            (step, _score_timeout_action(step, service)) for step in answer_steps
        ]
        ranked_steps = [
            step
            for step, score in sorted(
                scored_steps,
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0
        ]
        timeout_actions = ranked_steps
        actions.extend(timeout_actions)

        if not actions:
            target = service or "the affected service"
            actions.extend(
                [
                    f"Check {target} for 504, ReadTimeout, or ConnectTimeout errors",
                    f"Retry requests to {target} with exponential backoff",
                    (
                        "Validate downstream dependency health and consider "
                        "a temporary timeout increase"
                    ),
                ]
            )

    if "latency" in text and not timeout_detected:
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
            ]
        )
    if "file" in text and "failed" in text and "download" in text:
        file_name: str = None
        location: str = None

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
                " Use payload recommended in response, if available",
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
        "recommended_actions": infer_recommended_actions(query, answer_text),
    }
