from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx

FILE_SOURCE_RE = re.compile(r"\.(?:md|pdf|docx?|txt|xlsx?|csv|json|ya?ml)\b", re.IGNORECASE)
CITATION_MARKER_RE = re.compile(r"(?im)^\s*citations?\s*:|\b[A-Za-z0-9_-]+#chunk\d+\b")
MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
}
BENCHMARK_UPLOAD_COLLECTION_ID = "capability-benchmark-uploads"


@dataclass(frozen=True)
class CapabilityCase:
    test_id: str
    difficulty: str
    capability_area: str
    query: str
    guided_query: str
    expected_answer: str
    expected_sources: str
    collection_id: str
    graph_id: str
    success_criteria: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "CapabilityCase":
        return cls(
            test_id=str(row.get("test_id") or "").strip(),
            difficulty=str(row.get("difficulty") or "").strip(),
            capability_area=str(row.get("capability_area") or "").strip(),
            query=str(row.get("query") or "").strip(),
            guided_query=str(row.get("guided_query") or "").strip(),
            expected_answer=str(row.get("expected_answer") or "").strip(),
            expected_sources=str(row.get("expected_sources") or "").strip(),
            collection_id=str(row.get("collection_id") or "").strip(),
            graph_id=str(row.get("graph_id") or "").strip(),
            success_criteria=str(row.get("success_criteria") or "").strip(),
        )

    def prompt_for(self, prompt_field: str) -> str:
        requested = str(getattr(self, prompt_field, "") or "").strip()
        return requested or self.guided_query or self.query


@dataclass(frozen=True)
class JudgeScore:
    answer_correct: bool
    source_correct: bool
    reason: str = ""
    available: bool = True


@dataclass
class CapabilityResult:
    test_id: str
    prompt_variant: str
    difficulty: str
    capability_area: str
    prompt: str
    expected_answer: str
    expected_sources: str
    answer: str = ""
    error: str = ""
    deterministic_answer_correct: bool = False
    deterministic_source_correct: bool = False
    judge_answer_correct: bool | None = None
    judge_source_correct: bool | None = None
    judge_reason: str = ""
    needs_review: bool = False
    passed: bool = False
    latency_seconds: float = 0.0
    status_code: int | None = None
    stage_timings_ms: str = ""
    slowest_stage: str = ""
    budget_exhausted: bool = False
    retrieval_mode: str = ""
    tool_calls_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilitySummary:
    total: int
    passed: int
    failed: int
    needs_review: int
    errors: int
    accuracy: float
    deterministic_passed: int = 0
    deterministic_accuracy: float = 0.0
    judge_scored: int = 0
    judge_passed: int = 0
    judge_accuracy: float = 0.0
    judge_scored_accuracy: float = 0.0
    judge_delta_passes: int = 0
    judge_delta_accuracy: float = 0.0
    judge_new_passes: list[str] = field(default_factory=list)
    judge_new_failures: list[str] = field(default_factory=list)
    final_scorer: str = "deterministic"
    by_area: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_variant: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    stripped = re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold())
    return " ".join(stripped.split())


def _source_parts(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw or raw.casefold() == "none":
        return []
    return [item.strip() for item in raw.split("|") if item.strip()]


def _file_sources(value: str) -> list[str]:
    return [Path(item).name for item in _source_parts(value) if FILE_SOURCE_RE.search(item)]


def _rubric_expected(value: str) -> bool:
    normalized = str(value or "").strip().casefold()
    return normalized.startswith("the answer should") or normalized.startswith("answer should")


def deterministic_answer_match(answer: str, expected: str) -> bool:
    normalized_answer = _normalize_text(answer)
    normalized_expected = _normalize_text(expected)
    if not normalized_answer or not normalized_expected or _rubric_expected(expected):
        return False
    if normalized_expected in normalized_answer:
        return True
    expected_tokens = [
        token
        for token in normalized_expected.split()
        if (len(token) >= 3 or token.isdigit()) and token not in MATCH_STOPWORDS
    ]
    return bool(expected_tokens) and all(token in normalized_answer for token in expected_tokens)


def deterministic_source_match(answer: str, expected_sources: str) -> bool:
    expected = str(expected_sources or "").strip()
    if not expected or expected.casefold() == "none":
        return not CITATION_MARKER_RE.search(answer or "")
    file_sources = _file_sources(expected)
    if not file_sources:
        return True
    normalized_answer = _normalize_text(answer)
    return any(_normalize_text(source) in normalized_answer for source in file_sources)


def load_cases(
    suite_path: Path,
    *,
    difficulty: str = "Easy",
    test_ids: Sequence[str] | None = None,
) -> list[CapabilityCase]:
    wanted_ids = {str(item).strip() for item in (test_ids or []) if str(item).strip()}
    with suite_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cases = [CapabilityCase.from_row(row) for row in rows]
    if difficulty:
        cases = [case for case in cases if case.difficulty.casefold() == difficulty.casefold()]
    if wanted_ids:
        cases = [case for case in cases if case.test_id in wanted_ids]
    return cases


def _clean_collection_id(value: str) -> str:
    clean = str(value or "").strip()
    if clean in {"", "-", "*", "none"}:
        return ""
    return clean


def build_gateway_payload(
    case: CapabilityCase,
    *,
    prompt: str,
    model: str,
    prompt_variant: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "benchmark_test_id": case.test_id,
        "benchmark_prompt_variant": prompt_variant,
    }
    collection_id = _clean_collection_id(case.collection_id)
    if collection_id:
        metadata.update(
            {
                "upload_collection_id": BENCHMARK_UPLOAD_COLLECTION_ID,
                "kb_collection_id": collection_id,
                "requested_kb_collection_id": collection_id,
                "selected_kb_collection_id": collection_id,
                "available_kb_collection_ids": [collection_id],
                "search_collection_ids": [collection_id],
                "kb_collection_confirmed": True,
            }
        )
    if case.graph_id:
        metadata.update(
            {
                "graph_id": case.graph_id,
                "active_graph_ids": [case.graph_id],
                "selected_graph_ids": [case.graph_id],
            }
        )
    return {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "metadata": metadata,
    }


def conversation_id_for(case: CapabilityCase, *, run_id: str, prompt_variant: str) -> str:
    if case.test_id in {"MEM-E01", "MEM-E02"}:
        return f"capability-{run_id}-{prompt_variant}-memory"
    return f"capability-{run_id}-{prompt_variant}-{case.test_id.lower()}"


def user_id_for(case: CapabilityCase, *, run_id: str, prompt_variant: str) -> str:
    if case.test_id in {"MEM-E01", "MEM-E02"}:
        return f"capability-{run_id}-{prompt_variant}-memory-user"
    return f"capability-{run_id}-{prompt_variant}-{case.test_id.lower()}-user"


def build_gateway_headers(
    case: CapabilityCase,
    *,
    run_id: str,
    prompt_variant: str,
    token: str = "",
) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "X-Conversation-ID": conversation_id_for(case, run_id=run_id, prompt_variant=prompt_variant),
        "X-User-ID": user_id_for(case, run_id=run_id, prompt_variant=prompt_variant),
        "X-Request-ID": f"capability-{run_id}-{prompt_variant}-{case.test_id.lower()}",
    }
    collection_id = _clean_collection_id(case.collection_id)
    if collection_id:
        headers["X-Collection-ID"] = BENCHMARK_UPLOAD_COLLECTION_ID
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def extract_answer(response_payload: Mapping[str, Any]) -> str:
    choices = list(response_payload.get("choices") or [])
    if not choices:
        return ""
    message = dict(choices[0].get("message") or {})
    return str(message.get("content") or "")


def extract_rag_diagnostics(response_payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(response_payload.get("metadata") or {})
    summary = dict(metadata.get("rag_retrieval_summary") or {})
    timings = {
        str(key): float(value or 0.0)
        for key, value in dict(summary.get("stage_timings_ms") or {}).items()
        if str(key)
    }
    slowest_stage = ""
    if timings:
        name, elapsed = max(timings.items(), key=lambda item: item[1])
        slowest_stage = f"{name}:{elapsed:.0f}ms"
    return {
        "stage_timings_ms": json.dumps(timings, sort_keys=True) if timings else "",
        "slowest_stage": slowest_stage,
        "budget_exhausted": bool(summary.get("budget_exhausted", False)),
        "retrieval_mode": str(metadata.get("retrieval_mode") or summary.get("search_mode") or ""),
        "tool_calls_used": int(metadata.get("tool_calls_used") or summary.get("tool_calls_used") or 0),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def score_with_judge(judge_model: Any, case: CapabilityCase, answer: str) -> JudgeScore:
    prompt = (
        "Grade a benchmark answer. Return JSON only with keys "
        "answer_correct, source_correct, reason.\n\n"
        f"Test ID: {case.test_id}\n"
        f"Question: {case.query}\n"
        f"Expected answer: {case.expected_answer}\n"
        f"Expected sources: {case.expected_sources}\n"
        f"Success criteria: {case.success_criteria}\n"
        f"Actual answer:\n{answer}\n\n"
        "Use answer_correct=true only when the factual answer satisfies the expected answer. "
        "Use source_correct=true when citations/sources satisfy the expected sources, or when no sources are expected and none are fabricated."
    )
    response = judge_model.invoke(prompt, config={"callbacks": []})
    payload = _extract_json_object(getattr(response, "content", None) or str(response))
    if not payload:
        return JudgeScore(False, False, "Judge returned non-JSON output.", available=False)
    return JudgeScore(
        answer_correct=bool(payload.get("answer_correct")),
        source_correct=bool(payload.get("source_correct")),
        reason=str(payload.get("reason") or "").strip(),
        available=True,
    )


def build_optional_judge(*, dotenv_path: str = "") -> tuple[Any | None, str]:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv_path or None)
    except Exception:
        pass
    try:
        from agentic_chatbot_next.config import load_settings
        from agentic_chatbot_next.providers.factory import build_providers

        settings = load_settings(dotenv_path or None)
        return build_providers(settings).judge, ""
    except Exception as exc:
        return None, str(exc)


def score_case(
    case: CapabilityCase,
    *,
    prompt: str,
    answer: str,
    prompt_variant: str,
    judge_model: Any | None = None,
    error: str = "",
    status_code: int | None = None,
    latency_seconds: float = 0.0,
    diagnostics: Mapping[str, Any] | None = None,
) -> CapabilityResult:
    deterministic_answer_correct = deterministic_answer_match(answer, case.expected_answer)
    deterministic_source_correct = deterministic_source_match(answer, case.expected_sources)
    judge_score: JudgeScore | None = None
    if judge_model is not None and not error:
        try:
            judge_score = score_with_judge(judge_model, case, answer)
        except Exception as exc:
            judge_score = JudgeScore(False, False, str(exc), available=False)
    if judge_score is not None and judge_score.available:
        passed = judge_score.answer_correct and judge_score.source_correct
        needs_review = False
    elif error:
        passed = False
        needs_review = False
    elif _rubric_expected(case.expected_answer):
        passed = False
        needs_review = True
    else:
        passed = deterministic_answer_correct and deterministic_source_correct
        needs_review = False
    return CapabilityResult(
        test_id=case.test_id,
        prompt_variant=prompt_variant,
        difficulty=case.difficulty,
        capability_area=case.capability_area,
        prompt=prompt,
        expected_answer=case.expected_answer,
        expected_sources=case.expected_sources,
        answer=answer,
        error=error,
        deterministic_answer_correct=deterministic_answer_correct,
        deterministic_source_correct=deterministic_source_correct,
        judge_answer_correct=(judge_score.answer_correct if judge_score and judge_score.available else None),
        judge_source_correct=(judge_score.source_correct if judge_score and judge_score.available else None),
        judge_reason=(judge_score.reason if judge_score else ""),
        needs_review=needs_review,
        passed=passed,
        latency_seconds=latency_seconds,
        status_code=status_code,
        stage_timings_ms=str((diagnostics or {}).get("stage_timings_ms") or ""),
        slowest_stage=str((diagnostics or {}).get("slowest_stage") or ""),
        budget_exhausted=bool((diagnostics or {}).get("budget_exhausted", False)),
        retrieval_mode=str((diagnostics or {}).get("retrieval_mode") or ""),
        tool_calls_used=int((diagnostics or {}).get("tool_calls_used") or 0),
    )


def summarize_results(results: Sequence[CapabilityResult]) -> CapabilitySummary:
    total = len(results)
    passed = sum(1 for item in results if item.passed)
    needs_review = sum(1 for item in results if item.needs_review)
    errors = sum(1 for item in results if item.error)
    failed = total - passed
    deterministic_passed = sum(
        1
        for item in results
        if not item.error and item.deterministic_answer_correct and item.deterministic_source_correct
    )
    judge_scored = sum(
        1
        for item in results
        if item.judge_answer_correct is not None and item.judge_source_correct is not None
    )
    judge_passed = sum(
        1
        for item in results
        if item.judge_answer_correct is True and item.judge_source_correct is True
    )
    judge_new_passes = [
        f"{item.prompt_variant}:{item.test_id}"
        for item in results
        if item.judge_answer_correct is True
        and item.judge_source_correct is True
        and not (item.deterministic_answer_correct and item.deterministic_source_correct)
    ]
    judge_new_failures = [
        f"{item.prompt_variant}:{item.test_id}"
        for item in results
        if item.deterministic_answer_correct
        and item.deterministic_source_correct
        and item.judge_answer_correct is not None
        and item.judge_source_correct is not None
        and not (item.judge_answer_correct and item.judge_source_correct)
    ]

    def bucket(items: Iterable[CapabilityResult]) -> dict[str, Any]:
        materialized = list(items)
        bucket_total = len(materialized)
        bucket_passed = sum(1 for item in materialized if item.passed)
        bucket_deterministic_passed = sum(
            1
            for item in materialized
            if not item.error and item.deterministic_answer_correct and item.deterministic_source_correct
        )
        bucket_judge_scored = sum(
            1
            for item in materialized
            if item.judge_answer_correct is not None and item.judge_source_correct is not None
        )
        bucket_judge_passed = sum(
            1
            for item in materialized
            if item.judge_answer_correct is True and item.judge_source_correct is True
        )
        return {
            "total": bucket_total,
            "passed": bucket_passed,
            "failed": bucket_total - bucket_passed,
            "needs_review": sum(1 for item in materialized if item.needs_review),
            "errors": sum(1 for item in materialized if item.error),
            "accuracy": (bucket_passed / bucket_total) if bucket_total else 0.0,
            "deterministic_passed": bucket_deterministic_passed,
            "deterministic_accuracy": (bucket_deterministic_passed / bucket_total) if bucket_total else 0.0,
            "judge_scored": bucket_judge_scored,
            "judge_passed": bucket_judge_passed,
            "judge_accuracy": (bucket_judge_passed / bucket_total) if bucket_total else 0.0,
            "judge_scored_accuracy": (bucket_judge_passed / bucket_judge_scored) if bucket_judge_scored else 0.0,
            "judge_delta_passes": bucket_judge_passed - bucket_deterministic_passed,
            "judge_delta_accuracy": (
                ((bucket_judge_passed - bucket_deterministic_passed) / bucket_total)
                if bucket_total
                else 0.0
            ),
        }

    by_area = {area: bucket(item for item in results if item.capability_area == area) for area in sorted({item.capability_area for item in results})}
    by_variant = {variant: bucket(item for item in results if item.prompt_variant == variant) for variant in sorted({item.prompt_variant for item in results})}
    final_scorer = "judge" if judge_scored == total and total else "mixed" if judge_scored else "deterministic"
    return CapabilitySummary(
        total=total,
        passed=passed,
        failed=failed,
        needs_review=needs_review,
        errors=errors,
        accuracy=(passed / total) if total else 0.0,
        deterministic_passed=deterministic_passed,
        deterministic_accuracy=(deterministic_passed / total) if total else 0.0,
        judge_scored=judge_scored,
        judge_passed=judge_passed,
        judge_accuracy=(judge_passed / total) if total else 0.0,
        judge_scored_accuracy=(judge_passed / judge_scored) if judge_scored else 0.0,
        judge_delta_passes=judge_passed - deterministic_passed,
        judge_delta_accuracy=((judge_passed - deterministic_passed) / total) if total else 0.0,
        judge_new_passes=judge_new_passes,
        judge_new_failures=judge_new_failures,
        final_scorer=final_scorer,
        by_area=by_area,
        by_variant=by_variant,
    )


def run_gateway_case(
    client: httpx.Client,
    case: CapabilityCase,
    *,
    api_base: str,
    model: str,
    prompt_field: str,
    prompt_variant: str,
    run_id: str,
    token: str,
    judge_model: Any | None,
    raw_dir: Path,
) -> CapabilityResult:
    prompt = case.prompt_for(prompt_field)
    payload = build_gateway_payload(case, prompt=prompt, model=model, prompt_variant=prompt_variant)
    headers = build_gateway_headers(case, run_id=run_id, prompt_variant=prompt_variant, token=token)
    start = time.perf_counter()
    status_code: int | None = None
    try:
        response = client.post(
            f"{api_base.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        status_code = response.status_code
        response.raise_for_status()
        response_payload = response.json()
        answer = extract_answer(response_payload)
        diagnostics = extract_rag_diagnostics(response_payload)
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{prompt_variant}_{case.test_id}.json").write_text(
            json.dumps(
                {
                    "request": {"headers": {key: value for key, value in headers.items() if key.lower() != "authorization"}, "payload": payload},
                    "response": response_payload,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return score_case(
            case,
            prompt=prompt,
            answer=answer,
            prompt_variant=prompt_variant,
            judge_model=judge_model,
            status_code=status_code,
            latency_seconds=time.perf_counter() - start,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{prompt_variant}_{case.test_id}.json").write_text(
            json.dumps(
                {
                    "request": {"headers": {key: value for key, value in headers.items() if key.lower() != "authorization"}, "payload": payload},
                    "error": str(exc),
                    "status_code": status_code,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return score_case(
            case,
            prompt=prompt,
            answer="",
            prompt_variant=prompt_variant,
            judge_model=None,
            error=str(exc),
            status_code=status_code,
            latency_seconds=time.perf_counter() - start,
        )


def write_outputs(output_dir: Path, results: Sequence[CapabilityResult], summary: CapabilitySummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps({"summary": summary.to_dict(), "results": [item.to_dict() for item in results]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fieldnames = list(CapabilityResult.__dataclass_fields__.keys())
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item.to_dict() for item in results)
    lines = [
        "# Capability Benchmark Results",
        "",
        f"- Total: {summary.total}",
        f"- Passed: {summary.passed}",
        f"- Failed: {summary.failed}",
        f"- Needs review: {summary.needs_review}",
        f"- Errors: {summary.errors}",
        f"- Accuracy: {summary.accuracy:.1%}",
        f"- Final scorer: {summary.final_scorer}",
        f"- Deterministic accuracy: {summary.deterministic_passed}/{summary.total} ({summary.deterministic_accuracy:.1%})",
    ]
    if summary.judge_scored:
        lines.extend(
            [
                f"- Judge accuracy: {summary.judge_passed}/{summary.total} ({summary.judge_accuracy:.1%})",
                f"- Judge-scored accuracy: {summary.judge_passed}/{summary.judge_scored} ({summary.judge_scored_accuracy:.1%})",
                f"- Judge delta: {summary.judge_delta_passes:+d} pass(es) ({summary.judge_delta_accuracy:+.1%})",
                f"- Judge-only passes: {', '.join(summary.judge_new_passes) if summary.judge_new_passes else 'None'}",
                f"- Judge downgrades: {', '.join(summary.judge_new_failures) if summary.judge_new_failures else 'None'}",
            ]
        )
    lines.extend(
        [
            "",
            "| Test | Variant | Final Pass | Deterministic Pass | Judge Pass | Answer | Source | Area |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        answer_flag = result.judge_answer_correct if result.judge_answer_correct is not None else result.deterministic_answer_correct
        source_flag = result.judge_source_correct if result.judge_source_correct is not None else result.deterministic_source_correct
        deterministic_pass = result.deterministic_answer_correct and result.deterministic_source_correct and not result.error
        judge_pass = (
            result.judge_answer_correct and result.judge_source_correct
            if result.judge_answer_correct is not None and result.judge_source_correct is not None
            else ""
        )
        lines.append(
            f"| {result.test_id} | {result.prompt_variant} | {result.passed} | {deterministic_pass} | {judge_pass} | {answer_flag} | {source_flag} | {result.capability_area} |"
        )
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(summary: CapabilitySummary, *, output_dir: Path) -> None:
    print(f"Capability benchmark complete: {summary.passed}/{summary.total} passed ({summary.accuracy:.1%})")
    print(f"Final scorer: {summary.final_scorer}")
    print(
        "Deterministic score: "
        f"{summary.deterministic_passed}/{summary.total} ({summary.deterministic_accuracy:.1%})"
    )
    if summary.judge_scored:
        print(
            "Judge score: "
            f"{summary.judge_passed}/{summary.total} ({summary.judge_accuracy:.1%}); "
            f"delta {summary.judge_delta_passes:+d} ({summary.judge_delta_accuracy:+.1%})"
        )
        if summary.judge_new_passes:
            print(f"Judge-only passes: {', '.join(summary.judge_new_passes)}")
        if summary.judge_new_failures:
            print(f"Judge downgrades: {', '.join(summary.judge_new_failures)}")
    print(f"Needs review: {summary.needs_review}; errors: {summary.errors}")
    print(f"Artifacts: {output_dir}")
    for area, bucket in summary.by_area.items():
        print(f"- {area}: {bucket['passed']}/{bucket['total']} ({bucket['accuracy']:.1%})")


def run_suite(
    *,
    suite_path: Path,
    difficulty: str,
    prompt_field: str,
    compare_original: bool,
    api_base: str,
    model: str,
    token: str,
    judge: str,
    output_dir: Path,
    test_ids: Sequence[str],
    limit: int,
    timeout_seconds: float,
    fail_fast: bool,
    dotenv_path: str = "",
) -> CapabilitySummary:
    cases = load_cases(suite_path, difficulty=difficulty, test_ids=test_ids)
    if limit > 0:
        cases = cases[:limit]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    judge_model = None
    if judge != "off":
        judge_model, judge_error = build_optional_judge(dotenv_path=dotenv_path)
        if judge_model is None:
            print(f"Judge unavailable; using deterministic scoring only. Reason: {judge_error}")
    variants = [(prompt_field, prompt_field)]
    if compare_original and prompt_field != "query":
        variants.append(("query", "query"))

    results: list[CapabilityResult] = []
    raw_dir = output_dir / "raw"
    with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
        for field_name, variant in variants:
            for case in cases:
                result = run_gateway_case(
                    client,
                    case,
                    api_base=api_base,
                    model=model,
                    prompt_field=field_name,
                    prompt_variant=variant,
                    run_id=run_id,
                    token=token,
                    judge_model=judge_model,
                    raw_dir=raw_dir,
                )
                results.append(result)
                status = "PASS" if result.passed else "REVIEW" if result.needs_review else "FAIL"
                print(f"[{status}] {variant}:{case.test_id} {result.latency_seconds:.1f}s")
                if fail_fast and result.error:
                    break
            if fail_fast and results and results[-1].error:
                break

    summary = summarize_results(results)
    write_outputs(output_dir, results, summary)
    _print_summary(summary, output_dir=output_dir)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate capability-suite rows against the live agent gateway.")
    parser.add_argument("--suite", type=Path, default=Path("rag_system_capability_test_suite.csv"))
    parser.add_argument("--difficulty", default="Easy")
    parser.add_argument("--prompt-field", default="guided_query", choices=["guided_query", "query"])
    parser.add_argument("--compare-original", action="store_true")
    parser.add_argument("--api-base", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default=os.environ.get("GATEWAY_MODEL_ID", "enterprise-agent"))
    parser.add_argument("--token-env", default="GATEWAY_SHARED_BEARER_TOKEN")
    parser.add_argument("--token", default="")
    parser.add_argument("--judge", default="auto", choices=["auto", "off"])
    parser.add_argument("--dotenv", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=args.dotenv or None)
    except Exception:
        pass
    token = args.token or os.environ.get(args.token_env, "")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("data/runtime/capability_runs") / timestamp
    run_suite(
        suite_path=args.suite,
        difficulty=args.difficulty,
        prompt_field=args.prompt_field,
        compare_original=args.compare_original,
        api_base=args.api_base,
        model=args.model,
        token=token,
        judge=args.judge,
        output_dir=output_dir,
        test_ids=args.test_id,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        fail_fast=args.fail_fast,
        dotenv_path=args.dotenv,
    )
    return 0


__all__ = [
    "CapabilityCase",
    "CapabilityResult",
    "CapabilitySummary",
    "JudgeScore",
    "build_gateway_headers",
    "build_gateway_payload",
    "conversation_id_for",
    "deterministic_answer_match",
    "deterministic_source_match",
    "extract_answer",
    "load_cases",
    "main",
    "run_suite",
    "score_case",
    "summarize_results",
    "user_id_for",
    "write_outputs",
]
