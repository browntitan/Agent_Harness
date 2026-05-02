from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

from agentic_chatbot_next.benchmark.capability_suite import (
    CITATION_MARKER_RE,
    FILE_SOURCE_RE,
    MATCH_STOPWORDS,
    build_optional_judge,
    deterministic_answer_match,
    deterministic_source_match,
    extract_answer,
    extract_rag_diagnostics,
    score_with_judge,
)

PROFILE_LIMITS = {
    "smoke": 25,
    "diagnostic": 300,
    "full": 0,
}

DEFAULT_BENCHMARKS = ("beir:scifact", "hotpotqa", "ragbench")
DEFAULT_AVAILABLE_CAPABILITIES = ("chat", "rag")
DEFERRED_AGENT_BENCHMARKS = {"gaia", "webarena"}
NO_ANSWER_RE = re.compile(
    r"\b("
    r"cannot answer|can't answer|not enough information|insufficient evidence|"
    r"not in (?:the )?(?:context|documents|sources)|no evidence|unknown"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PublicBenchmarkCase:
    benchmark_id: str
    test_id: str
    task_type: str
    prompt: str
    gold_answer: str = ""
    gold_sources: tuple[str, ...] = ()
    collection_id: str = ""
    difficulty: str = ""
    tags: tuple[str, ...] = ()
    scorer: str = "answer_source"
    requires_capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PublicBenchmarkCase":
        return cls(
            benchmark_id=_clean(row.get("benchmark_id")),
            test_id=_clean(row.get("test_id")),
            task_type=_clean(row.get("task_type")) or "rag_qa",
            prompt=_clean(row.get("prompt")),
            gold_answer=_clean(row.get("gold_answer")),
            gold_sources=_coerce_tuple(row.get("gold_sources")),
            collection_id=_clean(row.get("collection_id")),
            difficulty=_clean(row.get("difficulty")),
            tags=_coerce_tuple(row.get("tags")),
            scorer=_clean(row.get("scorer")) or "answer_source",
            requires_capabilities=_coerce_tuple(row.get("requires_capabilities")),
            metadata=dict(row.get("metadata") or {}),
        )

    def gold_sources_text(self) -> str:
        return " | ".join(self.gold_sources)


@dataclass
class PublicBenchmarkResult:
    benchmark_id: str
    test_id: str
    task_type: str
    difficulty: str
    collection_id: str
    tags: str
    scorer: str
    prompt: str
    gold_answer: str
    gold_sources: str
    answer: str = ""
    error: str = ""
    skipped: bool = False
    skip_reason: str = ""
    status_code: int | None = None
    latency_seconds: float = 0.0
    answer_exact: bool = False
    answer_f1: float = 0.0
    answer_correct: bool = False
    source_hit: bool = False
    citation_source_match: bool = False
    no_answer_correct: bool = False
    tool_call_match: bool = False
    tool_argument_match: bool = False
    route_correct: bool | None = None
    judge_answer_correct: bool | None = None
    judge_source_correct: bool | None = None
    judge_reason: str = ""
    retrieval_recall_at_k: float | None = None
    retrieval_mrr: float | None = None
    retrieval_ndcg: float | None = None
    score: float = 0.0
    passed: bool = False
    failure_categories: str = ""
    stage_timings_ms: str = ""
    slowest_stage: str = ""
    budget_exhausted: bool = False
    retrieval_mode: str = ""
    tool_calls_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PublicBenchmarkSummary:
    profile: str
    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    accuracy: float
    pass_rate_excluding_skips: float
    by_benchmark: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_task_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_failure_category: dict[str, int] = field(default_factory=dict)
    latency_p50_seconds: float = 0.0
    latency_p95_seconds: float = 0.0
    final_scorer: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _coerce_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        parts = re.split(r"\s*\|\s*|\s*,\s*", value)
        return tuple(item.strip() for item in parts if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    stripped = re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold())
    return " ".join(stripped.split())


def _token_set(value: str) -> list[str]:
    return [
        token
        for token in _normalize_text(value).split()
        if (len(token) >= 3 or token.isdigit()) and token not in MATCH_STOPWORDS
    ]


def answer_token_f1(answer: str, gold_answer: str) -> float:
    answer_tokens = _token_set(answer)
    gold_tokens = _token_set(gold_answer)
    if not answer_tokens or not gold_tokens:
        return 0.0
    answer_counts: dict[str, int] = {}
    for token in answer_tokens:
        answer_counts[token] = answer_counts.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        count = answer_counts.get(token, 0)
        if count > 0:
            overlap += 1
            answer_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _case_to_capability_case(case: PublicBenchmarkCase) -> Any:
    from agentic_chatbot_next.benchmark.capability_suite import CapabilityCase

    return CapabilityCase(
        test_id=case.test_id,
        difficulty=case.difficulty or "",
        capability_area=case.task_type,
        query=case.prompt,
        guided_query=case.prompt,
        expected_answer=case.gold_answer,
        expected_sources=case.gold_sources_text() or "none",
        collection_id=case.collection_id,
        graph_id="",
        success_criteria=str(case.metadata.get("success_criteria") or ""),
    )


def _collection_for(benchmark_id: str, *, dataset: str = "", collection_prefix: str = "public") -> str:
    parts = [collection_prefix, benchmark_id]
    if dataset:
        parts.append(dataset)
    return "-".join(item.replace("_", "-").lower() for item in parts if item)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    yield payload


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "examples", "rows", "questions"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _profile_limit(profile: str) -> int:
    return PROFILE_LIMITS.get(profile, PROFILE_LIMITS["smoke"])


def _limit_cases(cases: list[PublicBenchmarkCase], profile: str) -> list[PublicBenchmarkCase]:
    limit = _profile_limit(profile)
    return cases if limit <= 0 else cases[:limit]


def _find_first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_beir_cases(
    data_root: Path,
    dataset: str,
    *,
    profile: str,
    collection_prefix: str = "public",
) -> list[PublicBenchmarkCase]:
    root = data_root / "beir" / dataset
    queries_path = _find_first_existing([root / "queries.jsonl", root / "queries.json"])
    qrels_path = _find_first_existing([root / "qrels" / "test.tsv", root / "qrels" / "dev.tsv", root / "qrels.tsv"])
    if queries_path is None or qrels_path is None:
        return []

    if queries_path.suffix == ".jsonl":
        query_rows = list(_iter_jsonl(queries_path))
    else:
        raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
        query_rows = []
        if isinstance(raw_queries, dict):
            for key, value in raw_queries.items():
                if isinstance(value, Mapping):
                    query_rows.append({"_id": key, "text": value.get("text") or value.get("query") or value.get("question")})
                else:
                    query_rows.append({"_id": key, "text": value})

    positives: dict[str, list[str]] = {}
    with qrels_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames and {"query-id", "corpus-id"}.issubset(set(reader.fieldnames)):
            for row in reader:
                score = float(row.get("score") or row.get("relevance") or 0)
                if score > 0:
                    positives.setdefault(str(row.get("query-id")), []).append(str(row.get("corpus-id")))
        else:
            handle.seek(0)
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] not in {"query-id", "query_id"}:
                    try:
                        score = float(parts[-1])
                    except ValueError:
                        score = 0.0
                    if score > 0:
                        positives.setdefault(parts[0], []).append(parts[1])

    cases: list[PublicBenchmarkCase] = []
    for row in query_rows:
        query_id = _clean(row.get("_id") or row.get("id") or row.get("query_id"))
        query = _clean(row.get("text") or row.get("query") or row.get("question"))
        if not query_id or not query:
            continue
        sources = tuple(positives.get(query_id, []))
        cases.append(
            PublicBenchmarkCase(
                benchmark_id=f"beir:{dataset}",
                test_id=query_id,
                task_type="retrieval_rag",
                prompt=f"Answer using the indexed {dataset} public benchmark collection and cite supporting sources: {query}",
                gold_answer="",
                gold_sources=sources,
                collection_id=_collection_for("beir", dataset=dataset, collection_prefix=collection_prefix),
                difficulty="public",
                tags=("beir", dataset, "retrieval"),
                scorer="retrieval_source",
                requires_capabilities=("rag",),
            )
        )
    return _limit_cases(cases, profile)


def load_hotpotqa_cases(
    data_root: Path,
    *,
    profile: str,
    collection_prefix: str = "public",
) -> list[PublicBenchmarkCase]:
    root = data_root / "hotpotqa"
    path = _find_first_existing(
        [
            root / "hotpot_dev_distractor_v1.json",
            root / "dev.json",
            root / "hotpotqa.json",
            root / "hotpotqa.jsonl",
        ]
    )
    if path is None:
        return []
    rows = list(_iter_jsonl(path)) if path.suffix == ".jsonl" else _load_json_list(path)
    cases: list[PublicBenchmarkCase] = []
    for index, row in enumerate(rows, start=1):
        test_id = _clean(row.get("_id") or row.get("id") or f"hotpotqa-{index}")
        question = _clean(row.get("question") or row.get("query"))
        answer = _clean(row.get("answer"))
        supporting = row.get("supporting_facts") or row.get("supporting_facts_titles") or []
        sources = []
        for item in supporting:
            if isinstance(item, (list, tuple)) and item:
                sources.append(str(item[0]))
            elif isinstance(item, str):
                sources.append(item)
        if not question:
            continue
        cases.append(
            PublicBenchmarkCase(
                benchmark_id="hotpotqa",
                test_id=test_id,
                task_type="multi_hop_qa",
                prompt=f"Answer this HotpotQA multi-hop question with cited evidence: {question}",
                gold_answer=answer,
                gold_sources=tuple(dict.fromkeys(sources)),
                collection_id=_collection_for("hotpotqa", collection_prefix=collection_prefix),
                difficulty=_clean(row.get("level")) or "public",
                tags=("hotpotqa", "multi-hop"),
                scorer="answer_source",
                requires_capabilities=("rag",),
                metadata={"question_type": _clean(row.get("type"))},
            )
        )
    return _limit_cases(cases, profile)


def load_ragbench_cases(
    data_root: Path,
    *,
    profile: str,
    collection_prefix: str = "public",
) -> list[PublicBenchmarkCase]:
    root = data_root / "ragbench"
    path = _find_first_existing(
        [
            root / "test.jsonl",
            root / "dev.jsonl",
            root / "validation.jsonl",
            root / "ragbench.jsonl",
            root / "ragbench.json",
        ]
    )
    if path is None:
        return []
    rows = list(_iter_jsonl(path)) if path.suffix == ".jsonl" else _load_json_list(path)
    cases: list[PublicBenchmarkCase] = []
    for index, row in enumerate(rows, start=1):
        test_id = _clean(row.get("id") or row.get("example_id") or f"ragbench-{index}")
        question = _clean(row.get("question") or row.get("query") or row.get("prompt"))
        answer = _clean(row.get("answer") or row.get("response") or row.get("reference_answer"))
        raw_sources = (
            row.get("source_ids")
            or row.get("gold_sources")
            or row.get("documents")
            or row.get("contexts")
            or row.get("passages")
            or []
        )
        sources = _coerce_source_names(raw_sources)
        if not question:
            continue
        cases.append(
            PublicBenchmarkCase(
                benchmark_id="ragbench",
                test_id=test_id,
                task_type="grounded_rag",
                prompt=f"Answer using the indexed RAGBench collection and cite supporting evidence: {question}",
                gold_answer=answer,
                gold_sources=sources,
                collection_id=_collection_for("ragbench", collection_prefix=collection_prefix),
                difficulty=_clean(row.get("difficulty")) or "public",
                tags=("ragbench", "faithfulness", "grounding"),
                scorer="answer_source",
                requires_capabilities=("rag",),
                metadata={
                    "faithfulness_label": row.get("faithfulness") or row.get("faithfulness_label"),
                    "support_label": row.get("support") or row.get("support_label"),
                    "completeness_label": row.get("completeness") or row.get("completeness_label"),
                },
            )
        )
    return _limit_cases(cases, profile)


def _coerce_source_names(raw_sources: Any) -> tuple[str, ...]:
    if isinstance(raw_sources, str):
        return _coerce_tuple(raw_sources)
    sources: list[str] = []
    for item in raw_sources or []:
        if isinstance(item, str):
            sources.append(item)
        elif isinstance(item, Mapping):
            value = item.get("id") or item.get("doc_id") or item.get("title") or item.get("source")
            if value:
                sources.append(str(value))
    return tuple(dict.fromkeys(item.strip() for item in sources if item.strip()))


def load_bfcl_cases(
    data_root: Path,
    *,
    profile: str,
    collection_prefix: str = "public",
) -> list[PublicBenchmarkCase]:
    del collection_prefix
    root = data_root / "bfcl"
    path = _find_first_existing([root / "test.jsonl", root / "bfcl.jsonl", root / "BFCL_v3.jsonl"])
    if path is None:
        return []
    cases: list[PublicBenchmarkCase] = []
    for index, row in enumerate(_iter_jsonl(path), start=1):
        test_id = _clean(row.get("id") or row.get("question_id") or f"bfcl-{index}")
        prompt = _clean(row.get("question") or row.get("prompt"))
        expected = row.get("ground_truth") or row.get("answer") or row.get("expected_tool_call") or {}
        expected_tool = _expected_tool_name(expected)
        expected_args = _expected_tool_arguments(expected)
        if not prompt:
            continue
        cases.append(
            PublicBenchmarkCase(
                benchmark_id="bfcl",
                test_id=test_id,
                task_type="tool_calling",
                prompt=(
                    "Return the tool call that should satisfy this request. "
                    "Use compact JSON with keys name and arguments only.\n\n"
                    f"Request: {prompt}"
                ),
                gold_answer=json.dumps({"name": expected_tool, "arguments": expected_args}, sort_keys=True),
                gold_sources=(),
                collection_id="",
                difficulty=_clean(row.get("category")) or "public",
                tags=("bfcl", "tool-calling"),
                scorer="tool_call",
                requires_capabilities=("tool_calling",),
                metadata={"expected_tool_name": expected_tool, "expected_arguments": expected_args},
            )
        )
    return _limit_cases(cases, profile)


def load_taubench_cases(
    data_root: Path,
    *,
    profile: str,
    collection_prefix: str = "public",
) -> list[PublicBenchmarkCase]:
    del collection_prefix
    root = data_root / "tau-bench"
    path = _find_first_existing([root / "test.jsonl", root / "taubench.jsonl", root / "tasks.jsonl"])
    if path is None:
        return []
    cases: list[PublicBenchmarkCase] = []
    for index, row in enumerate(_iter_jsonl(path), start=1):
        test_id = _clean(row.get("id") or row.get("task_id") or f"taubench-{index}")
        instruction = _clean(row.get("instruction") or row.get("user_intent") or row.get("prompt"))
        if not instruction:
            continue
        cases.append(
            PublicBenchmarkCase(
                benchmark_id="tau-bench",
                test_id=test_id,
                task_type="multi_turn_tool_policy",
                prompt=instruction,
                gold_answer=_clean(row.get("goal") or row.get("expected_outcome")),
                gold_sources=(),
                collection_id="",
                difficulty=_clean(row.get("domain")) or "public",
                tags=("tau-bench", "multi-turn", "tool-policy"),
                scorer="deferred_environment",
                requires_capabilities=("tool_environment_shim",),
                metadata={"deferred_reason": "tau-bench requires a stateful tool/user environment shim."},
            )
        )
    return _limit_cases(cases, profile)


def _expected_tool_name(expected: Any) -> str:
    if isinstance(expected, Mapping):
        for key in ("name", "function", "tool_name"):
            value = expected.get(key)
            if isinstance(value, str):
                return value
        function = expected.get("function")
        if isinstance(function, Mapping):
            return _clean(function.get("name"))
    if isinstance(expected, str):
        payload = _extract_json_object(expected)
        if payload:
            return _expected_tool_name(payload)
        match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", expected)
        return match.group(0) if match else ""
    return ""


def _expected_tool_arguments(expected: Any) -> dict[str, Any]:
    if isinstance(expected, Mapping):
        for key in ("arguments", "args", "parameters"):
            value = expected.get(key)
            if isinstance(value, Mapping):
                return dict(value)
            if isinstance(value, str):
                parsed = _extract_json_object(value)
                return parsed if parsed else {}
        function = expected.get("function")
        if isinstance(function, Mapping):
            return _expected_tool_arguments(function)
    if isinstance(expected, str):
        return _expected_tool_arguments(_extract_json_object(expected))
    return {}


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


def parse_benchmark_specs(benchmarks: str | Sequence[str]) -> list[tuple[str, str]]:
    raw_items: list[str] = []
    if isinstance(benchmarks, str):
        raw_items.extend(item.strip() for item in benchmarks.split(","))
    else:
        for item in benchmarks:
            raw_items.extend(part.strip() for part in str(item).split(","))
    specs: list[tuple[str, str]] = []
    for item in raw_items:
        if not item:
            continue
        benchmark, _, dataset = item.partition(":")
        specs.append((benchmark.strip().lower(), dataset.strip().lower()))
    return specs


def load_public_cases(
    *,
    benchmarks: str | Sequence[str],
    data_root: Path,
    profile: str,
    collection_prefix: str = "public",
) -> list[PublicBenchmarkCase]:
    cases: list[PublicBenchmarkCase] = []
    for benchmark, dataset in parse_benchmark_specs(benchmarks):
        if benchmark == "beir":
            for name in ([dataset] if dataset else ["scifact", "fiqa", "nq", "hotpotqa"]):
                cases.extend(
                    load_beir_cases(data_root, name, profile=profile, collection_prefix=collection_prefix)
                )
        elif benchmark == "hotpotqa":
            cases.extend(load_hotpotqa_cases(data_root, profile=profile, collection_prefix=collection_prefix))
        elif benchmark == "ragbench":
            cases.extend(load_ragbench_cases(data_root, profile=profile, collection_prefix=collection_prefix))
        elif benchmark == "bfcl":
            cases.extend(load_bfcl_cases(data_root, profile=profile, collection_prefix=collection_prefix))
        elif benchmark in {"tau-bench", "taubench", "tau3-bench", "tau3bench"}:
            cases.extend(load_taubench_cases(data_root, profile=profile, collection_prefix=collection_prefix))
        elif benchmark in DEFERRED_AGENT_BENCHMARKS:
            cases.append(
                PublicBenchmarkCase(
                    benchmark_id=benchmark,
                    test_id=f"{benchmark}-deferred",
                    task_type="deferred_full_autonomy",
                    prompt=f"{benchmark} adapter is deferred until browser/web-action tools are available.",
                    tags=(benchmark, "deferred"),
                    scorer="deferred_environment",
                    requires_capabilities=("browser_action_environment",),
                    metadata={"deferred_reason": "Full-autonomy browser benchmark support is intentionally deferred."},
                )
            )
    return cases


def build_gateway_payload(case: PublicBenchmarkCase, *, model: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "benchmark_id": case.benchmark_id,
        "benchmark_test_id": case.test_id,
        "benchmark_task_type": case.task_type,
        "benchmark_scorer": case.scorer,
    }
    if case.collection_id:
        metadata.update(
            {
                "kb_collection_id": case.collection_id,
                "requested_kb_collection_id": case.collection_id,
                "selected_kb_collection_id": case.collection_id,
                "available_kb_collection_ids": [case.collection_id],
                "search_collection_ids": [case.collection_id],
                "kb_collection_confirmed": True,
            }
        )
    metadata.update(dict(case.metadata.get("request_metadata") or {}))
    return {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": case.prompt}],
        "metadata": metadata,
    }


def build_gateway_headers(case: PublicBenchmarkCase, *, run_id: str, token: str = "") -> dict[str, str]:
    safe_benchmark = re.sub(r"[^a-z0-9_-]+", "-", case.benchmark_id.casefold()).strip("-")
    safe_test_id = re.sub(r"[^a-z0-9_-]+", "-", case.test_id.casefold()).strip("-")
    headers = {
        "Content-Type": "application/json",
        "X-Conversation-ID": f"public-benchmark-{run_id}-{safe_benchmark}-{safe_test_id}",
        "X-User-ID": f"public-benchmark-{run_id}-{safe_benchmark}-user",
        "X-Request-ID": f"public-benchmark-{run_id}-{safe_benchmark}-{safe_test_id}",
    }
    if case.collection_id:
        headers["X-Collection-ID"] = case.collection_id
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _retrieved_sources(response_payload: Mapping[str, Any]) -> list[str]:
    metadata = dict(response_payload.get("metadata") or {})
    summary = dict(metadata.get("rag_retrieval_summary") or {})
    candidates = (
        summary.get("retrieved_sources")
        or summary.get("source_documents")
        or summary.get("citations")
        or metadata.get("retrieved_sources")
        or metadata.get("citations")
        or []
    )
    sources: list[str] = []
    for item in candidates or []:
        if isinstance(item, str):
            sources.append(item)
        elif isinstance(item, Mapping):
            value = item.get("source") or item.get("title") or item.get("doc_id") or item.get("document_id")
            if value:
                sources.append(str(value))
    return sources


def retrieval_metrics(gold_sources: Sequence[str], retrieved_sources: Sequence[str]) -> tuple[float | None, float | None, float | None]:
    expected = [_normalize_text(item) for item in gold_sources if _normalize_text(item)]
    retrieved = [_normalize_text(item) for item in retrieved_sources if _normalize_text(item)]
    if not expected:
        return None, None, None
    if not retrieved:
        return 0.0, 0.0, 0.0
    expected_set = set(expected)
    hit_ranks = [index + 1 for index, value in enumerate(retrieved) if value in expected_set]
    recall = len(set(value for value in retrieved if value in expected_set)) / len(expected_set)
    mrr = (1.0 / min(hit_ranks)) if hit_ranks else 0.0
    dcg = 0.0
    for index, value in enumerate(retrieved, start=1):
        if value in expected_set:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(expected_set), len(retrieved))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return recall, mrr, ndcg


def _source_hit(case: PublicBenchmarkCase, answer: str, retrieved_sources: Sequence[str]) -> bool:
    if not case.gold_sources:
        return not CITATION_MARKER_RE.search(answer or "")
    expected = case.gold_sources_text()
    combined = "\n".join([answer, *retrieved_sources])
    return deterministic_source_match(combined, expected)


def _tool_call_scores(case: PublicBenchmarkCase, answer: str) -> tuple[bool, bool]:
    payload = _extract_json_object(answer)
    expected_tool = str(case.metadata.get("expected_tool_name") or "").strip()
    expected_args = dict(case.metadata.get("expected_arguments") or {})
    actual_tool = str(payload.get("name") or payload.get("tool_name") or payload.get("function") or "").strip()
    if isinstance(payload.get("function"), Mapping):
        actual_tool = str(payload["function"].get("name") or actual_tool)
    actual_args = payload.get("arguments") or payload.get("args") or payload.get("parameters") or {}
    if isinstance(actual_args, str):
        actual_args = _extract_json_object(actual_args)
    if not isinstance(actual_args, Mapping):
        actual_args = {}
    tool_match = bool(expected_tool) and actual_tool == expected_tool
    arg_match = tool_match and all(actual_args.get(key) == value for key, value in expected_args.items())
    return tool_match, arg_match


def _route_correct(case: PublicBenchmarkCase, response_payload: Mapping[str, Any]) -> bool | None:
    expected = str(case.metadata.get("expected_route") or case.metadata.get("expected_agent_path") or "").strip()
    if not expected:
        return None
    metadata = dict(response_payload.get("metadata") or {})
    route_context = dict(metadata.get("route_context") or {})
    actual = (
        str(metadata.get("suggested_agent") or "")
        or str(metadata.get("route") or "")
        or str(route_context.get("suggested_agent") or "")
        or str(route_context.get("route") or "")
    )
    return expected.casefold() in actual.casefold() if actual else False


def classify_failures(
    case: PublicBenchmarkCase,
    *,
    answer: str,
    answer_correct: bool,
    source_hit: bool,
    citation_source_match: bool,
    tool_call_match: bool,
    tool_argument_match: bool,
    route_correct: bool | None,
    latency_seconds: float,
    error: str = "",
) -> list[str]:
    failures: list[str] = []
    if error:
        failures.append("runtime_error")
    has_expected_sources = bool(case.gold_sources)
    has_citations = bool(CITATION_MARKER_RE.search(answer or "") or FILE_SOURCE_RE.search(answer or ""))
    if has_expected_sources and not source_hit:
        failures.append("retrieval_miss")
    if has_citations and has_expected_sources and not citation_source_match:
        failures.append("wrong_source")
    if source_hit and not answer_correct and case.scorer != "retrieval_source":
        failures.append("evidence_present_bad_synthesis")
    if not has_expected_sources and has_citations:
        failures.append("citation_fabrication")
    if case.scorer == "tool_call" and not tool_call_match:
        failures.append("tool_selection_error")
    if case.scorer == "tool_call" and tool_call_match and not tool_argument_match:
        failures.append("tool_argument_error")
    if route_correct is False:
        failures.append("routing_error")
    if ("multi-hop" in case.tags or case.task_type == "multi_hop_qa") and not answer_correct:
        failures.append("multi_hop_failure")
    if case.task_type in {"no_answer", "negative_evidence"} and not _is_no_answer(answer):
        failures.append("negative_evidence_failure")
    max_latency = float(case.metadata.get("max_latency_seconds") or 0)
    if max_latency > 0 and latency_seconds > max_latency:
        failures.append("latency_budget_failure")
    return list(dict.fromkeys(failures))


def _is_no_answer(answer: str) -> bool:
    return bool(NO_ANSWER_RE.search(answer or ""))


def _missing_capabilities(case: PublicBenchmarkCase, available_capabilities: Sequence[str]) -> list[str]:
    available = {item.strip() for item in available_capabilities if item.strip()}
    return [item for item in case.requires_capabilities if item not in available]


def score_public_case(
    case: PublicBenchmarkCase,
    *,
    answer: str,
    response_payload: Mapping[str, Any] | None = None,
    judge_model: Any | None = None,
    error: str = "",
    status_code: int | None = None,
    latency_seconds: float = 0.0,
    diagnostics: Mapping[str, Any] | None = None,
) -> PublicBenchmarkResult:
    response_payload = dict(response_payload or {})
    retrieved_sources = _retrieved_sources(response_payload)
    retrieval_recall, retrieval_mrr, retrieval_ndcg = retrieval_metrics(case.gold_sources, retrieved_sources)
    answer_exact = deterministic_answer_match(answer, case.gold_answer) if case.gold_answer else False
    answer_f1 = answer_token_f1(answer, case.gold_answer) if case.gold_answer else 0.0
    source_hit = _source_hit(case, answer, retrieved_sources)
    citation_source_match = deterministic_source_match(answer, case.gold_sources_text() or "none")
    no_answer_correct = _is_no_answer(answer) if case.task_type in {"no_answer", "negative_evidence"} else False
    tool_call_match, tool_argument_match = _tool_call_scores(case, answer) if case.scorer == "tool_call" else (False, False)
    route_correct = _route_correct(case, response_payload)
    judge_answer_correct: bool | None = None
    judge_source_correct: bool | None = None
    judge_reason = ""
    if judge_model is not None and not error and case.gold_answer:
        try:
            judge_score = score_with_judge(judge_model, _case_to_capability_case(case), answer)
            if judge_score.available:
                judge_answer_correct = judge_score.answer_correct
                judge_source_correct = judge_score.source_correct
                judge_reason = judge_score.reason
        except Exception as exc:
            judge_reason = str(exc)

    if case.scorer == "retrieval_source":
        answer_correct = True
        score = float(retrieval_recall if retrieval_recall is not None else source_hit)
        passed = source_hit and not error
    elif case.scorer == "tool_call":
        answer_correct = tool_argument_match
        score = 1.0 if tool_argument_match else 0.0
        passed = tool_argument_match and not error
    elif case.task_type in {"no_answer", "negative_evidence"}:
        answer_correct = no_answer_correct
        score = 1.0 if no_answer_correct and source_hit else 0.0
        passed = no_answer_correct and source_hit and not error
    else:
        if judge_answer_correct is not None:
            answer_correct = bool(judge_answer_correct)
            source_ok = bool(judge_source_correct)
        else:
            answer_correct = answer_exact or answer_f1 >= 0.7
            source_ok = source_hit
        score = (float(answer_correct) + float(source_ok)) / 2.0
        passed = bool(answer_correct and source_ok and not error)

    failures = classify_failures(
        case,
        answer=answer,
        answer_correct=answer_correct,
        source_hit=source_hit,
        citation_source_match=citation_source_match,
        tool_call_match=tool_call_match,
        tool_argument_match=tool_argument_match,
        route_correct=route_correct,
        latency_seconds=latency_seconds,
        error=error,
    )
    return PublicBenchmarkResult(
        benchmark_id=case.benchmark_id,
        test_id=case.test_id,
        task_type=case.task_type,
        difficulty=case.difficulty,
        collection_id=case.collection_id,
        tags=" | ".join(case.tags),
        scorer=case.scorer,
        prompt=case.prompt,
        gold_answer=case.gold_answer,
        gold_sources=case.gold_sources_text(),
        answer=answer,
        error=error,
        status_code=status_code,
        latency_seconds=latency_seconds,
        answer_exact=answer_exact,
        answer_f1=answer_f1,
        answer_correct=answer_correct,
        source_hit=source_hit,
        citation_source_match=citation_source_match,
        no_answer_correct=no_answer_correct,
        tool_call_match=tool_call_match,
        tool_argument_match=tool_argument_match,
        route_correct=route_correct,
        judge_answer_correct=judge_answer_correct,
        judge_source_correct=judge_source_correct,
        judge_reason=judge_reason,
        retrieval_recall_at_k=retrieval_recall,
        retrieval_mrr=retrieval_mrr,
        retrieval_ndcg=retrieval_ndcg,
        score=score,
        passed=passed,
        failure_categories=" | ".join(failures),
        stage_timings_ms=str((diagnostics or {}).get("stage_timings_ms") or ""),
        slowest_stage=str((diagnostics or {}).get("slowest_stage") or ""),
        budget_exhausted=bool((diagnostics or {}).get("budget_exhausted", False)),
        retrieval_mode=str((diagnostics or {}).get("retrieval_mode") or ""),
        tool_calls_used=int((diagnostics or {}).get("tool_calls_used") or 0),
    )


def skipped_result(case: PublicBenchmarkCase, missing_capabilities: Sequence[str]) -> PublicBenchmarkResult:
    reason = "missing required capabilities: " + ", ".join(missing_capabilities)
    if case.metadata.get("deferred_reason"):
        reason = str(case.metadata["deferred_reason"])
    return PublicBenchmarkResult(
        benchmark_id=case.benchmark_id,
        test_id=case.test_id,
        task_type=case.task_type,
        difficulty=case.difficulty,
        collection_id=case.collection_id,
        tags=" | ".join(case.tags),
        scorer=case.scorer,
        prompt=case.prompt,
        gold_answer=case.gold_answer,
        gold_sources=case.gold_sources_text(),
        skipped=True,
        skip_reason=reason,
    )


def run_gateway_case(
    client: httpx.Client,
    case: PublicBenchmarkCase,
    *,
    api_base: str,
    model: str,
    run_id: str,
    token: str,
    judge_model: Any | None,
    raw_dir: Path,
) -> PublicBenchmarkResult:
    payload = build_gateway_payload(case, model=model)
    headers = build_gateway_headers(case, run_id=run_id, token=token)
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
        (raw_dir / f"{_safe_filename(case.benchmark_id)}_{_safe_filename(case.test_id)}.json").write_text(
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
        return score_public_case(
            case,
            answer=answer,
            response_payload=response_payload,
            judge_model=judge_model,
            status_code=status_code,
            latency_seconds=time.perf_counter() - start,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{_safe_filename(case.benchmark_id)}_{_safe_filename(case.test_id)}.json").write_text(
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
        return score_public_case(
            case,
            answer="",
            response_payload={},
            judge_model=None,
            error=str(exc),
            status_code=status_code,
            latency_seconds=time.perf_counter() - start,
        )


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())[:120] or "case"


def summarize_results(results: Sequence[PublicBenchmarkResult], *, profile: str) -> PublicBenchmarkSummary:
    total = len(results)
    skipped = sum(1 for item in results if item.skipped)
    runnable_total = total - skipped
    passed = sum(1 for item in results if item.passed)
    errors = sum(1 for item in results if item.error)
    failed = total - passed - skipped
    latencies = sorted(item.latency_seconds for item in results if not item.skipped and not item.error)

    def bucket(items: Iterable[PublicBenchmarkResult]) -> dict[str, Any]:
        materialized = list(items)
        bucket_total = len(materialized)
        bucket_skipped = sum(1 for item in materialized if item.skipped)
        bucket_runnable = bucket_total - bucket_skipped
        bucket_passed = sum(1 for item in materialized if item.passed)
        return {
            "total": bucket_total,
            "passed": bucket_passed,
            "failed": bucket_total - bucket_passed - bucket_skipped,
            "skipped": bucket_skipped,
            "accuracy": (bucket_passed / bucket_total) if bucket_total else 0.0,
            "pass_rate_excluding_skips": (bucket_passed / bucket_runnable) if bucket_runnable else 0.0,
        }

    failure_counts: dict[str, int] = {}
    for result in results:
        for category in _coerce_tuple(result.failure_categories):
            failure_counts[category] = failure_counts.get(category, 0) + 1

    judge_scored = sum(1 for item in results if item.judge_answer_correct is not None or item.judge_source_correct is not None)
    return PublicBenchmarkSummary(
        profile=profile,
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        accuracy=(passed / total) if total else 0.0,
        pass_rate_excluding_skips=(passed / runnable_total) if runnable_total else 0.0,
        by_benchmark={
            key: bucket(item for item in results if item.benchmark_id == key)
            for key in sorted({item.benchmark_id for item in results})
        },
        by_task_type={
            key: bucket(item for item in results if item.task_type == key)
            for key in sorted({item.task_type for item in results})
        },
        by_failure_category=dict(sorted(failure_counts.items())),
        latency_p50_seconds=_percentile(latencies, 0.50),
        latency_p95_seconds=_percentile(latencies, 0.95),
        final_scorer="mixed" if judge_scored else "deterministic",
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[int(position)])
    lower_value = values[lower]
    upper_value = values[upper]
    return float(lower_value + (upper_value - lower_value) * (position - lower))


def write_outputs(output_dir: Path, results: Sequence[PublicBenchmarkResult], summary: PublicBenchmarkSummary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary.to_dict(), "results": [item.to_dict() for item in results]}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = list(PublicBenchmarkResult.__dataclass_fields__.keys())
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(item.to_dict() for item in results)
    (output_dir / "failure_slices.json").write_text(
        json.dumps(
            {
                "by_failure_category": summary.by_failure_category,
                "failed_cases": [
                    item.to_dict()
                    for item in results
                    if item.failure_categories and not item.skipped
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Public Benchmark Results",
        "",
        f"- Profile: {summary.profile}",
        f"- Total: {summary.total}",
        f"- Passed: {summary.passed}",
        f"- Failed: {summary.failed}",
        f"- Skipped: {summary.skipped}",
        f"- Errors: {summary.errors}",
        f"- Accuracy: {summary.accuracy:.1%}",
        f"- Runnable pass rate: {summary.pass_rate_excluding_skips:.1%}",
        f"- Latency p50/p95: {summary.latency_p50_seconds:.1f}s / {summary.latency_p95_seconds:.1f}s",
        "",
        "| Benchmark | Test | Task | Pass | Score | Failures |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        state = "SKIP" if result.skipped else "PASS" if result.passed else "FAIL"
        failures = result.skip_reason if result.skipped else result.failure_categories
        lines.append(
            f"| {result.benchmark_id} | {result.test_id} | {result.task_type} | {state} | {result.score:.2f} | {failures} |"
        )
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(summary: PublicBenchmarkSummary, *, output_dir: Path) -> None:
    print(
        "Public benchmark complete: "
        f"{summary.passed}/{summary.total} passed "
        f"({summary.accuracy:.1%}); {summary.skipped} skipped"
    )
    print(f"Runnable pass rate: {summary.pass_rate_excluding_skips:.1%}")
    print(f"Latency p50/p95: {summary.latency_p50_seconds:.1f}s / {summary.latency_p95_seconds:.1f}s")
    if summary.by_failure_category:
        failures = ", ".join(f"{key}={value}" for key, value in summary.by_failure_category.items())
        print(f"Failure slices: {failures}")
    print(f"Artifacts: {output_dir}")
    for benchmark, bucket in summary.by_benchmark.items():
        print(f"- {benchmark}: {bucket['passed']}/{bucket['total']} ({bucket['accuracy']:.1%}), skipped {bucket['skipped']}")


def run_public_benchmark_suite(
    *,
    benchmarks: str | Sequence[str],
    profile: str,
    data_root: Path,
    collection_prefix: str,
    api_base: str,
    model: str,
    token: str,
    judge: str,
    output_dir: Path,
    limit: int,
    timeout_seconds: float,
    fail_fast: bool,
    available_capabilities: Sequence[str] = DEFAULT_AVAILABLE_CAPABILITIES,
    dotenv_path: str = "",
) -> PublicBenchmarkSummary:
    cases = load_public_cases(
        benchmarks=benchmarks,
        data_root=data_root,
        profile=profile,
        collection_prefix=collection_prefix,
    )
    if limit > 0:
        cases = cases[:limit]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    judge_model = None
    if judge != "off":
        judge_model, judge_error = build_optional_judge(dotenv_path=dotenv_path)
        if judge_model is None:
            print(f"Judge unavailable; using deterministic scoring only. Reason: {judge_error}")

    results: list[PublicBenchmarkResult] = []
    raw_dir = output_dir / "raw"
    with httpx.Client(timeout=httpx.Timeout(timeout_seconds)) as client:
        for case in cases:
            missing = _missing_capabilities(case, available_capabilities)
            if missing:
                result = skipped_result(case, missing)
            else:
                result = run_gateway_case(
                    client,
                    case,
                    api_base=api_base,
                    model=model,
                    run_id=run_id,
                    token=token,
                    judge_model=judge_model,
                    raw_dir=raw_dir,
                )
            results.append(result)
            status = "SKIP" if result.skipped else "PASS" if result.passed else "FAIL"
            print(f"[{status}] {case.benchmark_id}:{case.test_id} {result.latency_seconds:.1f}s")
            if fail_fast and result.error:
                break

    summary = summarize_results(results, profile=profile)
    write_outputs(output_dir, results, summary)
    _print_summary(summary, output_dir=output_dir)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate public benchmark adapters against the live agent gateway.")
    parser.add_argument("--profile", default="smoke", choices=sorted(PROFILE_LIMITS))
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--data-root", type=Path, default=Path("data/public_benchmarks"))
    parser.add_argument("--collection-prefix", default="public")
    parser.add_argument("--api-base", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default=os.environ.get("GATEWAY_MODEL_ID", "enterprise-agent"))
    parser.add_argument("--token-env", default="GATEWAY_SHARED_BEARER_TOKEN")
    parser.add_argument("--token", default="")
    parser.add_argument("--judge", default="auto", choices=["auto", "off"])
    parser.add_argument("--dotenv", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--available-capability", action="append", default=None)
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
    output_dir = args.output_dir or Path("data/runtime/public_benchmark_runs") / timestamp
    run_public_benchmark_suite(
        benchmarks=args.benchmarks,
        profile=args.profile,
        data_root=args.data_root,
        collection_prefix=args.collection_prefix,
        api_base=args.api_base,
        model=args.model,
        token=token,
        judge=args.judge,
        output_dir=output_dir,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        fail_fast=args.fail_fast,
        available_capabilities=args.available_capability or DEFAULT_AVAILABLE_CAPABILITIES,
        dotenv_path=args.dotenv,
    )
    return 0


__all__ = [
    "DEFAULT_AVAILABLE_CAPABILITIES",
    "DEFAULT_BENCHMARKS",
    "PROFILE_LIMITS",
    "PublicBenchmarkCase",
    "PublicBenchmarkResult",
    "PublicBenchmarkSummary",
    "answer_token_f1",
    "build_gateway_headers",
    "build_gateway_payload",
    "classify_failures",
    "load_beir_cases",
    "load_bfcl_cases",
    "load_hotpotqa_cases",
    "load_public_cases",
    "load_ragbench_cases",
    "load_taubench_cases",
    "main",
    "parse_benchmark_specs",
    "retrieval_metrics",
    "run_public_benchmark_suite",
    "score_public_case",
    "summarize_results",
    "write_outputs",
]
