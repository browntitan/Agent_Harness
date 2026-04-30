from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from agentic_chatbot_next.rag.doc_targets import extract_named_document_targets
from agentic_chatbot_next.rag.hints import (
    coerce_controller_hints,
    infer_rag_execution_hints,
    normalize_structured_query,
)
from agentic_chatbot_next.rag.inventory import extract_requested_kb_collection_id
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_INDEXES,
    classify_inventory_query,
)
from agentic_chatbot_next.runtime.doc_focus import (
    active_doc_focus_controller_hints,
    active_doc_focus_from_metadata,
    is_active_doc_focus_followup,
)
from agentic_chatbot_next.runtime.task_decomposition import (
    is_clause_policy_workflow,
    is_mixed_utility_retrieval_request,
)
from agentic_chatbot_next.runtime.turn_contracts import (
    plan_satisfies_intent,
    resolve_turn_intent,
    resolved_turn_intent_from_metadata,
)

_VALID_EXECUTORS = {
    "rag_worker",
    "utility",
    "data_analyst",
    "general",
    "graph_manager",
    "verifier",
}
_VALID_MODES = {"sequential", "parallel"}
TERMINAL_TASK_STATUSES = {"completed", "failed", "stopped", "waiting_message"}
_WORKBOOK_ANALYSIS_HINTS = re.compile(
    r"\b(budget|schedule|staffing|scorecard|kpi|bom|cost|costs|training|spares|milestone|milestones|variance|supplier|suppliers|price|procurement|resource|deployment|deploy|rollout|ims|risks?)\b",
    flags=re.I,
)
_GRAPH_MUTATION_HINTS = re.compile(
    r"\b(create|build|index|import|refresh|rebuild|update)\s+(?:an?\s+|this\s+)?(?:knowledge\s+)?graph\b|"
    r"\bgraph\s+(?:build|index|import|refresh|rebuild)\b",
    flags=re.I,
)
_GRAPH_QUERY_HINTS = re.compile(
    r"\b(knowledge\s+graph|graph\s+catalog|existing\s+graphs?|what\s+graphs?|"
    r"which\s+graphs?|inspect\s+graph|search\s+graph|use\s+a?\s*graph|should\s+.*use\s+a?\s*graph)\b",
    flags=re.I,
)


@dataclass
class TaskSpec:
    id: str
    title: str
    executor: str
    mode: str
    depends_on: List[str] = field(default_factory=list)
    input: str = ""
    doc_scope: List[str] = field(default_factory=list)
    skill_queries: List[str] = field(default_factory=list)
    research_profile: str = ""
    coverage_goal: str = ""
    result_mode: str = ""
    answer_mode: str = "answer"
    controller_hints: Dict[str, Any] = field(default_factory=dict)
    produces_artifacts: List[str] = field(default_factory=list)
    consumes_artifacts: List[str] = field(default_factory=list)
    handoff_schema: str = ""
    input_artifact_ids: List[str] = field(default_factory=list)
    capability_requirements: Dict[str, Any] = field(default_factory=dict)
    evidence_scope: Dict[str, Any] = field(default_factory=dict)
    loop_over_artifact: str = ""
    parallelization_key: str = ""
    acceptance_criteria: List[str] = field(default_factory=list)
    permission_requirements: List[str] = field(default_factory=list)
    expected_artifacts: List[str] = field(default_factory=list)
    status: str = "pending"
    artifact_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "executor": self.executor,
            "mode": self.mode,
            "depends_on": list(self.depends_on),
            "input": self.input,
            "doc_scope": list(self.doc_scope),
            "skill_queries": list(self.skill_queries),
            "research_profile": self.research_profile,
            "coverage_goal": self.coverage_goal,
            "result_mode": self.result_mode,
            "answer_mode": self.answer_mode,
            "controller_hints": dict(self.controller_hints),
            "produces_artifacts": list(self.produces_artifacts),
            "consumes_artifacts": list(self.consumes_artifacts),
            "handoff_schema": self.handoff_schema,
            "input_artifact_ids": list(self.input_artifact_ids),
            "capability_requirements": dict(self.capability_requirements),
            "evidence_scope": dict(self.evidence_scope),
            "loop_over_artifact": self.loop_over_artifact,
            "parallelization_key": self.parallelization_key,
            "acceptance_criteria": list(self.acceptance_criteria),
            "permission_requirements": list(self.permission_requirements),
            "expected_artifacts": list(self.expected_artifacts),
            "status": self.status,
            "artifact_ref": self.artifact_ref or f"task:{self.id}",
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], *, index: int = 0) -> "TaskSpec":
        task_id = str(raw.get("id") or f"task_{index + 1}")
        title = str(raw.get("title") or f"Task {index + 1}")
        executor = str(raw.get("executor") or "general").strip()
        if executor not in _VALID_EXECUTORS:
            executor = _infer_executor(f"{title}\n{raw.get('input', '')}")
        mode = str(raw.get("mode") or "sequential").strip().lower()
        if mode not in _VALID_MODES:
            mode = "sequential"
        depends_on = [str(item) for item in (raw.get("depends_on") or []) if str(item)]
        doc_scope = [str(item) for item in (raw.get("doc_scope") or []) if str(item)]
        skill_queries = [str(item) for item in (raw.get("skill_queries") or []) if str(item)]
        produces_artifacts = [str(item) for item in (raw.get("produces_artifacts") or []) if str(item)]
        consumes_artifacts = [str(item) for item in (raw.get("consumes_artifacts") or []) if str(item)]
        input_artifact_ids = [str(item) for item in (raw.get("input_artifact_ids") or []) if str(item)]
        acceptance_criteria = [str(item) for item in (raw.get("acceptance_criteria") or []) if str(item)]
        permission_requirements = [str(item) for item in (raw.get("permission_requirements") or []) if str(item)]
        expected_artifacts = [str(item) for item in (raw.get("expected_artifacts") or []) if str(item)]
        status = str(raw.get("status") or "pending")
        artifact_ref = str(raw.get("artifact_ref") or f"task:{task_id}")
        task = cls(
            id=task_id,
            title=title,
            executor=executor,
            mode=mode,
            depends_on=depends_on,
            input=str(raw.get("input") or ""),
            doc_scope=doc_scope,
            skill_queries=skill_queries,
            research_profile=str(raw.get("research_profile") or ""),
            coverage_goal=str(raw.get("coverage_goal") or ""),
            result_mode=str(raw.get("result_mode") or ""),
            answer_mode=str(raw.get("answer_mode") or "answer"),
            controller_hints=coerce_controller_hints(raw.get("controller_hints") or {}),
            produces_artifacts=produces_artifacts,
            consumes_artifacts=consumes_artifacts,
            handoff_schema=str(raw.get("handoff_schema") or ""),
            input_artifact_ids=input_artifact_ids,
            capability_requirements=dict(raw.get("capability_requirements") or {}),
            evidence_scope=dict(raw.get("evidence_scope") or {}),
            loop_over_artifact=str(raw.get("loop_over_artifact") or ""),
            parallelization_key=str(raw.get("parallelization_key") or ""),
            acceptance_criteria=acceptance_criteria,
            permission_requirements=permission_requirements,
            expected_artifacts=expected_artifacts,
            status=status,
            artifact_ref=artifact_ref,
        )
        if task.executor == "rag_worker":
            return _apply_rag_task_defaults(task, query=task.input or f"{title}\n{raw.get('input', '')}")
        return task


@dataclass
class TaskResult:
    task_id: str
    title: str
    executor: str
    status: str
    output: str
    artifact_ref: str
    warnings: List[str] = field(default_factory=list)
    handoff_artifact_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "executor": self.executor,
            "status": self.status,
            "output": self.output,
            "artifact_ref": self.artifact_ref,
            "warnings": list(self.warnings),
            "handoff_artifact_ids": list(self.handoff_artifact_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class WorkerExecutionRequest:
    agent_name: str
    task_id: str
    title: str
    prompt: str
    instruction_prompt: str = ""
    semantic_query: str = ""
    context_summary: str = ""
    description: str = ""
    doc_scope: List[str] = field(default_factory=list)
    skill_queries: List[str] = field(default_factory=list)
    research_profile: str = ""
    coverage_goal: str = ""
    result_mode: str = ""
    answer_mode: str = "answer"
    controller_hints: Dict[str, Any] = field(default_factory=dict)
    artifact_refs: List[str] = field(default_factory=list)
    produces_artifacts: List[str] = field(default_factory=list)
    consumes_artifacts: List[str] = field(default_factory=list)
    handoff_schema: str = ""
    input_artifact_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "title": self.title,
            "prompt": self.prompt,
            "instruction_prompt": self.instruction_prompt,
            "semantic_query": self.semantic_query,
            "context_summary": self.context_summary,
            "description": self.description,
            "doc_scope": list(self.doc_scope),
            "skill_queries": list(self.skill_queries),
            "research_profile": self.research_profile,
            "coverage_goal": self.coverage_goal,
            "result_mode": self.result_mode,
            "answer_mode": self.answer_mode,
            "controller_hints": dict(self.controller_hints),
            "artifact_refs": list(self.artifact_refs),
            "produces_artifacts": list(self.produces_artifacts),
            "consumes_artifacts": list(self.consumes_artifacts),
            "handoff_schema": self.handoff_schema,
            "input_artifact_ids": list(self.input_artifact_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class VerificationResult:
    status: str = "pass"
    verdict: str = "PASS"
    summary: str = ""
    issues: List[str] = field(default_factory=list)
    feedback: str = ""
    parse_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "verdict": self.verdict,
            "summary": self.summary,
            "issues": list(self.issues),
            "feedback": self.feedback,
            "parse_failed": self.parse_failed,
        }


@dataclass
class TaskExecutionState:
    user_request: str
    planner_summary: str
    task_plan: List[Dict[str, Any]] = field(default_factory=list)
    task_results: List[Dict[str, Any]] = field(default_factory=list)
    partial_answer: str = ""
    final_answer: str = ""
    verification: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_request": self.user_request,
            "planner_summary": self.planner_summary,
            "task_plan": [dict(item) for item in self.task_plan],
            "task_results": [dict(item) for item in self.task_results],
            "partial_answer": self.partial_answer,
            "final_answer": self.final_answer,
            "verification": dict(self.verification),
        }


def _infer_executor(text: str) -> str:
    lower = text.lower()
    if _is_graph_mutation_request(text):
        return "general"
    if _is_graph_query_request(text):
        return "graph_manager"
    if any(token in lower for token in ("csv", "excel", "spreadsheet", "dataframe", "pandas")):
        return "data_analyst"
    if (
        any(token in lower for token in ("calculate", "math", "convert", "memory", "remember"))
        or re.search(r"\b(sum|difference|percent|percentage|total|average|mean)\b", lower)
    ):
        return "utility"
    if any(
        token in lower
        for token in ("document", "contract", "policy", "clause", "requirement", "kb", "knowledge base", "upload", "compare")
    ):
        return "rag_worker"
    return "general"


def _is_collection_reference_hint(query: str, hint: str) -> bool:
    candidate = str(hint or "").strip()
    if not candidate:
        return False
    escaped = re.escape(candidate)
    return bool(
        re.search(rf'["\'`]{escaped}["\'`]\s+collection\b', query, flags=re.I)
        or re.search(rf'\bcollection\s+["\'`]{escaped}["\'`]', query, flags=re.I)
        or re.search(rf'["\'`]{escaped}["\'`]\s+(?:knowledge\s*base|kb)\b', query, flags=re.I)
        or re.search(rf'\b(?:knowledge\s*base|kb)\s+["\'`]{escaped}["\'`]', query, flags=re.I)
    )


def _extract_doc_hints(query: str) -> List[str]:
    normalized_query = normalize_structured_query(query) or str(query or "")
    requested_collection_id = extract_requested_kb_collection_id(normalized_query)
    hints: List[str] = []
    seen: set[str] = set()

    def _add_hint(value: str) -> None:
        clean = str(value or "").strip()
        if re.search(r"\.[A-Za-z0-9]{2,8}[.,;:!?]+$", clean):
            clean = re.sub(r"[.,;:!?]+$", "", clean)
        if not clean or clean in seen:
            return
        seen.add(clean)
        hints.append(clean)

    for candidate in extract_named_document_targets(normalized_query):
        _add_hint(candidate)

    for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'|`([^`]+)`', normalized_query):
        candidate = str(match.group(1) or match.group(2) or match.group(3) or "").strip()
        if not candidate:
            continue
        if requested_collection_id and candidate.casefold() == requested_collection_id.casefold() and _is_collection_reference_hint(
            normalized_query,
            candidate,
        ):
            continue
        _add_hint(candidate)

    if hints:
        return hints

    for match in re.findall(r"\b[a-z0-9._/-]+\.(?:md|pdf|docx|txt|csv|xlsx|xls)\b", normalized_query, flags=re.I):
        _add_hint(match)

    if hints:
        return hints

    matches = re.findall(
        r"\b(?:doc(?:ument)?|contract|policy|runbook|file)\s+([a-z0-9._-]+)",
        normalized_query,
        flags=re.I,
    )
    return [match.strip() for match in matches if match.strip()]


def _is_document_research_campaign(query: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"identify\s+all\s+documents|which\s+documents|list\s+(?:all\s+)?(?:documents|files)|"
            r"across\s+(?:the\s+)?(?:corpus|documents|policies|sops)|every\s+document|"
            r"inventory|exhaustive|all\s+sops?|compare\s+all|cross[-\s]?document|"
            r"(?:provide|give|return)\s+(?:me\s+)?(?:only\s+)?(?:a\s+)?list\s+of\s+(?:potential\s+)?(?:documents|files)|"
            r"potential\s+(?:documents|files)\s+(?:about|for)|"
            r"(?:documents|files)\s+that\s+(?:have|contain)\s+information\s+about|"
            r"(?:documents|files)\s+that\s+(?:discuss|describe|cover|contain)|"
            r"search\s+across\s+(?:the\s+)?documents"
            r")\b",
            query,
            flags=re.I,
        )
    )


def _is_holistic_repository_intent(resolved_intent: Any) -> bool:
    answer_contract = getattr(resolved_intent, "answer_contract", None)
    return (
        answer_contract is not None
        and str(getattr(answer_contract, "kind", "") or "").strip() == "grounded_synthesis"
        and str(getattr(answer_contract, "coverage_profile", "") or "").strip() == "holistic_repository"
    )


def _is_graph_mutation_request(query: str) -> bool:
    return bool(_GRAPH_MUTATION_HINTS.search(query))


def _is_graph_query_request(query: str) -> bool:
    return bool(_GRAPH_QUERY_HINTS.search(query))


def _needs_workbook_followup(query: str) -> bool:
    return bool(_WORKBOOK_ANALYSIS_HINTS.search(query))


def _is_data_to_document_handoff_request(query: str) -> bool:
    lower = query.lower()
    data_hint = any(token in lower for token in ("csv", "excel", "spreadsheet", "dataset", "dataframe", "table", "workbook"))
    kb_hint = any(
        token in lower
        for token in (
            "knowledge base",
            "kb",
            "documents",
            "policies",
            "contracts",
            "sops",
            "similar patterns",
            "similar cases",
            "search the corpus",
        )
    )
    return data_hint and kb_hint


def _apply_rag_task_defaults(task: TaskSpec, *, query: str) -> TaskSpec:
    inferred = infer_rag_execution_hints(query, skill_queries=task.skill_queries)
    if not task.research_profile:
        task.research_profile = inferred.research_profile
    if not task.coverage_goal:
        task.coverage_goal = inferred.coverage_goal
    if not task.result_mode:
        task.result_mode = inferred.result_mode
    task.controller_hints = {
        **dict(inferred.controller_hints),
        **dict(task.controller_hints or {}),
    }
    return task


def _build_rag_task(
    *,
    task_id: str,
    title: str,
    mode: str,
    query: str,
    doc_scope: List[str] | None = None,
    skill_queries: List[str] | None = None,
    depends_on: List[str] | None = None,
    answer_mode: str = "answer",
    controller_hints: Dict[str, Any] | None = None,
    produces_artifacts: List[str] | None = None,
    consumes_artifacts: List[str] | None = None,
    handoff_schema: str = "",
) -> TaskSpec:
    task = TaskSpec(
        id=task_id,
        title=title,
        executor="rag_worker",
        mode=mode,
        depends_on=list(depends_on or []),
        input=query,
        doc_scope=list(doc_scope or []),
        skill_queries=list(skill_queries or []),
        answer_mode=str(answer_mode or "answer"),
        controller_hints=dict(controller_hints or {}),
        produces_artifacts=list(produces_artifacts or []),
        consumes_artifacts=list(consumes_artifacts or []),
        handoff_schema=handoff_schema,
    )
    return _apply_rag_task_defaults(task, query=query)


def _build_mixed_utility_retrieval_plan(
    query: str,
    *,
    doc_hints: List[str],
    max_tasks: int,
) -> List[Dict[str, Any]]:
    utility_task = TaskSpec(
        id="task_1",
        title="Calculate numeric result",
        executor="utility",
        mode="parallel",
        input=(
            "Use the calculator for the arithmetic slice of this request. "
            "Return the numeric result with a brief explanation and do not answer the document-search slice.\n\n"
            f"REQUEST:\n{query}"
        ),
        skill_queries=["calculator usage"],
        produces_artifacts=["calculation_result"],
        result_mode="calculation",
    )
    retrieval_task = _build_rag_task(
        task_id="task_2",
        title="Search indexed documents",
        mode="parallel",
        query=(
            "Search the indexed documents for the grounded document-search slice of this request. "
            "Return concise evidence, citations when available, and any retrieval warnings. "
            "Do not answer the arithmetic slice.\n\n"
            f"REQUEST:\n{query}"
        ),
        doc_scope=doc_hints,
        skill_queries=[
            "retrieval strategy selection",
            "citation hygiene and synthesis rules",
        ],
        produces_artifacts=["evidence_response"],
    )
    synthesis_task = TaskSpec(
        id="task_3",
        title="Combine mixed-intent results",
        executor="general",
        mode="sequential",
        depends_on=["task_1", "task_2"],
        input=(
            "Combine the completed arithmetic result and indexed-document search result into one answer. "
            "Return both results together, keep retrieval warnings visible, and do not invent missing evidence.\n\n"
            f"REQUEST:\n{query}"
        ),
        skill_queries=["result synthesis"],
        consumes_artifacts=["calculation_result", "evidence_response"],
        result_mode="mixed_intent_answer",
    )
    tasks = [utility_task, retrieval_task]
    if max_tasks >= 3:
        tasks.append(synthesis_task)
    return [task.to_dict() for task in tasks[:max_tasks]]


def _build_clause_redline_policy_plan(
    query: str,
    *,
    analysis_query: str,
    max_tasks: int,
    requested_collection_id: str = "",
) -> List[Dict[str, Any]]:
    requested_collection_id = requested_collection_id or extract_requested_kb_collection_id(analysis_query)
    policy_scope = [requested_collection_id] if requested_collection_id else []
    extract_task = TaskSpec(
        id="task_1",
        title="Extract clauses and redlines",
        executor="general",
        mode="sequential",
        input=(
            "Inspect the uploaded document artifacts and extract every clause plus associated redline or tracked-change context. "
            "Do not search policy guidance in this task. Return structured JSON only with this schema:\n"
            "{"
            '"clauses":[{"clause_id":"...", "clause_text":"...", "redline_text":"...", '
            '"redline_type":"insertion|deletion|modification|comment|unknown", '
            '"source_doc_id":"...", "location":"...", "confidence":0.0}], '
            '"warnings":["..."]'
            "}\n\n"
            f"REQUEST:\n{query}"
        ),
        skill_queries=[
            "document clause extraction",
            "tracked changes redline extraction",
            "structured legal review artifact",
        ],
        produces_artifacts=["clause_redline_inventory"],
        handoff_schema="clause_redline_policy_review",
        capability_requirements={"agents": ["general"], "tools": ["read_indexed_doc", "extract_requirement_statements"]},
        evidence_scope={"source": "uploads", "requires_redline_preservation": True},
        acceptance_criteria=[
            "Every extracted item has clause_id, clause_text, source_doc_id, location, and confidence.",
            "Redlines are represented separately from flattened clause text when available.",
            "Extraction warnings call out any unsupported tracked-change fidelity gaps.",
        ],
        expected_artifacts=["clause_redline_inventory"],
    )
    lookup_task = _build_rag_task(
        task_id="task_2",
        title="Search internal policy guidance per clause",
        mode="sequential",
        query=(
            "For each clause/redline in the clause_redline_inventory artifact, search the selected internal policy guidance collection. "
            "Return one policy evidence record per clause_id, including explicit no-evidence records when no policy guidance is found. "
            "Do not draft buyer-facing language in this task.\n\n"
            f"REQUEST:\n{query}"
        ),
        doc_scope=policy_scope,
        skill_queries=[
            "policy guidance retrieval",
            "per item evidence fanout",
            "citation hygiene and synthesis rules",
        ],
        depends_on=["task_1"],
        answer_mode="evidence_only",
        controller_hints={
            "research_workflow": "clause_redline_policy_review",
            "workflow_phase": "policy_lookup_fanout",
            "retrieval_scope_mode": "kb_only",
            "strict_kb_scope": bool(requested_collection_id),
            **({"requested_kb_collection_id": requested_collection_id, "search_collection_ids": [requested_collection_id]} if requested_collection_id else {}),
        },
        produces_artifacts=["policy_guidance_matches"],
        consumes_artifacts=["clause_redline_inventory"],
        handoff_schema="clause_redline_policy_review",
    )
    lookup_task.capability_requirements = {
        "agents": ["rag_worker"],
        "collections": policy_scope,
        "tools": ["search_indexed_docs", "read_indexed_doc"],
    }
    lookup_task.evidence_scope = {"source": "knowledge_base", "collection_id": requested_collection_id}
    lookup_task.loop_over_artifact = "clause_redline_inventory.clauses"
    lookup_task.parallelization_key = "clause_id"
    lookup_task.acceptance_criteria = [
        "Each clause_id has policy evidence or an explicit no_evidence result.",
        "Policy summaries preserve citation IDs and collection scope.",
        "No uploaded-document clause text is treated as policy guidance.",
    ]
    lookup_task.expected_artifacts = ["policy_guidance_matches"]

    verify_task = TaskSpec(
        id="task_3",
        title="Verify clause coverage",
        executor="verifier",
        mode="sequential",
        depends_on=["task_2"],
        input=(
            "Audit the clause_redline_inventory and policy_guidance_matches artifacts. "
            "Verify that every clause_id has either cited policy evidence or a documented no-evidence result. "
            "Return JSON only with verdict PASS, FAIL, or PARTIAL plus missing clause_ids and risks."
        ),
        consumes_artifacts=["clause_redline_inventory", "policy_guidance_matches"],
        produces_artifacts=["policy_coverage_verification"],
        handoff_schema="clause_redline_policy_review",
        capability_requirements={"agents": ["verifier", "general"], "tools": ["read_indexed_doc"]},
        evidence_scope={"source": "artifacts", "requires_complete_clause_coverage": True},
        acceptance_criteria=[
            "Verdict is PASS, FAIL, or PARTIAL.",
            "Missing or weak-evidence clause_ids are listed explicitly.",
        ],
        expected_artifacts=["policy_coverage_verification"],
    )
    synthesis_task = TaskSpec(
        id="task_4",
        title="Draft buyer recommendation table",
        executor="general",
        mode="sequential",
        depends_on=["task_3"],
        input=(
            "Use the verified clause/redline extraction and policy evidence to prepare a buyer-facing recommendation table. "
            "Columns must be: clause/redline, supplier position, internal policy guidance summary, recommended buyer response, "
            "risk level, citations/evidence, unresolved questions. Do not invent policy guidance where evidence is missing.\n\n"
            f"REQUEST:\n{query}"
        ),
        consumes_artifacts=["clause_redline_inventory", "policy_guidance_matches", "policy_coverage_verification"],
        produces_artifacts=["buyer_recommendation_table"],
        handoff_schema="clause_redline_policy_review",
        capability_requirements={"agents": ["general"], "collections": policy_scope},
        evidence_scope={"source": "mixed_upload_and_policy_kb", "requires_citations": True},
        acceptance_criteria=[
            "Final table includes all requested columns.",
            "Recommendations are grounded in cited internal guidance or marked unresolved.",
            "Risk level is provided for every clause/redline.",
        ],
        expected_artifacts=["buyer_recommendation_table"],
    )
    tasks = [extract_task, lookup_task, verify_task, synthesis_task]
    return [task.to_dict() for task in tasks[:max_tasks]]


def _research_inventory_controller_hints(
    analysis_query: str,
    *,
    workflow_phase: str,
    requested_collection_id: str = "",
    final_output_mode: str = "document_titles_only",
) -> Dict[str, Any]:
    hints: Dict[str, Any] = {
        "research_workflow": "multi_step_document_discovery",
        "workflow_phase": workflow_phase,
        "retrieval_scope_mode": "kb_only",
        "strict_kb_scope": True,
        "prefer_inventory_output": True,
        "prefer_windowed_keyword_followup": True,
        "final_output_mode": final_output_mode,
    }
    if requested_collection_id:
        hints["requested_kb_collection_id"] = requested_collection_id
        hints["kb_collection_id"] = requested_collection_id
        hints["search_collection_ids"] = [requested_collection_id]
    if re.search(r"\b(process\s+flows?|workflows?|handoff|approval\s+flows?|escalation)\b", analysis_query, flags=re.I):
        hints["prefer_process_flow_docs"] = True
    return hints


def _build_multi_step_research_inventory_plan(
    query: str,
    *,
    analysis_query: str,
    doc_hints: List[str],
    final_output_mode: str,
) -> List[Dict[str, Any]]:
    requested_collection_id = extract_requested_kb_collection_id(analysis_query)
    title_scan_hints = _research_inventory_controller_hints(
        analysis_query,
        workflow_phase="title_path_scan",
        requested_collection_id=requested_collection_id,
        final_output_mode=final_output_mode,
    )
    seed_hints = _research_inventory_controller_hints(
        analysis_query,
        workflow_phase="seed_scan",
        requested_collection_id=requested_collection_id,
        final_output_mode=final_output_mode,
    )
    seed_hints["round_budget"] = 2
    seed_hints["retrieval_strategies"] = ["hybrid", "keyword"]
    seed_hints["max_reflection_rounds"] = 2
    seed_hints["force_deep_search"] = True
    seed_hints["prefer_full_reads"] = True

    tasks: List[TaskSpec] = [
        TaskSpec(
            id="task_1",
            title="Scan title and path candidates",
            executor="general",
            mode="sequential",
            input=(
                "Use `search_indexed_docs` to scan the indexed knowledge base for likely candidate documents. "
                "Run the search on the normalized user request plus 1-3 focused key phrases when that improves recall. "
                "Prefer titles and paths that look directly relevant to the request, but keep a few near matches when they add a distinct subsystem or architecture angle. "
                "Return JSON only with this schema:\n"
                "{"
                '"documents":[{"doc_id":"...", "title":"...", "source_path":"...", "match_reason":"...", "score":0.0}], '
                '"query_variants":["..."], '
                f'"scope_collection_id":"{requested_collection_id or ""}"'
                "}\n"
                "Do not include prose outside JSON.\n\n"
                f"REQUEST:\n{query}"
            ),
            doc_scope=doc_hints,
            skill_queries=[
                "document title candidate search",
                "corpus discovery",
            ],
            produces_artifacts=["title_candidates"],
            handoff_schema="research_inventory",
            controller_hints=title_scan_hints,
        ),
        _build_rag_task(
            task_id="task_2",
            title="Seed corpus scan",
            mode="sequential",
            query=(
                "Search only the knowledge base corpus for the best seed documents for this research request. "
                "Use any title/path candidates as seed hints, but still search broadly enough to discover stronger evidence. "
                "Find the strongest grounded seed documents that likely describe the major subsystems, architecture, control flow, "
                "or system boundaries relevant to the request. Return evidence only for coordinator use.\n\n"
                f"REQUEST:\n{query}"
            ),
            doc_scope=doc_hints,
            skill_queries=[
                "corpus discovery",
                "cross document inventory",
                "coverage sufficiency audit",
            ],
            answer_mode="evidence_only",
            controller_hints=seed_hints,
            produces_artifacts=["doc_focus"],
            consumes_artifacts=["title_candidates"],
            handoff_schema="research_inventory",
        ),
        TaskSpec(
            id="task_3",
            title="Inspect seed documents and extract research facets",
            executor="general",
            mode="sequential",
            depends_on=["task_1", "task_2"],
            input=(
                "Use the structured title_candidates and doc_focus handoffs to select up to 3 promising seed documents. "
                "Use read_indexed_doc to inspect those files directly, prioritizing full or section-first coverage instead of a single overview pass. "
                "Extract 2-4 likely subsystem or architectural facets that should be searched independently across the KB. "
                "Also identify any unresolved questions and up to 6 preferred review documents for deeper per-document analysis. "
                "Return JSON only with this schema:\n"
                "{"
                '"facets":[{"name":"...", "aliases":["..."], "rationale":"...", "seed_doc_ids":["..."]}], '
                '"seed_documents":[{"doc_id":"...", "title":"..."}], '
                '"review_documents":[{"doc_id":"...", "title":"...", "source_path":"..."}], '
                '"unresolved_questions":["..."], '
                f'"scope_collection_id":"{requested_collection_id or ""}"'
                "}\n"
                "Do not include prose outside JSON."
            ),
            doc_scope=doc_hints,
            consumes_artifacts=["title_candidates", "doc_focus"],
            produces_artifacts=["research_facets"],
            handoff_schema="research_inventory",
            controller_hints=_research_inventory_controller_hints(
                analysis_query,
                workflow_phase="seed_inspection",
                requested_collection_id=requested_collection_id,
                final_output_mode=final_output_mode,
            ),
        ),
        TaskSpec(
            id="task_4",
            title="Expand facet searches",
            executor="general",
            mode="sequential",
            depends_on=["task_3"],
            input="Expand research facets into parallel KB facet searches.",
            consumes_artifacts=["research_facets"],
            handoff_schema="research_inventory",
            controller_hints={
                **_research_inventory_controller_hints(
                    analysis_query,
                    workflow_phase="facet_fanout",
                    requested_collection_id=requested_collection_id,
                    final_output_mode=final_output_mode,
                ),
                "dynamic_facet_fanout": True,
                "max_parallel_facets": 4,
            },
        ),
        TaskSpec(
            id="task_5",
            title="Expand shallow document triage",
            executor="general",
            mode="sequential",
            depends_on=["task_4"],
            input=(
                "Build a ranked shortlist from title candidates, seed evidence, and facet matches, then expand parallel shallow document triage tasks. "
                "Each triage task should produce a compact research_triage_note before any expensive full-document review."
            ),
            consumes_artifacts=["title_candidates", "doc_focus", "research_facets", "facet_matches"],
            handoff_schema="research_inventory",
            controller_hints={
                **_research_inventory_controller_hints(
                    analysis_query,
                    workflow_phase="doc_triage_fanout",
                    requested_collection_id=requested_collection_id,
                    final_output_mode=final_output_mode,
                ),
                "dynamic_triage_fanout": True,
                "max_parallel_triage": 6,
                "max_optional_triage": 2,
                "prefer_structured_final_answer": True,
            },
        ),
        TaskSpec(
            id="task_6",
            title="Expand document review",
            executor="general",
            mode="sequential",
            depends_on=["task_5"],
            input=(
                "Use the shallow research_triage_note handoffs to select only relevant or partially relevant documents for full review. "
                "Do not deeply review documents triaged as irrelevant unless they are the only available evidence for a unique facet."
            ),
            consumes_artifacts=["title_candidates", "doc_focus", "research_facets", "facet_matches", "research_triage_note"],
            handoff_schema="research_inventory",
            controller_hints={
                **_research_inventory_controller_hints(
                    analysis_query,
                    workflow_phase="doc_review_fanout",
                    requested_collection_id=requested_collection_id,
                    final_output_mode=final_output_mode,
                ),
                "dynamic_doc_review_fanout": True,
                "max_parallel_doc_reviews": 4,
                "max_optional_doc_reviews": 2,
                "prefer_structured_final_answer": True,
            },
        ),
        TaskSpec(
            id="task_7",
            title="Consolidate subsystem inventory",
            executor="general",
            mode="sequential",
            depends_on=["task_6"],
            input=(
                "Use the structured title_candidates, research_facets, facet_matches, research_triage_note, and doc_digest handoffs to build a reviewed shortlist and a subsystem inventory. "
                "Rank documents by reviewed relevance first, then facet breadth, then evidence strength, then title/path score. "
                "Exclude documents marked `irrelevant`. Keep `partial` documents only when they add a unique facet or subsystem. "
                "Return JSON only with this schema:\n"
                "{"
                '"subsystems":[{"name":"...", "aliases":["..."], "description":"...", '
                '"responsibilities":["..."], "interfaces":["..."], '
                '"supporting_documents":[{"doc_id":"...", "title":"...", "source_path":"...", "source_type":"..."}], '
                '"supporting_citation_ids":["..."], "coverage":"strong|thin"}], '
                '"source_documents":[{"doc_id":"...", "title":"...", "source_path":"...", "source_type":"..."}], '
                f'"scope_collection_id":"{requested_collection_id or ""}"'
                "}\n"
                "Do not include prose outside JSON."
            ),
            consumes_artifacts=["title_candidates", "research_facets", "facet_matches", "research_triage_note", "doc_digest"],
            produces_artifacts=["subsystem_inventory"],
            handoff_schema="research_inventory",
            controller_hints={
                **_research_inventory_controller_hints(
                    analysis_query,
                    workflow_phase="subsystem_inventory",
                    requested_collection_id=requested_collection_id,
                    final_output_mode=final_output_mode,
                ),
                "prefer_structured_final_answer": True,
            },
        ),
        TaskSpec(
            id="task_8",
            title="Expand subsystem evidence backfill",
            executor="general",
            mode="sequential",
            depends_on=["task_7"],
            input="Expand thin subsystem coverage into targeted KB evidence backfill tasks.",
            consumes_artifacts=["subsystem_inventory"],
            handoff_schema="research_inventory",
            controller_hints={
                **_research_inventory_controller_hints(
                    analysis_query,
                    workflow_phase="subsystem_backfill",
                    requested_collection_id=requested_collection_id,
                    final_output_mode=final_output_mode,
                ),
                "dynamic_subsystem_backfill": True,
                "max_parallel_subsystems": 4,
                "prefer_structured_final_answer": True,
            },
        ),
    ]
    return [task.to_dict() for task in tasks]


def _research_inventory_final_output_mode(resolved_intent: Any) -> str:
    answer_contract = getattr(resolved_intent, "answer_contract", None)
    if answer_contract is None:
        return "document_titles_only"
    if bool(getattr(answer_contract, "prefer_list_only", False)) or str(getattr(answer_contract, "kind", "")).strip() == "inventory":
        return "document_titles_only"
    if str(getattr(answer_contract, "kind", "")).strip() == "grounded_synthesis":
        return "detailed_subsystem_summary"
    return "document_titles_only"


def _active_doc_focus_summary_controller_hints(
    query: str,
    *,
    session_metadata: Dict[str, Any] | None,
    workflow_phase: str,
) -> Dict[str, Any]:
    hints = active_doc_focus_controller_hints(query, session_metadata)
    if not hints:
        return {}
    hints["research_workflow"] = "active_doc_focus_summary"
    hints["workflow_phase"] = workflow_phase
    hints["strict_doc_focus"] = True
    hints["doc_read_depth"] = "full"
    return hints


def _build_active_doc_focus_summary_plan(
    query: str,
    *,
    session_metadata: Dict[str, Any] | None,
    max_tasks: int,
) -> List[Dict[str, Any]]:
    active_doc_focus = active_doc_focus_from_metadata(session_metadata)
    if active_doc_focus is None:
        return []
    documents = [dict(item) for item in (active_doc_focus.get("documents") or []) if isinstance(item, dict)]
    if not documents:
        return []
    digest_docs = list(documents)
    digest_tasks: List[TaskSpec] = []
    digest_task_ids: List[str] = []

    for index, doc in enumerate(digest_docs, start=1):
        doc_id = str(doc.get("doc_id") or "").strip()
        title = str(doc.get("title") or doc_id or f"Document {index}").strip()
        digest_task_id = f"task_{index}"
        digest_task_ids.append(digest_task_id)
        digest_tasks.append(
            TaskSpec(
                id=digest_task_id,
                title=f"Digest {title}",
                executor="general",
                mode="parallel",
                input=(
                    "Inspect this exact indexed document only. Read it to coverage completion before summarizing. "
                    "Use read_indexed_doc(mode=\"full\") with cursor pagination until you have covered the entire file "
                    "or all major outline sections. Do not rely on a single overview read. Stay strictly inside the scoped doc. "
                    "Identify the major subsystems, responsibilities, interfaces, and cross-cutting architecture details described in the file. "
                    "Return JSON only with this schema:\n"
                    "{"
                    '"document":{"doc_id":"...", "title":"..."}, '
                    '"document_summary":"...", '
                    '"subsystems":[{"name":"...", "aliases":["..."], "description":"...", '
                    '"responsibilities":["..."], "interfaces":["..."], "supporting_citation_ids":["..."]}], '
                    '"responsibilities":["..."], '
                    '"interfaces":["..."], '
                    '"used_citation_ids":["..."]'
                    "}\n"
                    "Do not include prose outside JSON."
                ),
                doc_scope=[doc_id] if doc_id else [title],
                controller_hints=_active_doc_focus_summary_controller_hints(
                    query,
                    session_metadata=session_metadata,
                    workflow_phase="doc_digest",
                ),
                produces_artifacts=["doc_digest"],
                handoff_schema="active_doc_focus_summary",
            )
        )

    consolidation_task_id = f"task_{len(digest_tasks) + 1}"
    backfill_task_id = f"task_{len(digest_tasks) + 2}"
    source_documents_json = json.dumps(
        [
            {
                "doc_id": str(item.get("doc_id") or "").strip(),
                "title": str(item.get("title") or "").strip(),
            }
            for item in digest_docs
        ],
        ensure_ascii=False,
    )
    consolidation_task = TaskSpec(
        id=consolidation_task_id,
        title="Consolidate subsystem inventory",
        executor="general",
        mode="sequential",
        depends_on=list(digest_task_ids),
        input=(
            "Use the structured doc_digest handoffs to merge overlapping subsystem names, aliases, and responsibilities. "
            "Identify which subsystems have thin support and need targeted evidence backfill. Preserve cross-cutting systems "
            "such as routing, observability, memory, tools, and jobs when the doc digests support them. Return JSON only with this schema:\n"
            "{"
            '"subsystems":[{"name":"...", "aliases":["..."], "description":"...", '
            '"responsibilities":["..."], "interfaces":["..."], '
            '"supporting_documents":[{"doc_id":"...", "title":"..."}], '
            '"supporting_citation_ids":["..."], "coverage":"strong|thin"}], '
            f'"source_documents": {source_documents_json}, '
            f'"scope_collection_id":"{active_doc_focus.get("collection_id") or "default"}"'
            "}\n"
            "Do not include prose outside JSON."
        ),
        consumes_artifacts=["doc_digest"],
        produces_artifacts=["subsystem_inventory"],
        handoff_schema="active_doc_focus_summary",
        controller_hints=_active_doc_focus_summary_controller_hints(
            query,
            session_metadata=session_metadata,
            workflow_phase="subsystem_inventory",
        ),
    )
    backfill_task = TaskSpec(
        id=backfill_task_id,
        title="Expand subsystem evidence backfill",
        executor="general",
        mode="sequential",
        depends_on=[consolidation_task_id],
        input="Expand thin subsystem coverage into targeted KB evidence backfill tasks.",
        consumes_artifacts=["subsystem_inventory"],
        handoff_schema="active_doc_focus_summary",
        controller_hints={
            **_active_doc_focus_summary_controller_hints(
                query,
                session_metadata=session_metadata,
                workflow_phase="subsystem_backfill",
            ),
            "dynamic_subsystem_backfill": True,
            "max_parallel_subsystems": 4,
        },
    )
    tasks = [*digest_tasks, consolidation_task, backfill_task]
    return [task.to_dict() for task in tasks]


def build_fallback_plan(
    query: str,
    *,
    max_tasks: int = 8,
    session_metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    analysis_query = normalize_structured_query(query) or str(query or "")
    lower = analysis_query.lower()
    doc_hints = _extract_doc_hints(analysis_query)
    resolved_intent = resolved_turn_intent_from_metadata(session_metadata) or resolve_turn_intent(
        analysis_query,
        session_metadata or {},
    )

    if is_active_doc_focus_followup(analysis_query, session_metadata):
        plan = _build_active_doc_focus_summary_plan(
            query,
            session_metadata=session_metadata,
            max_tasks=max_tasks,
        )
        if plan:
            return plan

    if classify_inventory_query(analysis_query) == INVENTORY_QUERY_GRAPH_INDEXES:
        task = TaskSpec(
            id="task_1",
            title="List available graph indexes",
            executor="general",
            mode="sequential",
            input=(
                "Use `list_graph_indexes` first to list the managed graph indexes available to this tenant. "
                "Keep the answer lightweight and inventory-shaped. "
                "Only call `inspect_graph_index` if the user asks for details about a specific graph or readiness state. "
                "Do not call document-search, retrieval, worker-orchestration, or graph-search tools for a plain availability question.\n\n"
                f"REQUEST:\n{query}"
            ),
            skill_queries=[
                "graph catalog inspection",
                "graph inventory response shaping",
            ],
            result_mode="inventory",
            controller_hints={
                "preferred_sources": ["graph"],
                "prefer_inventory_output": True,
                "graph_inventory_only": True,
                "avoid_graph_search": True,
            },
        )
        return [task.to_dict()]

    if max_tasks >= 4 and is_clause_policy_workflow(
        analysis_query,
        session_metadata={
            **dict(session_metadata or {}),
            "has_uploads": bool(dict(session_metadata or {}).get("uploaded_doc_ids")),
        },
    ):
        return _build_clause_redline_policy_plan(
            query,
            analysis_query=analysis_query,
            max_tasks=max_tasks,
            requested_collection_id=str(
                dict(session_metadata or {}).get("requested_kb_collection_id")
                or dict(session_metadata or {}).get("kb_collection_id")
                or ""
            ).strip(),
        )

    if max_tasks >= 2 and is_mixed_utility_retrieval_request(
        analysis_query,
        session_metadata=session_metadata,
    ):
        return _build_mixed_utility_retrieval_plan(
            query,
            doc_hints=doc_hints,
            max_tasks=max_tasks,
        )

    if (
        _is_holistic_repository_intent(resolved_intent)
        and max_tasks >= 2
        and not _needs_workbook_followup(analysis_query)
    ):
        return _build_multi_step_research_inventory_plan(
            query,
            analysis_query=analysis_query,
            doc_hints=doc_hints,
            final_output_mode=_research_inventory_final_output_mode(resolved_intent),
        )

    if (
        len(doc_hints) == 1
        and not resolved_intent.answer_contract.broad_coverage
        and not _is_document_research_campaign(analysis_query)
    ):
        task = TaskSpec(
            id="task_1",
            title="Handle user request",
            executor="rag_worker",
            mode="sequential",
            input=query,
            doc_scope=doc_hints,
            skill_queries=[
                "document resolution and ambiguity handling",
                "retrieval strategy selection",
                "citation hygiene and synthesis rules",
            ],
        )
        task = _apply_rag_task_defaults(task, query=analysis_query)
        return [task.to_dict()]

    if (
        resolved_intent.answer_contract.kind == "grounded_synthesis"
        and resolved_intent.answer_contract.broad_coverage
        and max_tasks >= 3
        and not _needs_workbook_followup(analysis_query)
    ):
        return _build_multi_step_research_inventory_plan(
            query,
            analysis_query=analysis_query,
            doc_hints=doc_hints,
            final_output_mode=_research_inventory_final_output_mode(resolved_intent),
        )

    if _is_graph_mutation_request(analysis_query):
        task = TaskSpec(
            id="task_1",
            title="Explain admin-managed graph workflow",
            executor="general",
            mode="sequential",
            input=query,
            doc_scope=doc_hints,
            skill_queries=[
                "graph workspace admin workflow",
                "graph catalog inspection",
            ],
            controller_hints={"preferred_sources": ["graph"]},
        )
        return [task.to_dict()]

    if _is_graph_query_request(analysis_query):
        graph_skill_queries = [
            "graph catalog inspection",
            "graph retrieval planning",
            "source planning",
        ]
        controller_hints = {"preferred_sources": ["graph"]}
        if "should" in lower and "graph" in lower:
            controller_hints["prefer_graph"] = True
        task = TaskSpec(
            id="task_1",
            title="Handle graph lifecycle request",
            executor="graph_manager",
            mode="sequential",
            input=query,
            doc_scope=doc_hints,
            skill_queries=graph_skill_queries,
            controller_hints=controller_hints,
        )
        return [task.to_dict()]

    if _is_data_to_document_handoff_request(analysis_query) and max_tasks >= 2:
        tasks = [
            TaskSpec(
                id="task_1",
                title="Analyze structured data",
                executor="data_analyst",
                mode="sequential",
                input=query,
                skill_queries=["dataset inspection", "analysis planning", "return file handoff"],
                produces_artifacts=["analysis_summary", "entity_candidates", "keyword_windows"],
                handoff_schema="analysis_to_rag",
            ),
            _build_rag_task(
                task_id="task_2",
                title="Search corpus from analyst findings",
                mode="sequential",
                query=f"Use the analyst findings to search the knowledge base for relevant corroborating documents: {query}",
                doc_scope=doc_hints,
                skill_queries=[
                    "corpus discovery",
                    "windowed keyword followup",
                    "cross document inventory",
                ],
                depends_on=["task_1"],
            ),
        ]
        tasks[1].consumes_artifacts = ["analysis_summary", "entity_candidates", "keyword_windows"]
        tasks[1].handoff_schema = "analysis_to_rag"
        return [task.to_dict() for task in tasks[:max_tasks]]

    if any(token in lower for token in ("compare", "diff", "difference")) and len(doc_hints) >= 2:
        tasks: List[TaskSpec] = []
        for index, hint in enumerate(doc_hints[: max_tasks - 1], start=1):
            task_id = f"task_{index}"
            tasks.append(
                _build_rag_task(
                    task_id=task_id,
                    title=f"Analyze {hint}",
                    mode="parallel",
                    query=f"Analyze the document or source '{hint}' for the user's comparison request: {query}",
                    doc_scope=[hint],
                    skill_queries=[
                        "document resolution and ambiguity handling",
                        "comparison campaign",
                        "citation hygiene and synthesis rules",
                    ],
                )
            )
        return [task.to_dict() for task in tasks[:max_tasks]]

    if _is_document_research_campaign(analysis_query) and max_tasks >= 2:
        if "compare" not in lower and not _needs_workbook_followup(analysis_query):
            return _build_multi_step_research_inventory_plan(
                query,
                analysis_query=analysis_query,
                doc_hints=doc_hints,
                final_output_mode=_research_inventory_final_output_mode(resolved_intent),
            )

        if _needs_workbook_followup(analysis_query) and max_tasks >= 3:
            tasks = [
                _build_rag_task(
                    task_id="task_1",
                    title="Discover relevant documents",
                    mode="sequential",
                    query=f"Identify the most relevant documents, workbooks, and sheets for: {query}",
                    doc_scope=doc_hints,
                    skill_queries=[
                        "corpus discovery",
                        "coverage sufficiency audit",
                        "cross document inventory",
                    ],
                ),
                TaskSpec(
                    id="task_2",
                    title="Inspect workbook and table evidence",
                    executor="data_analyst",
                    mode="sequential",
                    depends_on=["task_1"],
                    input=f"Inspect the staged workbook and table evidence for: {query}",
                    skill_queries=[
                        "dataset inspection",
                        "analysis planning",
                        "return file handoff",
                    ],
                    consumes_artifacts=["doc_focus"],
                    produces_artifacts=["analysis_summary", "entity_candidates", "keyword_windows", "evidence_response"],
                    handoff_schema="doc_focus_to_analysis",
                ),
                _build_rag_task(
                    task_id="task_3",
                    title="Synthesize grounded answer",
                    mode="sequential",
                    query=f"Use the discovered documents and structured workbook findings to answer: {query}",
                    skill_queries=[
                        "windowed keyword followup",
                        "cross document inventory",
                        "coverage sufficiency audit",
                    ],
                    depends_on=["task_1", "task_2"],
                ),
            ]
            tasks[0].produces_artifacts = ["doc_focus"]
            tasks[0].handoff_schema = "doc_focus_to_analysis"
            tasks[2].consumes_artifacts = ["analysis_summary", "entity_candidates", "keyword_windows", "evidence_response", "doc_focus"]
            tasks[2].handoff_schema = "doc_focus_to_analysis"
            return [task.to_dict() for task in tasks[:max_tasks]]

        if "compare" in lower:
            tasks = [
                _build_rag_task(
                    task_id="task_1",
                    title="Discover relevant documents",
                    mode="sequential",
                    query=f"Identify the documents relevant to this comparison request: {query}",
                    doc_scope=doc_hints,
                    skill_queries=[
                        "corpus discovery",
                        "coverage sufficiency audit",
                        "negative evidence reporting",
                    ],
                ),
                _build_rag_task(
                    task_id="task_2",
                    title="Compare discovered documents",
                    mode="sequential",
                    query=f"Compare the discovered documents for this request and preserve per-document differences: {query}",
                    skill_queries=[
                        "comparison campaign",
                        "cross document inventory",
                        "coverage sufficiency audit",
                    ],
                    depends_on=["task_1"],
                ),
            ]
            return [task.to_dict() for task in tasks[:max_tasks]]

        tasks = [
            _build_rag_task(
                task_id="task_1",
                title="Semantic corpus discovery",
                mode="parallel",
                query=f"Search the corpus broadly and identify relevant documents for: {query}",
                doc_scope=doc_hints,
                skill_queries=[
                    "corpus discovery",
                    "coverage sufficiency audit",
                ],
            ),
            _build_rag_task(
                task_id="task_2",
                title="Keyword and heading sweep",
                mode="parallel",
                query=f"Use exact terminology, headings, and section labels to find matching documents for: {query}",
                doc_scope=doc_hints,
                skill_queries=[
                    "windowed keyword followup",
                    "cross document inventory",
                ],
            ),
        ]
        if re.search(r"\b(process\s+flows?|workflows?|flowcharts?|handoff|approval\s+flows?|escalation)\b", analysis_query, flags=re.I):
            tasks.append(
                _build_rag_task(
                    task_id="task_3",
                    title="Process-flow document detection",
                    mode="parallel",
                    query=f"Identify documents that contain explicit workflows, process flows, or approval handoffs for: {query}",
                    doc_scope=doc_hints,
                    skill_queries=[
                        "process flow identification",
                        "cross document inventory",
                    ],
                )
            )
        return [task.to_dict() for task in tasks[:max_tasks]]

    executor = _infer_executor(analysis_query)
    skill_queries: List[str] = []
    if executor == "rag_worker":
        skill_queries = [
            "document resolution and ambiguity handling",
            "retrieval strategy selection",
            "citation hygiene and synthesis rules",
        ]
    elif executor == "data_analyst":
        skill_queries = ["dataset inspection", "analysis planning", "safe code execution"]
    elif executor == "utility":
        skill_queries = ["calculator usage", "memory recall", "document listing"]

    task = TaskSpec(
        id="task_1",
        title="Handle user request",
        executor=executor,
        mode="sequential",
        input=query,
        doc_scope=doc_hints,
        skill_queries=skill_queries,
    )
    if executor == "rag_worker":
        task = _apply_rag_task_defaults(task, query=analysis_query)
    return [task.to_dict()]


def normalise_task_plan(
    raw_tasks: Any,
    *,
    query: str,
    max_tasks: int,
    session_metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    tasks = raw_tasks if isinstance(raw_tasks, list) else []
    if not tasks:
        return build_fallback_plan(query, max_tasks=max_tasks, session_metadata=session_metadata)
    resolved_intent = resolved_turn_intent_from_metadata(session_metadata) or resolve_turn_intent(
        query,
        session_metadata or {},
    )
    normalised = [
        TaskSpec.from_dict(task, index=index).to_dict()
        for index, task in enumerate(tasks[:max_tasks])
        if isinstance(task, dict)
    ]
    analysis_query = normalize_structured_query(query) or str(query or "")
    if is_mixed_utility_retrieval_request(
        analysis_query,
        session_metadata=session_metadata,
    ):
        has_utility = any(str(task.get("executor") or "").strip() == "utility" for task in normalised)
        has_retrieval = any(str(task.get("executor") or "").strip() == "rag_worker" for task in normalised)
        if not (has_utility and has_retrieval):
            return build_fallback_plan(query, max_tasks=max_tasks, session_metadata=session_metadata)
    if _is_document_research_campaign(analysis_query) or _is_holistic_repository_intent(resolved_intent):
        has_title_scan = any(
            "title_candidates" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
            for task in normalised
        )
        has_seed_scan = any(
            str(task.get("answer_mode") or "").strip().lower() == "evidence_only"
            and str(task.get("executor") or "").strip() == "rag_worker"
            for task in normalised
        )
        has_facet_phase = any(
            bool(dict(task.get("controller_hints") or {}).get("dynamic_facet_fanout"))
            or "research_facets" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
            for task in normalised
        )
        has_doc_review_phase = any(
            bool(dict(task.get("controller_hints") or {}).get("dynamic_doc_review_fanout"))
            or "doc_digest" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
            for task in normalised
        )
        has_subsystem_inventory = any(
            "subsystem_inventory" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)]
            for task in normalised
        )
        if not (has_title_scan and has_seed_scan and has_facet_phase and has_doc_review_phase and has_subsystem_inventory):
            return build_fallback_plan(query, max_tasks=max_tasks, session_metadata=session_metadata)
    if is_active_doc_focus_followup(analysis_query, session_metadata):
        has_doc_digest = any("doc_digest" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)] for task in normalised)
        has_subsystem_inventory = any("subsystem_inventory" in [str(item) for item in (task.get("produces_artifacts") or []) if str(item)] for task in normalised)
        if not (has_doc_digest and has_subsystem_inventory):
            return build_fallback_plan(query, max_tasks=max_tasks, session_metadata=session_metadata)
    if not plan_satisfies_intent(normalised, resolved_intent):
        return build_fallback_plan(query, max_tasks=max_tasks, session_metadata=session_metadata)
    return normalised or build_fallback_plan(query, max_tasks=max_tasks, session_metadata=session_metadata)


def completed_task_ids(task_results: List[Dict[str, Any]]) -> set[str]:
    return {
        str(result.get("task_id"))
        for result in task_results
        if isinstance(result, dict) and str(result.get("status")) == "completed"
    }


def attempted_task_ids(task_results: List[Dict[str, Any]]) -> set[str]:
    return {
        str(result.get("task_id"))
        for result in task_results
        if isinstance(result, dict) and str(result.get("task_id"))
    }


def select_execution_batch(
    task_plan: List[Dict[str, Any]],
    task_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    done = completed_task_ids(task_results)
    attempted = attempted_task_ids(task_results)

    ready: List[Dict[str, Any]] = []
    for task in task_plan:
        task_id = str(task.get("id", ""))
        if not task_id or task_id in attempted:
            continue
        dependencies = [str(dep) for dep in (task.get("depends_on") or []) if str(dep)]
        if any(dep not in done for dep in dependencies):
            continue
        ready.append(task)

    if not ready:
        return []

    first = ready[0]
    if str(first.get("mode", "sequential")) != "parallel":
        return [first]

    batch: List[Dict[str, Any]] = []
    for task in ready:
        if str(task.get("mode", "sequential")) != "parallel":
            break
        batch.append(task)
    return batch
