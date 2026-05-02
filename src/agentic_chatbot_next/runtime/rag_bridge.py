from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, TYPE_CHECKING

from agentic_chatbot_next.rag.fanout import (
    RagRuntimeBridge,
    RagSearchBatchResult,
    RagSearchTask,
    RagSearchTaskResult,
    TabularEvidenceBatchResult,
    TabularEvidenceResult,
    TabularEvidenceTask,
)
from agentic_chatbot_next.runtime.task_plan import WorkerExecutionRequest
from agentic_chatbot_next.utils.json_utils import extract_json

if TYPE_CHECKING:
    from agentic_chatbot_next.contracts.messages import SessionState
    from agentic_chatbot_next.runtime.kernel import RuntimeKernel


class KernelRagRuntimeBridge(RagRuntimeBridge):
    def __init__(self, kernel: "RuntimeKernel", session_state: "SessionState") -> None:
        self.kernel = kernel
        self.session_state = session_state

    def can_run_parallel(self, *, task_count: int) -> bool:
        if int(task_count) <= 1:
            return False
        if self.kernel._uses_local_ollama_workers():
            return False
        return self.kernel._can_identify_non_ollama_worker_runtime()

    def run_search_tasks(self, tasks: Sequence[RagSearchTask]) -> RagSearchBatchResult:
        if not tasks:
            return RagSearchBatchResult()

        parallel = self.can_run_parallel(task_count=len(tasks))
        jobs: List[tuple[str, RagSearchTask, Any]] = []
        route_context = dict(self.session_state.metadata.get("route_context") or {})

        for task in tasks:
            scoped_state = self.kernel._build_scoped_worker_state(self.session_state, agent_name="rag_worker")
            worker_request = WorkerExecutionRequest(
                agent_name="rag_worker",
                task_id=task.task_id,
                title=task.title,
                prompt=task.query,
                instruction_prompt=task.query,
                semantic_query=task.query,
                description=task.title[:120],
                doc_scope=list(task.doc_scope),
                research_profile=task.research_profile,
                coverage_goal=task.coverage_goal,
                result_mode=task.result_mode,
                controller_hints=dict(task.controller_hints),
                metadata={
                    "rag_search_task": task.to_dict(),
                    "answer_mode": task.answer_mode,
                },
            )
            job = self.kernel.job_manager.create_job(
                agent_name="rag_worker",
                prompt=task.query,
                session_id=self.session_state.session_id,
                description=task.title[:120],
                tenant_id=self.session_state.tenant_id,
                user_id=self.session_state.user_id,
                priority=str(task.controller_hints.get("priority") or "interactive"),
                queue_class=str(task.controller_hints.get("queue_class") or task.controller_hints.get("priority") or "interactive"),
                session_state=scoped_state.to_dict(),
                metadata={
                    "session_state": scoped_state.to_dict(),
                    "worker_request": worker_request.to_dict(),
                    "route_context": route_context,
                },
            )
            jobs.append((job.job_id, task, job))

        real_jobs = [record for _, _, record in jobs]
        if parallel:
            for job in real_jobs:
                self.kernel.job_manager.start_background_job(job, self.kernel._job_runner)
            self.kernel._wait_for_jobs([job.job_id for job in real_jobs])
        else:
            for job in real_jobs:
                self.kernel.job_manager.run_job_inline(job, self.kernel._job_runner)

        results: List[RagSearchTaskResult] = []
        warnings: List[str] = []
        for job_id, task, record in jobs:
            job = self.kernel.job_manager.get_job(job_id) or record
            payload = self._extract_result_payload(job)
            if payload:
                task_result = RagSearchTaskResult.from_dict(payload)
            else:
                task_result = RagSearchTaskResult(task_id=task.task_id)
            job_warning = str(getattr(job, "last_error", "") or "").strip()
            if job_warning:
                task_result.warnings.append(job_warning)
            if not payload and str(getattr(job, "result_summary", "") or "").strip():
                task_result.warnings.append(str(job.result_summary))
            if task_result.warnings:
                warnings.extend(task_result.warnings)
            results.append(task_result)
        return RagSearchBatchResult(
            results=results,
            warnings=warnings,
            parallel_workers_used=parallel,
        )

    def run_tabular_evidence_tasks(self, tasks: Sequence[TabularEvidenceTask]) -> TabularEvidenceBatchResult:
        if not tasks:
            return TabularEvidenceBatchResult()

        jobs: List[tuple[str, TabularEvidenceTask, Any, str]] = []
        route_context = dict(self.session_state.metadata.get("route_context") or {})
        for task in tasks:
            prompt = self._tabular_worker_prompt(task)
            scoped_state = self.kernel._build_scoped_worker_state(self.session_state, agent_name="data_analyst")
            worker_request = WorkerExecutionRequest(
                agent_name="data_analyst",
                task_id=task.task_id,
                title=f"Tabular evidence: {task.title}",
                prompt=prompt,
                instruction_prompt=prompt,
                semantic_query=task.query,
                description=f"Profile/analyze spreadsheet evidence for RAG: {task.title}"[:120],
                doc_scope=[task.doc_id] if task.doc_id else [],
                result_mode="tabular_evidence",
                controller_hints={"tabular_evidence": True},
                metadata={
                    "tabular_evidence_task": task.to_dict(),
                    "answer_mode": "tabular_evidence_json",
                },
            )
            job = self.kernel.job_manager.create_job(
                agent_name="data_analyst",
                prompt=prompt,
                session_id=self.session_state.session_id,
                description=f"Tabular evidence: {task.title}"[:120],
                tenant_id=self.session_state.tenant_id,
                user_id=self.session_state.user_id,
                priority="interactive",
                queue_class="interactive",
                session_state=scoped_state.to_dict(),
                metadata={
                    "session_state": scoped_state.to_dict(),
                    "worker_request": worker_request.to_dict(),
                    "route_context": route_context,
                },
            )
            result_text = self.kernel.job_manager.run_job_inline(job, self.kernel._job_runner)
            jobs.append((job.job_id, task, job, result_text))

        results: List[TabularEvidenceResult] = []
        warnings: List[str] = []
        for job_id, task, record, result_text in jobs:
            job = self.kernel.job_manager.get_job(job_id) or record
            payload = self._extract_tabular_result_payload(job, result_text)
            if payload:
                payload.setdefault("task_id", task.task_id)
                result = TabularEvidenceResult.from_dict(payload)
            else:
                result = TabularEvidenceResult(
                    task_id=task.task_id,
                    status="failed",
                    summary="The data analyst did not return structured tabular evidence.",
                    warnings=["TABULAR_ANALYST_UNSTRUCTURED_RESULT"],
                    confidence=0.0,
                )
            job_warning = str(getattr(job, "last_error", "") or "").strip()
            if job_warning:
                result.warnings.append(job_warning)
            if result.warnings:
                warnings.extend(result.warnings)
            results.append(result)
        return TabularEvidenceBatchResult(results=results, warnings=warnings)

    @staticmethod
    def _extract_result_payload(job: Any) -> Dict[str, Any]:
        metadata = dict(getattr(job, "metadata", {}) or {})
        result_metadata = dict(metadata.get("result_metadata") or {})
        payload = dict(result_metadata.get("rag_search_result") or {})
        if payload:
            return payload
        raw_text = str(getattr(job, "result_summary", "") or "").strip()
        if not raw_text:
            return {}
        try:
            parsed = json.loads(raw_text)
        except Exception:
            return {}
        if isinstance(parsed, dict) and isinstance(parsed.get("task_id"), str):
            return parsed
        return {}

    @staticmethod
    def _tabular_worker_prompt(task: TabularEvidenceTask) -> str:
        task_json = json.dumps(task.to_dict(), ensure_ascii=False, indent=2)
        return (
            "You are the data_analyst helping the RAG agent answer a grounded question from a spreadsheet or CSV.\n"
            "Inspect the dataset with `profile_dataset` first. Use `inspect_columns` or `execute_code` only if needed "
            "for the requested lookup, filter, aggregation, comparison, or profiling task.\n"
            "Return ONLY valid JSON matching this schema: "
            "{\"task_id\": string, \"status\": \"ok\"|\"partial\"|\"failed\", \"summary\": string, "
            "\"findings\": array, \"source_refs\": array, \"operations\": array, \"warnings\": array, "
            "\"confidence\": number}.\n"
            "Each source_refs item must include doc_id, title, and any known sheet_name, row_start, row_end, "
            "cell_range, and columns. Do not include markdown outside the JSON.\n"
            f"TABULAR_EVIDENCE_TASK:\n{task_json}\n"
        )

    @staticmethod
    def _extract_tabular_result_payload(job: Any, result_text: str = "") -> Dict[str, Any]:
        metadata = dict(getattr(job, "metadata", {}) or {})
        result_metadata = dict(metadata.get("result_metadata") or {})
        for key in ("tabular_evidence_result", "structured_result"):
            payload = result_metadata.get(key)
            if isinstance(payload, dict):
                return dict(payload)

        candidates = [str(result_text or "").strip(), str(getattr(job, "result_summary", "") or "").strip()]
        for raw_text in candidates:
            if not raw_text:
                continue
            try:
                parsed = json.loads(raw_text)
            except Exception:
                parsed = extract_json(raw_text)
            if isinstance(parsed, dict) and (
                "source_refs" in parsed or "findings" in parsed or "summary" in parsed
            ):
                return dict(parsed)
        return {}
