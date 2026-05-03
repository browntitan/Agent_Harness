from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_chatbot_next.authz import (
    access_summary_allowed_ids,
    access_summary_authz_enabled,
    normalize_user_email,
)
from agentic_chatbot_next.capabilities import (
    coerce_effective_capabilities,
    resolve_effective_capabilities,
)
from agentic_chatbot_next.agents.prompt_builder import PromptBuilder
from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.jobs import JobRecord, TaskNotification
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState, utc_now_iso
from agentic_chatbot_next.general_agent import build_react_agent_graph
from agentic_chatbot_next.memory.manager import MemorySelector, MemoryWriteManager
from agentic_chatbot_next.memory.projector import MemoryProjector
from agentic_chatbot_next.memory.extractor import MemoryExtractor
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.memory.scope import MemoryScope
from agentic_chatbot_next.observability.callbacks import (
    RuntimeTraceCallbackHandler,
    get_langchain_callbacks,
)
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.router.feedback_loop import RouterFeedbackLoop
from agentic_chatbot_next.runtime.clarification import (
    ClarificationRequest,
    clarification_turn_metadata,
    clarification_from_metadata,
    is_clarification_turn,
)
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.context_budget import ContextBudgetManager, ContextSection
from agentic_chatbot_next.runtime.doc_focus import doc_focus_result_from_metadata
from agentic_chatbot_next.runtime.event_sink import RuntimeEventSink
from agentic_chatbot_next.runtime.kernel_coordinator import KernelCoordinatorController
from agentic_chatbot_next.runtime.kernel_events import KernelEventController
from agentic_chatbot_next.runtime.kernel_providers import KernelProviderController
from agentic_chatbot_next.runtime.job_manager import AgentDispatchOutcome, RuntimeJobManager
from agentic_chatbot_next.runtime.long_output import LongOutputComposer, LongOutputOptions
from agentic_chatbot_next.providers.output_limits import coerce_optional_positive_int
from agentic_chatbot_next.runtime.notification_store import NotificationStore
from agentic_chatbot_next.runtime.openwebui_helpers import is_openwebui_helper_message
from agentic_chatbot_next.runtime.query_loop import QueryLoop, QueryLoopResult
from agentic_chatbot_next.runtime.rag_bridge import KernelRagRuntimeBridge
from agentic_chatbot_next.runtime.task_plan import (
    TaskResult,
    VerificationResult,
    WorkerExecutionRequest,
)
from agentic_chatbot_next.runtime.task_decomposition import is_clause_policy_workflow
from agentic_chatbot_next.runtime.turn_contracts import (
    filter_context_messages,
    resolve_turn_intent,
    resolved_turn_intent_from_metadata,
)
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore
from agentic_chatbot_next.skills.runtime import SkillRuntime
from agentic_chatbot_next.skills.execution import (
    EXECUTABLE_SKILL_KINDS,
    SkillExecutionConfig,
    SkillExecutionResult,
    build_skill_execution_preview,
)
from agentic_chatbot_next.skills.telemetry import (
    SkillTelemetryEvent,
    coerce_answer_quality,
    compute_skill_health,
    is_scored_answer_quality,
)
from agentic_chatbot_next.rag.retrieval_scope import resolve_upload_collection_id
from agentic_chatbot_next.rag.hints import normalize_structured_query
from agentic_chatbot_next.rag.inventory import (
    dispatch_authoritative_inventory,
    inventory_query_requests_grounded_analysis,
    is_authoritative_inventory_query_type,
)
from agentic_chatbot_next.rag.requirements_service import (
    REQUIREMENTS_WORKFLOW_KIND,
    RequirementExtractionService,
    is_requirements_extraction_request,
)
from agentic_chatbot_next.tools.base import ToolContext
from agentic_chatbot_next.tools.executor import build_agent_tools
from agentic_chatbot_next.tools.policy import ToolPolicyService, tool_allowed_by_selectors
from agentic_chatbot_next.tools.registry import build_tool_definitions
from agentic_chatbot_next.utils.json_utils import extract_json, make_json_compatible

logger = logging.getLogger(__name__)


@dataclass
class AgentRunResult:
    text: str
    messages: List[RuntimeMessage]
    metadata: Dict[str, Any]


def _conversation_history_messages(messages: List[RuntimeMessage]) -> List[Any]:
    return [message.to_langchain() for message in filter_context_messages(messages)]


def _authoritative_inventory_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    view = str(payload.get("view") or "").strip()
    summary: Dict[str, Any] = {"view": view}
    if view == "kb_collections":
        collections = [dict(item) for item in (payload.get("collections") or []) if isinstance(item, dict)]
        graphs = [dict(item) for item in (payload.get("graphs") or []) if isinstance(item, dict)]
        summary.update(
            {
                "collection_count": len(collections),
                "collection_ids": [
                    str(item.get("collection_id") or "").strip()
                    for item in collections
                    if str(item.get("collection_id") or "").strip()
                ],
                "graph_count": len(graphs),
                "graph_ids": [
                    str(item.get("graph_id") or "").strip()
                    for item in graphs
                    if str(item.get("graph_id") or "").strip()
                ],
            }
        )
    elif view == "graph_indexes":
        graphs = [dict(item) for item in (payload.get("graphs") or []) if isinstance(item, dict)]
        summary.update(
            {
                "graph_count": len(graphs),
                "graph_ids": [
                    str(item.get("graph_id") or "").strip()
                    for item in graphs
                    if str(item.get("graph_id") or "").strip()
                ],
                "collection_ids": sorted(
                    {
                        str(item.get("collection_id") or "").strip()
                        for item in graphs
                        if str(item.get("collection_id") or "").strip()
                    }
                ),
            }
        )
    elif view == "session_access":
        uploaded_documents = [dict(item) for item in (payload.get("uploaded_documents") or []) if isinstance(item, dict)]
        summary.update(
            {
                "kb_collection_id": str(payload.get("kb_collection_id") or "").strip(),
                "kb_collection_count": len(payload.get("kb_collections") or []),
                "uploaded_document_count": len(uploaded_documents),
                "has_uploads": bool(payload.get("has_uploads")),
            }
        )
    elif view == "kb_file_inventory":
        documents = [dict(item) for item in (payload.get("documents") or []) if isinstance(item, dict)]
        summary.update(
            {
                "kb_collection_id": str(payload.get("kb_collection_id") or "").strip(),
                "document_count": len(documents),
                "requested_collection_available": bool(payload.get("requested_collection_available", True)),
            }
        )
    elif view == "graph_file_inventory":
        documents = [dict(item) for item in (payload.get("documents") or []) if isinstance(item, dict)]
        summary.update(
            {
                "graph_id": str(payload.get("graph_id") or payload.get("requested_graph_id") or "").strip(),
                "document_count": len(documents),
                "requested_graph_available": bool(payload.get("requested_graph_available", True)),
                "query_ready": bool(payload.get("query_ready")),
            }
        )
    elif view == "namespace_combined_inventory":
        collections = [dict(item) for item in (payload.get("collections") or []) if isinstance(item, dict)]
        graphs = [dict(item) for item in (payload.get("graphs") or []) if isinstance(item, dict)]
        summary.update(
            {
                "namespace_query": str(payload.get("namespace_query") or "").strip(),
                "collection_count": len(collections),
                "graph_count": len(graphs),
            }
        )
    elif view in {"namespace_clarification", "namespace_not_found"}:
        summary["namespace_query"] = str(payload.get("namespace_query") or "").strip()
    return make_json_compatible(summary)


def _truncate_inline(value: Any, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _render_requirements_result(payload: Dict[str, Any]) -> str:
    if payload.get("error"):
        candidates = [
            item
            for item in list(payload.get("candidate_documents") or [])
            if isinstance(item, dict)
        ]
        if candidates:
            options = [
                str(item.get("title") or item.get("doc_id") or "").strip()
                for item in candidates[:8]
                if str(item.get("title") or item.get("doc_id") or "").strip()
            ]
            return (
                "<clarification_request>"
                + json.dumps(
                    {
                        "question": "Which document should I extract requirements from? You can also say \"all documents\".",
                        "reason": "requirements_document_selection",
                        "options": options,
                    },
                    ensure_ascii=False,
                )
                + "</clarification_request>"
            )
        return str(payload.get("error") or "Requirements extraction could not be completed.")

    lines: list[str] = [str(payload.get("summary_text") or "").strip()]
    mode = str(payload.get("mode") or "").strip()
    if mode:
        lines.append(f"Mode: `{mode}`.")
    documents = [item for item in list(payload.get("documents") or []) if isinstance(item, dict)]
    if documents:
        doc_titles = ", ".join(
            f"`{str(item.get('title') or item.get('doc_id') or '').strip()}`"
            for item in documents[:5]
            if str(item.get("title") or item.get("doc_id") or "").strip()
        )
        if doc_titles:
            suffix = "..." if len(documents) > 5 else ""
            lines.append(f"Covered documents: {doc_titles}{suffix}")

    preview_rows = [item for item in list(payload.get("preview_rows") or []) if isinstance(item, dict)]
    if preview_rows:
        columns = [str(item) for item in list(payload.get("preview_columns") or []) if str(item)]
        if not columns:
            columns = ["document_title", "source_location", "requirement_text", "confidence"]
        column_labels = {
            "document_title": "Document",
            "modality": "Modality",
            "location": "Location",
            "source_location": "Source",
            "statement_text": "Requirement",
            "requirement_text": "Requirement",
            "source_structure": "Structure",
            "source_excerpt": "Source Text",
            "risk_rationale": "Risk Rationale",
            "risk_label": "Risk",
            "binding_strength": "Binding",
            "confidence": "Confidence",
        }
        lines.append("")
        lines.append("| " + " | ".join(column_labels.get(column, column.replace("_", " ").title()) for column in columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in preview_rows:
            lines.append(
                "| "
                + " | ".join(
                    _truncate_inline(
                        row.get(column),
                        limit=220 if column in {"requirement_text", "statement_text", "source_excerpt", "risk_rationale"} else 80,
                    ).replace("|", "\\|")
                    for column in columns
                )
                + " |"
            )
    elif int(payload.get("statement_count") or 0) == 0:
        lines.append("No matching requirement statements were found in the covered document scope.")

    artifacts = [item for item in list(payload.get("artifacts") or []) if isinstance(item, dict)]
    if artifacts:
        lines.append("")
        lines.append("Downloadable outputs:")
        for artifact in artifacts:
            label = str(artifact.get("label") or artifact.get("filename") or "download").strip()
            if label:
                lines.append(f"- {label}")

    warnings = [str(item).strip() for item in list(payload.get("warnings") or []) if str(item).strip()]
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(line for line in lines if line is not None).strip()


class RuntimeKernel:
    def __init__(
        self,
        settings: Any,
        providers: Any | None = None,
        stores: Any | None = None,
        *,
        paths: Optional[RuntimePaths] = None,
        registry: Optional[AgentRegistry] = None,
        query_loop: Optional[QueryLoop] = None,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.stores = stores
        self.paths = paths or RuntimePaths.from_settings(settings)
        self.transcript_store = RuntimeTranscriptStore(
            self.paths,
            session_hydrate_window_messages=int(getattr(settings, "session_hydrate_window_messages", 40)),
            session_transcript_page_size=int(getattr(settings, "session_transcript_page_size", 100)),
        )
        self.event_controller = KernelEventController(settings, self.transcript_store)
        self.event_sink = self.event_controller.event_sink
        self.context_budget_manager = ContextBudgetManager(
            settings,
            transcript_store=self.transcript_store,
            event_sink=self.event_sink,
        )
        self.router_feedback = RouterFeedbackLoop(
            self.paths,
            settings,
            emit_event=lambda event_type, session_id, payload: self.event_controller.emit(
                event_type,
                session_id,
                agent_name="router_feedback",
                payload=payload,
            ),
        )
        self.notification_store = NotificationStore(self.transcript_store)
        self.job_manager = RuntimeJobManager(
            self.transcript_store,
            settings=self.settings,
            event_sink=self.event_sink,
            max_worker_concurrency=int(getattr(settings, "max_worker_concurrency", 4)),
        )
        agents_dir = Path(getattr(settings, "agents_dir", Path("data") / "agents"))
        agent_overlay_dir = getattr(settings, "control_panel_agent_overlays_dir", None)
        prompt_overlay_dir = getattr(settings, "control_panel_prompt_overlays_dir", None)
        self.registry = registry or AgentRegistry(
            agents_dir,
            overlay_dir=Path(agent_overlay_dir) if agent_overlay_dir is not None else None,
        )
        self.prompt_builder = PromptBuilder(
            Path(getattr(settings, "skills_dir", Path("data") / "skills")),
            overlay_dir=Path(prompt_overlay_dir) if prompt_overlay_dir is not None else None,
        )
        self.skill_runtime = SkillRuntime(settings, stores, self.prompt_builder) if stores is not None else None
        self.query_loop = query_loop or QueryLoop(
            settings=settings,
            providers=providers,
            stores=stores,
            skill_runtime=self.skill_runtime,
            context_budget_manager=self.context_budget_manager,
        )
        self.provider_controller = KernelProviderController(
            settings,
            providers,
            event_controller=self.event_controller,
        )
        self.provider_resolver = self.provider_controller.agent_resolver
        self.coordinator_controller = KernelCoordinatorController(self)
        self.tool_policy = ToolPolicyService()
        self.tool_definitions = build_tool_definitions(self)
        if bool(getattr(self.settings, "memory_enabled", True)):
            self.file_memory_store = FileMemoryStore(self.paths)
            self.memory_extractor = MemoryExtractor(self.file_memory_store)
            self.memory_store = getattr(self.stores, "memory_store", None) if self.stores is not None else None
            if self.memory_store is not None:
                self.memory_projector = MemoryProjector(self.memory_store, self.paths)
                self.memory_selector = MemorySelector(self.memory_store, self.settings)
                self.memory_write_manager = MemoryWriteManager(
                    self.memory_store,
                    self.settings,
                    selector=self.memory_selector,
                    projector=self.memory_projector,
                )
            else:
                self.memory_projector = None
                self.memory_selector = None
                self.memory_write_manager = None
        else:
            self.file_memory_store = None
            self.memory_extractor = None
            self.memory_store = None
            self.memory_projector = None
            self.memory_selector = None
            self.memory_write_manager = None
        self._validate_registry()

    def register_live_progress_sink(self, session_id: str, sink: RuntimeEventSink) -> None:
        self.event_controller.register_live_progress_sink(session_id, sink)

    def unregister_live_progress_sink(self, session_id: str, sink: RuntimeEventSink) -> None:
        self.event_controller.unregister_live_progress_sink(session_id, sink)

    def get_live_progress_sink(self, session_id: str) -> RuntimeEventSink | None:
        return self.event_controller.get_live_progress_sink(session_id)

    def build_rag_runtime_bridge(self, session_state: SessionState) -> KernelRagRuntimeBridge:
        return KernelRagRuntimeBridge(self, session_state)

    def hydrate_session_state(self, session: Any) -> SessionState:
        state = self._load_or_build_session_state(session)
        self._drain_pending_notifications(state)
        return state

    def emit_router_decision(
        self,
        session: Any,
        *,
        router_decision_id: str,
        route: str,
        confidence: float,
        reasons: List[str],
        router_method: str,
        suggested_agent: str,
        force_agent: bool,
        has_attachments: bool,
        requested_agent_override: str = "",
        requested_agent_override_applied: bool = False,
        router_evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.event_controller.emit_router_decision(
            session,
            router_decision_id=router_decision_id,
            route=route,
            confidence=confidence,
            reasons=reasons,
            router_method=router_method,
            suggested_agent=suggested_agent,
            force_agent=force_agent,
            has_attachments=has_attachments,
            requested_agent_override=requested_agent_override,
            requested_agent_override_applied=requested_agent_override_applied,
            router_evidence=router_evidence,
        )

    def build_callbacks(
        self,
        session_or_state: Any,
        *,
        trace_name: str,
        agent_name: str = "",
        job_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        base_callbacks: Optional[List[Any]] = None,
    ) -> List[Any]:
        return self.event_controller.build_callbacks(
            session_or_state,
            trace_name=trace_name,
            agent_name=agent_name,
            job_id=job_id,
            metadata=metadata,
            base_callbacks=base_callbacks,
        )

    def process_turn(
        self,
        session: Any,
        *,
        user_text: str,
        agent_name: Optional[str] = None,
    ) -> str:
        return self.process_agent_turn(session, user_text=user_text, agent_name=agent_name)

    def process_basic_turn(
        self,
        session: Any,
        *,
        user_text: str,
        system_prompt: str,
        chat_llm: Any,
        route_metadata: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        user_already_recorded: bool = False,
        user_message_metadata: Optional[Dict[str, Any]] = None,
        assistant_message_metadata: Optional[Dict[str, Any]] = None,
        skip_post_turn_memory: bool = False,
    ) -> str:
        from agentic_chatbot_next.basic_chat import run_basic_chat

        state = self.hydrate_session_state(session)
        route_metadata_dict = dict(route_metadata or {})
        user_metadata_dict = dict(user_message_metadata or {})
        assistant_metadata_dict = dict(assistant_message_metadata or {})
        internal_helper_turn = bool(
            route_metadata_dict.get("openwebui_helper_task_type")
            or user_metadata_dict.get("openwebui_internal")
            or assistant_metadata_dict.get("openwebui_internal")
        )
        if internal_helper_turn:
            state.metadata["last_openwebui_helper_route_context"] = route_metadata_dict
        else:
            state.metadata["route_context"] = route_metadata_dict
        pending_router_feedback_id = str(route_metadata_dict.get("router_decision_id") or "").strip()
        if pending_router_feedback_id:
            state.metadata["pending_router_feedback_id"] = pending_router_feedback_id
        latest_matches = bool(
            state.messages
            and state.messages[-1].role == "user"
            and state.messages[-1].content == user_text
        )
        if not user_already_recorded or not latest_matches:
            state.append_message("user", user_text, metadata=user_metadata_dict)
            if internal_helper_turn:
                resolved_turn_intent = resolve_turn_intent(user_text, {})
            else:
                resolved_turn_intent = resolve_turn_intent(user_text, dict(state.metadata or {}))
                state.metadata["resolved_turn_intent"] = resolved_turn_intent.to_dict()
            self._persist_state(state)
            self.transcript_store.append_session_transcript(
                state.session_id,
                {"kind": "message", "message": state.messages[-1].to_dict()},
            )
        else:
            if internal_helper_turn:
                resolved_turn_intent = resolve_turn_intent(user_text, {})
            else:
                resolved_turn_intent = resolve_turn_intent(user_text, dict(state.metadata or {}))
                state.metadata["resolved_turn_intent"] = resolved_turn_intent.to_dict()
        budgeted = self.context_budget_manager.prepare_turn(
            agent_name="basic",
            session_state=state,
            user_text=resolved_turn_intent.effective_user_text,
            sections=[
                ContextSection(
                    name="base_prompt",
                    content=system_prompt,
                    priority=100,
                    preserve=True,
                )
            ],
            history_messages=list(filter_context_messages(state.messages[:-1])),
            providers=type("ProviderView", (), {"judge": getattr(self.providers, "judge", None)})()
            if self.providers is not None
            else None,
            transcript_store=self.transcript_store,
            event_sink=self.event_sink,
        )
        model_messages = _conversation_history_messages(budgeted.history_messages)
        system_prompt = budgeted.system_prompt
        basic_callbacks = self.build_callbacks(
            state,
            trace_name="basic_turn",
            agent_name="basic",
            metadata={"route": "BASIC", **route_metadata_dict},
            base_callbacks=callbacks,
        )
        self._emit(
            "basic_turn_started",
            state.session_id,
            agent_name="basic",
            payload={
                "conversation_id": state.conversation_id,
                "user_text": user_text[:500],
                "effective_user_text": resolved_turn_intent.effective_user_text[:500],
                **route_metadata_dict,
            },
        )
        try:
            text = run_basic_chat(
                chat_llm,
                messages=model_messages,
                user_text=resolved_turn_intent.effective_user_text,
                system_prompt=system_prompt,
                callbacks=basic_callbacks,
            )
        except Exception as exc:
            self._persist_state(state)
            state.sync_to_session(session)
            self._emit(
                "basic_turn_failed",
                state.session_id,
                agent_name="basic",
                payload={
                    "conversation_id": state.conversation_id,
                    "error": str(exc)[:1000],
                    **route_metadata_dict,
                },
            )
            raise
        state.append_message(
            "assistant",
            text,
            metadata={
                "agent_name": "basic",
                "context_budget": budgeted.ledger.to_dict(),
                **assistant_metadata_dict,
            },
        )
        self.router_feedback.observe_turn_result(
            state,
            metadata=dict(state.messages[-1].metadata or {}),
            route_context=route_metadata_dict,
        )
        self._sync_pending_clarification(state)
        self._sync_pending_worker_request(state)
        self._sync_active_doc_focus(state)
        self._persist_state(state)
        self.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": state.messages[-1].to_dict()},
        )
        self._emit(
            "basic_turn_completed",
            state.session_id,
            agent_name="basic",
            payload={
                "conversation_id": state.conversation_id,
                "assistant_message_id": state.messages[-1].message_id,
                **route_metadata_dict,
            },
        )
        if not skip_post_turn_memory:
            self._run_post_turn_memory_maintenance(state, latest_text=user_text)
        state.sync_to_session(session)
        return text

    def process_agent_turn(
        self,
        session: Any,
        *,
        user_text: str,
        callbacks: Optional[List[Any]] = None,
        agent_name: Optional[str] = None,
        route_metadata: Optional[Dict[str, Any]] = None,
        chat_max_output_tokens: int | None = None,
    ) -> str:
        state = self.hydrate_session_state(session)
        state.metadata["route_context"] = dict(route_metadata or {})
        pending_router_feedback_id = str((route_metadata or {}).get("router_decision_id") or "").strip()
        if pending_router_feedback_id:
            state.metadata["pending_router_feedback_id"] = pending_router_feedback_id
        state.append_message("user", user_text)
        resolved_turn_intent = resolve_turn_intent(user_text, dict(state.metadata or {}))
        state.metadata["resolved_turn_intent"] = resolved_turn_intent.to_dict()
        self._persist_state(state)
        self.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": state.messages[-1].to_dict()},
        )
        chosen_agent = agent_name or ("coordinator" if getattr(self.settings, "enable_coordinator_mode", False) else "general")
        agent = self._resolve_agent(chosen_agent)
        state.active_agent = agent.name
        self._persist_state(state)
        runtime_callbacks = self.build_callbacks(
            state,
            trace_name="agent_turn",
            agent_name=agent.name,
            metadata={**dict(route_metadata or {}), "requested_agent": agent.name},
            base_callbacks=callbacks,
        )
        self._emit(
            "turn_accepted",
            state.session_id,
            agent_name=agent.name,
            payload={
                "user_text": user_text[:500],
                "effective_user_text": resolved_turn_intent.effective_user_text[:500],
                "answer_contract": resolved_turn_intent.answer_contract.to_dict(),
                "presentation_preferences": resolved_turn_intent.presentation_preferences.to_dict(),
            },
        )
        coverage_profile = str(getattr(resolved_turn_intent.answer_contract, "coverage_profile", "") or "").strip()
        if coverage_profile:
            self._emit(
                "deep_research_intent_classified",
                state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": state.conversation_id,
                    "coverage_profile": coverage_profile,
                    "broad_coverage": bool(resolved_turn_intent.answer_contract.broad_coverage),
                    "answer_contract": resolved_turn_intent.answer_contract.to_dict(),
                    **dict(route_metadata or {}),
                },
            )
        self._emit(
            "agent_run_started",
            state.session_id,
            agent_name=agent.name,
            payload={"mode": agent.mode},
        )
        self._emit(
            "agent_turn_started",
            state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": state.conversation_id,
                "user_text": user_text[:500],
                **dict(route_metadata or {}),
            },
        )
        try:
            result = self.run_agent(
                agent,
                state,
                user_text=resolved_turn_intent.effective_user_text,
                callbacks=runtime_callbacks,
                chat_max_output_tokens=chat_max_output_tokens,
            )
        except Exception as exc:
            self._persist_state(state)
            state.sync_to_session(session)
            self._emit(
                "agent_turn_failed",
                state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": state.conversation_id,
                    "error": str(exc)[:1000],
                    **dict(route_metadata or {}),
                },
            )
            self._emit(
                "turn_failed",
                state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": state.conversation_id,
                    "error": str(exc)[:1000],
                    **dict(route_metadata or {}),
                },
            )
            raise
        self._record_skill_telemetry(
            state,
            agent_name=agent.name,
            user_text=user_text,
            metadata=dict(result.metadata or {}),
        )
        self.router_feedback.observe_turn_result(
            state,
            metadata=dict(result.metadata or {}),
            route_context=dict(route_metadata or {}),
        )
        state.messages = result.messages
        self._sync_pending_clarification(state)
        self._sync_pending_worker_request(state)
        self._sync_active_doc_focus(state)
        self._persist_state(state)
        if state.messages:
            self.transcript_store.append_session_transcript(
                state.session_id,
                {"kind": "message", "message": state.messages[-1].to_dict()},
            )
        self._emit(
            "agent_run_completed",
            state.session_id,
            agent_name=agent.name,
            payload=dict(result.metadata),
        )
        self._emit(
            "agent_turn_completed",
            state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": state.conversation_id,
                **dict(route_metadata or {}),
                **dict(result.metadata),
            },
        )
        self._emit(
            "turn_completed",
            state.session_id,
            agent_name=agent.name,
            payload={"assistant_message_id": state.messages[-1].message_id if state.messages else ""},
        )
        self._run_post_turn_memory_maintenance(state, latest_text=user_text)
        state.sync_to_session(session)
        return result.text

    def _maybe_run_authoritative_inventory(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
    ) -> AgentRunResult | None:
        resolved_turn_intent = resolved_turn_intent_from_metadata(dict(session_state.metadata or {})) or resolve_turn_intent(
            user_text,
            dict(session_state.metadata or {}),
        )
        inventory_query_type = str(
            dict(getattr(resolved_turn_intent, "requested_scope", {}) or {}).get("inventory_query_type") or ""
        ).strip()
        if not inventory_query_type or not is_authoritative_inventory_query_type(inventory_query_type):
            return None
        if (
            str(getattr(resolved_turn_intent.answer_contract, "kind", "") or "").strip().lower() != "inventory"
            or not bool(getattr(resolved_turn_intent.answer_contract, "requires_authoritative_inventory", False))
        ):
            return None
        if inventory_query_requests_grounded_analysis(user_text, query_type=inventory_query_type):
            return None

        dispatched = dispatch_authoritative_inventory(
            self.settings,
            self.stores,
            session_state,
            query=user_text,
            query_type=inventory_query_type,
        )
        if not bool(dispatched.get("handled")):
            return None

        answer_payload = dict(dispatched.get("answer") or {})
        payload = dict(dispatched.get("payload") or {})
        text = str(answer_payload.get("answer") or "").strip()
        inventory_summary = _authoritative_inventory_summary(payload)
        metadata = {
            "agent_name": agent.name,
            "turn_outcome": "authoritative_inventory",
            "provenance": str(dispatched.get("provenance") or "authoritative_inventory"),
            "inventory_query_type": inventory_query_type,
            "inventory_view": str(dispatched.get("view") or payload.get("view") or "").strip(),
            "inventory_summary": inventory_summary,
        }
        self._emit(
            "authoritative_inventory_dispatched",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "inventory_query_type": inventory_query_type,
                "inventory_view": metadata["inventory_view"],
                "selected_agent": agent.name,
                "inventory_summary": inventory_summary,
            },
        )
        assistant_message = RuntimeMessage(role="assistant", content=text, metadata=dict(metadata))
        return AgentRunResult(
            text=text,
            messages=list(session_state.messages) + [assistant_message],
            metadata=dict(metadata),
        )

    def _maybe_run_requirements_extraction(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
    ) -> AgentRunResult | None:
        resolved_turn_intent = resolved_turn_intent_from_metadata(dict(session_state.metadata or {})) or resolve_turn_intent(
            user_text,
            dict(session_state.metadata or {}),
        )
        sanitized_user_query = (
            str(getattr(resolved_turn_intent, "normalized_user_objective", "") or "").strip()
            or normalize_structured_query(user_text)
            or str(user_text or "").strip()
        )
        effective_user_text = (
            str(getattr(resolved_turn_intent, "effective_user_text", "") or "").strip()
            or sanitized_user_query
            or str(user_text or "").strip()
        )
        answer_kind = str(getattr(resolved_turn_intent.answer_contract, "kind", "") or "").strip().lower()
        if answer_kind != REQUIREMENTS_WORKFLOW_KIND and not is_requirements_extraction_request(sanitized_user_query):
            return None

        self._emit(
            "requirements_extraction_started",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "label": "Extracting requirements",
                "detail": "Resolving source documents and extracting requirement statements.",
                "workflow": REQUIREMENTS_WORKFLOW_KIND,
                "sanitized_user_query": sanitized_user_query,
                "requested_scope": dict(getattr(resolved_turn_intent, "requested_scope", {}) or {}),
            },
        )
        agent_providers = self.resolve_providers_for_agent(agent.name)
        service = RequirementExtractionService(self.settings, self.stores, session_state, providers=agent_providers)
        payload = service.extract_for_user_request(
            sanitized_user_query,
            requested_scope=dict(getattr(resolved_turn_intent, "requested_scope", {}) or {}),
        )
        text = _render_requirements_result(payload)
        artifacts = [dict(item) for item in list(payload.get("artifacts") or []) if isinstance(item, dict)]
        artifact_refs = [str(item.get("artifact_ref") or "") for item in artifacts if str(item.get("artifact_ref") or "")]
        session_state.metadata["selected_requirement_doc_ids"] = list(payload.get("selected_doc_ids") or [])
        if payload.get("candidate_documents"):
            session_state.metadata["requirements_candidate_documents"] = list(payload.get("candidate_documents") or [])
        else:
            session_state.metadata.pop("requirements_candidate_documents", None)
        metadata: Dict[str, Any] = {
            "agent_name": agent.name,
            "turn_outcome": "requirements_extraction",
            "workflow": REQUIREMENTS_WORKFLOW_KIND,
            "sanitized_user_query": sanitized_user_query,
            "selected_requirement_doc_ids": list(payload.get("selected_doc_ids") or []),
            "requirements_candidate_documents": list(payload.get("candidate_documents") or []),
            "requirements_extraction": make_json_compatible(
                {
                    key: value
                    for key, value in dict(payload).items()
                    if key not in {"rows"}
                }
            ),
            "artifacts": artifacts,
        }
        if payload.get("error") and payload.get("candidate_documents"):
            metadata = clarification_turn_metadata(
                ClarificationRequest(
                    question="Which document should I extract requirements from? You can also say \"all documents\".",
                    reason="requirements_document_selection",
                    options=tuple(
                        str(item.get("title") or item.get("doc_id") or "").strip()
                        for item in list(payload.get("candidate_documents") or [])[:8]
                        if isinstance(item, dict) and str(item.get("title") or item.get("doc_id") or "").strip()
                    ),
                    source_agent=agent.name,
                    blocking=True,
                ),
                agent_name=agent.name,
                extra=metadata,
            )
        assistant_message = RuntimeMessage(
            role="assistant",
            content=text,
            artifact_refs=artifact_refs,
            metadata=dict(metadata),
        )
        session_state.metadata["pending_artifacts"] = []
        self._emit(
            "requirements_extraction_completed",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "label": "Requirements extraction complete",
                "detail": str(payload.get("summary_text") or payload.get("error") or "").strip(),
                "workflow": REQUIREMENTS_WORKFLOW_KIND,
                "statement_count": int(payload.get("statement_count") or 0),
                "document_count": int(payload.get("document_count") or 0),
                "artifact_count": len(artifacts),
                "sanitized_user_query": sanitized_user_query,
                "selected_doc_ids": list(payload.get("selected_doc_ids") or []),
                "mode": str(payload.get("mode") or ""),
                "candidate_count": int(payload.get("candidate_count") or 0),
                "kept_count": int(payload.get("kept_count") or 0),
                "dropped_count": int(payload.get("dropped_count") or 0),
                "dedupe_count": int(payload.get("dedupe_count") or 0),
                "artifact_names": list(payload.get("artifact_names") or []),
                "extractor_version": str(payload.get("extractor_version") or ""),
            },
        )
        return AgentRunResult(
            text=text,
            messages=list(session_state.messages) + [assistant_message],
            metadata=dict(metadata),
        )

    def run_agent(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        callbacks: List[Any],
        task_payload: Optional[Dict[str, Any]] = None,
        chat_max_output_tokens: int | None = None,
    ) -> AgentRunResult:
        inventory_result = self._maybe_run_authoritative_inventory(
            agent,
            session_state,
            user_text=user_text,
        )
        if inventory_result is not None:
            return inventory_result

        effective_capabilities = resolve_effective_capabilities(
            settings=self.settings,
            stores=self.stores,
            session=session_state,
            registry=self.registry,
        )
        session_state.metadata = {
            **dict(session_state.metadata or {}),
            "effective_capabilities": effective_capabilities.to_dict(),
        }

        if agent.mode == "coordinator":
            return self._run_coordinator(agent, session_state, user_text=user_text, callbacks=callbacks)

        if not is_clause_policy_workflow(user_text, session_metadata=session_state.metadata):
            requirements_result = self._maybe_run_requirements_extraction(
                agent,
                session_state,
                user_text=user_text,
            )
            if requirements_result is not None:
                return requirements_result

        skill_execution_payload = dict((task_payload or {}).get("skill_execution") or {})
        if skill_execution_payload:
            agent_providers = self.provider_controller.resolve_for_skill(
                agent.name,
                model_override=str(skill_execution_payload.get("model") or ""),
                effort=str(skill_execution_payload.get("effort") or ""),
                chat_max_output_tokens=chat_max_output_tokens,
            )
        else:
            agent_providers = self.resolve_providers_for_agent(
                agent.name,
                chat_max_output_tokens=chat_max_output_tokens,
            )
        tool_context = ToolContext(
            settings=self.settings,
            providers=agent_providers,
            stores=self.stores,
            session=session_state,
            paths=self.paths,
            callbacks=callbacks,
            transcript_store=self.transcript_store,
            job_manager=self.job_manager,
            event_sink=self.event_sink,
            kernel=self,
            active_agent=agent.name,
            active_definition=agent,
            file_memory_store=self.file_memory_store,
            memory_store=self.memory_store,
            progress_emitter=self.get_live_progress_sink(session_state.session_id),
            rag_runtime_bridge=self.build_rag_runtime_bridge(session_state) if agent.mode == "rag" else None,
            metadata={
                "task_payload": dict(task_payload or {}),
                "job_id": str((task_payload or {}).get("job_id") or ""),
                "effective_capabilities": effective_capabilities.to_dict(),
            },
        )
        tools = self._build_tools(agent, tool_context)
        loop_result: QueryLoopResult = self.query_loop.run(
            agent,
            session_state,
            user_text=user_text,
            providers=agent_providers,
            tool_context=tool_context,
            tools=tools,
            task_payload=task_payload,
        )
        return AgentRunResult(
            text=loop_result.text,
            messages=list(loop_result.messages or session_state.messages),
            metadata=dict(loop_result.metadata),
        )

    def export_langgraph_react_graph(self, agent_name: str = "") -> Dict[str, Any]:
        selected_agent_name = str(agent_name or "").strip() or self.registry.get_default_agent_name()
        warnings: List[str] = []

        def unavailable(message: str) -> Dict[str, Any]:
            warning_list = [*warnings, message] if message else list(warnings)
            return {
                "status": "unavailable",
                "generated_at": utc_now_iso(),
                "agent_name": selected_agent_name,
                "mermaid": "",
                "nodes": [],
                "edges": [],
                "warnings": warning_list,
            }

        try:
            agent = self.registry.get(selected_agent_name)
            if agent is None:
                return unavailable(f"Agent '{selected_agent_name}' was not found in the live registry.")
            if agent.mode in {"basic", "coordinator"}:
                return unavailable(f"Agent '{selected_agent_name}' uses {agent.mode} execution rather than the ReAct LangGraph loop.")
            if str(agent.metadata.get("execution_strategy") or "").strip().lower() == "plan_execute":
                return unavailable(f"Agent '{selected_agent_name}' is configured for plan-execute fallback.")

            providers = self.resolve_providers_for_agent(agent.name)
            chat_llm = getattr(providers, "chat", None) if providers is not None else None
            if chat_llm is None:
                return unavailable(f"Agent '{selected_agent_name}' has no chat provider available for graph export.")

            session_state = SessionState(
                tenant_id=str(getattr(self.settings, "default_tenant_id", "local-dev") or "local-dev"),
                user_id=str(getattr(self.settings, "default_user_id", "control-panel") or "control-panel"),
                conversation_id="architecture-graph-preview",
                user_email="control-panel@example.local",
                active_agent=agent.name,
                metadata={"source": "control_panel_architecture_export"},
            )
            tool_context = ToolContext(
                settings=self.settings,
                providers=providers,
                stores=self.stores,
                session=session_state,
                paths=self.paths,
                callbacks=[],
                transcript_store=self.transcript_store,
                job_manager=self.job_manager,
                event_sink=self.event_sink,
                kernel=self,
                active_agent=agent.name,
                active_definition=agent,
                file_memory_store=self.file_memory_store,
                memory_store=self.memory_store,
                progress_emitter=self.get_live_progress_sink(session_state.session_id),
                rag_runtime_bridge=self.build_rag_runtime_bridge(session_state) if agent.mode == "rag" else None,
                metadata={"source": "control_panel_architecture_export"},
            )
            tools = self._build_tools(agent, tool_context)
            if tools:
                bind_tools = getattr(chat_llm, "bind_tools", None)
                if callable(bind_tools):
                    bind_tools(tools)
                else:
                    warnings.append(f"Agent '{selected_agent_name}' chat provider does not expose bind_tools().")

            compiled_graph = build_react_agent_graph(
                chat_llm,
                tools=tools,
                max_tool_calls=agent.max_tool_calls,
                max_parallel_tool_calls=getattr(self.settings, "max_parallel_tool_calls", 4),
                context_budget_manager=self.context_budget_manager,
                tool_context=tool_context,
                providers=providers,
            )
            graph_view = compiled_graph.get_graph()
            mermaid = graph_view.draw_mermaid() if hasattr(graph_view, "draw_mermaid") else ""
            return {
                "status": "available",
                "generated_at": utc_now_iso(),
                "agent_name": agent.name,
                "mermaid": str(mermaid or ""),
                "nodes": self._serialize_langgraph_nodes(graph_view),
                "edges": self._serialize_langgraph_edges(graph_view),
                "warnings": warnings,
            }
        except Exception as exc:
            logger.warning("LangGraph architecture export failed for agent '%s': %s", selected_agent_name, exc)
            return unavailable(str(exc))

    @staticmethod
    def _safe_langgraph_json(value: Any) -> Any:
        normalized = make_json_compatible(value)
        try:
            json.dumps(normalized)
            return normalized
        except TypeError:
            return str(normalized)

    @classmethod
    def _serialize_langgraph_nodes(cls, graph_view: Any) -> List[Dict[str, Any]]:
        raw_nodes = getattr(graph_view, "nodes", {}) or {}
        items = raw_nodes.items() if isinstance(raw_nodes, dict) else enumerate(raw_nodes)
        serialized: List[Dict[str, Any]] = []
        for fallback_id, node in items:
            node_id = str(getattr(node, "id", fallback_id) or fallback_id)
            node_data = getattr(node, "data", None)
            serialized.append(
                {
                    "id": node_id,
                    "name": str(getattr(node, "name", node_id) or node_id),
                    "data_type": node_data.__class__.__name__ if node_data is not None else "",
                    "metadata": cls._safe_langgraph_json(getattr(node, "metadata", {}) or {}),
                }
            )
        return serialized

    @classmethod
    def _serialize_langgraph_edges(cls, graph_view: Any) -> List[Dict[str, Any]]:
        raw_edges = getattr(graph_view, "edges", []) or []
        serialized: List[Dict[str, Any]] = []
        for index, edge in enumerate(raw_edges):
            serialized.append(
                {
                    "id": f"langgraph-edge-{index + 1}",
                    "source": str(getattr(edge, "source", "") or ""),
                    "target": str(getattr(edge, "target", "") or ""),
                    "conditional": bool(getattr(edge, "conditional", False)),
                    "data": cls._safe_langgraph_json(getattr(edge, "data", None)),
                }
            )
        return serialized

    def resolve_providers_for_agent(self, agent_name: str, *, chat_max_output_tokens: int | None = None) -> Any | None:
        return self.provider_controller.resolve_for_agent(
            agent_name,
            chat_max_output_tokens=chat_max_output_tokens,
        )

    def resolve_base_providers(self) -> Any | None:
        return self.provider_controller.resolve_base_providers()

    def compact_session_context(self, session_id: str, *, preview: bool = False, reason: str = "manual") -> Dict[str, Any]:
        state = self.transcript_store.load_session_state(session_id)
        if state is None:
            raise ValueError("Session not found.")
        page = self.transcript_store.load_session_message_page(
            session_id,
            page_size=max(1000, int(getattr(self.settings, "session_transcript_page_size", 200) or 200) * 20),
        )
        messages = [
            message
            for message in list(page.get("messages") or [])
            if isinstance(message, RuntimeMessage)
        ] or list(state.messages or [])
        providers = self.resolve_base_providers()
        payload = self.context_budget_manager.manual_compact_session(
            session_state=state,
            messages=messages,
            providers=providers,
            preview=preview,
            reason=reason,
        )
        if not preview:
            self.transcript_store.persist_session_state(state)
        return payload

    def is_bundle_role_open(self, bundle: Any | None, role: str) -> bool:
        return self.provider_controller.is_bundle_role_open(bundle, role)

    def bundle_role_identity(self, bundle: Any | None, role: str) -> tuple[str, str, str]:
        return self.provider_controller.bundle_role_identity(bundle, role)

    def execute_skill_from_tool(
        self,
        tool_context: ToolContext,
        *,
        skill_id: str,
        input_text: str = "",
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not bool(getattr(self.settings, "executable_skills_enabled", False)):
            return {"object": "skill.execution_result", "skill_id": skill_id, "status": "error", "error": "Executable skills are disabled."}

        active_definition = tool_context.active_definition
        if active_definition is None or "execute_skill" not in set(active_definition.allowed_tools):
            return {"object": "skill.execution_result", "skill_id": skill_id, "status": "error", "error": "Current agent is not allowed to execute skills."}

        task_payload = dict((tool_context.metadata or {}).get("task_payload") or {})
        if task_payload.get("skill_execution"):
            return {"object": "skill.execution_result", "skill_id": skill_id, "status": "error", "error": "Recursive skill execution is not allowed."}

        skill_store = getattr(self.stores, "skill_store", None)
        if skill_store is None:
            return {"object": "skill.execution_result", "skill_id": skill_id, "status": "error", "error": "Skill store is not configured."}

        clean_skill_id = str(skill_id or "").strip()
        if not clean_skill_id:
            return {"object": "skill.execution_result", "skill_id": "", "status": "error", "error": "skill_id is required."}

        session = tool_context.session
        access_summary = dict((getattr(session, "metadata", {}) or {}).get("access_summary") or getattr(session, "access_summary", {}) or {})
        accessible_skill_family_ids = (
            list(access_summary_allowed_ids(access_summary, "skill_family", action="use"))
            if access_summary_authz_enabled(access_summary)
            else None
        )
        record = skill_store.get_skill_pack(
            clean_skill_id,
            tenant_id=session.tenant_id,
            owner_user_id=session.user_id,
            accessible_skill_family_ids=accessible_skill_family_ids,
        )
        if record is None:
            return {"object": "skill.execution_result", "skill_id": clean_skill_id, "status": "error", "error": "Skill not found or not visible."}
        effective_capabilities = coerce_effective_capabilities(
            dict(tool_context.metadata or {}).get("effective_capabilities")
            or dict(getattr(session, "metadata", {}) or {}).get("effective_capabilities")
        )
        family_id = str(getattr(record, "version_parent", "") or clean_skill_id)
        if effective_capabilities is not None and not effective_capabilities.allows_skill(clean_skill_id, family_id=family_id):
            return {
                "object": "skill.execution_result",
                "skill_id": clean_skill_id,
                "status": "error",
                "error": "Skill pack is disabled by the effective capability profile.",
            }
        if str(getattr(record, "kind", "retrievable") or "retrievable").strip().lower() not in EXECUTABLE_SKILL_KINDS:
            return {"object": "skill.execution_result", "skill_id": clean_skill_id, "status": "error", "error": "Skill is not executable."}
        if not bool(getattr(record, "enabled", False)) or str(getattr(record, "status", "") or "").strip().lower() != "active":
            return {"object": "skill.execution_result", "skill_id": clean_skill_id, "status": "error", "error": "Skill is not active."}
        invoking_scope = str(active_definition.skill_scope or "").strip()
        skill_scope = str(getattr(record, "agent_scope", "") or "").strip()
        if not invoking_scope or skill_scope != invoking_scope:
            return {
                "object": "skill.execution_result",
                "skill_id": clean_skill_id,
                "status": "error",
                "error": f"Skill scope '{skill_scope}' does not match agent scope '{invoking_scope}'.",
            }

        config = SkillExecutionConfig.from_record(record)
        skill_allowed_tools = list(config.allowed_tools)
        if "execute_skill" in set(skill_allowed_tools):
            return {"object": "skill.execution_result", "skill_id": clean_skill_id, "status": "error", "error": "Executable skills cannot allow execute_skill."}
        caller_tools = list(active_definition.allowed_tools or [])
        disallowed_by_caller = sorted(tool for tool in skill_allowed_tools if not tool_allowed_by_selectors(caller_tools, tool))
        if disallowed_by_caller:
            return {
                "object": "skill.execution_result",
                "skill_id": clean_skill_id,
                "status": "error",
                "error": "Skill requests tools that the current agent is not allowed to use.",
                "disallowed_tools": disallowed_by_caller,
            }
        if effective_capabilities is not None:
            tool_definitions = getattr(self, "tool_definitions", {}) or {}
            disallowed_by_capability = []
            for tool_name in skill_allowed_tools:
                definition = tool_definitions.get(tool_name)
                if definition is None:
                    continue
                if not effective_capabilities.allows_tool(
                    tool_name,
                    group=str(getattr(definition, "group", "") or ""),
                    read_only=bool(getattr(definition, "read_only", False)),
                    destructive=bool(getattr(definition, "destructive", False)),
                    metadata=dict(getattr(definition, "metadata", {}) or {}),
                ):
                    disallowed_by_capability.append(tool_name)
            if disallowed_by_capability:
                return {
                    "object": "skill.execution_result",
                    "skill_id": clean_skill_id,
                    "status": "error",
                    "error": "Skill requests tools disabled by the effective capability profile.",
                    "disallowed_tools": sorted(disallowed_by_capability),
                }

        preview = build_skill_execution_preview(
            record,
            input_text=input_text,
            arguments=dict(arguments or {}),
        )
        if config.context == "inline":
            return SkillExecutionResult(
                skill_id=clean_skill_id,
                context="inline",
                status="inline_ready",
                result=preview.rendered_prompt,
                allowed_tools=skill_allowed_tools,
                model=config.model,
                effort=config.effort,
            ).to_dict()

        clean_agent = str(config.agent or "utility").strip() or "utility"
        allowed_agents = set(active_definition.allowed_worker_agents or [])
        if clean_agent not in allowed_agents:
            return {
                "object": "skill.execution_result",
                "skill_id": clean_skill_id,
                "context": "fork",
                "status": "error",
                "error": f"Agent '{clean_agent}' is not allowed for this skill fork.",
                "allowed_agents": sorted(allowed_agents),
            }
        worker_agent = self._resolve_agent(clean_agent)
        worker_tools = list(worker_agent.allowed_tools or [])
        disallowed_by_worker = sorted(tool for tool in skill_allowed_tools if not tool_allowed_by_selectors(worker_tools, tool))
        if disallowed_by_worker:
            return {
                "object": "skill.execution_result",
                "skill_id": clean_skill_id,
                "context": "fork",
                "status": "error",
                "error": "Skill requests tools that the forked worker is not allowed to use.",
                "disallowed_tools": disallowed_by_worker,
            }

        max_steps = min(worker_agent.max_steps, config.max_steps) if config.max_steps is not None else worker_agent.max_steps
        max_tool_calls = (
            min(worker_agent.max_tool_calls, config.max_tool_calls)
            if config.max_tool_calls is not None
            else worker_agent.max_tool_calls
        )
        skill_execution_payload = {
            "skill_id": clean_skill_id,
            "skill_name": getattr(record, "name", ""),
            "skill_family_id": str(getattr(record, "version_parent", "") or clean_skill_id),
            "context": "fork",
            "invoking_agent": active_definition.name,
            "allowed_tools": skill_allowed_tools,
            "model": config.model,
            "effort": config.effort,
            "max_steps": max_steps,
            "max_tool_calls": max_tool_calls,
        }
        job = self._create_tool_worker_job(
            tool_context,
            agent_name=clean_agent,
            prompt=preview.rendered_prompt,
            description=f"Skill: {getattr(record, 'name', clean_skill_id)}",
            run_in_background=False,
            parent_job_id=str((tool_context.metadata or {}).get("job_id") or ""),
            extra_metadata={"skill_execution": skill_execution_payload},
        )
        result = self.job_manager.run_job_inline(job, self._job_runner)
        refreshed = self.job_manager.get_job(job.job_id) or job
        return SkillExecutionResult(
            skill_id=clean_skill_id,
            context="fork",
            status=str(getattr(refreshed, "status", "") or "completed"),
            result=str(result or ""),
            job_id=job.job_id,
            allowed_tools=skill_allowed_tools,
            model=config.model,
            effort=config.effort,
        ).to_dict()

    def spawn_worker_from_tool(
        self,
        tool_context: ToolContext,
        *,
        prompt: str,
        agent_name: str = "utility",
        description: str = "",
        run_in_background: bool = False,
    ) -> Dict[str, Any]:
        active_definition = tool_context.active_definition
        allowed_agents = set(active_definition.allowed_worker_agents if active_definition is not None else [])
        clean_agent = (agent_name or "utility").strip()
        effective_capabilities = coerce_effective_capabilities(
            dict(tool_context.metadata or {}).get("effective_capabilities")
            or dict(getattr(tool_context.session, "metadata", {}) or {}).get("effective_capabilities")
        )
        if effective_capabilities is not None and not effective_capabilities.allows_agent(clean_agent):
            return {"error": f"Agent '{clean_agent}' is disabled by the effective capability profile.", "agent_name": clean_agent}
        if clean_agent not in allowed_agents:
            return {"error": f"Agent '{clean_agent}' is not allowed.", "allowed_agents": sorted(allowed_agents)}
        if clean_agent == "memory_maintainer" and not bool(getattr(self.settings, "memory_enabled", True)):
            return {
                "error": "Agent 'memory_maintainer' is unavailable because MEMORY_ENABLED is false.",
                "agent_name": clean_agent,
                "background": False,
            }
        worker_agent = self._resolve_agent(clean_agent)
        if run_in_background and not worker_agent.allow_background_jobs:
            return {
                "error": f"Agent '{clean_agent}' does not allow background jobs.",
                "background": False,
                "agent_name": clean_agent,
            }
        job = self._create_tool_worker_job(
            tool_context,
            agent_name=clean_agent,
            prompt=prompt,
            description=description,
            run_in_background=run_in_background,
        )
        if run_in_background:
            self.job_manager.start_background_job(job, self._job_runner)
            return {"job_id": job.job_id, "status": "queued", "agent_name": clean_agent, "background": True}
        result = self.job_manager.run_job_inline(job, self._job_runner)
        refreshed = self.job_manager.get_job(job.job_id) or job
        return {
            "job_id": job.job_id,
            "status": refreshed.status,
            "agent_name": clean_agent,
            "background": False,
            "result": result,
            "output_path": refreshed.output_path,
        }

    def invoke_agent_from_tool(
        self,
        tool_context: ToolContext,
        *,
        agent_name: str,
        message: str,
        description: str = "",
        job_id: str = "",
        reuse_running_job: bool = True,
        team_channel_id: str = "",
    ) -> Dict[str, Any]:
        active_definition = tool_context.active_definition
        allowed_agents = set(active_definition.allowed_worker_agents if active_definition is not None else [])
        clean_agent = str(agent_name or "").strip()
        clean_message = str(message or "").strip()
        clean_job_id = str(job_id or "").strip()
        effective_capabilities = coerce_effective_capabilities(
            dict(tool_context.metadata or {}).get("effective_capabilities")
            or dict(getattr(tool_context.session, "metadata", {}) or {}).get("effective_capabilities")
        )
        if effective_capabilities is not None and not effective_capabilities.allows_agent(clean_agent):
            return {"error": f"Agent '{clean_agent}' is disabled by the effective capability profile.", "agent_name": clean_agent}
        if not clean_agent:
            return {"error": "agent_name is required."}
        if not clean_message:
            return {"error": "message is required."}
        if clean_agent not in allowed_agents:
            return {"error": f"Agent '{clean_agent}' is not allowed.", "allowed_agents": sorted(allowed_agents)}
        if clean_agent == "memory_maintainer" and not bool(getattr(self.settings, "memory_enabled", True)):
            return {
                "error": "Agent 'memory_maintainer' is unavailable because MEMORY_ENABLED is false.",
                "agent_name": clean_agent,
            }
        try:
            dispatch = self._dispatch_agent_message(
                tool_context,
                target_agent=clean_agent,
                message=clean_message,
                description=description,
                target_job_id=clean_job_id,
                reuse_running_job=reuse_running_job,
                team_channel_id=team_channel_id,
            )
        except ValueError as exc:
            return {"error": str(exc), "agent_name": clean_agent}
        return {
            "job_id": dispatch.job.job_id,
            "target_agent": dispatch.job.agent_name,
            "status": dispatch.job.status,
            "reused_existing_job": dispatch.reused_existing_job,
            "queued": dispatch.queued,
        }

    def message_worker_from_tool(
        self,
        tool_context: ToolContext,
        *,
        job_id: str,
        message: str,
        resume: bool = True,
    ) -> Dict[str, Any]:
        sender = str(getattr(tool_context, "active_agent", "") or "").strip() or "parent"
        source_job_id = str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or "").strip()
        mailbox_message = self.job_manager.enqueue_message(
            job_id,
            message,
            sender=sender,
            metadata={
                "peer_dispatch": False,
                "source_agent": sender,
                "source_job_id": source_job_id,
            },
        )
        if mailbox_message is None:
            return {"error": f"Job '{job_id}' was not found."}
        if resume:
            self.job_manager.continue_job(job_id, self._job_runner)
        job = self.job_manager.get_job(job_id)
        return {
            "job_id": job_id,
            "message_id": getattr(mailbox_message, "message_id", ""),
            "status": getattr(job, "status", "unknown"),
            "queued": True,
        }

    def request_parent_question_from_tool(
        self,
        tool_context: ToolContext,
        *,
        question: str,
        reason: str = "",
        options: Optional[List[str]] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        job_id = str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or "").strip()
        if not job_id:
            return {"error": "request_parent_question is only available inside worker jobs."}
        clean_question = str(question or "").strip()
        if not clean_question:
            return {"error": "question is required."}
        message = self.job_manager.open_worker_request(
            job_id,
            request_type="question_request",
            content=clean_question,
            sender=str(getattr(tool_context, "active_agent", "") or "worker"),
            subject=clean_question[:120],
            payload={
                "question": clean_question,
                "reason": str(reason or "").strip(),
                "options": [str(item).strip() for item in (options or []) if str(item).strip()],
                "context": str(context or "").strip(),
            },
        )
        if message is None:
            return {"error": f"Job '{job_id}' was not found."}
        job = self.job_manager.get_job(job_id)
        if job is not None:
            self._sync_pending_worker_request_for_session(job.session_id)
        return {
            "object": "worker_mailbox_request",
            "job_id": job_id,
            "request_id": message.message_id,
            "request_type": message.message_type,
            "status": message.status,
            "question": clean_question,
            "waiting": True,
        }

    def request_parent_approval_from_tool(
        self,
        tool_context: ToolContext,
        *,
        action: str,
        reason: str,
        tool_name: str = "",
        arguments: Optional[Dict[str, Any]] = None,
        risk: str = "",
        context: str = "",
    ) -> Dict[str, Any]:
        job_id = str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or "").strip()
        if not job_id:
            return {"error": "request_parent_approval is only available inside worker jobs."}
        clean_action = str(action or "").strip()
        clean_reason = str(reason or "").strip()
        if not clean_action:
            return {"error": "action is required."}
        if not clean_reason:
            return {"error": "reason is required."}
        message = self.job_manager.open_worker_request(
            job_id,
            request_type="approval_request",
            content=f"{clean_action}\n\nReason: {clean_reason}",
            sender=str(getattr(tool_context, "active_agent", "") or "worker"),
            subject=clean_action[:120],
            payload={
                "action": clean_action,
                "reason": clean_reason,
                "tool_name": str(tool_name or "").strip(),
                "arguments": dict(arguments or {}),
                "risk": str(risk or "").strip(),
                "context": str(context or "").strip(),
            },
        )
        if message is None:
            return {"error": f"Job '{job_id}' was not found."}
        job = self.job_manager.get_job(job_id)
        if job is not None:
            self._sync_pending_worker_request_for_session(job.session_id)
        return {
            "object": "worker_mailbox_request",
            "job_id": job_id,
            "request_id": message.message_id,
            "request_type": message.message_type,
            "status": message.status,
            "action": clean_action,
            "waiting": True,
        }

    def list_worker_requests_from_tool(
        self,
        tool_context: ToolContext,
        *,
        job_id: str = "",
        status_filter: str = "open",
        request_type: str = "",
    ) -> List[Dict[str, Any]]:
        session_id = tool_context.session.session_id
        clean_job_id = str(job_id or "").strip()
        jobs = [self.job_manager.get_job(clean_job_id)] if clean_job_id else self.job_manager.list_jobs(session_id=session_id)
        rows: List[Dict[str, Any]] = []
        for job in jobs:
            if job is None or job.session_id != session_id:
                continue
            for item in self.job_manager.list_mailbox_requests(
                job.job_id,
                status_filter=status_filter,
                request_type=request_type,
            ):
                rows.append(
                    {
                        "job_id": job.job_id,
                        "agent_name": job.agent_name,
                        **item.to_dict(),
                    }
                )
        return rows

    def respond_worker_request_from_tool(
        self,
        tool_context: ToolContext,
        *,
        job_id: str,
        request_id: str,
        response: str,
        decision: str = "",
        resume: bool = True,
    ) -> Dict[str, Any]:
        if str(decision or "").strip():
            return {"error": "Approval requests require operator/API approval; agents can only answer question requests."}
        job = self.job_manager.get_job(job_id)
        if job is None or job.session_id != tool_context.session.session_id:
            return {"error": f"Job '{job_id}' was not found."}
        try:
            result = self.job_manager.respond_to_request(
                job_id,
                request_id,
                response=response,
                responder=str(getattr(tool_context, "active_agent", "") or "parent"),
                decision="",
                allow_approval=False,
                metadata={
                    "source_agent": str(getattr(tool_context, "active_agent", "") or ""),
                    "source_job_id": str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or ""),
                },
            )
        except (PermissionError, ValueError) as exc:
            return {"error": str(exc)}
        if result is None:
            return {"error": f"Job '{job_id}' was not found."}
        request, response_message = result
        if resume:
            self.job_manager.continue_job(job_id, self._job_runner)
        self._sync_pending_worker_request_for_session(job.session_id)
        refreshed = self.job_manager.get_job(job_id) or job
        return {
            "job_id": job_id,
            "request_id": request.message_id,
            "response_message_id": response_message.message_id,
            "status": getattr(refreshed, "status", "unknown"),
            "queued": bool(resume),
        }

    def _team_mailbox_enabled(self) -> bool:
        return bool(getattr(self.settings, "team_mailbox_enabled", False))

    def _team_actor(self, tool_context: ToolContext) -> tuple[str, str]:
        return (
            str(getattr(tool_context, "active_agent", "") or "").strip(),
            str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or "").strip(),
        )

    def _is_team_mailbox_admin(self, tool_context: ToolContext) -> bool:
        agent = getattr(tool_context, "active_definition", None)
        active_agent, active_job_id = self._team_actor(tool_context)
        metadata = dict(getattr(agent, "metadata", {}) or {}) if agent is not None else {}
        role_kind = str(metadata.get("role_kind") or "").strip()
        return (
            active_agent == "coordinator"
            or getattr(agent, "mode", "") == "coordinator"
            or (not active_job_id and role_kind in {"top_level", "manager"})
        )

    def _team_channel_visible(self, tool_context: ToolContext, channel: Any) -> bool:
        if self._is_team_mailbox_admin(tool_context):
            return True
        active_agent, active_job_id = self._team_actor(tool_context)
        return (
            active_agent in set(getattr(channel, "member_agents", []) or [])
            or (active_job_id and active_job_id in set(getattr(channel, "member_job_ids", []) or []))
        )

    def _validate_team_targets(self, tool_context: ToolContext, target_agents: List[str]) -> List[str]:
        clean_targets = [str(item).strip() for item in list(target_agents or []) if str(item).strip()]
        active_definition = getattr(tool_context, "active_definition", None)
        allowed = set(getattr(active_definition, "allowed_worker_agents", []) or [])
        active_agent = str(getattr(tool_context, "active_agent", "") or "").strip()
        disallowed = sorted(target for target in clean_targets if target != active_agent and target not in allowed)
        if disallowed:
            raise ValueError(f"Target agent(s) are not allowed: {', '.join(disallowed)}")
        return clean_targets

    def _validate_team_job_ids(self, session_id: str, job_ids: List[str], *, channel: Any = None) -> List[str]:
        clean_job_ids = [str(item).strip() for item in list(job_ids or []) if str(item).strip()]
        if not clean_job_ids:
            return []
        channel_job_ids = set(getattr(channel, "member_job_ids", []) or []) if channel is not None else set()
        if channel_job_ids:
            outside_channel = sorted(job_id for job_id in clean_job_ids if job_id not in channel_job_ids)
            if outside_channel:
                raise ValueError(f"Target job(s) are not members of this team mailbox channel: {', '.join(outside_channel)}")
        wrong_session: List[str] = []
        for job_id in clean_job_ids:
            job = self.job_manager.get_job(job_id)
            if job is None or str(getattr(job, "session_id", "") or "") != str(session_id or ""):
                wrong_session.append(job_id)
        if wrong_session:
            raise ValueError(f"Target job(s) are not in this session: {', '.join(sorted(wrong_session))}")
        return clean_job_ids

    @staticmethod
    def _team_message_summary(messages: List[Any], channels: List[Any]) -> Dict[str, Any]:
        latest = messages[-1].to_dict() if messages else {}
        return {
            "active_channel_count": len(channels),
            "open_message_count": len(messages),
            "pending_question_count": sum(1 for item in messages if getattr(item, "message_type", "") == "question_request"),
            "pending_approval_count": sum(1 for item in messages if getattr(item, "message_type", "") == "approval_request"),
            "open_handoff_count": sum(1 for item in messages if getattr(item, "message_type", "") == "handoff"),
            "latest_open_message": latest,
        }

    def create_team_channel_from_tool(
        self,
        tool_context: ToolContext,
        *,
        name: str,
        purpose: str = "",
        member_agents: Optional[List[str]] = None,
        member_job_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self._team_mailbox_enabled():
            return {"error": "Team mailbox is disabled.", "enabled": False}
        active_agent, active_job_id = self._team_actor(tool_context)
        try:
            clean_member_agents = self._validate_team_targets(tool_context, list(member_agents or []))
        except ValueError as exc:
            return {"error": str(exc)}
        if active_agent and active_agent not in clean_member_agents:
            clean_member_agents.insert(0, active_agent)
        try:
            clean_member_job_ids = self._validate_team_job_ids(
                tool_context.session.session_id,
                list(member_job_ids or []),
            )
        except ValueError as exc:
            return {"error": str(exc)}
        if active_job_id and active_job_id not in clean_member_job_ids:
            clean_member_job_ids.insert(0, active_job_id)
        try:
            channel = self.job_manager.create_team_channel(
                session_id=tool_context.session.session_id,
                name=name,
                purpose=purpose,
                created_by_job_id=active_job_id,
                member_agents=clean_member_agents,
                member_job_ids=clean_member_job_ids,
                metadata={"created_by_agent": active_agent, "source": "tool"},
            )
        except ValueError as exc:
            return {"error": str(exc)}
        return {"object": "team_mailbox_channel", "channel": channel.to_dict()}

    def post_team_message_from_tool(
        self,
        tool_context: ToolContext,
        *,
        channel_id: str,
        content: str,
        message_type: str = "message",
        target_agents: Optional[List[str]] = None,
        target_job_ids: Optional[List[str]] = None,
        subject: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._team_mailbox_enabled():
            return {"error": "Team mailbox is disabled.", "enabled": False}
        channel = self.job_manager.transcript_store.load_team_channel(tool_context.session.session_id, channel_id)
        if channel is None:
            return {"error": f"Team mailbox channel '{channel_id}' was not found."}
        if not self._team_channel_visible(tool_context, channel):
            return {"error": "Current agent is not a member of this team mailbox channel."}
        try:
            clean_targets = self._validate_team_targets(tool_context, list(target_agents or []))
            clean_target_job_ids = self._validate_team_job_ids(
                tool_context.session.session_id,
                list(target_job_ids or []),
                channel=channel,
            )
            active_agent, active_job_id = self._team_actor(tool_context)
            message = self.job_manager.post_team_message(
                session_id=tool_context.session.session_id,
                channel_id=channel.channel_id,
                content=content,
                source_agent=active_agent,
                source_job_id=active_job_id,
                target_agents=clean_targets,
                target_job_ids=clean_target_job_ids,
                message_type=message_type,
                subject=subject,
                payload=dict(payload or {}),
                metadata={"source": "tool"},
            )
        except ValueError as exc:
            return {"error": str(exc)}
        return {"object": "team_mailbox_message", "message": message.to_dict()}

    def list_team_messages_from_tool(
        self,
        tool_context: ToolContext,
        *,
        channel_id: str = "",
        message_type: str = "",
        status_filter: str = "open",
        limit: int = 20,
    ) -> Dict[str, Any]:
        if not self._team_mailbox_enabled():
            return {"error": "Team mailbox is disabled.", "enabled": False}
        channels = (
            [self.job_manager.transcript_store.load_team_channel(tool_context.session.session_id, channel_id)]
            if str(channel_id or "").strip()
            else self.job_manager.list_team_channels(tool_context.session.session_id, status_filter="active")
        )
        visible_channels = [channel for channel in channels if channel is not None and self._team_channel_visible(tool_context, channel)]
        rows: List[Any] = []
        max_rows = max(1, min(int(limit or 20), 200))
        for channel in visible_channels:
            rows.extend(
                self.job_manager.list_team_messages(
                    tool_context.session.session_id,
                    channel_id=channel.channel_id,
                    message_type=message_type,
                    status_filter=status_filter,
                    limit=max_rows,
                )
            )
        rows.sort(key=lambda item: getattr(item, "created_at", ""))
        rows = rows[-max_rows:]
        return {
            "object": "team_mailbox_messages",
            "channels": [channel.to_dict() for channel in visible_channels],
            "summary": self._team_message_summary(rows, visible_channels),
            "data": [item.to_dict() for item in rows],
        }

    def claim_team_messages_from_tool(
        self,
        tool_context: ToolContext,
        *,
        channel_id: str,
        limit: int = 0,
        message_type: str = "",
    ) -> Dict[str, Any]:
        if not self._team_mailbox_enabled():
            return {"error": "Team mailbox is disabled.", "enabled": False}
        channel = self.job_manager.transcript_store.load_team_channel(tool_context.session.session_id, channel_id)
        if channel is None:
            return {"error": f"Team mailbox channel '{channel_id}' was not found."}
        if not self._team_channel_visible(tool_context, channel):
            return {"error": "Current agent is not a member of this team mailbox channel."}
        active_agent, active_job_id = self._team_actor(tool_context)
        try:
            messages = self.job_manager.claim_team_messages(
                tool_context.session.session_id,
                channel.channel_id,
                claimant_agent=active_agent,
                claimant_job_id=active_job_id,
                limit=limit,
                message_type=message_type,
            )
        except ValueError as exc:
            return {"error": str(exc)}
        return {"object": "team_mailbox_claim", "channel_id": channel.channel_id, "data": [item.to_dict() for item in messages]}

    def respond_team_message_from_tool(
        self,
        tool_context: ToolContext,
        *,
        channel_id: str,
        message_id: str,
        response: str,
        decision: str = "",
        resolve: bool = True,
    ) -> Dict[str, Any]:
        if not self._team_mailbox_enabled():
            return {"error": "Team mailbox is disabled.", "enabled": False}
        if str(decision or "").strip():
            return {"error": "Approval requests require operator/API approval; agents can only answer question requests."}
        channel = self.job_manager.transcript_store.load_team_channel(tool_context.session.session_id, channel_id)
        if channel is None:
            return {"error": f"Team mailbox channel '{channel_id}' was not found."}
        if not self._team_channel_visible(tool_context, channel):
            return {"error": "Current agent is not a member of this team mailbox channel."}
        active_agent, active_job_id = self._team_actor(tool_context)
        try:
            result = self.job_manager.respond_team_message(
                tool_context.session.session_id,
                channel.channel_id,
                message_id,
                response=response,
                responder_agent=active_agent,
                responder_job_id=active_job_id,
                decision="",
                allow_approval=False,
                resolve=resolve,
                metadata={"source": "tool"},
            )
        except (PermissionError, ValueError) as exc:
            return {"error": str(exc)}
        if result is None:
            return {"error": f"Team mailbox message '{message_id}' was not found."}
        request, response_message = result
        return {
            "object": "team_mailbox_response",
            "channel_id": channel.channel_id,
            "request": request.to_dict(),
            "response": response_message.to_dict(),
        }

    def list_jobs_from_tool(
        self,
        tool_context: ToolContext,
        *,
        status_filter: str = "",
    ) -> List[Dict[str, Any]]:
        jobs = self.job_manager.list_jobs(session_id=tool_context.session.session_id)
        rows: List[Dict[str, Any]] = []
        for job in jobs:
            if status_filter and job.status != status_filter:
                continue
            rows.append(
                {
                    "job_id": job.job_id,
                    "agent_name": job.agent_name,
                    "status": job.status,
                    "scheduler_state": getattr(job, "scheduler_state", ""),
                    "queue_class": getattr(job, "queue_class", ""),
                    "priority": getattr(job, "priority", ""),
                    "description": job.description,
                    "result_summary": job.result_summary,
                    "output_path": job.output_path,
                    "mailbox": self.job_manager.mailbox_summary(job.job_id),
                }
            )
        return rows

    def stop_job_from_tool(
        self,
        tool_context: ToolContext,
        *,
        job_id: str,
    ) -> Dict[str, Any]:
        del tool_context
        job = self.job_manager.stop_job(job_id)
        if job is None:
            return {"error": f"Job '{job_id}' was not found."}
        return {"job_id": job.job_id, "status": job.status}

    def _run_coordinator(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        callbacks: List[Any],
    ) -> AgentRunResult:
        return self.coordinator_controller.run(
            agent,
            session_state,
            user_text=user_text,
            callbacks=callbacks,
        )

    def _coordinator_skill_queries(self, task_plan: List[Dict[str, Any]]) -> List[str]:
        return self.coordinator_controller.coordinator_skill_queries(task_plan)

    def _build_tools(self, agent: AgentDefinition, tool_context: ToolContext) -> List[Any]:
        if tool_context.providers is None or tool_context.stores is None:
            return []
        return build_agent_tools(agent, tool_context, policy_service=self.tool_policy)

    def _build_scoped_worker_state(self, parent: SessionState, *, agent_name: str) -> SessionState:
        return self.coordinator_controller.build_scoped_worker_state(parent, agent_name=agent_name)

    def _recent_context_summary(self, session_state: SessionState, limit: int = 4) -> str:
        return self.coordinator_controller.recent_context_summary(session_state, limit=limit)

    def _build_worker_request(
        self,
        *,
        task: Dict[str, Any],
        user_request: str,
        session_state: SessionState,
        artifact_refs: Optional[List[str]] = None,
    ) -> WorkerExecutionRequest:
        return self.coordinator_controller.build_worker_request(
            task=task,
            user_request=user_request,
            session_state=session_state,
            artifact_refs=artifact_refs,
        )

    def _run_task_batch(
        self,
        *,
        agent: AgentDefinition,
        session_state: SessionState,
        user_request: str,
        callbacks: List[Any],
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self.coordinator_controller.run_task_batch(
            agent=agent,
            session_state=session_state,
            user_request=user_request,
            callbacks=callbacks,
            batch=batch,
        )

    def _should_run_task_batch_in_parallel(self, *, batch: List[Dict[str, Any]], real_jobs: List[Any]) -> bool:
        return self.coordinator_controller.should_run_task_batch_in_parallel(batch=batch, real_jobs=real_jobs)

    def _uses_local_ollama_workers(self) -> bool:
        return self.provider_controller.uses_local_ollama_workers()

    def _can_identify_non_ollama_worker_runtime(self) -> bool:
        return self.provider_controller.can_identify_non_ollama_worker_runtime()

    @staticmethod
    def _normalize_provider_name(value: Any) -> str:
        return KernelProviderController._normalize_provider_name(value)

    @classmethod
    def _model_runtime_name(cls, model: Any) -> str:
        return KernelProviderController._model_runtime_name(model)

    def _wait_for_jobs(self, job_ids: List[str], *, timeout_seconds: float = 300.0) -> None:
        if timeout_seconds == 300.0:
            timeout_seconds = float(getattr(self.settings, "worker_job_wait_timeout_seconds", 600.0))
        self.coordinator_controller.wait_for_jobs(job_ids, timeout_seconds=timeout_seconds)

    def _build_task_result(self, job: JobRecord, worker_request: WorkerExecutionRequest) -> TaskResult:
        return self.coordinator_controller.build_task_result(job, worker_request)

    def _build_partial_answer(self, task_results: List[Dict[str, Any]]) -> str:
        return self.coordinator_controller.build_partial_answer(task_results)

    def _parse_verification_result(self, result: AgentRunResult) -> VerificationResult:
        return self.coordinator_controller.parse_verification_result(result)

    def _refresh_session_access(self, session_state: SessionState) -> None:
        authz_service = getattr(self.stores, "authorization_service", None) if self.stores is not None else None
        if authz_service is None:
            normalized_email = normalize_user_email(
                getattr(session_state, "user_email", "")
                or dict(session_state.metadata or {}).get("user_email")
            )
            if normalized_email:
                session_state.user_email = normalized_email
                session_state.metadata = {
                    **dict(session_state.metadata or {}),
                    "user_email": normalized_email,
                }
            return
        snapshot = authz_service.apply_access_snapshot(
            session_state,
            tenant_id=session_state.tenant_id,
            user_id=session_state.user_id,
            user_email=normalize_user_email(
                getattr(session_state, "user_email", "")
                or dict(session_state.metadata or {}).get("user_email")
            ),
            session_upload_collection_id=resolve_upload_collection_id(self.settings, session_state),
            display_name=session_state.user_id,
        )
        session_state.metadata = {
            **dict(session_state.metadata or {}),
            "access_summary": snapshot.to_summary(),
            "role_ids": list(snapshot.role_ids),
            "user_email": snapshot.user_email,
            "auth_provider": snapshot.auth_provider,
            "principal_id": snapshot.principal_id,
        }

    def _job_runner(self, job: JobRecord) -> str:
        session_payload = dict(job.metadata.get("session_state") or job.session_state or {})
        session_state = SessionState.from_dict(session_payload)
        session_state.metadata["route_context"] = dict(job.metadata.get("route_context") or {})
        self._refresh_session_access(session_state)
        if isinstance(job.metadata.get("long_output"), dict):
            return self._run_long_output_job(job, session_state)
        agent = self._resolve_agent(job.agent_name)
        skill_execution = dict(job.metadata.get("skill_execution") or {})
        if skill_execution:
            skill_allowed_tools = [
                str(item).strip()
                for item in (skill_execution.get("allowed_tools") or [])
                if str(item).strip()
            ]
            max_steps = int(skill_execution.get("max_steps") or agent.max_steps)
            max_tool_calls = int(skill_execution.get("max_tool_calls") or agent.max_tool_calls)
            agent = replace(
                agent,
                allowed_tools=[
                    tool
                    for tool in agent.allowed_tools
                    if tool_allowed_by_selectors(skill_allowed_tools, tool)
                    or any(tool_allowed_by_selectors([tool], skill_tool) for skill_tool in skill_allowed_tools)
                ],
                max_steps=max(1, min(agent.max_steps, max_steps)),
                max_tool_calls=max(0, min(agent.max_tool_calls, max_tool_calls)),
                metadata={
                    **dict(agent.metadata or {}),
                    "skill_execution": {
                        "skill_id": str(skill_execution.get("skill_id") or ""),
                        "skill_family_id": str(skill_execution.get("skill_family_id") or ""),
                        "invoking_agent": str(skill_execution.get("invoking_agent") or ""),
                    },
                },
            )
        worker_request = dict(job.metadata.get("worker_request") or {})
        mailbox = self.job_manager.drain_mailbox(job.job_id)
        mailbox_prompt = self.job_manager.render_mailbox_prompt(mailbox)
        prompt = mailbox_prompt if mailbox_prompt else job.prompt
        callbacks = self.build_callbacks(
            session_state,
            trace_name="worker_job",
            agent_name=agent.name,
            job_id=job.job_id,
            metadata={
                **dict(session_state.metadata.get("route_context") or {}),
                "worker_request": worker_request,
            },
        )
        self._emit(
            "worker_agent_started",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "job_id": job.job_id,
                "task_id": str(worker_request.get("task_id") or ""),
                "title": str(worker_request.get("title") or ""),
                "doc_scope": list(worker_request.get("doc_scope") or []),
                "detail": str(worker_request.get("description") or ""),
                "parallel_group_id": str(dict(worker_request.get("metadata") or {}).get("parallel_group_id") or ""),
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        result = self.run_agent(
            agent,
            session_state,
            user_text=prompt,
            callbacks=callbacks,
            task_payload={
                "job_id": job.job_id,
                "worker_request": worker_request,
                "skill_queries": worker_request.get("skill_queries") or [],
                "skill_execution": skill_execution,
            },
        )
        session_state.messages = list(result.messages)
        refreshed = self.job_manager.get_job(job.job_id) or job
        if refreshed.status == "waiting_message":
            refreshed.metadata["session_state"] = session_state.to_dict()
            refreshed.metadata["result_metadata"] = dict(result.metadata or {})
            refreshed.result_summary = refreshed.result_summary or result.text[:2000]
            self.transcript_store.persist_job_state(refreshed)
            self.transcript_store.append_job_transcript(
                job.job_id,
                {
                    "kind": "assistant",
                    "content": result.text,
                    "agent_name": agent.name,
                    "status": "waiting_message",
                },
            )
            self._sync_pending_worker_request_for_session(session_state.session_id)
            return result.text
        refreshed.result_summary = result.text[:2000]
        refreshed.metadata["session_state"] = session_state.to_dict()
        refreshed.metadata["result_metadata"] = dict(result.metadata or {})
        self.transcript_store.persist_job_state(refreshed)
        self.transcript_store.append_job_transcript(
            job.job_id,
            {"kind": "assistant", "content": result.text, "agent_name": agent.name},
        )
        self._emit(
            "worker_agent_completed",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "job_id": job.job_id,
                "task_id": str(worker_request.get("task_id") or ""),
                "title": str(worker_request.get("title") or ""),
                "doc_scope": list(worker_request.get("doc_scope") or []),
                "detail": str(worker_request.get("description") or ""),
                "status": str(refreshed.status or "completed"),
                "output_path": refreshed.output_path,
                "parallel_group_id": str(dict(worker_request.get("metadata") or {}).get("parallel_group_id") or ""),
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        notification = self.job_manager.build_notification(refreshed)
        self.job_manager.append_session_notification(session_state.session_id, notification)
        return result.text

    def _create_tool_worker_job(
        self,
        tool_context: ToolContext,
        *,
        agent_name: str,
        prompt: str,
        description: str = "",
        run_in_background: bool = True,
        parent_job_id: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        scoped_state = self._build_scoped_worker_state(tool_context.session, agent_name=agent_name)
        worker_request = WorkerExecutionRequest(
            agent_name=agent_name,
            task_id="manual",
            title=description or prompt[:80] or agent_name,
            prompt=prompt,
            instruction_prompt=prompt,
            semantic_query=prompt,
            description=description or prompt[:120],
        )
        metadata = {
            "session_state": scoped_state.to_dict(),
            "worker_request": worker_request.to_dict(),
            "route_context": dict(tool_context.session.metadata.get("route_context") or {}),
            **dict(extra_metadata or {}),
        }
        return self.job_manager.create_job(
            agent_name=agent_name,
            prompt=prompt,
            session_id=tool_context.session.session_id,
            description=description or prompt[:120],
            tenant_id=tool_context.session.tenant_id,
            user_id=tool_context.session.user_id,
            priority="background" if run_in_background else "interactive",
            queue_class="background" if run_in_background else "interactive",
            session_state=scoped_state.to_dict(),
            metadata=metadata,
            parent_job_id=parent_job_id,
        )

    def _dispatch_agent_message(
        self,
        tool_context: ToolContext,
        *,
        target_agent: str,
        message: str,
        description: str = "",
        target_job_id: str = "",
        reuse_running_job: bool = True,
        team_channel_id: str = "",
    ) -> AgentDispatchOutcome:
        source_job_id = str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or "").strip()
        source_job = self.job_manager.get_job(source_job_id) if source_job_id else None
        source_depth = int((getattr(source_job, "metadata", {}) or {}).get("delegation_depth") or 0)
        outcome = self.job_manager.enqueue_agent_message(
            session_id=tool_context.session.session_id,
            source_agent=str(getattr(tool_context, "active_agent", "") or ""),
            target_agent=target_agent,
            content=message,
            description=description,
            allowed_target_agents=list(getattr(tool_context.active_definition, "allowed_worker_agents", []) or []),
            source_job_id=source_job_id,
            target_job_id=target_job_id,
            reuse_running_job=reuse_running_job,
            source_delegation_depth=source_depth,
            create_job=lambda: self._create_tool_worker_job(
                tool_context,
                agent_name=target_agent,
                prompt=message,
                description=description,
                run_in_background=True,
                parent_job_id=source_job_id,
                extra_metadata={
                    "peer_dispatch": {
                        "source_agent": str(getattr(tool_context, "active_agent", "") or ""),
                        "source_job_id": source_job_id,
                        "description": description,
                    }
                },
            ),
        )
        if outcome.reused_existing_job:
            self.job_manager.continue_job(outcome.job.job_id, self._job_runner)
        else:
            self.job_manager.start_background_job(outcome.job, self._job_runner)
        clean_team_channel_id = str(team_channel_id or "").strip()
        if clean_team_channel_id and self._team_mailbox_enabled():
            channel = self.job_manager.transcript_store.load_team_channel(
                tool_context.session.session_id,
                clean_team_channel_id,
            )
            if channel is not None and self._team_channel_visible(tool_context, channel):
                self.job_manager.post_team_message(
                    session_id=tool_context.session.session_id,
                    channel_id=channel.channel_id,
                    content=message,
                    source_agent=str(getattr(tool_context, "active_agent", "") or ""),
                    source_job_id=source_job_id,
                    target_agents=[target_agent],
                    target_job_ids=[outcome.job.job_id],
                    message_type="handoff",
                    subject=description or f"Peer request for {target_agent}",
                    payload={
                        "peer_dispatch": True,
                        "target_agent": target_agent,
                        "target_job_id": outcome.job.job_id,
                        "reused_existing_job": outcome.reused_existing_job,
                    },
                    metadata={"source": "invoke_agent"},
                )
        return outcome

    def build_agent_system_prompt(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.query_loop._build_system_prompt(
            agent,
            session_state,
            providers=self.providers,
            task_payload=task_payload,
        )

    def _run_long_output_job(self, job: JobRecord, session_state: SessionState) -> str:
        long_output_meta = dict(job.metadata.get("long_output") or {})
        agent_name = str(long_output_meta.get("agent_name") or job.agent_name or "general")
        agent = self._resolve_agent(agent_name)
        session_state.active_agent = agent.name
        options = LongOutputOptions.from_metadata(long_output_meta)
        providers = (
            self.resolve_providers_for_agent(
                agent.name,
                chat_max_output_tokens=coerce_optional_positive_int(
                    long_output_meta.get("chat_max_output_tokens")
                ),
            )
            or self.resolve_base_providers()
            or self.providers
        )
        if providers is None or getattr(providers, "chat", None) is None:
            raise RuntimeError("Long-form generation requires configured chat providers.")
        callbacks = self.build_callbacks(
            session_state,
            trace_name="long_output_job",
            agent_name=agent.name,
            job_id=job.job_id,
            metadata={
                **dict(session_state.metadata.get("route_context") or {}),
                "long_output": options.to_dict(),
            },
        )
        self._emit(
            "worker_agent_started",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "job_id": job.job_id,
                "task_id": "long_output",
                "title": str(long_output_meta.get("title") or "Long-form draft"),
                "detail": "Generating long-form output",
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        composer = LongOutputComposer(
            settings=self.settings,
            chat_llm=providers.chat,
            agent=agent,
            system_prompt=self.build_agent_system_prompt(agent, session_state),
            session_or_state=session_state,
            callbacks=callbacks,
            progress_sink=self.get_live_progress_sink(session_state.session_id),
            metadata={"job_id": job.job_id},
        )
        result = composer.compose(user_text=job.prompt, options=options)
        refreshed = self.job_manager.get_job(job.job_id) or job
        refreshed.output_path = str(Path(session_state.workspace_root) / result.output_filename)
        refreshed.result_summary = result.summary_text[:2000]
        refreshed.metadata["session_state"] = session_state.to_dict()
        refreshed.metadata["artifacts"] = [dict(result.artifact)]
        refreshed.metadata["long_output_result"] = {
            "summary_text": result.summary_text,
            "output_filename": result.output_filename,
            "manifest_filename": result.manifest_filename,
            "title": result.title,
            "section_count": result.section_count,
        }
        self.transcript_store.persist_job_state(refreshed)
        self.transcript_store.append_job_transcript(
            job.job_id,
            {"kind": "assistant", "content": result.summary_text, "agent_name": agent.name},
        )
        self._emit(
            "worker_agent_completed",
            session_state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": session_state.conversation_id,
                "job_id": job.job_id,
                "task_id": "long_output",
                "title": result.title or str(long_output_meta.get("title") or "Long-form draft"),
                "detail": result.output_filename,
                "status": "completed",
                "output_path": refreshed.output_path,
                **dict(session_state.metadata.get("route_context") or {}),
            },
        )
        notification = self.job_manager.build_notification(refreshed)
        notification.summary = result.summary_text
        notification.metadata = {
            **dict(notification.metadata or {}),
            "agent_name": agent.name,
            "artifacts": [dict(result.artifact)],
            "long_output": dict(refreshed.metadata.get("long_output_result") or {}),
        }
        self._append_notification(notification, session_state.session_id)
        return result.summary_text

    def validate_registry(self, registry: AgentRegistry | None = None) -> None:
        tool_definitions = build_tool_definitions(None)
        available_tools = set(tool_definitions.keys())
        effective_registry = registry or self.registry
        available_agents = {definition.name for definition in effective_registry.list()}
        prompt_dir = Path(getattr(self.settings, "skills_dir", Path("data") / "skills"))
        prompt_overlay_dir = getattr(self.settings, "control_panel_prompt_overlays_dir", None)
        resolved_prompt_overlay_dir = Path(prompt_overlay_dir) if prompt_overlay_dir is not None else None
        errors: List[str] = []
        for tool_name, definition in tool_definitions.items():
            for issue in definition.validate_metadata():
                errors.append(f"tool {tool_name!r} {issue}")
        for agent in effective_registry.list():
            for tool_name in agent.allowed_tools:
                if str(tool_name or "").strip().endswith("*"):
                    continue
                if tool_name not in available_tools:
                    errors.append(f"agent {agent.name!r} references unknown tool {tool_name!r}")
            for worker_name in agent.allowed_worker_agents:
                if worker_name not in available_agents:
                    errors.append(f"agent {agent.name!r} references unknown worker {worker_name!r}")
            for scope in agent.memory_scopes:
                try:
                    MemoryScope(scope)
                except ValueError:
                    errors.append(f"agent {agent.name!r} declares invalid memory scope {scope!r}")
            prompt_exists = (prompt_dir / agent.prompt_file).exists()
            if resolved_prompt_overlay_dir is not None:
                prompt_exists = prompt_exists or (resolved_prompt_overlay_dir / agent.prompt_file).exists()
            if agent.prompt_file and not prompt_exists:
                errors.append(f"agent {agent.name!r} prompt file {agent.prompt_file!r} was not found")
        if errors:
            raise ValueError("Invalid next-runtime agent configuration:\n- " + "\n- ".join(errors))

    def _validate_registry(self) -> None:
        self.validate_registry()

    def _import_legacy_memory_for_session(self, session_state: SessionState) -> None:
        if self.memory_store is None or self.file_memory_store is None:
            return
        try:
            payload = {}
            for scope in (MemoryScope.user.value, MemoryScope.conversation.value):
                entries = self.file_memory_store.list_entries(
                    tenant_id=session_state.tenant_id,
                    user_id=session_state.user_id,
                    conversation_id=session_state.conversation_id,
                    scope=scope,
                )
                payload[scope] = [(entry.key, entry.value) for entry in entries]
            self.memory_store.import_legacy_for_session(
                tenant_id=session_state.tenant_id,
                user_id=session_state.user_id,
                conversation_id=session_state.conversation_id,
                session_id=session_state.session_id,
                file_entries_by_scope=payload,
            )
        except Exception as exc:
            logger.warning("Legacy memory import failed for session %s: %s", session_state.session_id, exc)

    def _run_post_turn_memory_maintenance(self, session_state: SessionState, *, latest_text: str) -> None:
        if not bool(getattr(self.settings, "memory_enabled", True)):
            return
        if not latest_text.strip():
            return
        self._import_legacy_memory_for_session(session_state)
        managed_result = None
        if self.memory_write_manager is not None:
            try:
                managed_result = self.memory_write_manager.process_turn(
                    session_state=session_state,
                    latest_user_text=latest_text,
                    providers=self.providers,
                )
                self._emit(
                    "memory_manager_completed",
                    session_state.session_id,
                    agent_name="memory_maintainer",
                    payload={
                        "conversation_id": session_state.conversation_id,
                        "mode": managed_result.mode,
                        "shadow": managed_result.shadow,
                        "applied_count": managed_result.applied_count,
                        "skipped_count": managed_result.skipped_count,
                        "operation_count": len(managed_result.operations),
                        "errors": list(managed_result.errors or []),
                    },
                )
            except Exception as exc:
                logger.warning("Managed memory write manager failed: %s", exc)
                self._emit(
                    "memory_manager_failed",
                    session_state.session_id,
                    agent_name="memory_maintainer",
                    payload={
                        "conversation_id": session_state.conversation_id,
                        "error": str(exc)[:1000],
                    },
                )
        if self.memory_extractor is None:
            return
        managed_live = (
            managed_result is not None
            and str(getattr(managed_result, "mode", "") or "").strip().lower() == "live"
            and int(getattr(managed_result, "applied_count", 0) or 0) > 0
        )
        if managed_live:
            return
        entries = self.memory_extractor.extract_entries(latest_text)
        if not entries:
            self._emit(
                "memory_extraction_skipped",
                session_state.session_id,
                agent_name="memory_maintainer",
                payload={
                    "conversation_id": session_state.conversation_id,
                    "reason": "no_structured_entries",
                    "mode": "heuristic",
                },
            )
            return
        scopes = [MemoryScope.conversation.value]
        if self.memory_extractor.has_explicit_memory_intent(latest_text):
            scopes.append(MemoryScope.user.value)
        self._emit(
            "memory_extraction_started",
            session_state.session_id,
            agent_name="memory_maintainer",
            payload={
                "conversation_id": session_state.conversation_id,
                "scopes": scopes,
                "mode": "heuristic",
            },
        )
        try:
            saved = self.memory_extractor.apply_from_text(session_state, latest_text, scopes=scopes)
        except Exception as exc:
            self._emit(
                "memory_extraction_failed",
                session_state.session_id,
                agent_name="memory_maintainer",
                payload={
                    "conversation_id": session_state.conversation_id,
                    "error": str(exc)[:1000],
                    "scopes": scopes,
                    "mode": "heuristic",
                },
            )
            return
        if saved:
            self._emit(
                "memory_extraction_completed",
                session_state.session_id,
                agent_name="memory_maintainer",
                payload={
                    "conversation_id": session_state.conversation_id,
                    "saved_entries": saved,
                    "scopes": scopes,
                    "mode": "heuristic",
                },
            )

    def _skill_store(self) -> Any | None:
        return getattr(self.stores, "skill_store", None) if self.stores is not None else None

    def _resolve_skill_family_id(
        self,
        skill_id: str,
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> str:
        clean_skill_id = str(skill_id or "").strip()
        if not clean_skill_id:
            return ""
        store = self._skill_store()
        if store is None:
            return clean_skill_id
        getter = getattr(store, "get_skill_pack", None)
        if not callable(getter):
            return clean_skill_id
        try:
            record = getter(clean_skill_id, tenant_id=tenant_id, owner_user_id=owner_user_id)
        except Exception:
            record = None
        if record is None:
            return clean_skill_id
        return str(getattr(record, "version_parent", "") or clean_skill_id)

    def _collect_skill_usage_entries(
        self,
        payload: Any,
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> List[Dict[str, str]]:
        collected: Dict[str, Dict[str, str]] = {}

        def add_skill(skill_id: Any, skill_family_id: Any = "") -> None:
            clean_skill_id = str(skill_id or "").strip()
            if not clean_skill_id:
                return
            clean_family_id = str(skill_family_id or "").strip() or self._resolve_skill_family_id(
                clean_skill_id,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
            )
            collected.setdefault(
                clean_skill_id,
                {
                    "skill_id": clean_skill_id,
                    "skill_family_id": clean_family_id or clean_skill_id,
                },
            )

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                skill_resolution = node.get("skill_resolution")
                if isinstance(skill_resolution, dict):
                    for match in skill_resolution.get("matches") or []:
                        if isinstance(match, dict):
                            add_skill(
                                match.get("skill_id"),
                                match.get("skill_family_id") or match.get("version_parent"),
                            )
                rag_execution_hints = node.get("rag_execution_hints")
                if isinstance(rag_execution_hints, dict):
                    for matched_skill_id in rag_execution_hints.get("matched_skill_ids") or []:
                        add_skill(matched_skill_id)
                for value in node.values():
                    visit(value)
                return
            if isinstance(node, list):
                for item in node:
                    visit(item)

        visit(payload)
        return list(collected.values())

    def _record_skill_telemetry(
        self,
        session_state: SessionState,
        *,
        agent_name: str,
        user_text: str,
        metadata: Dict[str, Any],
    ) -> None:
        verification = dict(metadata.get("verification") or {})
        answer_quality = coerce_answer_quality(verification.get("status"))
        if not is_scored_answer_quality(answer_quality):
            return
        store = self._skill_store()
        if store is None:
            return
        append_event = getattr(store, "append_skill_telemetry_event", None)
        if not callable(append_event):
            return
        try:
            skill_entries = self._collect_skill_usage_entries(
                metadata,
                tenant_id=session_state.tenant_id,
                owner_user_id=session_state.user_id,
            )
            if not skill_entries:
                return
            list_events = getattr(store, "list_skill_telemetry_events", None)
            review_status_before: Dict[str, str] = {}
            touched_families = sorted(
                {
                    str(item.get("skill_family_id") or "")
                    for item in skill_entries
                    if str(item.get("skill_family_id") or "")
                }
            )
            if callable(list_events):
                for family_id in touched_families:
                    before = compute_skill_health(
                        list_events(
                            tenant_id=session_state.tenant_id,
                            skill_family_id=family_id,
                            limit=200,
                        )
                    )
                    review_status_before[family_id] = before.review_status
            for entry in skill_entries:
                append_event(
                    SkillTelemetryEvent.build(
                        tenant_id=session_state.tenant_id,
                        skill_id=str(entry.get("skill_id") or ""),
                        skill_family_id=str(entry.get("skill_family_id") or ""),
                        query=user_text,
                        answer_quality=answer_quality,
                        agent_name=agent_name,
                        session_id=session_state.session_id,
                    )
                )
            if callable(list_events):
                for family_id in touched_families:
                    after = compute_skill_health(
                        list_events(
                            tenant_id=session_state.tenant_id,
                            skill_family_id=family_id,
                            limit=200,
                        )
                    )
                    before_status = review_status_before.get(family_id, "insufficient_data")
                    if before_status != "flagged" and after.review_status == "flagged":
                        self._emit(
                            "skill_review_flagged",
                            session_state.session_id,
                            agent_name=agent_name,
                            payload={
                                "skill_family_id": family_id,
                                "success_rate": after.success_rate,
                                "scored_uses": after.scored_uses,
                            },
                        )
                    elif before_status == "flagged" and after.review_status != "flagged":
                        self._emit(
                            "skill_review_cleared",
                            session_state.session_id,
                            agent_name=agent_name,
                            payload={
                                "skill_family_id": family_id,
                                "success_rate": after.success_rate,
                                "scored_uses": after.scored_uses,
                                "review_status": after.review_status,
                            },
                        )
        except Exception as exc:
            logger.warning("Could not record skill telemetry: %s", exc)

    def _resolve_agent(self, agent_name: str) -> AgentDefinition:
        agent = self.registry.get(agent_name)
        if agent is None:
            raise ValueError(f"Runtime agent {agent_name!r} is not defined.")
        return agent

    def _load_or_build_session_state(self, session: Any) -> SessionState:
        incoming = SessionState.from_session(session)
        stored = self.transcript_store.load_session_state(incoming.session_id)
        if stored is None:
            original_count = len(incoming.messages)
            if incoming.messages:
                self.transcript_store.ensure_session_transcript_seeded(incoming.session_id, incoming.messages)
            window = max(1, int(getattr(self.settings, "session_hydrate_window_messages", 40)))
            if original_count > window:
                incoming.messages = incoming.messages[-window:]
            metadata = dict(incoming.metadata or {})
            metadata["history_total_messages"] = max(original_count, int(metadata.get("history_total_messages") or 0))
            metadata["history_stored_window_messages"] = len(incoming.messages)
            metadata["has_earlier_history"] = int(metadata["history_total_messages"]) > len(incoming.messages)
            incoming.metadata = metadata
        state = stored or incoming
        if stored is not None and incoming.metadata:
            state.metadata = {
                **dict(state.metadata or {}),
                **dict(incoming.metadata or {}),
            }
        if stored is not None and incoming.uploaded_doc_ids:
            merged_uploads = list(state.uploaded_doc_ids or [])
            for doc_id in incoming.uploaded_doc_ids:
                if doc_id and doc_id not in merged_uploads:
                    merged_uploads.append(doc_id)
            state.uploaded_doc_ids = merged_uploads
        if incoming.user_email:
            state.user_email = incoming.user_email
        if incoming.auth_provider:
            state.auth_provider = incoming.auth_provider
        if incoming.principal_id:
            state.principal_id = incoming.principal_id
        if incoming.access_summary:
            state.access_summary = dict(incoming.access_summary)
        if not state.workspace_root:
            state.workspace_root = str(self.paths.workspace_dir(state.session_id))
        state.metadata.setdefault("runtime_kind", "next")
        return state

    def _persist_state(self, state: SessionState) -> None:
        self.transcript_store.persist_session_state(state)

    def _sync_pending_clarification(self, state: SessionState) -> None:
        latest_assistant = None
        for message in reversed(state.messages):
            if message.role == "assistant":
                if is_openwebui_helper_message(message):
                    continue
                latest_assistant = message
                break
        if latest_assistant is None:
            state.metadata.pop("pending_clarification", None)
            return
        metadata = dict(latest_assistant.metadata or {})
        if not is_clarification_turn(metadata):
            state.metadata.pop("pending_clarification", None)
            return
        clarification = clarification_from_metadata(metadata)
        if clarification is None:
            state.metadata.pop("pending_clarification", None)
            return
        route_context = dict(state.metadata.get("route_context") or {})
        resolved_turn_intent = dict(state.metadata.get("resolved_turn_intent") or {})
        semantic_routing = dict(
            route_context.get("semantic_routing")
            or state.metadata.get("semantic_routing")
            or {}
        )
        state.metadata["pending_clarification"] = {
            **clarification.to_dict(),
            "message_id": latest_assistant.message_id,
            "selected_agent": str(metadata.get("agent_name") or state.active_agent or route_context.get("suggested_agent") or "").strip(),
            "route_context": route_context,
            "semantic_routing": semantic_routing,
            "resolved_turn_intent": resolved_turn_intent,
            "original_user_text": str(
                resolved_turn_intent.get("normalized_user_objective")
                or resolved_turn_intent.get("source_user_text")
                or ""
            ).strip(),
        }

    def _sync_pending_worker_request(self, state: SessionState) -> None:
        open_questions: List[Dict[str, Any]] = []
        open_approvals: List[Dict[str, Any]] = []
        for job in self.job_manager.list_jobs(session_id=state.session_id):
            for item in self.job_manager.list_mailbox_requests(job.job_id, status_filter="open"):
                row = {
                    "job_id": job.job_id,
                    "agent_name": job.agent_name,
                    **item.to_dict(),
                }
                if item.message_type == "question_request":
                    open_questions.append(row)
                elif item.message_type == "approval_request":
                    open_approvals.append(row)
        if open_questions:
            state.metadata["pending_worker_question"] = open_questions[-1]
        else:
            state.metadata.pop("pending_worker_question", None)
        if open_approvals:
            state.metadata["pending_worker_approval"] = open_approvals[-1]
        else:
            state.metadata.pop("pending_worker_approval", None)
        state.metadata["pending_worker_request_counts"] = {
            "questions": len(open_questions),
            "approvals": len(open_approvals),
        }

    def _sync_pending_worker_request_for_session(self, session_id: str) -> None:
        clean_session_id = str(session_id or "").strip()
        if not clean_session_id:
            return
        state = self.transcript_store.load_session_state(clean_session_id)
        if state is None:
            return
        self._sync_pending_worker_request(state)
        self._persist_state(state)

    def _sync_active_doc_focus(self, state: SessionState) -> None:
        latest_assistant = None
        for message in reversed(state.messages):
            if message.role == "assistant":
                latest_assistant = message
                break
        if latest_assistant is None:
            return
        payload = doc_focus_result_from_metadata(dict(latest_assistant.metadata or {}))
        if payload is None:
            return
        state.metadata["active_doc_focus"] = {
            **payload,
            "message_id": latest_assistant.message_id,
        }

    def _append_notification(self, notification: TaskNotification, session_id: str) -> None:
        self.transcript_store.append_session_transcript(
            session_id,
            {"kind": "notification", "notification": notification.to_dict()},
        )
        self.notification_store.append(session_id, notification)
        self._emit(
            "notification_appended",
            session_id,
            agent_name=str(notification.metadata.get("agent_name") or ""),
            payload={"job_id": notification.job_id, "status": notification.status},
            job_id=notification.job_id,
        )

    def _drain_pending_notifications(self, session_state: SessionState) -> None:
        for notification in self.notification_store.drain(session_state.session_id):
            session_state.add_notification(notification)

    def _emit(
        self,
        event_type: str,
        session_id: str,
        *,
        agent_name: str = "",
        payload: Optional[Dict[str, Any]] = None,
        tool_name: str = "",
        job_id: str = "",
    ) -> None:
        self.event_controller.emit(
            event_type,
            session_id,
            agent_name=agent_name,
            payload=payload,
            tool_name=tool_name,
            job_id=job_id,
        )

    def persist_manual_assistant_response(
        self,
        session: Any,
        *,
        text: str,
        agent_name: str,
        route_metadata: Optional[Dict[str, Any]] = None,
        message_metadata: Optional[Dict[str, Any]] = None,
        artifact_refs: Optional[List[str]] = None,
    ) -> str:
        state = self.hydrate_session_state(session)
        state.metadata["route_context"] = dict(route_metadata or {})
        pending_router_feedback_id = str((route_metadata or {}).get("router_decision_id") or "").strip()
        if pending_router_feedback_id:
            state.metadata["pending_router_feedback_id"] = pending_router_feedback_id
        metadata = {"agent_name": agent_name, **dict(message_metadata or {})}
        message = state.append_message(
            "assistant",
            text,
            metadata=metadata,
            artifact_refs=[str(item) for item in (artifact_refs or []) if str(item)],
        )
        self.router_feedback.observe_turn_result(
            state,
            metadata=dict(message.metadata or {}),
            route_context=dict(route_metadata or {}),
        )
        self._sync_active_doc_focus(state)
        self._persist_state(state)
        self.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": message.to_dict()},
        )
        state.sync_to_session(session)
        return text
