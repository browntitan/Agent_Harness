from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from langchain_core.messages import AIMessage

from agentic_chatbot_next.capabilities import resolve_effective_capabilities
from agentic_chatbot_next.authz import (
    access_summary_allows,
    access_summary_authz_enabled,
    normalize_user_email,
)
from agentic_chatbot_next.config import runtime_settings_diagnostics
from agentic_chatbot_next.context import build_local_context
from agentic_chatbot_next.providers.circuit_breaker import CircuitBreakerOpenError
from agentic_chatbot_next.rag import (
    KBCoverageStatus,
    KnowledgeStores,
    SkillIndexSync,
    ensure_kb_indexed,
    get_kb_coverage_status,
    ingest_paths,
    load_basic_chat_skills,
    load_stores,
)
from agentic_chatbot_next.rag.doc_targets import extract_named_document_targets
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_FILE,
    INVENTORY_QUERY_GRAPH_INDEXES,
    classify_inventory_query,
    extract_requested_kb_collection_id,
    inventory_query_requests_grounded_analysis,
    is_authoritative_inventory_query_type,
    sync_session_kb_collection_state,
)
from agentic_chatbot_next.rag.retrieval_scope import (
    decide_retrieval_scope,
    document_source_policy_requires_repository,
    has_upload_evidence,
    merge_scope_metadata,
    query_requests_upload_scope,
    repository_upload_doc_ids,
    resolve_upload_collection_id,
)
from agentic_chatbot_next.providers.factory import ProviderBundle
from agentic_chatbot_next.rag.engine import render_rag_contract, run_rag_contract
from agentic_chatbot_next.router.llm_router import route_turn
from agentic_chatbot_next.router.patterns import load_router_patterns, patterns_path_from_settings
from agentic_chatbot_next.router.policy import choose_agent_name
from agentic_chatbot_next.router.router import RouterDecision
from agentic_chatbot_next.router.semantic import (
    SemanticRoutingContract,
    build_deterministic_semantic_contract,
    default_agent_for_semantic_contract,
    semantic_contract_requires_agent,
)
from agentic_chatbot_next.runtime.deep_rag import decide_deep_rag_policy
from agentic_chatbot_next.runtime.long_output import LongOutputComposer, LongOutputOptions
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.runtime.openwebui_helpers import (
    infer_openwebui_helper_task_type,
    is_openwebui_helper_message,
    normalize_openwebui_helper_task_type,
    openwebui_helper_system_prompt,
)
from agentic_chatbot_next.runtime.research_packet import build_research_packet
from agentic_chatbot_next.runtime.task_decomposition import decide_task_decomposition
from agentic_chatbot_next.runtime.turn_contracts import resolve_turn_intent
from agentic_chatbot_next.sandbox.workspace import SessionWorkspace

logger = logging.getLogger(__name__)

_BROAD_ANALYSIS_HINTS = re.compile(
    r"\b("
    r"analy(?:s|z)e|architecture|architectural|subsystems?|system\s+boundaries|"
    r"control\s+flow|cross-cutting|synthes(?:is|ize)|thorough|comprehensive|detailed|"
    r"deep\s+dive|major\s+subsystems"
    r")\b",
    re.IGNORECASE,
)
_CLAUSE_POLICY_WORKFLOW_HINTS = re.compile(
    r"\b(clause|clauses|redline|redlines|marked\s+changes?|tracked\s+changes?)\b",
    re.IGNORECASE,
)
_DIRECT_RAG_REQUEST_HINTS = re.compile(
    r"\b("
    r"search|cite|cites?|citation|citations|source|sources|grounded|knowledge\s+base|"
    r"default\s+kb|indexed\s+(?:documents|knowledge\s+base)|check\s+(?:indexed|uploaded)\s+sources"
    r")\b",
    re.IGNORECASE,
)
_POLICY_LOOKUP_WORKFLOW_HINTS = re.compile(
    r"\b(policy|policies|guidance|knowledge\s+base|kb|collection|internal\s+policy)\b",
    re.IGNORECASE,
)
_PER_ITEM_WORKFLOW_HINTS = re.compile(
    r"\b(each|every|all|per[-\s]?item|loop|fan\s*out|for\s+each)\b",
    re.IGNORECASE,
)


def _degraded_service_text() -> str:
    return (
        "I’m sorry, but the model service is temporarily degraded right now. "
        "I couldn’t complete the request safely. Please retry in a moment."
    )


def _summarise_history(messages: List[Any], n: int = 2) -> str:
    human_ai_pairs: List[str] = []
    for msg in reversed(messages):
        if is_openwebui_helper_message(msg):
            continue
        role = getattr(msg, "type", "")
        content = str(getattr(msg, "content", "") or "").strip()
        if role in ("human", "ai") and content:
            prefix = "User" if role == "human" else "Assistant"
            human_ai_pairs.append(f"{prefix}: {content[:120]}")
        if len(human_ai_pairs) >= n * 2:
            break
    return "\n".join(reversed(human_ai_pairs))


def _build_history_summary(
    *,
    messages: List[Any],
    metadata: Dict[str, Any] | None = None,
    uploaded_doc_ids: List[str] | None = None,
) -> str:
    session_like = SimpleNamespace(
        messages=list(messages or []),
        metadata=dict(metadata or {}),
        uploaded_doc_ids=list(uploaded_doc_ids or []),
    )
    packet = build_research_packet(
        session_like,
        recent_messages=8,
        message_char_limit=220,
        retrieval_limit=4,
    )
    if packet.strip():
        return packet
    return _summarise_history(messages, n=2)


def _coerce_positive_output_tokens(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_router_decision(
    raw_decision: Any,
    *,
    user_text: str,
    session_metadata: Dict[str, Any] | None = None,
) -> RouterDecision:
    route = str(getattr(raw_decision, "route", "BASIC") or "BASIC").strip().upper() or "BASIC"
    try:
        confidence = float(getattr(raw_decision, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    reasons = [
        str(reason).strip()
        for reason in list(getattr(raw_decision, "reasons", []) or [])
        if str(reason).strip()
    ]
    suggested_agent = str(getattr(raw_decision, "suggested_agent", "") or "").strip().lower()
    router_method = str(getattr(raw_decision, "router_method", "deterministic") or "deterministic").strip() or "deterministic"
    router_decision_id = str(getattr(raw_decision, "router_decision_id", "") or "").strip()
    router_evidence = dict(getattr(raw_decision, "router_evidence", {}) or {})
    semantic_contract = SemanticRoutingContract.from_value(
        getattr(raw_decision, "semantic_contract", None),
        default_route=route,
        default_confidence=confidence,
        default_reasoning="; ".join(reasons),
        default_suggested_agent=suggested_agent,
    )
    if getattr(raw_decision, "semantic_contract", None) is None:
        semantic_contract = build_deterministic_semantic_contract(
            user_text=user_text,
            route=route,
            suggested_agent=suggested_agent,
            confidence=confidence,
            reasoning="; ".join(reasons),
            session_metadata=session_metadata,
        )
    kwargs: Dict[str, Any] = {
        "route": route,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasons": reasons,
        "suggested_agent": suggested_agent,
        "router_method": router_method,
        "router_evidence": router_evidence,
        "semantic_contract": semantic_contract,
    }
    if router_decision_id:
        kwargs["router_decision_id"] = router_decision_id
    return RouterDecision(**kwargs)


def _should_shortcut_to_rag_worker(
    *,
    user_text: str,
    request_metadata: Dict[str, Any],
    scope_mode: str,
    requested_agent_override: str,
    force_agent: bool,
    helper_task_type: str,
) -> bool:
    if force_agent or requested_agent_override or helper_task_type:
        return False
    if str(scope_mode or "").strip().lower() in {"none", "ambiguous"}:
        return False
    lowered = str(user_text or "").casefold()
    if re.search(r"\b(calculator|compute|remember|memory|save\s+this|recall)\b", lowered):
        return False
    has_index_scope = any(
        str(request_metadata.get(key) or "").strip()
        for key in (
            "kb_collection_id",
            "requested_kb_collection_id",
            "selected_kb_collection_id",
            "upload_collection_id",
            "collection_id",
        )
    )
    text_names_index = bool(
        re.search(r"\b(knowledge\s+base|default\s+kb|indexed|uploaded\s+(?:document|file|source))\b", lowered)
    )
    return bool(_DIRECT_RAG_REQUEST_HINTS.search(user_text) and (has_index_scope or text_names_index))


def _should_default_to_coordinator_for_broad_grounded_analysis(
    user_text: str,
    *,
    session_metadata: Dict[str, Any] | None = None,
) -> bool:
    inventory_query_type = classify_inventory_query(user_text)
    if is_authoritative_inventory_query_type(inventory_query_type) and not inventory_query_requests_grounded_analysis(
        user_text,
        query_type=inventory_query_type,
    ):
        return False
    if extract_named_document_targets(user_text) or re.search(
        r"\b[a-z0-9._/-]+\.(?:md|pdf|docx|txt|csv|xlsx|xls)\b",
        str(user_text or ""),
        flags=re.I,
    ):
        return False
    resolved_intent = resolve_turn_intent(user_text, session_metadata or {})
    if str(resolved_intent.answer_contract.kind or "").strip() != "grounded_synthesis":
        return False
    if not bool(resolved_intent.answer_contract.broad_coverage):
        return False
    if str(resolved_intent.answer_contract.depth or "").strip() == "deep":
        return True
    return bool(_BROAD_ANALYSIS_HINTS.search(str(user_text or "")))


def _should_default_to_coordinator_for_capability_workflow(
    user_text: str,
    *,
    session_metadata: Dict[str, Any] | None = None,
) -> bool:
    text = str(user_text or "")
    metadata = dict(session_metadata or {})
    has_upload_context = bool(metadata.get("uploaded_doc_ids") or metadata.get("has_uploads"))
    mixed_clause_policy = bool(_CLAUSE_POLICY_WORKFLOW_HINTS.search(text)) and bool(
        _POLICY_LOOKUP_WORKFLOW_HINTS.search(text)
    )
    per_item_policy = bool(_PER_ITEM_WORKFLOW_HINTS.search(text)) and bool(_POLICY_LOOKUP_WORKFLOW_HINTS.search(text))
    buyer_response = bool(re.search(r"\b(buyer|supplier|write\s+back|recommended\s+action|recommendation)\b", text, re.I))
    return bool((has_upload_context or "uploaded" in text.lower()) and mixed_clause_policy and (per_item_policy or buyer_response))


@dataclass
class AppContext:
    settings: Any
    providers: ProviderBundle
    stores: KnowledgeStores


class RuntimeService:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._basic_chat_system_prompt = load_basic_chat_skills(ctx.settings)
        self._kb_status_by_tenant: Dict[str, KBCoverageStatus] = {}
        self.skill_index_sync_summary: Dict[str, Any] = {}
        self.skill_subsystem_degraded = False
        self._skill_hot_reload_thread: threading.Thread | None = None
        load_router_patterns(patterns_path_from_settings(ctx.settings))
        self.kernel = RuntimeKernel(ctx.settings, providers=ctx.providers, stores=ctx.stores)
        if getattr(ctx.settings, "agent_runtime_mode", ""):
            logger.info(
                "AGENT_RUNTIME_MODE=%s is deprecated and ignored by agentic_chatbot_next.",
                ctx.settings.agent_runtime_mode,
            )
        if getattr(ctx.settings, "agent_definitions_json", ""):
            logger.info("AGENT_DEFINITIONS_JSON is deprecated and ignored by agentic_chatbot_next.")
        try:
            sync_summary = SkillIndexSync(self.ctx.settings, self.ctx.stores).sync(
                tenant_id=self.ctx.settings.default_tenant_id,
            )
            self.skill_index_sync_summary = dict(sync_summary or {})
            self.skill_subsystem_degraded = not bool(self.skill_index_sync_summary.get("valid", True))
            dependency_graph = dict(self.skill_index_sync_summary.get("dependency_graph") or {})
            if self.skill_subsystem_degraded:
                logger.warning(
                    "Skill pack dependency validation failed at startup: %s",
                    dependency_graph,
                )
                startup_ctx = build_local_context(
                    self.ctx.settings,
                    conversation_id="skill-index-sync",
                    request_id="startup",
                )
                self.kernel._emit(
                    "skill_validation_warning",
                    startup_ctx.session_id,
                    agent_name="system",
                    payload={
                        "dependency_graph": dependency_graph,
                        "indexed_count": int(self.skill_index_sync_summary.get("count", 0)),
                    },
                )
        except Exception as exc:
            logger.warning("Could not sync skill packs at startup: %s", exc)
        self._start_skill_hot_reload_watcher()
        try:
            self.kernel.router_feedback.finalize_stale_decisions()
            self.kernel.router_feedback.generate_quarterly_retrain_artifacts(force=False)
        except Exception as exc:
            logger.warning("Could not refresh router feedback artifacts at startup: %s", exc)
        self._ensure_kb_ready(self.ctx.settings.default_tenant_id)

    def _start_skill_hot_reload_watcher(self) -> None:
        if not bool(getattr(self.ctx.settings, "skill_packs_hot_reload_enabled", False)):
            return
        if self._skill_hot_reload_thread is not None:
            return

        interval = max(1, int(getattr(self.ctx.settings, "skill_packs_hot_reload_interval_seconds", 5) or 5))
        tenant_id = self.ctx.settings.default_tenant_id

        def _watch() -> None:
            syncer = SkillIndexSync(self.ctx.settings, self.ctx.stores)
            while True:
                time.sleep(interval)
                try:
                    summary = syncer.sync_changed(tenant_id=tenant_id)
                    if int(summary.get("changed_count", 0) or 0):
                        self.skill_index_sync_summary = dict(summary or {})
                        self.skill_subsystem_degraded = not bool(summary.get("valid", True))
                        logger.info("Hot-reloaded %s changed skill pack(s).", summary.get("changed_count", 0))
                except Exception as exc:
                    logger.warning("Could not hot-reload skill packs: %s", exc)

        self._skill_hot_reload_thread = threading.Thread(
            target=_watch,
            name="skill-pack-hot-reload",
            daemon=True,
        )
        self._skill_hot_reload_thread.start()

    @classmethod
    def create(cls, settings: Any, providers: ProviderBundle) -> "RuntimeService":
        stores = load_stores(settings, providers.embeddings)
        return cls(AppContext(settings=settings, providers=providers, stores=stores))

    def _ensure_workspace(self, session: Any) -> None:
        if getattr(session, "workspace", None) is not None:
            return
        if getattr(self.ctx.settings, "workspace_dir", None) is None:
            return
        try:
            workspace = SessionWorkspace.for_session(session.session_id, self.ctx.settings.workspace_dir)
            workspace.open()
            session.workspace = workspace
            logger.debug("Opened session workspace at %s", workspace.root)
        except Exception as exc:
            logger.warning("Could not open session workspace: %s", exc)

    def _authorization_service(self) -> Any | None:
        return getattr(self.ctx.stores, "authorization_service", None)

    def _refresh_session_access(
        self,
        session: Any,
        *,
        request_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metadata = merge_scope_metadata(
            self.ctx.settings,
            {
                **dict(getattr(session, "metadata", {}) or {}),
                **dict(request_metadata or {}),
            },
        )
        session.metadata = metadata
        normalized_email = normalize_user_email(
            dict(request_metadata or {}).get("user_email")
            or metadata.get("user_email")
            or getattr(session, "user_email", "")
        )
        if hasattr(session, "user_email"):
            session.user_email = normalized_email
        authz_service = self._authorization_service()
        if authz_service is None:
            session.metadata = {
                **metadata,
                "user_email": normalized_email,
                "access_summary": dict(getattr(session, "access_summary", {}) or metadata.get("access_summary") or {}),
            }
            return dict(session.metadata.get("access_summary") or {})

        snapshot = authz_service.apply_access_snapshot(
            session,
            tenant_id=str(getattr(session, "tenant_id", self.ctx.settings.default_tenant_id) or self.ctx.settings.default_tenant_id),
            user_id=str(getattr(session, "user_id", self.ctx.settings.default_user_id) or self.ctx.settings.default_user_id),
            user_email=normalized_email,
            session_upload_collection_id=resolve_upload_collection_id(self.ctx.settings, session),
            display_name=str(getattr(session, "user_id", "") or normalized_email),
        )
        summary = snapshot.to_summary()
        session.metadata = {
            **dict(getattr(session, "metadata", {}) or {}),
            "access_summary": summary,
            "role_ids": list(snapshot.role_ids),
            "user_email": snapshot.user_email,
            "auth_provider": snapshot.auth_provider,
            "principal_id": snapshot.principal_id,
        }
        return summary

    def _require_collection_use_access(
        self,
        access_summary: Dict[str, Any] | None,
        *,
        collection_id: str,
    ) -> None:
        normalized_collection_id = str(collection_id or "").strip()
        if not normalized_collection_id or not access_summary_authz_enabled(access_summary):
            return
        if access_summary_allows(
            access_summary,
            "collection",
            normalized_collection_id,
            action="use",
            implicit_resource_id=str(dict(access_summary or {}).get("session_upload_collection_id") or ""),
        ):
            return
        raise HTTPException(
            status_code=403,
            detail=f"User is not allowed to use collection '{normalized_collection_id}'.",
        )

    def _ensure_kb_ready(self, tenant_id: str, *, attempt_sync: bool | None = None) -> KBCoverageStatus | None:
        if self.ctx.stores is None or not hasattr(self.ctx.stores, "doc_store"):
            return None
        try:
            status = ensure_kb_indexed(
                self.ctx.settings,
                self.ctx.stores,
                tenant_id=tenant_id,
                collection_id=self.ctx.settings.default_collection_id,
                attempt_sync=attempt_sync,
            )
            self._kb_status_by_tenant[tenant_id] = status
            if not status.ready:
                logger.warning(
                    "KB not ready for tenant=%s collection=%s reason=%s missing=%d",
                    tenant_id,
                    status.collection_id,
                    status.reason,
                    len(status.missing_source_paths),
                )
            return status
        except Exception as exc:
            logger.warning("Could not ensure KB index readiness: %s", exc)
            return None

    def get_kb_status(
        self,
        tenant_id: str | None = None,
        *,
        refresh: bool = False,
        attempt_sync: bool = False,
    ) -> KBCoverageStatus | None:
        effective_tenant_id = tenant_id or self.ctx.settings.default_tenant_id
        if self.ctx.stores is None or not hasattr(self.ctx.stores, "doc_store"):
            return None
        if refresh or effective_tenant_id not in self._kb_status_by_tenant:
            if attempt_sync:
                return self._ensure_kb_ready(effective_tenant_id, attempt_sync=True)
            try:
                status = get_kb_coverage_status(
                    self.ctx.settings,
                    self.ctx.stores,
                    tenant_id=effective_tenant_id,
                    collection_id=self.ctx.settings.default_collection_id,
                )
            except Exception as exc:
                logger.warning("Could not read KB coverage status: %s", exc)
                return self._kb_status_by_tenant.get(effective_tenant_id)
            self._kb_status_by_tenant[effective_tenant_id] = status
        return self._kb_status_by_tenant.get(effective_tenant_id)

    def list_requested_agent_overrides(self) -> List[str]:
        registry = getattr(self.kernel, "registry", None)
        if registry is None or not hasattr(registry, "list_routable"):
            return ["coordinator", "data_analyst", "general", "rag_worker"]

        allowed: List[str] = []
        for agent in registry.list_routable():
            if str(getattr(agent, "mode", "") or "").strip().lower() == "basic":
                continue
            name = str(getattr(agent, "name", "") or "").strip().lower()
            if name == "memory_maintainer" and not bool(getattr(self.ctx.settings, "memory_enabled", True)):
                continue
            if name and name not in allowed:
                allowed.append(name)
        return allowed

    def _process_openwebui_helper_turn(
        self,
        session: Any,
        *,
        user_text: str,
        helper_task_type: str,
        callbacks: Optional[List[Any]] = None,
    ) -> str:
        route_metadata = {
            "route": "BASIC",
            "router_confidence": 1.0,
            "router_reasons": ["openwebui_helper_task"],
            "router_method": "metadata_helper_task",
            "suggested_agent": "basic",
            "has_attachments": False,
            "uploaded_doc_ids": list(getattr(session, "uploaded_doc_ids", []) or []),
            "tenant_id": session.tenant_id,
            "user_id": session.user_id,
            "conversation_id": session.conversation_id,
            "request_id": session.request_id,
            "requested_agent_override": "",
            "requested_agent_override_applied": False,
            "long_output_requested": False,
            "openwebui_helper_task_type": helper_task_type,
        }
        decision_record = self.kernel.router_feedback.register_decision(
            session,
            user_text=user_text,
            route="BASIC",
            confidence=1.0,
            reasons=["openwebui_helper_task"],
            router_method="metadata_helper_task",
            suggested_agent="basic",
            force_agent=False,
            has_attachments=False,
            router_evidence={"openwebui_helper_task_type": helper_task_type},
        )
        route_metadata["router_decision_id"] = decision_record.router_decision_id
        route_metadata["router_evidence"] = dict(decision_record.router_evidence or {})
        self.kernel.emit_router_decision(
            session,
            router_decision_id=decision_record.router_decision_id,
            route="BASIC",
            confidence=1.0,
            reasons=["openwebui_helper_task"],
            router_method="metadata_helper_task",
            suggested_agent="basic",
            force_agent=False,
            has_attachments=False,
            requested_agent_override="",
            requested_agent_override_applied=False,
            router_evidence=dict(decision_record.router_evidence or {}),
        )
        basic_providers = self.kernel.resolve_providers_for_agent("basic") or self.kernel.resolve_base_providers() or self.ctx.providers
        helper_metadata = {"openwebui_helper_task_type": helper_task_type, "openwebui_internal": True}
        return self.kernel.process_basic_turn(
            session,
            user_text=user_text,
            system_prompt=openwebui_helper_system_prompt(helper_task_type),
            chat_llm=basic_providers.chat,
            route_metadata=route_metadata,
            callbacks=callbacks,
            user_message_metadata=helper_metadata,
            assistant_message_metadata=helper_metadata,
            skip_post_turn_memory=True,
        )

    def _answer_pending_worker_question(
        self,
        session: Any,
        *,
        pending_worker_question: Dict[str, Any],
        user_text: str,
    ) -> str:
        job_id = str(pending_worker_question.get("job_id") or "").strip()
        request_id = str(pending_worker_question.get("message_id") or "").strip()
        if not job_id or not request_id:
            return ""
        job = self.kernel.job_manager.get_job(job_id)
        if job is None or str(job.session_id or "") != str(getattr(session, "session_id", "") or ""):
            return ""
        state = self.kernel.hydrate_session_state(session)
        user_message = state.append_message(
            "user",
            user_text,
            metadata={
                "worker_request_response": {
                    "job_id": job_id,
                    "request_id": request_id,
                    "message_type": "question_response",
                }
            },
        )
        self.kernel._persist_state(state)
        self.kernel.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": user_message.to_dict()},
        )
        try:
            result = self.kernel.job_manager.respond_to_request(
                job_id,
                request_id,
                response=user_text,
                responder="user",
                allow_approval=False,
                metadata={"source": "pending_worker_question"},
            )
        except (PermissionError, ValueError):
            return ""
        if result is None:
            return ""
        self.kernel.job_manager.continue_job(job_id, self.kernel._job_runner)
        self.kernel._sync_pending_worker_request(state)
        response_text = f"I sent that answer to `{job.agent_name}` and resumed worker job `{job_id}`."
        assistant_message = state.append_message(
            "assistant",
            response_text,
            metadata={
                "agent_name": "worker_mailbox",
                "turn_outcome": "worker_question_response",
                "worker_request_response": {
                    "job_id": job_id,
                    "request_id": request_id,
                },
            },
        )
        self.kernel._persist_state(state)
        self.kernel.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": assistant_message.to_dict()},
        )
        state.sync_to_session(session)
        return response_text

    def _normalize_requested_agent_override(self, requested_agent: str | None) -> str:
        clean = str(requested_agent or "").strip().lower()
        if not clean:
            return ""
        allowed = set(self.list_requested_agent_overrides())
        if clean not in allowed:
            raise ValueError(
                "Unsupported requested_agent "
                f"{clean!r}. Allowed values: {', '.join(sorted(allowed))}"
            )
        return clean

    def ingest_and_summarize_uploads(
        self,
        session: Any,
        upload_paths: List[Path],
        *,
        progress_sink: Any | None = None,
    ) -> Tuple[List[str], str]:
        callbacks = self.kernel.build_callbacks(
            session,
            trace_name="upload_ingest",
            agent_name="rag_worker",
            metadata={
                "num_files": len(upload_paths),
                "tenant_id": session.tenant_id,
                "user_id": session.user_id,
                "conversation_id": session.conversation_id,
                "request_id": session.request_id,
            },
        )

        doc_ids = ingest_paths(
            self.ctx.settings,
            self.ctx.stores,
            upload_paths,
            source_type="upload",
            tenant_id=session.tenant_id,
            collection_id=str(
                dict(getattr(session, "metadata", {}) or {}).get("collection_id")
                or getattr(self.ctx.settings, "default_collection_id", "default")
            ),
        )
        session.uploaded_doc_ids.extend([doc_id for doc_id in doc_ids if doc_id not in session.uploaded_doc_ids])

        if getattr(session, "workspace", None) is not None:
            for upload_path in upload_paths:
                try:
                    session.workspace.copy_file(upload_path)
                    logger.debug("Copied %s into session workspace", upload_path.name)
                except Exception as exc:
                    logger.warning("Could not copy %s into workspace: %s", upload_path.name, exc)

        if not doc_ids:
            return [], "No documents were ingested (files missing or already indexed)."

        rag_providers = self.kernel.resolve_providers_for_agent("rag_worker") or self.ctx.providers
        summary_query = (
            "Summarize the uploaded documents. Provide:\n"
            "1) A 6-bullet executive summary\n"
            "2) Key definitions / terminology\n"
            "3) Important numbers / constraints (if any)\n"
            "4) Open questions / ambiguities\n"
            "5) 5 suggested questions the user can ask next\n"
            "Cite evidence inline using (citation_id)."
        )
        rag_kwargs = {
            "session": session,
            "query": summary_query,
            "conversation_context": "User uploaded documents.",
            "preferred_doc_ids": doc_ids,
            "providers": rag_providers,
            "callbacks": callbacks,
        }
        if progress_sink is not None:
            rag_kwargs["progress_sink"] = progress_sink
        rag_out = self._call_rag_direct(
            **rag_kwargs,
        )
        rendered = render_rag_contract(rag_out)
        session.messages.append(AIMessage(content=rendered))
        return doc_ids, rendered

    def _call_rag_direct(
        self,
        *,
        session: Any,
        query: str,
        conversation_context: str,
        preferred_doc_ids: List[str],
        providers: ProviderBundle,
        callbacks: Optional[List[Any]] = None,
        progress_sink: Any | None = None,
    ) -> Dict[str, Any]:
        if progress_sink is not None and hasattr(progress_sink, "emit_progress"):
            progress_sink.emit_progress(
                "phase_start",
                label="Summarizing uploaded files",
                detail=f"{len(preferred_doc_ids)} document(s)",
                agent="rag_worker",
            )
        rag_kwargs: Dict[str, Any] = {
            "providers": providers,
            "session": session,
            "query": query,
            "conversation_context": conversation_context,
            "preferred_doc_ids": preferred_doc_ids,
            "must_include_uploads": True,
            "top_k_vector": self.ctx.settings.rag_top_k_vector,
            "top_k_keyword": self.ctx.settings.rag_top_k_keyword,
            "max_retries": self.ctx.settings.rag_max_retries,
            "callbacks": callbacks or [],
            "search_mode": "auto",
            "max_search_rounds": max(2, int(getattr(self.ctx.settings, "rag_max_retries", 1)) + 1),
        }
        if progress_sink is not None:
            rag_kwargs["progress_emitter"] = progress_sink
        contract = run_rag_contract(
            self.ctx.settings,
            self.ctx.stores,
            **rag_kwargs,
        )
        if progress_sink is not None and hasattr(progress_sink, "emit_progress"):
            progress_sink.emit_progress(
                "phase_end",
                label="Upload summary ready",
                detail="Grounded summary complete",
                agent="rag_worker",
            )
        return contract.to_dict()

    def process_turn(
        self,
        session: Any,
        *,
        user_text: str,
        upload_paths: Optional[List[Path]] = None,
        force_agent: bool = False,
        requested_agent: str = "",
        extra_callbacks: Optional[List[Any]] = None,
        progress_sink: Any | None = None,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        upload_paths = upload_paths or []
        request_metadata = dict(request_metadata or {})
        request_scope_metadata = {
            key: value
            for key, value in request_metadata.items()
            if key in {"collection_id", "upload_collection_id", "kb_collection_id", "requested_kb_collection_id", "user_email"}
        }
        access_summary = self._refresh_session_access(
            session,
            request_metadata=request_scope_metadata,
        )
        explicit_requested_collection_id = str(
            request_metadata.get("requested_kb_collection_id")
            or request_metadata.get("kb_collection_id")
            or request_metadata.get("collection_id")
            or extract_requested_kb_collection_id(user_text)
            or ""
        ).strip()
        if explicit_requested_collection_id:
            self._require_collection_use_access(access_summary, collection_id=explicit_requested_collection_id)
        helper_task_type = normalize_openwebui_helper_task_type(
            request_metadata.get("openwebui_helper_task_type")
        )
        if not helper_task_type and bool(request_metadata.get("openwebui_client")):
            helper_task_type = infer_openwebui_helper_task_type(user_text)
            if helper_task_type:
                request_metadata["openwebui_helper_task_type"] = helper_task_type
        if helper_task_type:
            return self._process_openwebui_helper_turn(
                session,
                user_text=user_text,
                helper_task_type=helper_task_type,
                callbacks=extra_callbacks,
            )
        request_chat_max_output_tokens = _coerce_positive_output_tokens(
            request_metadata.get("chat_max_output_tokens")
        )
        long_output_options = LongOutputOptions.from_metadata(request_metadata.get("long_output"))
        requested_agent_override = self._normalize_requested_agent_override(requested_agent)
        stored_state = None
        if getattr(session, "session_id", ""):
            stored_state = self.kernel.transcript_store.load_session_state(session.session_id)
            if stored_state is not None and str((stored_state.metadata or {}).get("pending_router_feedback_id") or "").strip():
                self.kernel.router_feedback.observe_followup_user_turn(stored_state, user_text=user_text)
                self.kernel._persist_state(stored_state)
        preflight_uploads = list(getattr(stored_state, "uploaded_doc_ids", []) or [])
        for doc_id in list(getattr(session, "uploaded_doc_ids", []) or []):
            if doc_id and doc_id not in preflight_uploads:
                preflight_uploads.append(doc_id)
        raw_request_uploaded_doc_ids = request_metadata.get("uploaded_doc_ids") or []
        if isinstance(raw_request_uploaded_doc_ids, str):
            raw_request_uploaded_doc_ids = [raw_request_uploaded_doc_ids]
        for doc_id in list(raw_request_uploaded_doc_ids or []):
            doc_text = str(doc_id or "").strip()
            if doc_text and doc_text not in preflight_uploads:
                preflight_uploads.append(doc_text)
        session.uploaded_doc_ids = list(preflight_uploads)
        preflight_metadata = {
            **dict(getattr(stored_state, "metadata", {}) or {}),
            **dict(getattr(session, "metadata", {}) or {}),
            "uploaded_doc_ids": list(preflight_uploads),
            "user_email": str(getattr(session, "user_email", "") or ""),
            "auth_provider": str(getattr(session, "auth_provider", "") or ""),
            "principal_id": str(getattr(session, "principal_id", "") or ""),
            "access_summary": dict(access_summary or {}),
        }
        kb_scope = sync_session_kb_collection_state(
            self.ctx.settings,
            self.ctx.stores,
            SimpleNamespace(
                tenant_id=session.tenant_id,
                metadata=dict(preflight_metadata),
            ),
            query=user_text,
            requested_collection_id=str(
                request_metadata.get("requested_kb_collection_id")
                or request_metadata.get("kb_collection_id")
                or ""
            ),
        )
        kb_scope_patch = {
            key: value
            for key, value in dict(kb_scope or {}).items()
            if key != "collections"
        }
        preflight_metadata = {
            **preflight_metadata,
            **kb_scope_patch,
        }
        session.metadata = {
            **dict(getattr(session, "metadata", {}) or {}),
            **kb_scope_patch,
            "access_summary": dict(access_summary or {}),
        }
        effective_capabilities = resolve_effective_capabilities(
            settings=self.ctx.settings,
            stores=self.ctx.stores,
            session=SimpleNamespace(
                tenant_id=session.tenant_id,
                user_id=session.user_id,
                access_summary=dict(access_summary or {}),
                metadata=preflight_metadata,
            ),
            registry=self.kernel.registry,
            access_summary=dict(access_summary or {}),
        )
        preflight_metadata = {
            **preflight_metadata,
            "effective_capabilities": effective_capabilities.to_dict(),
            "permission_mode": effective_capabilities.permission_mode,
            "fast_path_policy": effective_capabilities.fast_path_policy,
        }
        session.metadata = {
            **dict(getattr(session, "metadata", {}) or {}),
            "effective_capabilities": effective_capabilities.to_dict(),
            "permission_mode": effective_capabilities.permission_mode,
            "fast_path_policy": effective_capabilities.fast_path_policy,
        }
        pending_worker_question = dict(preflight_metadata.get("pending_worker_question") or {})
        if pending_worker_question and str(pending_worker_question.get("message_type") or "") == "question_request":
            routed_response = self._answer_pending_worker_question(
                session,
                pending_worker_question=pending_worker_question,
                user_text=user_text,
            )
            if routed_response:
                return routed_response
        preflight_resolved_intent = resolve_turn_intent(user_text, preflight_metadata)
        pending_clarification = dict(preflight_metadata.get("pending_clarification") or {})
        clarification_resume_applied = bool(
            pending_clarification
            and str(preflight_resolved_intent.clarification_response or "").strip()
        )
        routing_user_text = (
            preflight_resolved_intent.effective_user_text
            if clarification_resume_applied
            else user_text
        )
        preflight_scope = decide_retrieval_scope(
            self.ctx.settings,
            SimpleNamespace(metadata=preflight_metadata, uploaded_doc_ids=preflight_uploads),
            query=routing_user_text,
            has_uploads=has_upload_evidence(SimpleNamespace(metadata=preflight_metadata, uploaded_doc_ids=preflight_uploads)),
            kb_available=False,
        )
        if (
            document_source_policy_requires_repository(preflight_metadata)
            and query_requests_upload_scope(routing_user_text)
            and not repository_upload_doc_ids(SimpleNamespace(metadata=preflight_metadata, uploaded_doc_ids=preflight_uploads))
        ):
            text = (
                "I cannot analyze the uploaded document yet because it has not been ingested into the agent document repository. "
                "Please upload the file again or ingest it through the control panel, then rerun the request."
            )
            return self.kernel.persist_manual_assistant_response(
                session,
                text=text,
                agent_name="rag_worker",
                route_metadata={
                    "route": "AGENT",
                    "suggested_agent": "rag_worker",
                    "router_reasons": ["agent_repository_upload_missing"],
                    "document_source_policy": "agent_repository_only",
                    "effective_user_text": routing_user_text[:500],
                },
                message_metadata={
                    "turn_outcome": "upload_ingestion_missing",
                    "document_source_policy": "agent_repository_only",
                },
            )
        self._ensure_workspace(session)
        if preflight_scope.mode not in {"uploads_only", "none"}:
            self._ensure_kb_ready(
                session.tenant_id,
                attempt_sync=bool(getattr(self.ctx.settings, "seed_demo_kb_on_startup", True)),
            )
        if _should_shortcut_to_rag_worker(
            user_text=routing_user_text,
            request_metadata=request_metadata,
            scope_mode=preflight_scope.mode,
            requested_agent_override=requested_agent_override,
            force_agent=force_agent,
            helper_task_type=helper_task_type,
        ):
            route_metadata = {
                "route": "AGENT",
                "router_confidence": 1.0,
                "router_reasons": ["direct_grounded_rag_request"],
                "router_method": "metadata_direct_rag",
                "suggested_agent": "rag_worker",
                "requested_agent_override": "",
                "requested_agent_override_applied": False,
                "has_attachments": bool(upload_paths),
                "uploaded_doc_ids": list(getattr(session, "uploaded_doc_ids", []) or []),
                "tenant_id": session.tenant_id,
                "user_id": session.user_id,
                "conversation_id": session.conversation_id,
                "request_id": session.request_id,
                "effective_user_text": routing_user_text[:500],
                "runtime_diagnostics": runtime_settings_diagnostics(self.ctx.settings),
            }
            return self.kernel.process_agent_turn(
                session,
                user_text=routing_user_text,
                callbacks=extra_callbacks,
                agent_name="rag_worker",
                route_metadata=route_metadata,
                chat_max_output_tokens=request_chat_max_output_tokens,
            )
        registered_live_sink = False
        if progress_sink is not None and getattr(session, "session_id", ""):
            self.kernel.register_live_progress_sink(session.session_id, progress_sink)
            registered_live_sink = True

        try:
            if upload_paths:
                self.ingest_and_summarize_uploads(session, upload_paths, progress_sink=progress_sink)

            route_providers = self.kernel.resolve_base_providers() or self.ctx.providers
            decision = route_turn(
                self.ctx.settings,
                route_providers,
                user_text=routing_user_text,
                has_attachments=bool(upload_paths),
                history_summary=_build_history_summary(
                    messages=list(session.messages or []),
                    metadata={**preflight_metadata, **dict(getattr(session, "metadata", {}) or {})},
                    uploaded_doc_ids=list(getattr(session, "uploaded_doc_ids", []) or []),
                ),
                force_agent=force_agent,
                registry=self.kernel.registry,
                session_id=str(getattr(session, "session_id", "") or ""),
                session_metadata=preflight_metadata,
            )
            decision = _coerce_router_decision(
                decision,
                user_text=routing_user_text,
                session_metadata=preflight_metadata,
            )
            semantic_contract = SemanticRoutingContract.from_value(
                getattr(decision, "semantic_contract", None),
                default_route=decision.route,
                default_confidence=decision.confidence,
                default_reasoning="; ".join(decision.reasons),
                default_suggested_agent=decision.suggested_agent,
            )
            basic_candidate_upgraded = False
            if decision.route == "BASIC" and semantic_contract_requires_agent(semantic_contract):
                basic_candidate_upgraded = True
                upgraded_agent = default_agent_for_semantic_contract(
                    semantic_contract,
                    fallback_suggested_agent=decision.suggested_agent,
                )
                semantic_contract = SemanticRoutingContract.from_value(
                    {
                        **semantic_contract.to_dict(),
                        "route": "AGENT",
                        "suggested_agent": upgraded_agent,
                        "reasoning": semantic_contract.reasoning or "runtime_semantic_gate",
                    },
                    default_route="AGENT",
                    default_confidence=decision.confidence,
                    default_reasoning="runtime_semantic_gate",
                    default_suggested_agent=upgraded_agent,
                )
                decision = _coerce_router_decision(
                    SimpleNamespace(
                        route="AGENT",
                        confidence=decision.confidence,
                        reasons=[*list(decision.reasons or []), "runtime_semantic_gate"],
                        suggested_agent=upgraded_agent,
                        router_method=getattr(decision, "router_method", "deterministic"),
                        router_decision_id=getattr(decision, "router_decision_id", ""),
                        router_evidence={
                            **dict(getattr(decision, "router_evidence", {}) or {}),
                            "basic_candidate_upgraded": True,
                        },
                        semantic_contract=semantic_contract,
                    ),
                    user_text=routing_user_text,
                    session_metadata=preflight_metadata,
                )
            elif decision.route == "AGENT" and not str(decision.suggested_agent or "").strip():
                upgraded_agent = default_agent_for_semantic_contract(semantic_contract)
                decision = _coerce_router_decision(
                    SimpleNamespace(
                        route="AGENT",
                        confidence=decision.confidence,
                        reasons=list(decision.reasons or []),
                        suggested_agent=upgraded_agent,
                        router_method=getattr(decision, "router_method", "deterministic"),
                        router_decision_id=getattr(decision, "router_decision_id", ""),
                        router_evidence=dict(getattr(decision, "router_evidence", {}) or {}),
                        semantic_contract=SemanticRoutingContract.from_value(
                            {
                                **semantic_contract.to_dict(),
                                "route": "AGENT",
                                "suggested_agent": upgraded_agent,
                            },
                            default_route="AGENT",
                            default_confidence=decision.confidence,
                            default_reasoning="; ".join(decision.reasons),
                            default_suggested_agent=upgraded_agent,
                        ),
                    ),
                    user_text=routing_user_text,
                    session_metadata=preflight_metadata,
                )
                semantic_contract = decision.semantic_contract
            semantic_routing_payload = semantic_contract.to_dict()
            semantic_metadata_patch: Dict[str, Any] = {
                "semantic_routing": semantic_routing_payload,
            }
            requested_collection_id = str(semantic_routing_payload.get("requested_collection_id") or "").strip()
            if requested_collection_id:
                self._require_collection_use_access(access_summary, collection_id=requested_collection_id)
                semantic_metadata_patch.update(
                    {
                        "requested_kb_collection_id": requested_collection_id,
                        "kb_collection_id": requested_collection_id,
                        "kb_collection_confirmed": True,
                        "search_collection_ids": [requested_collection_id],
                    }
                )
            preflight_metadata = merge_scope_metadata(
                self.ctx.settings,
                {
                    **preflight_metadata,
                    **semantic_metadata_patch,
                },
            )
            session.metadata = merge_scope_metadata(
                self.ctx.settings,
                {
                    **dict(getattr(session, "metadata", {}) or {}),
                    **preflight_metadata,
                    **semantic_metadata_patch,
                },
            )
            router_evidence = {
                **dict(getattr(decision, "router_evidence", {}) or {}),
                "semantic_routing": semantic_routing_payload,
            }
            if basic_candidate_upgraded:
                router_evidence["basic_candidate_upgraded"] = True

            meta = {
                "route": decision.route,
                "router_confidence": decision.confidence,
                "router_reasons": decision.reasons,
                "router_method": getattr(decision, "router_method", "deterministic"),
                "suggested_agent": getattr(decision, "suggested_agent", ""),
                "has_attachments": bool(upload_paths),
                "uploaded_doc_ids": list(getattr(session, "uploaded_doc_ids", []) or []),
                "tenant_id": session.tenant_id,
                "user_id": session.user_id,
                "conversation_id": session.conversation_id,
                "request_id": session.request_id,
                "requested_agent_override": requested_agent_override,
                "requested_agent_override_applied": False,
                "long_output_requested": bool(long_output_options.enabled),
                "semantic_routing": semantic_routing_payload,
                "basic_candidate_upgraded": basic_candidate_upgraded,
                "clarification_resume_applied": clarification_resume_applied,
                "clarification_response": str(preflight_resolved_intent.clarification_response or "").strip(),
                "effective_user_text": routing_user_text[:500],
                "runtime_diagnostics": runtime_settings_diagnostics(self.ctx.settings),
            }
            decision_record = self.kernel.router_feedback.register_decision(
                session,
                user_text=routing_user_text,
                route=decision.route,
                confidence=decision.confidence,
                reasons=list(decision.reasons),
                router_method=getattr(decision, "router_method", "deterministic"),
                suggested_agent=getattr(decision, "suggested_agent", ""),
                force_agent=force_agent,
                has_attachments=bool(upload_paths),
                requested_agent_override=requested_agent_override,
                requested_agent_override_applied=False,
                router_decision_id=str(getattr(decision, "router_decision_id", "") or ""),
                router_evidence=router_evidence,
            )
            meta["router_decision_id"] = decision_record.router_decision_id
            meta["router_evidence"] = dict(decision_record.router_evidence or {})
            deep_rag_policy = decide_deep_rag_policy(
                self.ctx.settings,
                getattr(route_providers, "judge", None),
                user_text=routing_user_text,
                route=decision.route,
                suggested_agent=str(getattr(decision, "suggested_agent", "") or ""),
                has_attachments=bool(upload_paths),
                research_packet=_build_history_summary(
                    messages=list(session.messages or []),
                    metadata={**preflight_metadata, **dict(getattr(session, "metadata", {}) or {})},
                    uploaded_doc_ids=list(getattr(session, "uploaded_doc_ids", []) or []),
                ),
                session_metadata={**preflight_metadata, **dict(getattr(session, "metadata", {}) or {})},
                request_metadata=request_metadata,
            )
            meta["deep_rag"] = deep_rag_policy.to_dict()
            self.kernel.emit_router_decision(
                session,
                router_decision_id=decision_record.router_decision_id,
                route=decision.route,
                confidence=decision.confidence,
                reasons=list(decision.reasons),
                router_method=getattr(decision, "router_method", "deterministic"),
                suggested_agent=getattr(decision, "suggested_agent", ""),
                force_agent=force_agent,
                has_attachments=bool(upload_paths),
                requested_agent_override=requested_agent_override,
                requested_agent_override_applied=False,
                router_evidence=dict(decision_record.router_evidence or {}),
            )
            if "llm_router_circuit_open" in set(decision.reasons):
                self.kernel._emit(
                    "router_degraded_to_deterministic",
                    str(getattr(session, "session_id", "") or ""),
                    agent_name="router",
                    payload={
                        "conversation_id": session.conversation_id,
                        "router_method": getattr(decision, "router_method", ""),
                        "reasons": list(decision.reasons),
                    },
                )

            selected_agent = requested_agent_override or choose_agent_name(
                self.ctx.settings,
                decision,
                registry=self.kernel.registry,
            ) or "general"
            coordinator_default_applied = False
            scope_kind = str(semantic_routing_payload.get("requested_scope_kind") or "").strip().lower()
            answer_origin = str(semantic_routing_payload.get("answer_origin") or "").strip().lower()
            graph_evidence_required = bool(semantic_routing_payload.get("requires_external_evidence")) or answer_origin in {
                "retrieval",
                "ambiguous",
            }
            inventory_query_type = classify_inventory_query(routing_user_text)
            is_graph_inventory = (
                scope_kind == "graph_indexes"
                and not graph_evidence_required
                and inventory_query_type in {INVENTORY_QUERY_GRAPH_INDEXES, INVENTORY_QUERY_GRAPH_FILE}
            )
            if decision.route == "AGENT" and not requested_agent_override:
                if is_graph_inventory:
                    selected_agent = "general"
                elif scope_kind == "graph_indexes" and graph_evidence_required:
                    selected_agent = "graph_manager"
                elif _should_default_to_coordinator_for_broad_grounded_analysis(
                    routing_user_text,
                    session_metadata={**preflight_metadata, **dict(getattr(session, "metadata", {}) or {})},
                ):
                    if selected_agent != "coordinator":
                        selected_agent = "coordinator"
                        coordinator_default_applied = True
                elif _should_default_to_coordinator_for_capability_workflow(
                    routing_user_text,
                    session_metadata={
                        **preflight_metadata,
                        **dict(getattr(session, "metadata", {}) or {}),
                        "has_uploads": bool(getattr(session, "uploaded_doc_ids", []) or upload_paths),
                    },
                ) and effective_capabilities.allows_agent("coordinator"):
                    if selected_agent != "coordinator":
                        selected_agent = "coordinator"
                        coordinator_default_applied = True
                elif (
                    not str(getattr(decision, "suggested_agent", "") or "").strip()
                    and str(deep_rag_policy.preferred_agent or "").strip()
                ):
                    selected_agent = str(deep_rag_policy.preferred_agent).strip()
                task_decomposition = decide_task_decomposition(
                    routing_user_text,
                    current_agent=selected_agent,
                    route=decision.route,
                    suggested_agent=str(getattr(decision, "suggested_agent", "") or ""),
                    session_metadata={**preflight_metadata, **dict(getattr(session, "metadata", {}) or {})},
                    explicit_override=bool(requested_agent_override)
                    or bool(getattr(self.ctx.settings, "enable_coordinator_mode", False)),
                )
                if task_decomposition.applied and task_decomposition.selected_agent:
                    selected_agent = task_decomposition.selected_agent
                    meta["task_decomposition_applied"] = True
                else:
                    meta["task_decomposition_applied"] = False
                if task_decomposition.is_mixed_intent or task_decomposition.applied:
                    payload = task_decomposition.to_dict()
                    meta["task_decomposition"] = payload
                    self.kernel._emit(
                        "decomposition_decision",
                        str(getattr(session, "session_id", "") or ""),
                        agent_name=selected_agent,
                        payload={
                            "conversation_id": session.conversation_id,
                            **payload,
                        },
                    )
            if not effective_capabilities.allows_agent(selected_agent):
                if selected_agent != "coordinator" and effective_capabilities.allows_agent("coordinator"):
                    selected_agent = "coordinator"
                    coordinator_default_applied = True
                else:
                    return self.kernel.persist_manual_assistant_response(
                        session,
                        text=(
                            f"The selected agent `{selected_agent}` is disabled by your current capability profile. "
                            "Turn it back on or enable the coordinator to run this request."
                        ),
                        agent_name="router",
                        route_metadata={
                            **meta,
                            "capability_blocked_agent": selected_agent,
                            "effective_capabilities": effective_capabilities.to_dict(),
                        },
                        message_metadata={"turn_outcome": "capability_blocked_agent"},
                    )
            if decision.route == "BASIC" and not requested_agent_override:
                selected_agent = "basic"
            meta["requested_agent_override_applied"] = bool(requested_agent_override)
            meta["coordinator_default_applied"] = coordinator_default_applied
            meta["effective_capabilities"] = effective_capabilities.to_dict()

            if long_output_options.enabled:
                try:
                    return self._process_long_output_turn(
                        session,
                        user_text=user_text,
                        selected_agent=selected_agent,
                        route_metadata=meta,
                        callbacks=extra_callbacks,
                        progress_sink=progress_sink,
                        options=long_output_options,
                        chat_max_output_tokens=request_chat_max_output_tokens,
                    )
                except CircuitBreakerOpenError:
                    degraded_meta = {
                        **meta,
                        "degraded_service": True,
                        "degraded_reason": "llm_circuit_open",
                        "degraded_from_agent_name": selected_agent,
                    }
                    self.kernel._emit(
                        "degraded_response_returned",
                        str(getattr(session, "session_id", "") or ""),
                        agent_name=selected_agent,
                        payload={
                            "conversation_id": session.conversation_id,
                            "route": decision.route,
                            "reason": "llm_circuit_open",
                            "from_agent": selected_agent,
                        },
                    )
                    return self.kernel.persist_manual_assistant_response(
                        session,
                        text=_degraded_service_text(),
                        agent_name=selected_agent,
                        route_metadata=degraded_meta,
                    )

            if decision.route == "BASIC":
                basic_providers = (
                    self.kernel.resolve_providers_for_agent(
                        "basic",
                        chat_max_output_tokens=request_chat_max_output_tokens,
                    )
                    or self.kernel.resolve_base_providers()
                    or self.ctx.providers
                )
                try:
                    text = self.kernel.process_basic_turn(
                        session,
                        user_text=user_text,
                        system_prompt=self._basic_chat_system_prompt,
                        chat_llm=basic_providers.chat,
                        route_metadata=meta,
                        callbacks=extra_callbacks,
                    )
                except CircuitBreakerOpenError:
                    degraded_meta = {
                        **meta,
                        "degraded_service": True,
                        "degraded_reason": "llm_circuit_open",
                    }
                    self.kernel._emit(
                        "degraded_response_returned",
                        str(getattr(session, "session_id", "") or ""),
                        agent_name="basic",
                        payload={
                            "conversation_id": session.conversation_id,
                            "route": "BASIC",
                            "reason": "llm_circuit_open",
                        },
                    )
                    text = self.kernel.persist_manual_assistant_response(
                        session,
                        text=_degraded_service_text(),
                        agent_name="basic",
                        route_metadata=degraded_meta,
                    )
                if getattr(self.ctx.settings, "clear_scratchpad_per_turn", False):
                    session.clear_scratchpad()
                return text

            override_applied = bool(requested_agent_override)
            meta["requested_agent_override_applied"] = override_applied
            requested_providers = (
                self.kernel.resolve_providers_for_agent(
                    selected_agent,
                    chat_max_output_tokens=request_chat_max_output_tokens,
                )
                or self.kernel.resolve_base_providers()
                or self.ctx.providers
            )
            try:
                try:
                    text = self.kernel.process_agent_turn(
                        session,
                        user_text=user_text,
                        callbacks=extra_callbacks,
                        agent_name=selected_agent,
                        route_metadata=meta,
                        chat_max_output_tokens=request_chat_max_output_tokens,
                    )
                except CircuitBreakerOpenError:
                    basic_providers = (
                        self.kernel.resolve_providers_for_agent(
                            "basic",
                            chat_max_output_tokens=request_chat_max_output_tokens,
                        )
                        or self.kernel.resolve_base_providers()
                        or self.ctx.providers
                    )
                    degraded_meta = {
                        **meta,
                        "degraded_from_agent": True,
                        "degraded_from_agent_name": selected_agent,
                        "degraded_reason": "llm_circuit_open",
                    }
                    can_downgrade = bool(basic_providers) and (
                        self.kernel.bundle_role_identity(requested_providers, "chat")
                        != self.kernel.bundle_role_identity(basic_providers, "chat")
                        or not self.kernel.is_bundle_role_open(basic_providers, "chat")
                    )
                    if can_downgrade:
                        self.kernel._emit(
                            "agent_downgraded_to_basic",
                            str(getattr(session, "session_id", "") or ""),
                            agent_name=selected_agent,
                            payload={
                                "conversation_id": session.conversation_id,
                                "from_agent": selected_agent,
                                "to_agent": "basic",
                                "reason": "llm_circuit_open",
                            },
                        )
                        try:
                            text = self.kernel.process_basic_turn(
                                session,
                                user_text=user_text,
                                system_prompt=self._basic_chat_system_prompt,
                                chat_llm=basic_providers.chat,
                                route_metadata=degraded_meta,
                                callbacks=extra_callbacks,
                                user_already_recorded=True,
                            )
                        except CircuitBreakerOpenError:
                            self.kernel._emit(
                                "degraded_response_returned",
                                str(getattr(session, "session_id", "") or ""),
                                agent_name="basic",
                                payload={
                                    "conversation_id": session.conversation_id,
                                    "route": "AGENT",
                                    "reason": "llm_circuit_open",
                                    "from_agent": selected_agent,
                                },
                            )
                            text = self.kernel.persist_manual_assistant_response(
                                session,
                                text=_degraded_service_text(),
                                agent_name="basic",
                                route_metadata=degraded_meta,
                            )
                    else:
                        self.kernel._emit(
                            "degraded_response_returned",
                            str(getattr(session, "session_id", "") or ""),
                            agent_name=selected_agent,
                            payload={
                                "conversation_id": session.conversation_id,
                                "route": "AGENT",
                                "reason": "llm_circuit_open",
                                "from_agent": selected_agent,
                            },
                        )
                        text = self.kernel.persist_manual_assistant_response(
                            session,
                            text=_degraded_service_text(),
                            agent_name=selected_agent,
                            route_metadata=degraded_meta,
                        )
            finally:
                if getattr(self.ctx.settings, "clear_scratchpad_per_turn", False):
                    session.clear_scratchpad()
            return text
        finally:
            if registered_live_sink:
                self.kernel.unregister_live_progress_sink(session.session_id, progress_sink)

    def _process_long_output_turn(
        self,
        session: Any,
        *,
        user_text: str,
        selected_agent: str,
        route_metadata: Dict[str, Any],
        callbacks: Optional[List[Any]],
        progress_sink: Any | None,
        options: LongOutputOptions,
        chat_max_output_tokens: int | None = None,
    ) -> str:
        state = self.kernel.hydrate_session_state(session)
        state.metadata["route_context"] = dict(route_metadata or {})
        state.append_message("user", user_text)
        self.kernel._persist_state(state)
        self.kernel.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": state.messages[-1].to_dict()},
        )
        agent = self.kernel._resolve_agent(selected_agent)
        state.active_agent = agent.name
        self.kernel._persist_state(state)
        runtime_callbacks = self.kernel.build_callbacks(
            state,
            trace_name="long_output_turn",
            agent_name=agent.name,
            metadata={
                **dict(route_metadata or {}),
                "requested_agent": agent.name,
                "long_output": options.to_dict(),
            },
            base_callbacks=callbacks,
        )
        self.kernel._emit(
            "turn_accepted",
            state.session_id,
            agent_name=agent.name,
            payload={"user_text": user_text[:500]},
        )
        self.kernel._emit(
            "agent_run_started",
            state.session_id,
            agent_name=agent.name,
            payload={"mode": f"{agent.mode}_long_output"},
        )
        self.kernel._emit(
            "agent_turn_started",
            state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": state.conversation_id,
                "user_text": user_text[:500],
                **dict(route_metadata or {}),
            },
        )

        if options.should_run_in_background() and agent.allow_background_jobs:
            job = self.kernel.job_manager.create_job(
                agent_name=agent.name,
                prompt=user_text,
                session_id=state.session_id,
                description=f"Long-form draft for: {user_text[:120]}",
                tenant_id=state.tenant_id,
                user_id=state.user_id,
                priority="background",
                queue_class="background",
                session_state=state.to_dict(),
                metadata={
                    "session_state": state.to_dict(),
                    "route_context": dict(route_metadata or {}),
                    "long_output": {
                        **options.to_dict(),
                        "agent_name": agent.name,
                        "title": f"Long-form draft for {agent.name}",
                        **(
                            {"chat_max_output_tokens": chat_max_output_tokens}
                            if chat_max_output_tokens is not None
                            else {}
                        ),
                    },
                },
            )
            self.kernel.job_manager.start_background_job(job, self.kernel._job_runner)
            if progress_sink is not None and hasattr(progress_sink, "emit_progress"):
                progress_sink.emit_progress(
                    "phase_end",
                    label="Long-form job queued",
                    detail=job.job_id,
                    agent=agent.name,
                    job_id=job.job_id,
                    status="queued",
                )
            text = (
                "I started a background long-form generation job. "
                f"Job ID: {job.job_id}. I’ll save the full draft as a downloadable artifact when it finishes."
            )
            state.append_message(
                "assistant",
                text,
                metadata={
                    "agent_name": agent.name,
                    "job_id": job.job_id,
                    "long_output": {
                        "background": True,
                        "status": "queued",
                        "delivery_mode": options.delivery_mode,
                    },
                },
            )
            self.kernel._persist_state(state)
            self.kernel.transcript_store.append_session_transcript(
                state.session_id,
                {"kind": "message", "message": state.messages[-1].to_dict()},
            )
            self.kernel._emit(
                "agent_run_completed",
                state.session_id,
                agent_name=agent.name,
                payload={"background": True, "job_id": job.job_id, "agent_name": agent.name},
            )
            self.kernel._emit(
                "agent_turn_completed",
                state.session_id,
                agent_name=agent.name,
                payload={
                    "conversation_id": state.conversation_id,
                    "job_id": job.job_id,
                    "background": True,
                    **dict(route_metadata or {}),
                },
            )
            self.kernel._emit(
                "turn_completed",
                state.session_id,
                agent_name=agent.name,
                payload={"assistant_message_id": state.messages[-1].message_id if state.messages else ""},
            )
            self.kernel._run_post_turn_memory_maintenance(state, latest_text=user_text)
            state.sync_to_session(session)
            return text

        providers = (
            self.kernel.resolve_providers_for_agent(
                agent.name,
                chat_max_output_tokens=chat_max_output_tokens,
            )
            or self.kernel.resolve_base_providers()
            or self.ctx.providers
        )
        if providers is None or getattr(providers, "chat", None) is None:
            raise RuntimeError("Long-form generation requires configured providers.")
        composer = LongOutputComposer(
            settings=self.ctx.settings,
            chat_llm=providers.chat,
            agent=agent,
            system_prompt=self.kernel.build_agent_system_prompt(agent, state),
            session_or_state=state,
            callbacks=runtime_callbacks,
            progress_sink=progress_sink,
        )
        result = composer.compose(user_text=user_text, options=options)
        state.append_message(
            "assistant",
            result.summary_text,
            metadata={
                "agent_name": agent.name,
                "artifacts": [dict(result.artifact)],
                **result.to_metadata(),
            },
            artifact_refs=[str(result.artifact.get("artifact_ref") or "")],
        )
        self.kernel._persist_state(state)
        self.kernel.transcript_store.append_session_transcript(
            state.session_id,
            {"kind": "message", "message": state.messages[-1].to_dict()},
        )
        self.kernel._emit(
            "agent_run_completed",
            state.session_id,
            agent_name=agent.name,
            payload={
                "agent_name": agent.name,
                "artifacts": [dict(result.artifact)],
                "long_output": result.to_metadata().get("long_output", {}),
            },
        )
        self.kernel._emit(
            "agent_turn_completed",
            state.session_id,
            agent_name=agent.name,
            payload={
                "conversation_id": state.conversation_id,
                **dict(route_metadata or {}),
                "artifacts": [dict(result.artifact)],
                "long_output": result.to_metadata().get("long_output", {}),
            },
        )
        self.kernel._emit(
            "turn_completed",
            state.session_id,
            agent_name=agent.name,
            payload={"assistant_message_id": state.messages[-1].message_id if state.messages else ""},
        )
        self.kernel._run_post_turn_memory_maintenance(state, latest_text=user_text)
        state.sync_to_session(session)
        return result.summary_text

    @classmethod
    def from_settings(cls, settings: Any, providers: Optional[ProviderBundle] = None) -> "RuntimeService":
        from agentic_chatbot_next.providers.factory import build_providers

        resolved_providers = providers or build_providers(settings)
        return cls.create(settings, resolved_providers)

    @classmethod
    def create_local_session(cls, settings: Any, *, conversation_id: Optional[str] = None) -> Any:
        from agentic_chatbot_next.session import ChatSession

        ctx = build_local_context(settings, conversation_id=conversation_id)
        return ChatSession.from_context(ctx)
