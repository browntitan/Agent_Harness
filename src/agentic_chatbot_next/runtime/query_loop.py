from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_chatbot_next.utils.json_utils import extract_json
from agentic_chatbot_next.basic_chat import run_basic_chat
from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import RuntimeMessage, SessionState
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.general_agent import run_general_agent
from agentic_chatbot_next.memory.context_builder import MemoryContextBuilder
from agentic_chatbot_next.memory.manager import MemoryCandidateRetriever, MemorySelector, MemoryWriteManager
from agentic_chatbot_next.memory.extractor import MemoryExtractor
from agentic_chatbot_next.memory.file_store import FileMemoryStore
from agentic_chatbot_next.rag.doc_targets import resolve_indexed_docs as resolve_named_indexed_docs
from agentic_chatbot_next.rag.hints import coerce_controller_hints
from agentic_chatbot_next.rag.adaptive import run_retrieval_controller
from agentic_chatbot_next.rag.engine import render_rag_contract, run_rag_contract
from agentic_chatbot_next.rag.fanout import RagSearchTask, serialize_document, serialize_graded_chunk
from agentic_chatbot_next.rag.retrieval_scope import (
    repository_upload_doc_ids,
    resolve_search_collection_ids,
)
from agentic_chatbot_next.rag.skill_policy import resolve_rag_execution_hints
from agentic_chatbot_next.runtime.artifacts import pop_pending_artifacts
from agentic_chatbot_next.runtime.clarification import (
    ClarificationRequest,
    build_clarification_policy_text,
    clarification_turn_metadata,
    contract_clarification_request,
    parse_clarification_request,
    pending_clarification_prompt_block,
)
from agentic_chatbot_next.runtime.deep_rag import (
    deep_rag_controller_hints,
    deep_rag_search_mode,
)
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.context_budget import (
    BudgetedTurn,
    ContextBudgetManager,
    ContextSection,
)
from agentic_chatbot_next.runtime.doc_focus import (
    active_doc_focus_controller_hints,
    active_doc_focus_doc_ids,
    active_doc_focus_prompt_block,
    build_doc_focus_result,
)
from agentic_chatbot_next.runtime.research_packet import build_research_packet
from agentic_chatbot_next.runtime.task_plan import normalise_task_plan
from agentic_chatbot_next.runtime.turn_contracts import (
    filter_context_messages,
    resolved_turn_intent_prompt_block,
)
from agentic_chatbot_next.prompt_fallbacks import compose_fallback_prompt

logger = logging.getLogger(__name__)


def _recent_conversation_context(session: SessionState, limit: int = 6) -> str:
    parts: List[str] = []
    for message in reversed(filter_context_messages(session.messages)):
        if message.role in {"user", "assistant"} and message.content.strip():
            parts.append(f"{message.role}: {message.content[:300]}")
        if len(parts) >= limit:
            break
    return "\n".join(reversed(parts))


def _conversation_history_messages(session: SessionState) -> list[Any]:
    return [message.to_langchain() for message in filter_context_messages(session.messages[:-1])]


def _conversation_history_runtime_messages(session: SessionState) -> list[RuntimeMessage]:
    return list(filter_context_messages(session.messages[:-1]))


def _doc_focus_from_documents(documents: List[Any], limit: int = 6) -> List[Dict[str, str]]:
    seen: set[str] = set()
    items: List[Dict[str, str]] = []
    for doc in documents or []:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        doc_id = str(metadata.get("doc_id") or "")
        key = doc_id or str(metadata.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "doc_id": doc_id,
                "title": str(metadata.get("title") or ""),
                "source_path": str(metadata.get("source_path") or ""),
                "source_type": str(metadata.get("source_type") or ""),
            }
        )
        if len(items) >= limit:
            break
    return items


def _tenant_id(settings: Any, session: SessionState) -> str:
    return str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")


def _get_document_by_id(doc_store: Any, *, doc_id: str, tenant_id: str) -> Any | None:
    if doc_store is None or not hasattr(doc_store, "get_document"):
        return None
    try:
        return doc_store.get_document(doc_id, tenant_id)
    except TypeError:
        try:
            return doc_store.get_document(doc_id=doc_id, tenant_id=tenant_id)
        except Exception:
            return None
    except Exception:
        return None


def _resolve_worker_doc_scope(
    *,
    settings: Any | None,
    stores: Any | None,
    session: SessionState,
    doc_scope: List[str],
) -> List[str]:
    candidates = [str(item).strip() for item in (doc_scope or []) if str(item).strip()]
    if not candidates or settings is None or stores is None:
        return []

    tenant_id = _tenant_id(settings, session)
    doc_store = getattr(stores, "doc_store", None)
    resolved_doc_ids: List[str] = []
    unresolved_names: List[str] = []
    seen: set[str] = set()
    uploaded_doc_ids = set(repository_upload_doc_ids(session))

    for candidate in candidates:
        if candidate in uploaded_doc_ids:
            if candidate not in seen:
                seen.add(candidate)
                resolved_doc_ids.append(candidate)
            continue
        record = _get_document_by_id(doc_store, doc_id=candidate, tenant_id=tenant_id)
        if record is not None:
            if candidate not in seen:
                seen.add(candidate)
                resolved_doc_ids.append(candidate)
            continue
        unresolved_names.append(candidate)

    if unresolved_names:
        try:
            resolution = resolve_named_indexed_docs(
                stores,
                settings=settings,
                tenant_id=tenant_id,
                names=unresolved_names,
                collection_ids=resolve_search_collection_ids(settings, session),
            )
        except Exception:
            resolution = None
        for item in list(getattr(resolution, "resolved", ()) or ()):
            doc_id = str(getattr(item, "doc_id", "") or "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                resolved_doc_ids.append(doc_id)

    return resolved_doc_ids


@dataclass
class QueryLoopResult:
    text: str
    messages: List[RuntimeMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryLoop:
    def __init__(
        self,
        *,
        settings: Any | None = None,
        providers: Any | None = None,
        stores: Any | None = None,
        skill_runtime: Any | None = None,
        context_budget_manager: ContextBudgetManager | None = None,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.stores = stores
        self.skill_runtime = skill_runtime
        self.context_budget_manager = context_budget_manager or ContextBudgetManager(settings)
        self._last_context_ledger: Dict[str, Any] = {}
        self._paths = RuntimePaths.from_settings(settings) if settings is not None else None
        memory_enabled = bool(getattr(settings, "memory_enabled", True))
        self._legacy_memory_store = FileMemoryStore(self._paths) if self._paths is not None and memory_enabled else None
        self._managed_memory_store = getattr(stores, "memory_store", None) if memory_enabled and stores is not None else None
        self._memory_selector = MemorySelector(self._managed_memory_store, settings) if self._managed_memory_store is not None else None
        self._memory_candidates = (
            MemoryCandidateRetriever(self._managed_memory_store, settings) if self._managed_memory_store is not None else None
        )
        self._memory_write_manager = (
            MemoryWriteManager(
                self._managed_memory_store,
                settings,
                selector=self._memory_selector,
            )
            if self._managed_memory_store is not None and self._memory_selector is not None
            else None
        )
        if self._managed_memory_store is not None or self._legacy_memory_store is not None:
            self._memory_context = MemoryContextBuilder(
                self._managed_memory_store or self._legacy_memory_store,
                fallback_store=self._legacy_memory_store,
                candidate_retriever=self._memory_candidates,
                selector=self._memory_selector,
                settings=settings,
            )
        else:
            self._memory_context = None
        self._memory_extractor = MemoryExtractor(self._legacy_memory_store) if self._legacy_memory_store is not None else None

    def run(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        providers: Any | None = None,
        tool_context: Any | None = None,
        tools: Optional[List[Any]] = None,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> QueryLoopResult:
        callbacks = list(getattr(tool_context, "callbacks", []) or [])
        if tool_context is not None:
            self.context_budget_manager.bind_runtime(
                transcript_store=getattr(tool_context, "transcript_store", None),
                event_sink=getattr(tool_context, "event_sink", None),
            )
        active_providers = providers or self.providers
        if agent.mode != "memory_maintainer":
            if active_providers is None or getattr(active_providers, "chat", None) is None:
                raise RuntimeError("QueryLoop requires configured providers for live execution.")

        skill_context = ""
        skill_resolution = None
        if tool_context is not None and self.skill_runtime is not None:
            skill_resolution = self.skill_runtime.resolve_context(
                agent,
                session_state,
                user_text=user_text,
                task_payload=task_payload,
            )
            skill_context = skill_resolution.text
            tool_context.skill_context = skill_context
            tool_context.skill_resolution = skill_resolution
            if getattr(skill_resolution, "warnings", None) and getattr(tool_context, "event_sink", None) is not None:
                tool_context.event_sink.emit(
                    RuntimeEvent(
                        event_type="skill_resolution_warning",
                        session_id=session_state.session_id,
                        agent_name=agent.name,
                        payload={
                            "conversation_id": session_state.conversation_id,
                            "warnings": list(skill_resolution.warnings),
                            "resolved_skill_families": list(getattr(skill_resolution, "resolved_skill_families", []) or []),
                        },
                    )
                )

        result: QueryLoopResult
        if agent.mode == "basic":
            result = self._run_basic(
                agent,
                session_state,
                user_text=user_text,
                skill_context=skill_context,
                callbacks=callbacks,
                providers=active_providers,
                tool_context=tool_context,
            )
        elif agent.mode == "rag":
            result = self._run_rag(
                agent,
                session_state,
                user_text=user_text,
                skill_context=skill_context,
                callbacks=callbacks,
                providers=active_providers,
                tool_context=tool_context,
                task_payload=dict(task_payload or {}),
            )
        elif agent.mode == "memory_maintainer":
            result = self._run_memory_maintainer(
                agent,
                session_state,
                user_text=user_text,
                providers=active_providers,
            )
        elif agent.mode == "planner":
            result = self._run_planner(
                agent,
                session_state,
                user_text=user_text,
                skill_context=skill_context,
                callbacks=callbacks,
                providers=active_providers,
            )
        elif agent.mode == "finalizer":
            result = self._run_finalizer(
                agent,
                session_state,
                user_text=user_text,
                skill_context=skill_context,
                task_payload=dict(task_payload or {}),
                callbacks=callbacks,
                providers=active_providers,
            )
        elif agent.mode == "verifier":
            result = self._run_verifier(
                agent,
                session_state,
                user_text=user_text,
                skill_context=skill_context,
                task_payload=dict(task_payload or {}),
                callbacks=callbacks,
                providers=active_providers,
            )
        else:
            result = self._run_react(
                agent,
                session_state,
                user_text=user_text,
                skill_context=skill_context,
                providers=active_providers,
                tool_context=tool_context,
                tools=list(tools or []),
            )
        if skill_resolution is not None:
            result.metadata = {
                "skill_resolution": skill_resolution.to_dict(),
                **dict(result.metadata or {}),
            }
        return result

    def _build_task_context(self, task_payload: Dict[str, Any] | None) -> str:
        payload = dict(task_payload or {})
        worker_request = dict(payload.get("worker_request") or {})
        if not worker_request:
            return ""
        lines: List[str] = []
        task_id = str(worker_request.get("task_id") or "").strip()
        title = str(worker_request.get("title") or "").strip()
        description = str(worker_request.get("description") or "").strip()
        if task_id:
            lines.append(f"task_id: {task_id}")
        if title:
            lines.append(f"title: {title}")
        if description:
            lines.append(f"description: {description}")
        instruction_prompt = str(worker_request.get("instruction_prompt") or "").strip()
        semantic_query = str(worker_request.get("semantic_query") or "").strip()
        context_summary = str(worker_request.get("context_summary") or "").strip()
        if semantic_query:
            lines.append(f"semantic_query: {semantic_query}")
        if context_summary:
            lines.append(f"context_summary: {context_summary[:400]}")
        if instruction_prompt and instruction_prompt != str(worker_request.get("prompt") or "").strip():
            lines.append(f"instruction_prompt: {instruction_prompt[:400]}")
        doc_scope = [str(item).strip() for item in (worker_request.get("doc_scope") or []) if str(item).strip()]
        if doc_scope:
            lines.append("doc_scope: " + ", ".join(doc_scope))
        artifact_refs = [str(item).strip() for item in (worker_request.get("artifact_refs") or []) if str(item).strip()]
        if artifact_refs:
            lines.append("artifact_refs: " + ", ".join(artifact_refs))
        handoff_artifacts = [
            dict(item)
            for item in (worker_request.get("metadata") or {}).get("handoff_artifacts", [])
            if isinstance(item, dict)
        ]
        if handoff_artifacts:
            lines.append(
                "handoff_artifacts: "
                + json.dumps(
                    [
                        {
                            "artifact_type": str(item.get("artifact_type") or ""),
                            "handoff_schema": str(item.get("handoff_schema") or ""),
                            "summary": str(item.get("summary") or ""),
                        }
                        for item in handoff_artifacts
                    ],
                    ensure_ascii=False,
                )
            )
        research_profile = str(worker_request.get("research_profile") or "").strip()
        coverage_goal = str(worker_request.get("coverage_goal") or "").strip()
        result_mode = str(worker_request.get("result_mode") or "").strip()
        answer_mode = str(worker_request.get("answer_mode") or "").strip()
        controller_hints = coerce_controller_hints(worker_request.get("controller_hints") or {})
        if research_profile:
            lines.append(f"research_profile: {research_profile}")
        if coverage_goal:
            lines.append(f"coverage_goal: {coverage_goal}")
        if result_mode:
            lines.append(f"result_mode: {result_mode}")
        if answer_mode:
            lines.append(f"answer_mode: {answer_mode}")
        if controller_hints:
            lines.append("controller_hints: " + json.dumps(controller_hints, ensure_ascii=False))
        return "\n".join(lines).strip()

    def _build_prompt_sections(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str = "",
        providers: Any | None = None,
        skill_context: str = "",
        task_payload: Dict[str, Any] | None = None,
        deferred_tool_summary: Dict[str, Any] | None = None,
    ) -> List[ContextSection]:
        prompt = ""
        if self.skill_runtime is not None:
            prompt = self.skill_runtime.build_prompt(agent).strip()
        if not prompt:
            prompt = compose_fallback_prompt(agent.prompt_file) or f"You are the {agent.name} agent."
        task_context = self._build_task_context(task_payload)
        memory_context = ""
        if self._memory_context is not None:
            memory_context = self._memory_context.build_for_agent(
                agent,
                session_state,
                user_text=user_text,
                providers=providers,
            )
        sections = [ContextSection(name="base_prompt", content=prompt, priority=100, preserve=True)]
        sections.append(
            ContextSection(
                name="clarification_policy",
                title="Clarification Policy",
                content=self._clarification_policy_text(),
                priority=95,
                preserve=True,
            )
        )
        pending_clarification = pending_clarification_prompt_block(dict(session_state.metadata or {}))
        if pending_clarification:
            sections.append(ContextSection(name="pending_clarification", content=pending_clarification, priority=96, preserve=True))
        resolved_intent = resolved_turn_intent_prompt_block(dict(session_state.metadata or {}))
        if resolved_intent:
            sections.append(ContextSection(name="resolved_turn_intent", content=resolved_intent, priority=88))
        active_doc_focus = active_doc_focus_prompt_block(dict(session_state.metadata or {}))
        if active_doc_focus:
            sections.append(ContextSection(name="active_doc_focus", content=active_doc_focus, priority=90, preserve=True))
        if task_context:
            sections.append(ContextSection(name="task_context", title="Task Context", content=task_context, priority=75))
        deferred_tool_block = self._deferred_tool_prompt_block(deferred_tool_summary)
        if deferred_tool_block:
            sections.append(ContextSection(name="deferred_tools", title="Deferred Tool Discovery", content=deferred_tool_block, priority=74))
        if skill_context:
            sections.append(ContextSection(name="skill_context", title="Skill Context", content=skill_context, priority=70))
        if memory_context:
            sections.append(ContextSection(name="memory_context", title="Memory Context", content=memory_context, priority=72))
        return sections

    def _deferred_tool_prompt_block(self, summary: Dict[str, Any] | None) -> str:
        data = dict(summary or {})
        if not data.get("enabled") or int(data.get("count") or 0) <= 0:
            return ""
        groups = ", ".join(str(item) for item in (data.get("groups") or []) if str(item))
        tools = ", ".join(str(item) for item in (data.get("tools") or []) if str(item))
        lines = [
            "Some allowed tools are deferred to keep the initial context small.",
            "Use discover_tools with a natural-language query before calling any deferred capability.",
            "Use call_deferred_tool only for a tool returned by discover_tools in this turn; normal tool policy still applies.",
        ]
        if groups:
            lines.append(f"Deferred groups available: {groups}.")
        if tools:
            lines.append(f"Examples of deferred tools: {tools}.")
        return "\n".join(lines)

    def _prepare_budgeted_turn(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str = "",
        providers: Any | None = None,
        skill_context: str = "",
        task_payload: Dict[str, Any] | None = None,
        history_messages: List[RuntimeMessage] | None = None,
        tool_context: Any | None = None,
    ) -> BudgetedTurn:
        sections = self._build_prompt_sections(
            agent,
            session_state,
            user_text=user_text,
            providers=providers,
            skill_context=skill_context,
            task_payload=task_payload,
            deferred_tool_summary=dict((getattr(tool_context, "metadata", {}) or {}).get("deferred_tool_discovery") or {}),
        )
        budgeted = self.context_budget_manager.prepare_turn(
            agent_name=agent.name,
            session_state=session_state,
            user_text=user_text,
            sections=sections,
            history_messages=list(history_messages or []),
            providers=providers,
            transcript_store=getattr(tool_context, "transcript_store", None),
            event_sink=getattr(tool_context, "event_sink", None),
            job_id=str((getattr(tool_context, "metadata", {}) or {}).get("job_id") or ""),
        )
        self._last_context_ledger = budgeted.ledger.to_dict()
        return budgeted

    def _build_system_prompt(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str = "",
        providers: Any | None = None,
        skill_context: str = "",
        task_payload: Dict[str, Any] | None = None,
    ) -> str:
        budgeted = self._prepare_budgeted_turn(
            agent,
            session_state,
            user_text=user_text,
            providers=providers,
            skill_context=skill_context,
            task_payload=task_payload,
            history_messages=[],
        )
        return budgeted.system_prompt

    def _clarification_policy_text(self) -> str:
        return build_clarification_policy_text(getattr(self.settings, "clarification_sensitivity", 50))

    def _maybe_delegate_rag_peer(
        self,
        *,
        agent: AgentDefinition,
        session_state: SessionState,
        user_text: str,
        providers: Any,
        tool_context: Any | None,
        callbacks: List[Any],
        execution_hints: Any,
        preferred_doc_ids: List[str],
        handoff_artifacts: List[Dict[str, Any]],
        conversation_context: str,
    ) -> Dict[str, Any] | None:
        if tool_context is None or getattr(tool_context, "kernel", None) is None:
            return None
        allowed_agents = [
            str(item).strip()
            for item in getattr(tool_context.active_definition, "allowed_worker_agents", []) or []
            if str(item).strip()
        ]
        if not allowed_agents:
            return None
        judge_model = getattr(providers, "judge", None) or getattr(providers, "chat", None)
        if judge_model is None:
            return None
        prompt = (
            "Decide whether this direct RAG turn should answer now or enqueue one asynchronous peer request.\n"
            "Return JSON only with keys: action, agent_name, description, message, rationale.\n"
            "Set action to 'answer' unless one peer can materially improve the outcome and the work can continue asynchronously.\n"
            f"Allowed peer agents: {', '.join(allowed_agents)}.\n"
            f"USER_REQUEST: {user_text}\n"
            f"RESEARCH_PROFILE: {getattr(execution_hints, 'research_profile', '')}\n"
            f"COVERAGE_GOAL: {getattr(execution_hints, 'coverage_goal', '')}\n"
            f"RESULT_MODE: {getattr(execution_hints, 'result_mode', '')}\n"
            f"PREFERRED_DOC_IDS: {preferred_doc_ids[:8]}\n"
            f"HANDOFF_ARTIFACT_TYPES: {[str(item.get('artifact_type') or '') for item in handoff_artifacts[:8]]}\n"
            f"CONTROLLER_HINTS: {json.dumps(dict(getattr(execution_hints, 'controller_hints', {}) or {}), ensure_ascii=False)}\n"
            f"CONVERSATION_CONTEXT: {conversation_context[:1500]}\n"
        )
        try:
            response = judge_model.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are deciding whether to continue synchronously or queue one bounded peer request. "
                            "Never choose invoke_agent unless a specialist follow-up is clearly better than answering now."
                        )
                    ),
                    HumanMessage(content=prompt),
                ],
                config={"callbacks": callbacks},
            )
            text = getattr(response, "content", None) or str(response)
        except Exception as exc:
            logger.warning("RAG peer-dispatch decision failed: %s", exc)
            return None
        payload = extract_json(text or "") or {}
        if not isinstance(payload, dict):
            return None
        action = str(payload.get("action") or "").strip().lower()
        if action != "invoke_agent":
            return None
        target_agent = str(payload.get("agent_name") or "").strip()
        message = str(payload.get("message") or "").strip()
        description = str(payload.get("description") or "").strip()
        if target_agent not in allowed_agents or not message:
            return None
        dispatch = tool_context.kernel.invoke_agent_from_tool(
            tool_context,
            agent_name=target_agent,
            message=message,
            description=description,
            reuse_running_job=True,
        )
        if isinstance(dispatch, dict) and dispatch.get("error"):
            logger.warning("RAG peer-dispatch enqueue failed: %s", dispatch.get("error"))
            return None
        return {
            **dict(dispatch or {}),
            "agent_name": target_agent,
            "description": description,
            "message": message,
            "rationale": str(payload.get("rationale") or "").strip(),
        }

    @staticmethod
    def _render_rag_peer_dispatch_message(dispatch: Dict[str, Any]) -> str:
        target_agent = str(dispatch.get("agent_name") or dispatch.get("target_agent") or "worker").strip() or "worker"
        description = str(dispatch.get("description") or "").strip()
        if description:
            return (
                f"I queued a background `{target_agent}` follow-up to continue this request: {description}. "
                "Results will arrive through the runtime notification stream."
            )
        return (
            f"I queued a background `{target_agent}` follow-up to continue this request. "
            "Results will arrive through the runtime notification stream."
        )

    def _finalize_assistant_turn(
        self,
        agent: AgentDefinition,
        *,
        text: str,
        metadata: Dict[str, Any] | None = None,
        clarification: ClarificationRequest | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        clean_text = str(text or "").strip()
        parsed_text, parsed_request = parse_clarification_request(
            clean_text,
            source_agent=agent.name,
        )
        final_request = clarification or parsed_request
        final_text = parsed_text if final_request is not None else clean_text
        final_metadata = clarification_turn_metadata(
            final_request,
            agent_name=agent.name,
            extra=metadata,
        )
        return final_text, final_metadata

    def _run_basic(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        callbacks: List[Any],
        providers: Any,
        tool_context: Any | None = None,
    ) -> QueryLoopResult:
        budgeted = self._prepare_budgeted_turn(
            agent,
            session_state,
            user_text=user_text,
            providers=providers,
            skill_context=skill_context,
            history_messages=_conversation_history_runtime_messages(session_state),
            tool_context=tool_context,
        )
        text = run_basic_chat(
            providers.chat,
            messages=[message.to_langchain() for message in budgeted.history_messages],
            user_text=user_text,
            system_prompt=budgeted.system_prompt,
            callbacks=callbacks,
        )
        text, assistant_metadata = self._finalize_assistant_turn(agent, text=text)
        assistant_metadata = {
            **dict(assistant_metadata),
            "context_budget": budgeted.ledger.to_dict(),
        }
        return QueryLoopResult(
            text=text,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=text, metadata=assistant_metadata)],
            metadata=dict(assistant_metadata),
        )

    def _run_react(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        providers: Any,
        tool_context: Any,
        tools: List[Any],
    ) -> QueryLoopResult:
        if tool_context is None:
            raise ValueError("React execution requires a tool context.")
        budgeted = self._prepare_budgeted_turn(
            agent,
            session_state,
            user_text=user_text,
            providers=providers,
            skill_context=skill_context,
            task_payload=dict(getattr(tool_context, "metadata", {}) or {}).get("task_payload") or {},
            history_messages=_conversation_history_runtime_messages(session_state),
            tool_context=tool_context,
        )
        final_text, updated_messages, run_stats = run_general_agent(
            providers.chat,
            tools=tools,
            messages=[message.to_langchain() for message in budgeted.history_messages],
            user_text=user_text,
            system_prompt=budgeted.system_prompt,
            callbacks=tool_context.callbacks,
            max_steps=agent.max_steps,
            max_tool_calls=agent.max_tool_calls,
            max_parallel_tool_calls=getattr(self.settings, "max_parallel_tool_calls", 4),
            force_plan_execute=str(agent.metadata.get("execution_strategy") or "").lower() == "plan_execute",
            context_budget_manager=self.context_budget_manager,
            tool_context=tool_context,
            providers=providers,
        )
        final_text, assistant_metadata = self._finalize_assistant_turn(agent, text=final_text)
        messages = [RuntimeMessage.from_langchain(message) for message in updated_messages]
        tool_context.refresh_from_session_handle()
        pending_artifacts = pop_pending_artifacts(session_state)
        if pending_artifacts:
            artifact_refs = [str(item.get("artifact_ref") or "") for item in pending_artifacts if str(item.get("artifact_ref") or "")]
            assistant_message = None
            for message in reversed(messages):
                if message.role == "assistant":
                    assistant_message = message
                    break
            if assistant_message is None:
                assistant_message = RuntimeMessage(role="assistant", content=final_text)
                messages.append(assistant_message)
            assistant_message.artifact_refs = artifact_refs
            assistant_message.metadata = dict(assistant_message.metadata or {})
            assistant_message.metadata["artifacts"] = pending_artifacts
        assistant_message = None
        for message in reversed(messages):
            if message.role == "assistant":
                assistant_message = message
                break
        if assistant_message is None:
            assistant_message = RuntimeMessage(role="assistant", content=final_text)
            messages.append(assistant_message)
        assistant_message.content = final_text
        assistant_message.metadata = {
            **dict(assistant_message.metadata or {}),
            **assistant_metadata,
            "context_budget": budgeted.ledger.to_dict(),
        }
        return QueryLoopResult(
            text=final_text,
            messages=messages,
            metadata={
                "run_stats": run_stats,
                "artifacts": pending_artifacts,
                "context_budget": budgeted.ledger.to_dict(),
                **dict(assistant_metadata),
            },
        )

    def _run_rag(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        callbacks: List[Any],
        providers: Any,
        tool_context: Any | None = None,
        task_payload: Dict[str, Any] | None = None,
    ) -> QueryLoopResult:
        payload = dict(task_payload or {})
        worker_request = dict(payload.get("worker_request") or {})
        task_metadata = dict(worker_request.get("metadata") or {})
        handoff_artifacts = [
            dict(item)
            for item in (task_metadata.get("handoff_artifacts") or [])
            if isinstance(item, dict)
        ]
        rag_search_task = None
        if isinstance(task_metadata.get("rag_search_task"), dict):
            rag_search_task = RagSearchTask.from_dict(dict(task_metadata.get("rag_search_task") or {}))
        answer_mode = str(
            task_metadata.get("answer_mode")
            or (rag_search_task.answer_mode if rag_search_task is not None else "")
            or "answer"
        ).strip().lower()
        semantic_query = str(
            worker_request.get("semantic_query")
            or task_metadata.get("semantic_query")
            or (rag_search_task.query if rag_search_task is not None else "")
            or user_text
        ).strip() or str(user_text or "").strip()
        instruction_prompt = str(
            worker_request.get("instruction_prompt")
            or task_metadata.get("instruction_prompt")
            or worker_request.get("prompt")
            or user_text
        ).strip() or str(user_text or "").strip()
        skill_queries = [
            str(item).strip()
            for item in (payload.get("skill_queries") or [])
            if str(item).strip()
        ]
        skill_queries.extend(
            str(item).strip()
            for item in (worker_request.get("skill_queries") or [])
            if str(item).strip()
        )
        explicit_controller_hints: Dict[str, Any] = {}
        for raw_hints in (
            payload.get("controller_hints"),
            worker_request.get("controller_hints"),
            task_metadata.get("controller_hints"),
            rag_search_task.controller_hints if rag_search_task is not None else None,
        ):
            explicit_controller_hints.update(coerce_controller_hints(raw_hints))
        active_doc_focus_hints = active_doc_focus_controller_hints(user_text, dict(session_state.metadata or {}))
        if active_doc_focus_hints:
            explicit_controller_hints = {
                **dict(active_doc_focus_hints),
                **dict(explicit_controller_hints),
            }
        handoff_context_parts: List[str] = []
        handoff_doc_ids: List[str] = []
        route_context = dict(session_state.metadata.get("route_context") or {})
        for artifact in handoff_artifacts:
            artifact_type = str(artifact.get("artifact_type") or "")
            artifact_data = dict(artifact.get("data") or {})
            if artifact_type == "analysis_summary":
                summary = str(artifact_data.get("summary") or "").strip()
                if summary:
                    handoff_context_parts.append(f"analysis_summary:\n{summary[:2000]}")
                if artifact_data.get("keywords"):
                    explicit_controller_hints.setdefault("seed_keywords", list(artifact_data.get("keywords") or []))
            elif artifact_type == "entity_candidates":
                entities = [str(item) for item in (artifact_data.get("entities") or []) if str(item)]
                if entities:
                    explicit_controller_hints.setdefault("entity_candidates", entities)
                    handoff_context_parts.append("entity_candidates: " + ", ".join(entities[:12]))
            elif artifact_type == "keyword_windows":
                keywords = [str(item) for item in (artifact_data.get("keywords") or []) if str(item)]
                if keywords:
                    explicit_controller_hints.setdefault("seed_keywords", keywords)
                    handoff_context_parts.append("keyword_windows: " + ", ".join(keywords[:12]))
            elif artifact_type == "title_candidates":
                candidate_titles: List[str] = []
                for doc in artifact_data.get("documents") or []:
                    if not isinstance(doc, dict):
                        continue
                    doc_id = str(doc.get("doc_id") or "")
                    title = str(doc.get("title") or doc_id)
                    if doc_id:
                        handoff_doc_ids.append(doc_id)
                    if title:
                        candidate_titles.append(title)
                query_variants = [str(item).strip() for item in (artifact_data.get("query_variants") or []) if str(item).strip()]
                if query_variants:
                    explicit_controller_hints["seed_keywords"] = [
                        *[
                            str(item).strip()
                            for item in (explicit_controller_hints.get("seed_keywords") or [])
                            if str(item).strip()
                        ],
                        *query_variants,
                    ][:12]
                    handoff_context_parts.append("title_candidate_queries: " + "; ".join(query_variants[:6]))
                if candidate_titles:
                    explicit_controller_hints["prefer_doc_focus"] = True
                    handoff_context_parts.append("title_candidates: " + ", ".join(candidate_titles[:8]))
            elif artifact_type == "doc_focus":
                for doc in artifact_data.get("documents") or []:
                    if not isinstance(doc, dict):
                        continue
                    doc_id = str(doc.get("doc_id") or "")
                    if doc_id:
                        handoff_doc_ids.append(doc_id)
                if handoff_doc_ids:
                    explicit_controller_hints["prefer_doc_focus"] = True
            elif artifact_type == "evidence_request":
                request_query = str(artifact_data.get("query") or "").strip()
                if request_query:
                    handoff_context_parts.append(f"evidence_request:\n{request_query}")
            elif artifact_type == "evidence_response":
                summary = str(artifact_data.get("summary") or "").strip()
                if summary:
                    handoff_context_parts.append(f"evidence_response:\n{summary[:2000]}")
        execution_hints = resolve_rag_execution_hints(
            self.settings,
            self.stores,
            session=session_state,
            pinned_skill_ids=list(agent.preload_skill_packs),
            query=(
                rag_search_task.query if rag_search_task is not None and rag_search_task.query else semantic_query
            )
            + ("\n" + "\n".join(handoff_context_parts) if handoff_context_parts else ""),
            skill_queries=skill_queries,
            research_profile=str(
                payload.get("research_profile")
                or worker_request.get("research_profile")
                or task_metadata.get("research_profile")
                or (rag_search_task.research_profile if rag_search_task is not None else "")
                or ""
            ),
            coverage_goal=str(
                payload.get("coverage_goal")
                or worker_request.get("coverage_goal")
                or task_metadata.get("coverage_goal")
                or (rag_search_task.coverage_goal if rag_search_task is not None else "")
                or ""
            ),
            result_mode=str(
                payload.get("result_mode")
                or worker_request.get("result_mode")
                or task_metadata.get("result_mode")
                or (rag_search_task.result_mode if rag_search_task is not None else "")
                or ""
            ),
            controller_hints=explicit_controller_hints,
        )
        execution_hints.controller_hints = {
            **deep_rag_controller_hints(route_context),
            **dict(execution_hints.controller_hints or {}),
        }
        kb_only_scope = str(execution_hints.controller_hints.get("retrieval_scope_mode") or "").strip().lower() == "kb_only"
        preferred_doc_ids = _resolve_worker_doc_scope(
            settings=self.settings,
            stores=self.stores,
            session=session_state,
            doc_scope=list(rag_search_task.doc_scope) if rag_search_task is not None else [],
        )
        if (
            str(explicit_controller_hints.get("summary_scope") or "").strip().lower() == "active_doc_focus"
            and not preferred_doc_ids
        ):
            preferred_doc_ids = list(active_doc_focus_doc_ids(dict(session_state.metadata or {})))
        for doc_id in handoff_doc_ids:
            if doc_id and doc_id not in preferred_doc_ids:
                preferred_doc_ids.append(doc_id)
        conversation_context = build_research_packet(
            session_state,
            recent_messages=10,
            message_char_limit=320,
            retrieval_limit=4,
        ) or _recent_conversation_context(session_state)
        if self._memory_context is not None:
            memory_context = self._memory_context.build_for_agent(
                agent,
                session_state,
                user_text=semantic_query,
                providers=providers,
            )
            if memory_context:
                conversation_context = "\n\n".join(
                    part
                    for part in [
                        conversation_context,
                        f"managed_memory_context:\n{memory_context}",
                    ]
                    if str(part).strip()
                )
        if handoff_context_parts:
            conversation_context = "\n\n".join(part for part in [conversation_context, *handoff_context_parts] if part.strip())
        conversation_context = self.context_budget_manager.budget_text_block(
            "rag_conversation_context",
            conversation_context,
            max_tokens=max(1200, int(getattr(self.settings, "context_microcompact_target_tokens", 2400) or 2400)),
        )
        progress_emitter = getattr(tool_context, "progress_emitter", None) if tool_context is not None else None
        runtime_bridge = getattr(tool_context, "rag_runtime_bridge", None) if tool_context is not None else None

        if answer_mode == "evidence_only":
            retrieval_kwargs: Dict[str, Any] = {
                "providers": providers,
                "session": session_state,
                "query": rag_search_task.query if rag_search_task is not None and rag_search_task.query else semantic_query,
                "conversation_context": conversation_context,
                "preferred_doc_ids": preferred_doc_ids,
                "must_include_uploads": bool(session_state.uploaded_doc_ids) and not bool(preferred_doc_ids) and not kb_only_scope,
                "top_k_vector": self.settings.rag_top_k_vector,
                "top_k_keyword": self.settings.rag_top_k_keyword,
                "max_retries": self.settings.rag_max_retries,
                "callbacks": callbacks,
                "search_mode": "deep",
                "max_search_rounds": max(1, int(rag_search_task.round_budget if rag_search_task is not None else 1)),
                "allow_internal_fanout": False,
                "research_profile": execution_hints.research_profile,
                "coverage_goal": execution_hints.coverage_goal,
                "result_mode": execution_hints.result_mode,
                "controller_hints": execution_hints.controller_hints,
            }
            if progress_emitter is not None:
                retrieval_kwargs["progress_emitter"] = progress_emitter
            if tool_context is not None and getattr(tool_context, "event_sink", None) is not None:
                retrieval_kwargs["event_sink"] = tool_context.event_sink
            retrieval_run = run_retrieval_controller(
                self.settings,
                self.stores,
                **retrieval_kwargs,
            )
            search_result = {
                "task_id": str(rag_search_task.task_id if rag_search_task is not None else worker_request.get("task_id") or ""),
                "evidence_entries": list((retrieval_run.evidence_ledger or {}).get("entries") or []),
                "candidate_docs": [serialize_document(doc) for doc in retrieval_run.candidate_docs],
                "graded_chunks": [serialize_graded_chunk(item) for item in retrieval_run.graded],
                "warnings": [],
                "doc_focus": _doc_focus_from_documents(retrieval_run.selected_docs or retrieval_run.candidate_docs),
            }
            text = json.dumps(search_result, ensure_ascii=False)
            return QueryLoopResult(
                text=text,
                messages=list(session_state.messages)
                + [RuntimeMessage(role="assistant", content=text, metadata={"agent_name": agent.name, "answer_mode": answer_mode})],
                metadata={"rag_search_result": search_result, "agent_name": agent.name},
            )

        contract_kwargs: Dict[str, Any] = {
            "providers": providers,
            "session": session_state,
            "query": semantic_query,
            "conversation_context": conversation_context,
            "preferred_doc_ids": preferred_doc_ids,
            "must_include_uploads": bool(session_state.uploaded_doc_ids) and not kb_only_scope,
            "top_k_vector": self.settings.rag_top_k_vector,
            "top_k_keyword": self.settings.rag_top_k_keyword,
            "max_retries": self.settings.rag_max_retries,
            "callbacks": callbacks,
            "base_guidance": self.skill_runtime.build_prompt(agent).strip() if self.skill_runtime is not None else "",
            "skill_context": skill_context,
            "task_context": instruction_prompt,
            "search_mode": deep_rag_search_mode(route_context, default="auto"),
            "max_search_rounds": max(1, int(getattr(self.settings, "max_rag_agent_steps", 16) or 16)),
            "research_profile": execution_hints.research_profile,
            "coverage_goal": execution_hints.coverage_goal,
            "result_mode": execution_hints.result_mode,
            "controller_hints": execution_hints.controller_hints,
        }
        if runtime_bridge is not None:
            contract_kwargs["runtime_bridge"] = runtime_bridge
        if progress_emitter is not None:
            contract_kwargs["progress_emitter"] = progress_emitter
        if tool_context is not None and getattr(tool_context, "event_sink", None) is not None:
            contract_kwargs["event_sink"] = tool_context.event_sink
        dispatch = self._maybe_delegate_rag_peer(
            agent=agent,
            session_state=session_state,
            user_text=semantic_query,
            providers=providers,
            tool_context=tool_context,
            callbacks=callbacks,
            execution_hints=execution_hints,
            preferred_doc_ids=preferred_doc_ids,
            handoff_artifacts=handoff_artifacts,
            conversation_context=conversation_context,
        )
        if dispatch is not None:
            text = self._render_rag_peer_dispatch_message(dispatch)
            text, assistant_metadata = self._finalize_assistant_turn(
                agent,
                text=text,
                metadata={
                    "agent_name": agent.name,
                    "peer_dispatch": dispatch,
                    "turn_outcome": "background_delegated",
                },
            )
            return QueryLoopResult(
                text=text,
                messages=list(session_state.messages)
                + [RuntimeMessage(role="assistant", content=text, metadata=assistant_metadata)],
                metadata=dict(assistant_metadata),
            )
        contract = run_rag_contract(
            self.settings,
            self.stores,
            **contract_kwargs,
        )
        clarification = contract_clarification_request(
            answer=contract.answer,
            followups=getattr(contract, "followups", []),
            warnings=getattr(contract, "warnings", []),
            source_agent=agent.name,
        )
        rendered_text = contract.answer if clarification is not None else render_rag_contract(contract)
        text, assistant_metadata = self._finalize_assistant_turn(
            agent,
            text=rendered_text,
            clarification=clarification,
        )
        doc_focus_result = None
        if str(explicit_controller_hints.get("summary_scope") or "").strip().lower() == "active_doc_focus":
            doc_focus_result = build_doc_focus_result(
                collection_id=str(
                    explicit_controller_hints.get("requested_kb_collection_id")
                    or explicit_controller_hints.get("kb_collection_id")
                    or dict(session_state.metadata or {}).get("kb_collection_id")
                    or "default"
                ),
                documents=dict(session_state.metadata or {}).get("active_doc_focus", {}).get("documents") or [],
                source_query=semantic_query,
                result_mode="answer",
            )
        elif execution_hints.result_mode == "inventory" or bool(execution_hints.controller_hints.get("prefer_inventory_output")):
            doc_focus_result = build_doc_focus_result(
                collection_id=str(
                    execution_hints.controller_hints.get("requested_kb_collection_id")
                    or execution_hints.controller_hints.get("kb_collection_id")
                    or dict(session_state.metadata or {}).get("kb_collection_id")
                    or "default"
                ),
                documents=[
                    {
                        "doc_id": str(citation.doc_id or ""),
                        "title": str(citation.title or ""),
                        "source_type": str(citation.source_type or ""),
                        "source_path": "",
                    }
                    for citation in getattr(contract, "citations", []) or []
                ],
                source_query=semantic_query,
                result_mode=execution_hints.result_mode,
            )
        if doc_focus_result is not None:
            assistant_metadata = {
                **dict(assistant_metadata),
                "doc_focus_result": doc_focus_result,
            }
        return QueryLoopResult(
            text=text,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=text, metadata=assistant_metadata)],
            metadata={
                "rag_contract": contract.to_dict(),
                "rag_execution_hints": execution_hints.to_dict(),
                **dict(assistant_metadata),
            },
        )

    def _run_memory_maintainer(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        providers: Any | None = None,
    ) -> QueryLoopResult:
        if self._memory_write_manager is not None:
            managed_result = self._memory_write_manager.process_turn(
                session_state=session_state,
                latest_user_text=user_text,
                providers=providers,
            )
            if not managed_result.shadow:
                saved = managed_result.applied_count
                mode = managed_result.mode or "managed"
                if saved:
                    text = f"Saved {saved} managed memory entr{'y' if saved == 1 else 'ies'} via {mode} mode."
                else:
                    text = "No managed memory entries were validated for persistence."
                text, assistant_metadata = self._finalize_assistant_turn(agent, text=text)
                return QueryLoopResult(
                    text=text,
                    messages=list(session_state.messages)
                    + [RuntimeMessage(role="assistant", content=text, metadata=assistant_metadata)],
                    metadata={
                        "saved_entries": saved,
                        "memory_write_result": {
                            "applied_count": managed_result.applied_count,
                            "skipped_count": managed_result.skipped_count,
                            "shadow": managed_result.shadow,
                            "mode": managed_result.mode,
                            "errors": list(managed_result.errors),
                        },
                        **dict(assistant_metadata),
                    },
                )
        if self._memory_extractor is None:
            raise RuntimeError("Memory maintainer requires a configured memory store.")
        scopes = list(agent.memory_scopes or ["conversation"])
        saved = self._memory_extractor.apply_from_messages(
            session_state,
            session_state.messages[-8:],
            scopes=scopes,
        )
        if not saved:
            saved = self._memory_extractor.apply_from_text(session_state, user_text, scopes=scopes)
        if saved:
            text = f"Saved {saved} memory entr{'y' if saved == 1 else 'ies'} across scopes: {', '.join(scopes)}."
        else:
            text = "No structured memory entries were detected in the request."
        text, assistant_metadata = self._finalize_assistant_turn(agent, text=text)
        return QueryLoopResult(
            text=text,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=text, metadata=assistant_metadata)],
            metadata={"saved_entries": saved, **dict(assistant_metadata)},
        )

    def _run_planner(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        callbacks: List[Any],
        providers: Any,
    ) -> QueryLoopResult:
        system_prompt = self._build_system_prompt(
            agent,
            session_state,
            user_text=user_text,
            providers=providers,
            skill_context=skill_context,
        )
        prompt = (
            "Return JSON only with this schema:\n"
            "{"
            '"summary": "short summary",'
            '"tasks": ['
            '{"id": "task_1", "title": "...", "executor": "rag_worker|utility|data_analyst|general|graph_manager", '
            '"mode": "sequential|parallel", "depends_on": [], "input": "...", "doc_scope": [], '
            '"skill_queries": [], "research_profile": "", "coverage_goal": "", '
            '"result_mode": "", "answer_mode": "answer|evidence_only", "controller_hints": {}, "produces_artifacts": [], '
            '"consumes_artifacts": [], "handoff_schema": "", "input_artifact_ids": []}'
            "]}\n\n"
            f"Limit the number of tasks to {self.settings.planner_max_tasks}.\n"
            "Only mark tasks as parallel when they are truly independent.\n\n"
            f"USER_REQUEST:\n{user_text}"
        )
        text = ""
        try:
            response = providers.chat.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=prompt)],
                config={"callbacks": callbacks},
            )
            text = getattr(response, "content", None) or str(response)
        except Exception as exc:
            logger.warning("Planner agent failed: %s", exc)
        text, assistant_metadata = self._finalize_assistant_turn(agent, text=text)
        if str(assistant_metadata.get("turn_outcome") or "") == "clarification_request":
            return QueryLoopResult(
                text=text,
                messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=text, metadata=assistant_metadata)],
                metadata=dict(assistant_metadata),
            )
        obj = extract_json(text or "") or {}
        raw_tasks = obj.get("tasks") if isinstance(obj.get("tasks"), list) else []
        task_plan = normalise_task_plan(
            raw_tasks,
            query=user_text,
            max_tasks=self.settings.planner_max_tasks,
            session_metadata=dict(session_state.metadata or {}),
        )
        raw_shape = [
            (
                str(task.get("id") or f"task_{index + 1}"),
                str(task.get("executor") or "general"),
                str(task.get("mode") or "sequential"),
            )
            for index, task in enumerate(raw_tasks)
            if isinstance(task, dict)
        ]
        normalized_shape = [
            (
                str(task.get("id") or f"task_{index + 1}"),
                str(task.get("executor") or "general"),
                str(task.get("mode") or "sequential"),
            )
            for index, task in enumerate(task_plan)
            if isinstance(task, dict)
        ]
        plan_repair_applied = raw_shape != normalized_shape
        payload = {
            "summary": str(obj.get("summary") or f"Planned {len(task_plan)} task(s)."),
            "tasks": task_plan,
        }
        rendered = json.dumps(payload, ensure_ascii=False)
        rendered, assistant_metadata = self._finalize_assistant_turn(agent, text=rendered)
        return QueryLoopResult(
            text=rendered,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=rendered, metadata=assistant_metadata)],
            metadata={
                "planner_payload": payload,
                "planner_raw_task_count": len(raw_tasks),
                "planner_normalized_task_count": len(task_plan),
                "plan_repair_applied": plan_repair_applied,
                "context_budget": dict(self._last_context_ledger or {}),
                **dict(assistant_metadata),
            },
        )

    def _run_finalizer(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        task_payload: Dict[str, Any],
        callbacks: List[Any],
        providers: Any,
    ) -> QueryLoopResult:
        system_prompt = self._build_system_prompt(
            agent,
            session_state,
            user_text=user_text,
            providers=providers,
            skill_context=skill_context,
            task_payload=task_payload,
        )
        execution_digest = dict(task_payload.get("execution_digest") or {})
        control_notes = {
            key: value
            for key, value in {
                "revision_feedback": task_payload.get("revision_feedback"),
                "prefer_structured_final_answer": task_payload.get("prefer_structured_final_answer"),
                "finalizer_retry_reason": task_payload.get("finalizer_retry_reason"),
            }.items()
            if value not in (None, "", [], {})
        }
        prompt_payload = execution_digest or dict(task_payload or {})
        prompt_payload_text = self.context_budget_manager.budget_text_block(
            "finalizer_execution_digest",
            json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            max_tokens=max(1600, int(getattr(self.settings, "context_microcompact_target_tokens", 2400) or 2400)),
        )
        final_text = ""
        try:
            response = providers.chat.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            f"USER_REQUEST:\n{user_text}\n\n"
                            + (
                                "CONTROL_NOTES:\n"
                                f"{json.dumps(control_notes, ensure_ascii=False, indent=2)}\n\n"
                                if control_notes
                                else ""
                            )
                            + "EXECUTION_DIGEST:\n"
                            + prompt_payload_text
                        )
                    ),
                ],
                config={"callbacks": callbacks},
            )
            final_text = str(getattr(response, "content", None) or response).strip()
        except Exception as exc:
            logger.warning("Finalizer agent failed: %s", exc)
        if not final_text:
            final_text = str(task_payload.get("partial_answer") or "")
        final_text, assistant_metadata = self._finalize_assistant_turn(agent, text=final_text)
        return QueryLoopResult(
            text=final_text.strip(),
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=final_text.strip(), metadata=assistant_metadata)],
            metadata={
                "task_payload": task_payload,
                "execution_digest": prompt_payload,
                "context_budget": dict(self._last_context_ledger or {}),
                **dict(assistant_metadata),
            },
        )

    def _run_verifier(
        self,
        agent: AgentDefinition,
        session_state: SessionState,
        *,
        user_text: str,
        skill_context: str,
        task_payload: Dict[str, Any],
        callbacks: List[Any],
        providers: Any,
    ) -> QueryLoopResult:
        system_prompt = self._build_system_prompt(
            agent,
            session_state,
            user_text=user_text,
            providers=providers,
            skill_context=skill_context,
            task_payload=task_payload,
        )
        execution_digest = dict(task_payload.get("execution_digest") or {})
        prompt_payload = execution_digest or dict(task_payload or {})
        prompt_payload_text = self.context_budget_manager.budget_text_block(
            "verifier_execution_digest",
            json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            max_tokens=max(1600, int(getattr(self.settings, "context_microcompact_target_tokens", 2400) or 2400)),
        )
        verification = {
            "status": "pass",
            "summary": "No verification issues detected.",
            "issues": [],
            "feedback": "",
            "parse_failed": False,
        }
        assistant_metadata: Dict[str, Any] = {}
        try:
            response = providers.chat.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            "Return JSON only with this schema:\n"
                            '{'
                            '"status": "pass|revise", '
                            '"summary": "short verification summary", '
                            '"issues": ["issue 1"], '
                            '"feedback": "clear revision guidance"'
                            "}\n\n"
                            f"USER_REQUEST:\n{user_text}\n\n"
                            "EXECUTION_DIGEST:\n"
                            f"{prompt_payload_text}"
                        )
                    ),
                ],
                config={"callbacks": callbacks},
            )
            text = str(getattr(response, "content", None) or response).strip()
            text, assistant_metadata = self._finalize_assistant_turn(agent, text=text)
            if str(assistant_metadata.get("turn_outcome") or "") == "clarification_request":
                return QueryLoopResult(
                    text=text,
                    messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=text, metadata=assistant_metadata)],
                    metadata=dict(assistant_metadata),
                )
            payload = extract_json(text or "") or {}
            if not payload:
                verification = {
                    "status": "pass",
                    "summary": "Verifier returned unstructured output; keeping the current answer.",
                    "issues": [],
                    "feedback": "",
                    "parse_failed": True,
                }
            else:
                status = str(payload.get("status") or "pass").strip().lower()
                if status not in {"pass", "revise"}:
                    status = "pass"
                issues = [str(item) for item in (payload.get("issues") or []) if str(item).strip()]
                summary = str(payload.get("summary") or text or verification["summary"]).strip()
                feedback = str(payload.get("feedback") or "\n".join(issues) or summary).strip()
                verification = {
                    "status": status,
                    "summary": summary,
                    "issues": issues,
                    "feedback": feedback,
                    "parse_failed": False,
                }
        except Exception as exc:
            logger.warning("Verifier agent failed: %s", exc)
        rendered = json.dumps(verification, ensure_ascii=False)
        if not assistant_metadata:
            _, assistant_metadata = self._finalize_assistant_turn(agent, text=rendered)
        return QueryLoopResult(
            text=rendered,
            messages=list(session_state.messages) + [RuntimeMessage(role="assistant", content=rendered, metadata=assistant_metadata)],
            metadata={
                "verification": verification,
                "execution_digest": prompt_payload,
                "context_budget": dict(self._last_context_ledger or {}),
                **dict(assistant_metadata),
            },
        )
