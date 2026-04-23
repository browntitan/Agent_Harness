from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_chatbot_next.observability.callbacks import (
    RuntimeTraceCallbackHandler,
    get_langchain_callbacks,
)
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.runtime.event_sink import CompositeRuntimeEventSink, NullEventSink, RuntimeEventSink
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore


class TranscriptEventSink(RuntimeEventSink):
    def __init__(self, transcript_store: RuntimeTranscriptStore) -> None:
        self.transcript_store = transcript_store

    def emit(self, event: RuntimeEvent) -> None:
        self.transcript_store.append_session_event(event)
        if event.job_id:
            self.transcript_store.append_job_event(event)


class KernelEventController:
    def __init__(self, settings: Any, transcript_store: RuntimeTranscriptStore) -> None:
        self.settings = settings
        self.transcript_store = transcript_store
        if getattr(settings, "runtime_events_enabled", True):
            base_event_sink: RuntimeEventSink = TranscriptEventSink(transcript_store)
        else:
            base_event_sink = NullEventSink()
        self.event_sink = CompositeRuntimeEventSink(base_event_sink)

    def register_live_progress_sink(self, session_id: str, sink: RuntimeEventSink) -> None:
        self.event_sink.register_session_sink(session_id, sink)

    def unregister_live_progress_sink(self, session_id: str, sink: RuntimeEventSink) -> None:
        self.event_sink.unregister_session_sink(session_id, sink)

    def get_live_progress_sink(self, session_id: str) -> RuntimeEventSink | None:
        return self.event_sink.get_session_sink(session_id)

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
        conversation_id = str(getattr(session, "conversation_id", "") or "")
        session_id = str(getattr(session, "session_id", "") or "")
        tenant_id = str(getattr(session, "tenant_id", "") or "")
        user_id = str(getattr(session, "user_id", "") or "")
        request_id = str(getattr(session, "request_id", "") or "")
        self.emit(
            "router_decision",
            session_id,
            agent_name="router",
            payload={
                "router_decision_id": router_decision_id,
                "conversation_id": conversation_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "request_id": request_id,
                "route": route,
                "confidence": confidence,
                "reasons": list(reasons),
                "router_method": router_method,
                "suggested_agent": suggested_agent,
                "force_agent": force_agent,
                "has_attachments": has_attachments,
                "requested_agent_override": requested_agent_override,
                "requested_agent_override_applied": requested_agent_override_applied,
                "router_evidence": dict(router_evidence or {}),
            },
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
        session_id = str(getattr(session_or_state, "session_id", "") or "")
        conversation_id = str(getattr(session_or_state, "conversation_id", "") or "")
        combined_metadata = {
            "conversation_id": conversation_id,
            **dict(metadata or {}),
        }
        callbacks = list(base_callbacks or [])
        callbacks.extend(
            get_langchain_callbacks(
                self.settings,
                session_id=session_id,
                trace_name=trace_name,
                metadata=combined_metadata,
            )
        )
        if getattr(self.settings, "runtime_events_enabled", True):
            callbacks.append(
                RuntimeTraceCallbackHandler(
                    event_sink=self.event_sink,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    trace_name=trace_name,
                    agent_name=agent_name,
                    job_id=job_id,
                    metadata=combined_metadata,
                )
            )
        return callbacks

    def emit(
        self,
        event_type: str,
        session_id: str,
        *,
        agent_name: str = "",
        payload: Optional[Dict[str, Any]] = None,
        tool_name: str = "",
        job_id: str = "",
    ) -> None:
        if not session_id:
            return
        self.event_sink.emit(
            RuntimeEvent(
                event_type=event_type,
                session_id=session_id,
                agent_name=agent_name,
                tool_name=tool_name,
                job_id=job_id,
                payload=dict(payload or {}),
            )
        )


__all__ = ["KernelEventController", "TranscriptEventSink"]
