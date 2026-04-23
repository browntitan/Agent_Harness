from __future__ import annotations

import json
import inspect
import os
import threading
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.observability.token_usage import extract_token_usage
from agentic_chatbot_next.runtime.event_sink import RuntimeEventSink


def get_langchain_callbacks(
    settings: Settings,
    *,
    session_id: str,
    trace_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    metadata = metadata or {}
    public_key = getattr(settings, "langfuse_public_key", None)
    secret_key = getattr(settings, "langfuse_secret_key", None)
    host = getattr(settings, "langfuse_host", None)
    debug = bool(getattr(settings, "langfuse_debug", False))

    if not (public_key and secret_key):
        return []

    try:
        try:
            from langfuse.langchain import CallbackHandler
        except Exception:
            from langfuse.callback import CallbackHandler

        params = inspect.signature(CallbackHandler.__init__).parameters
        if "secret_key" in params:
            return [
                CallbackHandler(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                    debug=debug,
                    session_id=session_id,
                    trace_name=trace_name,
                    metadata=metadata,
                )
            ]

        if host:
            os.environ["LANGFUSE_HOST"] = host
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key or ""
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key or ""
        try:
            from langfuse import Langfuse

            Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                debug=debug,
            )
        except Exception:
            pass

        try:
            handler = CallbackHandler(public_key=public_key, update_trace=True)
        except TypeError:
            handler = CallbackHandler(public_key=public_key)
        return [handler]
    except Exception:
        return []


def _preview(value: Any, *, limit: int = 500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class RuntimeTraceCallbackHandler(BaseCallbackHandler):
    """Persist model and tool lifecycle events into the next-runtime trace store."""

    raise_error = False

    def __init__(
        self,
        *,
        event_sink: RuntimeEventSink,
        session_id: str,
        conversation_id: str,
        trace_name: str,
        agent_name: str = "",
        job_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.event_sink = event_sink
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.trace_name = trace_name
        self.agent_name = agent_name
        self.job_id = job_id
        self.metadata = dict(metadata or {})
        self._tool_runs: Dict[str, Dict[str, Any]] = {}
        self._model_runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        run_key = str(run_id)
        model_name = (
            str(serialized.get("name") or "")
            or str((serialized.get("kwargs") or {}).get("model") or "")
            or "chat_model"
        )
        message_count = sum(len(batch) for batch in messages)
        run_metadata = {
            "model_name": model_name,
            "message_count": message_count,
            "tags": list(tags or []),
            "callback_metadata": dict(metadata or {}),
            "parent_run_id": str(parent_run_id or ""),
        }
        with self._lock:
            self._model_runs[run_key] = run_metadata
        self._emit("model_start", payload=run_metadata)
        return None

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        run_key = str(run_id)
        with self._lock:
            run_metadata = self._model_runs.pop(run_key, {})
        llm_output = dict(getattr(response, "llm_output", {}) or {})
        output_preview = ""
        generations = getattr(response, "generations", None) or []
        if generations and generations[0]:
            first = generations[0][0]
            message = getattr(first, "message", None)
            if message is not None:
                output_preview = _preview(getattr(message, "content", ""))
            else:
                output_preview = _preview(getattr(first, "text", ""))
        payload = {
            **run_metadata,
            "parent_run_id": str(parent_run_id or ""),
            "output_preview": output_preview,
            "llm_output": llm_output,
            "token_usage": extract_token_usage(llm_output),
        }
        self._emit("model_end", payload=payload)
        return None

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        with self._lock:
            run_metadata = self._model_runs.pop(str(run_id), {})
        self._emit(
            "model_error",
            payload={
                **run_metadata,
                "parent_run_id": str(parent_run_id or ""),
                "error": _preview(str(error), limit=1000),
            },
        )
        return None

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        run_key = str(run_id)
        tool_name = str(serialized.get("name") or "tool")
        run_metadata = {
            "tool_name": tool_name,
            "input_preview": _preview(inputs if inputs is not None else input_str),
            "tags": list(tags or []),
            "callback_metadata": dict(metadata or {}),
            "parent_run_id": str(parent_run_id or ""),
        }
        with self._lock:
            self._tool_runs[run_key] = run_metadata
        self._emit("tool_start", tool_name=tool_name, payload=run_metadata)
        return None

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        with self._lock:
            run_metadata = self._tool_runs.pop(str(run_id), {})
        tool_name = str(run_metadata.get("tool_name") or "")
        payload = {
            **run_metadata,
            "parent_run_id": str(parent_run_id or ""),
            "output_preview": _preview(output),
        }
        self._emit("tool_end", tool_name=tool_name, payload=payload)
        return None

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        with self._lock:
            run_metadata = self._tool_runs.pop(str(run_id), {})
        tool_name = str(run_metadata.get("tool_name") or "")
        payload = {
            **run_metadata,
            "parent_run_id": str(parent_run_id or ""),
            "error": _preview(str(error), limit=1000),
        }
        self._emit("tool_error", tool_name=tool_name, payload=payload)
        return None

    def _emit(self, event_type: str, *, payload: Dict[str, Any], tool_name: str = "") -> None:
        base_payload = {
            "conversation_id": self.conversation_id,
            "trace_name": self.trace_name,
            "route": self.metadata.get("route", ""),
            "router_method": self.metadata.get("router_method", ""),
            "suggested_agent": self.metadata.get("suggested_agent", ""),
            **self.metadata,
        }
        merged_payload = {**base_payload, **dict(payload or {})}
        self.event_sink.emit(
            RuntimeEvent(
                event_type=event_type,
                session_id=self.session_id,
                agent_name=self.agent_name,
                job_id=self.job_id,
                tool_name=tool_name,
                payload=merged_payload,
            )
        )
