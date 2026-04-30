from __future__ import annotations

import json
import inspect
import os
import re
import socket
import threading
from time import perf_counter
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.observability.token_usage import extract_token_usage
from agentic_chatbot_next.runtime.event_sink import RuntimeEventSink
from agentic_chatbot_next.utils.json_utils import make_json_compatible


_TOOL_PREVIEW_LIMIT = 500
_TOOL_DETAIL_CHAR_LIMIT = 12_000
_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|bearer|cookie|credential|password|secret|session[_-]?id|token)",
    re.IGNORECASE,
)


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
    if not _langfuse_endpoint_resolvable(host):
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


def _langfuse_endpoint_resolvable(host: str | None) -> bool:
    if os.getenv("LANGFUSE_ALLOW_UNREACHABLE", "").lower() in {"1", "true", "yes", "y"}:
        return True
    parsed = urlparse(str(host or ""))
    hostname = parsed.hostname or str(host or "").split(":", 1)[0]
    if not hostname:
        return False
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        socket.getaddrinfo(hostname, parsed.port or 80)
        return True
    except OSError:
        return False


def _preview(value: Any, *, limit: int = _TOOL_PREVIEW_LIMIT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _redact_sensitive(value: Any, *, depth: int = 0, path: str = "", redacted_fields: Optional[List[str]] = None) -> Any:
    redacted_fields = redacted_fields if redacted_fields is not None else []
    if depth > 8:
        return _preview(value, limit=240)
    compatible = make_json_compatible(value)
    if isinstance(compatible, dict):
        redacted: Dict[str, Any] = {}
        for key, item in compatible.items():
            key_text = str(key)
            field_path = f"{path}.{key_text}" if path else key_text
            if _SENSITIVE_KEY_RE.search(key_text):
                redacted[key_text] = "[redacted]"
                if field_path not in redacted_fields:
                    redacted_fields.append(field_path)
            else:
                redacted[key_text] = _redact_sensitive(
                    item,
                    depth=depth + 1,
                    path=field_path,
                    redacted_fields=redacted_fields,
                )
        return redacted
    if isinstance(compatible, list):
        return [
            _redact_sensitive(
                item,
                depth=depth + 1,
                path=f"{path}[{index}]" if path else f"[{index}]",
                redacted_fields=redacted_fields,
            )
            for index, item in enumerate(compatible)
        ]
    return compatible


def _bounded_tool_value(value: Any) -> tuple[Any, str, bool, List[str]]:
    redacted_fields: List[str] = []
    redacted = _redact_sensitive(value, redacted_fields=redacted_fields)
    text = redacted if isinstance(redacted, str) else json.dumps(redacted, ensure_ascii=False, default=str)
    text = str(text)
    if len(text) <= _TOOL_DETAIL_CHAR_LIMIT:
        return redacted, _preview(redacted), False, redacted_fields
    return text[: _TOOL_DETAIL_CHAR_LIMIT - 3] + "...", _preview(text), True, redacted_fields


def _prefixed_fields(prefix: str, fields: List[str]) -> List[str]:
    return [f"{prefix}.{field}" if field else prefix for field in fields]


def _merge_unique_fields(*field_lists: Any) -> List[str]:
    merged: List[str] = []
    for fields in field_lists:
        for field in list(fields or []):
            value = str(field or "").strip()
            if value and value not in merged:
                merged.append(value)
    return merged


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _merge_truncation(existing: Any, *field_flags: tuple[str, bool]) -> tuple[bool, List[str]]:
    fields: List[str] = []
    if isinstance(existing, dict):
        fields.extend(str(key) for key, value in existing.items() if value and str(key))
    elif isinstance(existing, list):
        fields.extend(str(item) for item in existing if str(item))
    elif bool(existing):
        fields.append("unknown")
    for field, truncated in field_flags:
        if truncated and field not in fields:
            fields.append(field)
    if "unknown" in fields and len(fields) > 1:
        fields = [field for field in fields if field != "unknown"]
    return bool(fields), fields


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
        input_value = inputs if inputs is not None else input_str
        safe_input, input_preview, input_truncated, input_redacted_fields = _bounded_tool_value(input_value)
        callback_metadata = dict(metadata or {})
        parent_agent = str(
            callback_metadata.get("parent_agent")
            or self.metadata.get("parent_agent")
            or self.metadata.get("suggested_agent")
            or ""
        )
        parallel_group_id = str(
            callback_metadata.get("agentic_parallel_group_id")
            or callback_metadata.get("parallel_group_id")
            or ""
        )
        started_at = utc_now_iso()
        run_metadata = {
            "tool_name": tool_name,
            "tool_call_id": run_key,
            "input_preview": input_preview,
            "input": safe_input,
            "tags": list(tags or []),
            "callback_metadata": callback_metadata,
            "parent_run_id": str(parent_run_id or ""),
            "parent_agent": parent_agent,
            "parallel_group_id": parallel_group_id,
            "parallel_execution_mode": str(callback_metadata.get("agentic_parallel_execution_mode") or ""),
            "parallel_group_size": _safe_int(callback_metadata.get("agentic_parallel_group_size")),
            "payload_limit_chars": _TOOL_DETAIL_CHAR_LIMIT,
            "redacted_fields": _prefixed_fields("input", input_redacted_fields),
            "started_at": started_at,
            "completed_at": "",
            "duration_ms": None,
            "status": "running",
            "truncated": input_truncated,
            "truncated_fields": ["input"] if input_truncated else [],
        }
        with self._lock:
            self._tool_runs[run_key] = {
                **run_metadata,
                "_started_perf": perf_counter(),
            }
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
        started_perf = run_metadata.pop("_started_perf", None)
        duration_ms = (
            max(0, int((perf_counter() - float(started_perf)) * 1000))
            if isinstance(started_perf, (int, float))
            else None
        )
        safe_output, output_preview, output_truncated, output_redacted_fields = _bounded_tool_value(output)
        truncated, truncated_fields = _merge_truncation(
            run_metadata.get("truncated_fields") or run_metadata.get("truncated"),
            ("output", output_truncated),
        )
        payload = {
            **run_metadata,
            "parent_run_id": str(parent_run_id or ""),
            "output_preview": output_preview,
            "output": safe_output,
            "redacted_fields": _merge_unique_fields(
                run_metadata.get("redacted_fields"),
                _prefixed_fields("output", output_redacted_fields),
            ),
            "completed_at": utc_now_iso(),
            "duration_ms": duration_ms,
            "status": "completed",
            "truncated": truncated,
            "truncated_fields": truncated_fields,
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
        started_perf = run_metadata.pop("_started_perf", None)
        duration_ms = (
            max(0, int((perf_counter() - float(started_perf)) * 1000))
            if isinstance(started_perf, (int, float))
            else None
        )
        error_text, error_preview, error_truncated, error_redacted_fields = _bounded_tool_value(str(error))
        truncated, truncated_fields = _merge_truncation(
            run_metadata.get("truncated_fields") or run_metadata.get("truncated"),
            ("error", error_truncated),
        )
        payload = {
            **run_metadata,
            "parent_run_id": str(parent_run_id or ""),
            "error": error_text,
            "error_preview": error_preview,
            "redacted_fields": _merge_unique_fields(
                run_metadata.get("redacted_fields"),
                _prefixed_fields("error", error_redacted_fields),
            ),
            "completed_at": utc_now_iso(),
            "duration_ms": duration_ms,
            "status": "error",
            "truncated": truncated,
            "truncated_fields": truncated_fields,
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
