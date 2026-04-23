"""LangChain callback handler that captures agent progress events to a queue.

Used by the streaming endpoint to emit real-time SSE progress events while
process_turn() is running in a background thread.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from agentic_chatbot_next.api.live_progress import LiveProgressSink
from agentic_chatbot_next.api.status_tracker import (
    PHASE_SEARCHING,
    PHASE_STARTING,
    PHASE_SYNTHESIZING,
)

logger = logging.getLogger(__name__)

# LangGraph node names we surface to the user.
_GRAPH_NODES = {
    "supervisor", "rag_agent", "utility_agent", "parallel_planner",
    "rag_worker", "rag_synthesizer", "data_analyst", "clarify", "evaluator",
}

# Human-readable labels for nodes
_NODE_LABELS = {
    "supervisor": "Routing request",
    "rag_agent": "Searching documents",
    "utility_agent": "Running utility tools",
    "parallel_planner": "Planning parallel search",
    "rag_worker": "Searching (parallel worker)",
    "rag_synthesizer": "Synthesizing results",
    "data_analyst": "Analyzing data",
    "clarify": "Requesting clarification",
    "evaluator": "Evaluating answer quality",
}

_NODE_PHASES = {
    "supervisor": PHASE_STARTING,
    "rag_agent": PHASE_SEARCHING,
    "parallel_planner": PHASE_SEARCHING,
    "rag_worker": PHASE_SEARCHING,
    "rag_synthesizer": PHASE_SYNTHESIZING,
    "evaluator": PHASE_SYNTHESIZING,
}


class ProgressCallback(BaseCallbackHandler):
    """Pushes typed progress events to a queue.Queue for SSE emission.

    Thread-safe: process_turn() runs in a background thread; the SSE
    generator reads from self.events on the main thread.
    """

    def __init__(self, sink: LiveProgressSink | None = None) -> None:
        super().__init__()
        self.sink = sink or LiveProgressSink()
        self.events = self.sink.events
        self._start_times: Dict[str, float] = {}
        self._active_tool_names: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _push(self, event: Dict[str, Any]) -> None:
        self.sink.emit_progress(**event)

    def mark_done(self) -> None:
        """Push the sentinel that tells the SSE generator to stop reading."""
        self.sink.mark_done()

    # ── LangGraph chain (node) callbacks ──────────────────────────────────

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "")
        if name not in _GRAPH_NODES:
            return
        with self._lock:
            self._start_times[str(run_id)] = time.time()
        self._push({
            "event_type": "decision_point",
            "node": name,
            "agent": name,
            "active_agent": name,
            "selected_agent": name,
            "label": _NODE_LABELS.get(name, name),
            "detail": "Preparing next action",
            "why": f"{_NODE_LABELS.get(name, name)} is the current execution step.",
            "phase": _NODE_PHASES.get(name, ""),
            "timestamp": int(time.time() * 1000),
        })
        self._push({
            "event_type": "agent_start",
            "node": name,
            "agent": name,
            "active_agent": name,
            "selected_agent": name,
            "label": _NODE_LABELS.get(name, name),
            "phase": _NODE_PHASES.get(name, ""),
            "timestamp": int(time.time() * 1000),
        })

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        # We don't push chain end events — they clutter the UI
        with self._lock:
            self._start_times.pop(str(run_id), None)

    # ── Tool callbacks ────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "unknown_tool")
        with self._lock:
            self._start_times[str(run_id)] = time.time()
            self._active_tool_names[str(run_id)] = name
        # Parse input as JSON if possible for nicer display
        try:
            parsed_input: Any = json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            parsed_input = input_str
        self._push({
            "event_type": "tool_intent",
            "id": str(run_id),
            "tool": name,
            "label": f"Preparing {name}",
            "detail": _summarize_tool_intent(name, parsed_input),
            "why": _why_for_tool(name),
            "phase": PHASE_SEARCHING,
            "timestamp": int(time.time() * 1000),
        })
        self._push({
            "event_type": "tool_call",
            "id": str(run_id),
            "tool": name,
            "input": parsed_input,
            "phase": PHASE_SEARCHING,
            "timestamp": int(time.time() * 1000),
        })

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        with self._lock:
            start = self._start_times.pop(run_key, None)
            tool_name = self._active_tool_names.pop(run_key, "tool")
        duration_ms = int((time.time() - start) * 1000) if start else None
        output_str = str(output) if output is not None else ""
        # Truncate large outputs — the full output is in the final answer
        display_output = output_str[:800] + "…" if len(output_str) > 800 else output_str
        self._push({
            "event_type": "tool_result",
            "id": run_key,
            "tool": tool_name,
            "output": display_output,
            "duration_ms": duration_ms,
            "phase": PHASE_SEARCHING,
            "timestamp": int(time.time() * 1000),
        })

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        with self._lock:
            start = self._start_times.pop(run_key, None)
            tool_name = self._active_tool_names.pop(run_key, "tool")
        duration_ms = int((time.time() - start) * 1000) if start else None
        self._push({
            "event_type": "tool_error",
            "id": run_key,
            "tool": tool_name,
            "error": str(error)[:200],
            "duration_ms": duration_ms,
            "phase": PHASE_SEARCHING,
            "timestamp": int(time.time() * 1000),
        })

    # ── LLM callbacks ─────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Don't push — too noisy; multiple LLM calls happen per node
        pass

    def raise_error(self) -> bool:
        return False  # Don't suppress exceptions


def _summarize_tool_intent(tool_name: str, parsed_input: Any) -> str:
    if isinstance(parsed_input, dict):
        if "query" in parsed_input:
            return str(parsed_input.get("query") or "")[:160]
        if "doc_id" in parsed_input:
            return str(parsed_input.get("doc_id") or "")[:160]
    return str(tool_name)


def _why_for_tool(tool_name: str) -> str:
    mapping = {
        "search_document": "Checking a specific document for grounded evidence.",
        "search_all_documents": "Scanning the broader corpus for relevant evidence.",
        "full_text_search_document": "Looking for exact phrase or keyword matches nearby.",
        "fetch_chunk_window": "Reading the surrounding context around a likely hit.",
        "graph_search_local": "Traversing relationship-aware graph candidates near the query entities.",
        "graph_search_global": "Exploring graph-wide cross-document relationships.",
    }
    return mapping.get(tool_name, "Using a tool to gather or verify grounded evidence.")
