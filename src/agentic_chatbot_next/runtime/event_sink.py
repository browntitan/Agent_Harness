from __future__ import annotations

import threading
from collections import defaultdict
from typing import DefaultDict, List

from agentic_chatbot_next.observability.events import RuntimeEvent


class RuntimeEventSink:
    def emit(self, event: RuntimeEvent) -> None:
        raise NotImplementedError


class NullEventSink(RuntimeEventSink):
    def emit(self, event: RuntimeEvent) -> None:
        return None


class CompositeRuntimeEventSink(RuntimeEventSink):
    def __init__(self, *sinks: RuntimeEventSink) -> None:
        self._base_sinks: List[RuntimeEventSink] = [sink for sink in sinks if sink is not None]
        self._session_sinks: DefaultDict[str, List[RuntimeEventSink]] = defaultdict(list)
        self._lock = threading.Lock()

    def emit(self, event: RuntimeEvent) -> None:
        sinks = list(self._base_sinks)
        if event.session_id:
            with self._lock:
                sinks.extend(self._session_sinks.get(event.session_id, []))
        for sink in sinks:
            sink.emit(event)

    def register_session_sink(self, session_id: str, sink: RuntimeEventSink) -> None:
        if not session_id or sink is None:
            return
        with self._lock:
            if sink not in self._session_sinks[session_id]:
                self._session_sinks[session_id].append(sink)

    def unregister_session_sink(self, session_id: str, sink: RuntimeEventSink) -> None:
        if not session_id or sink is None:
            return
        with self._lock:
            sinks = self._session_sinks.get(session_id, [])
            if sink in sinks:
                sinks.remove(sink)
            if not sinks and session_id in self._session_sinks:
                del self._session_sinks[session_id]

    def get_session_sink(self, session_id: str) -> RuntimeEventSink | None:
        if not session_id:
            return None
        with self._lock:
            sinks = self._session_sinks.get(session_id, [])
            return sinks[0] if sinks else None
