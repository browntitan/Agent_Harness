from __future__ import annotations

from typing import List

from agentic_chatbot_next.contracts.jobs import TaskNotification
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore


class NotificationStore:
    def __init__(self, transcript_store: RuntimeTranscriptStore):
        self.transcript_store = transcript_store

    def append(self, session_id: str, notification: TaskNotification) -> None:
        self.transcript_store.append_notification(session_id, notification)

    def drain(self, session_id: str) -> List[TaskNotification]:
        return self.transcript_store.drain_notifications(session_id)
