from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.utils.json_utils import make_json_compatible


@dataclass
class RuntimeEvent:
    event_type: str
    session_id: str
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:16]}")
    created_at: str = field(default_factory=utc_now_iso)
    job_id: str = ""
    agent_name: str = ""
    tool_name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return make_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RuntimeEvent":
        return cls(
            event_id=str(raw.get("event_id") or f"evt_{uuid.uuid4().hex[:16]}"),
            event_type=str(raw.get("event_type") or ""),
            session_id=str(raw.get("session_id") or ""),
            created_at=str(raw.get("created_at") or utc_now_iso()),
            job_id=str(raw.get("job_id") or ""),
            agent_name=str(raw.get("agent_name") or ""),
            tool_name=str(raw.get("tool_name") or ""),
            payload=dict(raw.get("payload") or {}),
        )
