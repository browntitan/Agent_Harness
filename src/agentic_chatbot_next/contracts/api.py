from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OpenAIMessage:
    role: str
    content: Any = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatCompletionsRequest:
    model: str
    messages: List[OpenAIMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["messages"] = [message.to_dict() for message in self.messages]
        return payload


@dataclass
class IngestDocumentsRequest:
    paths: List[str] = field(default_factory=list)
    source_type: str = "upload"
    conversation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
