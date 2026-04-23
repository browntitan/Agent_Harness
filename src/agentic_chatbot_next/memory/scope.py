from __future__ import annotations

from enum import Enum


class MemoryScope(str, Enum):
    conversation = "conversation"
    user = "user"
