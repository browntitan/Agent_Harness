from __future__ import annotations

from typing import Any

from agentic_chatbot_next.app.service import RuntimeService


class ApiAdapter:
    @staticmethod
    def create_service(settings: Any, providers: Any) -> RuntimeService:
        return RuntimeService.create(settings, providers)
