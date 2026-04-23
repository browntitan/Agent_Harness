from __future__ import annotations

import os
import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List

from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.app.api_adapter import ApiAdapter
from agentic_chatbot_next.config import Settings, load_settings
from agentic_chatbot_next.control_panel.overlay_store import OverlayStore
from agentic_chatbot_next.providers import (
    ProviderConfigurationError,
    ProviderDependencyError,
    build_providers,
)
from agentic_chatbot_next.runtime.registry_diagnostics import build_runtime_error_payload


logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _patched_environ() -> Iterator[None]:
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


@dataclass
class RuntimeSnapshot:
    settings: Settings
    bot: Any


class RuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot: RuntimeSnapshot | None = None
        self._last_reload: Dict[str, Any] = {
            "status": "never_loaded",
            "timestamp": "",
            "reason": "startup",
            "actor": "system",
            "changed_keys": [],
            "error": "",
        }

    def _load_settings(self, *, env_overrides: Dict[str, str | None] | None = None) -> Settings:
        with _patched_environ():
            return load_settings(env_overrides=env_overrides)

    def _build_snapshot(self, *, env_overrides: Dict[str, str | None] | None = None) -> RuntimeSnapshot:
        settings = self._load_settings(env_overrides=env_overrides)
        providers = build_providers(settings)
        bot = ApiAdapter.create_service(settings, providers)
        return RuntimeSnapshot(settings=settings, bot=bot)

    def _record_reload(
        self,
        *,
        status: str,
        reason: str,
        actor: str,
        changed_keys: List[str] | None = None,
        error: str = "",
    ) -> Dict[str, Any]:
        self._last_reload = {
            "status": status,
            "timestamp": _utc_now(),
            "reason": str(reason or "").strip() or "manual",
            "actor": str(actor or "").strip() or "control-panel",
            "changed_keys": [str(item) for item in (changed_keys or []) if str(item)],
            "error": str(error or "").strip(),
        }
        return dict(self._last_reload)

    def get_snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            if self._snapshot is None:
                try:
                    self._snapshot = self._build_snapshot()
                    self._record_reload(status="success", reason="startup", actor="system")
                except Exception as exc:
                    payload = build_runtime_error_payload(exc)
                    logger.error("Runtime startup failed: %s", payload)
                    self._record_reload(
                        status="failed",
                        reason="startup",
                        actor="system",
                        error=json.dumps(payload, ensure_ascii=False),
                    )
                    raise
            return self._snapshot

    def get_settings(self) -> Settings:
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot.settings
        return self._load_settings()

    def get_overlay_store(self) -> OverlayStore:
        return OverlayStore.from_settings(self.get_settings())

    def preview_snapshot(self, *, env_overrides: Dict[str, str | None] | None = None) -> RuntimeSnapshot:
        return self._build_snapshot(env_overrides=env_overrides)

    def reload_runtime(
        self,
        *,
        reason: str,
        actor: str = "control-panel",
        changed_keys: List[str] | None = None,
    ) -> Dict[str, Any]:
        try:
            snapshot = self._build_snapshot()
        except (ProviderDependencyError, ProviderConfigurationError, Exception) as exc:
            payload = build_runtime_error_payload(exc)
            logger.error("Runtime reload failed: %s", payload)
            return self._record_reload(
                status="failed",
                reason=reason,
                actor=actor,
                changed_keys=changed_keys,
                error=json.dumps(payload, ensure_ascii=False),
            )

        with self._lock:
            self._snapshot = snapshot
            return self._record_reload(
                status="success",
                reason=reason,
                actor=actor,
                changed_keys=changed_keys,
            )

    def reload_agents(
        self,
        *,
        actor: str = "control-panel",
        changed_keys: List[str] | None = None,
    ) -> Dict[str, Any]:
        with self._lock:
            snapshot = self.get_snapshot()
            settings = snapshot.settings
            kernel = snapshot.bot.kernel
            try:
                temp_registry = AgentRegistry(
                    Path(getattr(settings, "agents_dir")),
                    overlay_dir=Path(getattr(settings, "control_panel_agent_overlays_dir")),
                )
                kernel.validate_registry(temp_registry)
            except Exception as exc:
                return self._record_reload(
                    status="failed",
                    reason="agent_reload",
                    actor=actor,
                    changed_keys=changed_keys,
                    error=str(exc),
                )
            kernel.registry = temp_registry
            return self._record_reload(
                status="success",
                reason="agent_reload",
                actor=actor,
                changed_keys=changed_keys,
            )

    def last_reload_summary(self) -> Dict[str, Any]:
        return dict(self._last_reload)


@lru_cache(maxsize=1)
def get_runtime_manager() -> RuntimeManager:
    return RuntimeManager()
