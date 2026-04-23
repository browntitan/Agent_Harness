from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import dotenv_values


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, content: str) -> None:
    _ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _safe_name(name: str) -> str:
    value = str(name or "").strip()
    if not value:
        raise ValueError("name must not be empty")
    if Path(value).name != value:
        raise ValueError("name must not include path separators")
    return value


def _encode_env_value(value: str) -> str:
    text = str(value)
    if text == "":
        return '""'
    if all(ch.isalnum() or ch in "._-/:@+" for ch in text):
        return text
    return json.dumps(text, ensure_ascii=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class OverlayStore:
    root_dir: Path
    runtime_env_path: Path
    prompt_dir: Path
    agent_dir: Path
    audit_log_path: Path

    @classmethod
    def from_settings(cls, settings: Any) -> "OverlayStore":
        return cls(
            root_dir=Path(getattr(settings, "control_panel_overlay_dir")),
            runtime_env_path=Path(getattr(settings, "control_panel_runtime_env_path")),
            prompt_dir=Path(getattr(settings, "control_panel_prompt_overlays_dir")),
            agent_dir=Path(getattr(settings, "control_panel_agent_overlays_dir")),
            audit_log_path=Path(getattr(settings, "control_panel_audit_log_path")),
        )

    def ensure_layout(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def read_runtime_env(self) -> Dict[str, str]:
        if not self.runtime_env_path.exists():
            return {}
        values = dotenv_values(self.runtime_env_path)
        return {
            str(key): str(value)
            for key, value in values.items()
            if str(key).strip() and value is not None
        }

    def write_runtime_env(self, values: Dict[str, str]) -> None:
        self.ensure_layout()
        normalized = {
            str(key).strip(): str(value)
            for key, value in dict(values or {}).items()
            if str(key).strip()
        }
        if not normalized:
            if self.runtime_env_path.exists():
                self.runtime_env_path.unlink()
            return
        lines = [f"{key}={_encode_env_value(normalized[key])}" for key in sorted(normalized)]
        _atomic_write_text(self.runtime_env_path, "\n".join(lines) + "\n")

    def apply_runtime_env_changes(self, changes: Dict[str, str | None]) -> Dict[str, str]:
        merged = dict(self.read_runtime_env())
        for key, value in dict(changes or {}).items():
            env_name = str(key or "").strip()
            if not env_name:
                continue
            if value is None or str(value) == "":
                merged.pop(env_name, None)
                continue
            merged[env_name] = str(value)
        return merged

    def prompt_overlay_path(self, prompt_file: str) -> Path:
        return self.prompt_dir / _safe_name(prompt_file)

    def list_prompt_overlays(self) -> List[str]:
        if not self.prompt_dir.exists():
            return []
        return sorted(path.name for path in self.prompt_dir.iterdir() if path.is_file())

    def read_prompt_overlay(self, prompt_file: str) -> str:
        path = self.prompt_overlay_path(prompt_file)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_prompt_overlay(self, prompt_file: str, content: str) -> Path:
        path = self.prompt_overlay_path(prompt_file)
        _atomic_write_text(path, str(content))
        return path

    def delete_prompt_overlay(self, prompt_file: str) -> bool:
        path = self.prompt_overlay_path(prompt_file)
        if not path.exists():
            return False
        path.unlink()
        return True

    def agent_overlay_path(self, agent_name: str) -> Path:
        filename = _safe_name(agent_name)
        stem = filename[:-3] if filename.endswith(".md") else filename
        return self.agent_dir / f"{stem}.md"

    def list_agent_overlays(self) -> List[str]:
        if not self.agent_dir.exists():
            return []
        return sorted(path.stem for path in self.agent_dir.glob("*.md") if path.is_file())

    def read_agent_overlay(self, agent_name: str) -> str:
        path = self.agent_overlay_path(agent_name)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_agent_overlay(self, agent_name: str, content: str) -> Path:
        path = self.agent_overlay_path(agent_name)
        _atomic_write_text(path, str(content))
        return path

    def delete_agent_overlay(self, agent_name: str) -> bool:
        path = self.agent_overlay_path(agent_name)
        if not path.exists():
            return False
        path.unlink()
        return True

    def append_audit_event(self, *, action: str, actor: str, details: Dict[str, Any]) -> None:
        self.ensure_layout()
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": _utc_now(),
                        "action": str(action or "").strip(),
                        "actor": str(actor or "").strip() or "control-panel",
                        "details": dict(details or {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def read_audit_events(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.audit_log_path.exists():
            return []
        lines = self.audit_log_path.read_text(encoding="utf-8").splitlines()
        events: List[Dict[str, Any]] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(events) >= max(1, int(limit)):
                break
        return list(reversed(events))
