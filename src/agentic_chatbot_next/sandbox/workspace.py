"""Session workspace for the next runtime.

Each session gets a persistent host-side directory under:

    data/workspaces/<filesystem_key(session_id)>/

The directory is safe for host-side file operations and can be bind-mounted into
the Docker sandbox at ``/workspace`` for data-analyst execution.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from agentic_chatbot_next.runtime.context import filesystem_key

logger = logging.getLogger(__name__)

_MAX_READ_BYTES = 5 * 1024 * 1024
_MAX_FILENAME_LEN = 128


class WorkspacePathError(ValueError):
    """Raised when a caller supplies an unsafe or invalid filename."""


def _safe_filename(filename: str) -> str:
    name = filename.strip()
    if not name:
        raise WorkspacePathError("filename must not be empty")
    if len(name) > _MAX_FILENAME_LEN:
        raise WorkspacePathError(f"filename too long (max {_MAX_FILENAME_LEN} chars)")
    if "/" in name or "\\" in name:
        raise WorkspacePathError("filename must not contain path separators")
    if "\x00" in name:
        raise WorkspacePathError("filename must not contain null bytes")
    if name.startswith(".."):
        raise WorkspacePathError("filename must not start with '..'")
    return name


@dataclass
class SessionWorkspace:
    session_id: str
    root: Path

    @classmethod
    def for_session(cls, session_id: str, workspace_dir: Path) -> "SessionWorkspace":
        root = Path(workspace_dir) / filesystem_key(session_id)
        return cls(session_id=session_id, root=root)

    def open(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._write_meta()
        logger.info("Session workspace opened: %s", self.root)

    def close(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
            logger.info("Session workspace closed and removed: %s", self.root)

    @property
    def is_open(self) -> bool:
        return self.root.exists()

    def copy_file(self, src: Path, filename: Optional[str] = None) -> Path:
        target_name = _safe_filename(filename or src.name)
        dest = self.root / target_name
        shutil.copy2(src, dest)
        logger.debug("Workspace: copied %s -> %s", src, dest)
        return dest

    def write_text(self, filename: str, content: str, encoding: str = "utf-8") -> Path:
        target_name = _safe_filename(filename)
        dest = self.root / target_name
        dest.write_text(content, encoding=encoding)
        logger.debug("Workspace: wrote text file %s (%d chars)", dest, len(content))
        return dest

    def read_text(self, filename: str, encoding: str = "utf-8") -> str:
        target_name = _safe_filename(filename)
        path = self.root / target_name
        if not path.exists():
            raise FileNotFoundError(f"Workspace file not found: {filename!r}")
        size = path.stat().st_size
        if size > _MAX_READ_BYTES:
            logger.warning(
                "Workspace file %s is %.1f MiB; truncating to %.1f MiB",
                filename,
                size / 1e6,
                _MAX_READ_BYTES / 1e6,
            )
            raw = path.read_bytes()[:_MAX_READ_BYTES]
            return raw.decode(encoding, errors="replace") + "\n...[truncated]"
        return path.read_text(encoding=encoding)

    def read_bytes(self, filename: str) -> bytes:
        target_name = _safe_filename(filename)
        path = self.root / target_name
        if not path.exists():
            raise FileNotFoundError(f"Workspace file not found: {filename!r}")
        return path.read_bytes()

    def exists(self, filename: str) -> bool:
        try:
            target_name = _safe_filename(filename)
        except WorkspacePathError:
            return False
        return (self.root / target_name).exists()

    def list_files(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted(
            item.name
            for item in self.root.iterdir()
            if item.is_file() and not item.name.startswith(".meta")
        )

    def delete_file(self, filename: str) -> bool:
        target_name = _safe_filename(filename)
        path = self.root / target_name
        if path.exists():
            path.unlink()
            logger.debug("Workspace: deleted %s", path)
            return True
        return False

    def _write_meta(self) -> None:
        meta = {
            "session_id": self.session_id,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.root / ".meta").write_text(json.dumps(meta), encoding="utf-8")
