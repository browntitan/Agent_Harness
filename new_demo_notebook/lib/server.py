from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


@dataclass
class BackendServerManager:
    repo_root: Path
    artifacts_dir: Path
    host: str = "127.0.0.1"
    port: int = 0
    readiness_path: str = "/health/ready"
    python_executable: str = field(default_factory=lambda: sys.executable)
    _process: Optional[subprocess.Popen[str]] = field(default=None, init=False, repr=False)
    _log_handle: Optional[object] = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        if not self.port:
            raise RuntimeError("Server port is not set yet.")
        return f"http://{self.host}:{self.port}"

    @property
    def log_path(self) -> Path:
        return self.artifacts_dir / "server.log"

    def __enter__(self) -> "BackendServerManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(force=True)

    def _resolve_ready_timeout(self, timeout_seconds: Optional[float]) -> float:
        if timeout_seconds is not None:
            return float(timeout_seconds)
        raw_timeout = os.getenv("NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS")
        if raw_timeout in (None, ""):
            return 90.0
        try:
            return float(raw_timeout)
        except ValueError:
            return 90.0

    def start(self, *, timeout_seconds: Optional[float] = None) -> str:
        if self._process is not None and self._process.poll() is None:
            return self.base_url
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.port = self.port or find_free_port(self.host)
        effective_timeout = self._resolve_ready_timeout(timeout_seconds)
        command = [
            self.python_executable,
            "run.py",
            "serve-api",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            self.wait_until_ready(timeout_seconds=effective_timeout)
        except Exception:
            self.stop(force=True)
            raise RuntimeError(
                "Backend server failed to become ready.\n\n"
                f"Log tail:\n{self.read_log_tail()}"
            )
        return self.base_url

    def wait_until_ready(self, *, timeout_seconds: Optional[float] = None) -> None:
        effective_timeout = self._resolve_ready_timeout(timeout_seconds)
        deadline = time.monotonic() + effective_timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"Backend server exited early with code {self._process.poll()}.")
            try:
                response = httpx.get(f"{self.base_url}{self.readiness_path}", timeout=2.0)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"Timed out waiting for backend readiness at {self.base_url}{self.readiness_path}.")

    def read_log_tail(self, *, max_chars: int = 4000) -> str:
        if not self.log_path.exists():
            return ""
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return text if len(text) <= max_chars else text[-max_chars:]

    def stop(self, *, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0 if not force else 1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
