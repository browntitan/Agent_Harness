"""Sandbox-specific exceptions."""
from __future__ import annotations


class SandboxUnavailableError(RuntimeError):
    """Raised when Docker is not available or not responding on the host."""
