from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from agentic_chatbot_next.sandbox.images import (
    DEFAULT_SANDBOX_IMAGE,
    build_sandbox_image,
    check_docker_availability,
    probe_sandbox_image,
)


def test_check_docker_availability_reports_missing_cli(monkeypatch):
    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.shutil.which", lambda name: None)

    result = check_docker_availability()

    assert result.ok is False
    assert "docker binary not found" in result.detail


def test_probe_sandbox_image_fails_when_image_missing(monkeypatch):
    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.check_docker_availability", lambda timeout_seconds=8.0: type("R", (), {"ok": True, "detail": "ok", "remediation": ""})())
    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.shutil.which", lambda name: "/usr/local/bin/docker")

    def fake_run(command, **kwargs):
        return CompletedProcess(command, 1, "", "No such image")

    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.subprocess.run", fake_run)

    result = probe_sandbox_image(DEFAULT_SANDBOX_IMAGE)

    assert result.ok is False
    assert DEFAULT_SANDBOX_IMAGE in result.detail
    assert "build-sandbox-image" in result.remediation


def test_probe_sandbox_image_fails_when_offline_import_probe_fails(monkeypatch):
    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.check_docker_availability", lambda timeout_seconds=8.0: type("R", (), {"ok": True, "detail": "ok", "remediation": ""})())
    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.shutil.which", lambda name: "/usr/local/bin/docker")
    calls = {"count": 0}

    def fake_run(command, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return CompletedProcess(command, 0, "[]", "")
        return CompletedProcess(command, 1, "", "pandas: ModuleNotFoundError: No module named 'pandas'")

    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.subprocess.run", fake_run)

    result = probe_sandbox_image(DEFAULT_SANDBOX_IMAGE)

    assert result.ok is False
    assert "failed the offline import probe" in result.detail
    assert "pandas" in result.detail


def test_build_sandbox_image_builds_then_verifies(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.check_docker_availability", lambda timeout_seconds=8.0: type("R", (), {"ok": True, "detail": "ok", "remediation": ""})())
    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.shutil.which", lambda name: "/usr/local/bin/docker")
    monkeypatch.setattr(
        "agentic_chatbot_next.sandbox.images.probe_sandbox_image",
        lambda image, timeout_seconds=30.0: type("R", (), {"ok": True, "detail": "probe ok", "remediation": "", "command": "docker run"})(),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 0, "build ok", "")

    monkeypatch.setattr("agentic_chatbot_next.sandbox.images.subprocess.run", fake_run)
    dockerfile = tmp_path / "docker" / "sandbox.Dockerfile"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text("FROM python:3.12-slim\n", encoding="utf-8")

    result = build_sandbox_image(tmp_path, image=DEFAULT_SANDBOX_IMAGE, dockerfile_path=dockerfile)

    assert result.ok is True
    assert "Built sandbox image" in result.detail
    assert calls
