from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_next_docker_executor_resolves_workspace_path_to_absolute(monkeypatch, tmp_path: Path):
    from agentic_chatbot_next.sandbox.docker_exec import DockerSandboxExecutor

    recorded = {}

    class _Container:
        def start(self):
            return None

        def wait(self, timeout=None):
            del timeout
            return {"StatusCode": 0}

        def logs(self, stdout=True, stderr=False):
            del stdout, stderr
            return b""

        def remove(self, force=False):
            del force
            return None

    class _Containers:
        def create(self, image, command, network_disabled, mem_limit, working_dir, detach, auto_remove, volumes):
            del image, command, network_disabled, mem_limit, working_dir, detach, auto_remove
            recorded["volumes"] = volumes
            return _Container()

    class _Client:
        def __init__(self):
            self.containers = _Containers()

    fake_docker = SimpleNamespace(from_env=lambda: _Client())
    monkeypatch.setitem(__import__("sys").modules, "docker", fake_docker)

    executor = DockerSandboxExecutor()
    relative_workspace = Path("tmp") / "relative-workspace"
    result = executor.execute("print('hello')", workspace_path=relative_workspace)

    assert result.success is True
    assert recorded["volumes"] is not None
    host_path = next(iter(recorded["volumes"].keys()))
    assert Path(host_path).is_absolute()
    assert host_path == str(relative_workspace.resolve())

