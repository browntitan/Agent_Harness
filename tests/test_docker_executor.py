"""Unit tests for DockerSandboxExecutor.

All Docker SDK calls are mocked — no real Docker daemon is required to run
these tests.
"""
from __future__ import annotations

import io
import tarfile
from unittest.mock import MagicMock, patch, call

import pytest

from agentic_chatbot_next.sandbox.exceptions import SandboxUnavailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor(**kwargs):
    """Create a DockerSandboxExecutor with test-friendly defaults."""
    from agentic_chatbot_next.sandbox.docker_exec import DockerSandboxExecutor
    return DockerSandboxExecutor(
        image="agentic-chatbot-sandbox:py312",
        timeout_seconds=10,
        memory_limit="256m",
        network_disabled=True,
        **kwargs,
    )


def _make_container_mock(
    exit_code: int = 0,
    stdout: bytes = b"hello",
    stderr: bytes = b"",
):
    """Return a mock Docker container object."""
    container = MagicMock()
    container.wait.return_value = {"StatusCode": exit_code}
    container.logs.side_effect = lambda stdout=True, stderr=False: (
        stdout and b"stdout: " + b"hello" or stderr and b""
    )
    # More precise: match exactly what _extract_output calls
    def logs_side_effect(**kwargs):
        if kwargs.get("stdout") and not kwargs.get("stderr"):
            return stdout
        if kwargs.get("stderr") and not kwargs.get("stdout"):
            return stderr
        return b""
    container.logs.side_effect = logs_side_effect
    return container


# ---------------------------------------------------------------------------
# SandboxUnavailableError when docker not installed
# ---------------------------------------------------------------------------

class TestDockerNotInstalled:
    def test_raises_when_docker_package_missing(self):
        executor = _make_executor()
        with patch.dict("sys.modules", {"docker": None}):
            with pytest.raises(SandboxUnavailableError, match="not installed"):
                executor.execute("print('hello')")


# ---------------------------------------------------------------------------
# SandboxUnavailableError when daemon not running
# ---------------------------------------------------------------------------

class TestDockerDaemonUnavailable:
    def test_raises_when_from_env_fails(self):
        executor = _make_executor()
        mock_docker = MagicMock()
        mock_docker.from_env.side_effect = Exception("Cannot connect to Docker")
        with patch.dict("sys.modules", {"docker": mock_docker}):
            with pytest.raises(SandboxUnavailableError, match="not available"):
                executor.execute("print('hello')")


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

class TestExecuteSimpleCode:
    def test_returns_sandbox_result(self):
        from agentic_chatbot_next.sandbox.docker_exec import DockerSandboxExecutor, SandboxResult

        executor = _make_executor()
        mock_docker = MagicMock()
        container = _make_container_mock(exit_code=0, stdout=b"42\n", stderr=b"")
        mock_docker.from_env.return_value.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = executor.execute("print(42)")

        assert isinstance(result, SandboxResult)
        assert result.exit_code == 0
        assert result.success is True
        assert result.stdout == "42"

    def test_container_is_always_removed(self):
        executor = _make_executor()
        mock_docker = MagicMock()
        container = _make_container_mock(exit_code=0, stdout=b"ok\n")
        mock_docker.from_env.return_value.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            executor.execute("print('ok')")

        container.remove.assert_called_once_with(force=True)

    def test_container_started(self):
        executor = _make_executor()
        mock_docker = MagicMock()
        container = _make_container_mock(exit_code=0, stdout=b"ok\n")
        mock_docker.from_env.return_value.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            executor.execute("print('ok')")

        container.start.assert_called_once()

    def test_network_disabled_in_create(self):
        executor = _make_executor()
        mock_docker = MagicMock()
        mock_client = mock_docker.from_env.return_value
        container = _make_container_mock()
        mock_client.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            executor.execute("print('ok')")

        _, kwargs = mock_client.containers.create.call_args
        assert kwargs.get("network_disabled") is True

    def test_memory_limit_passed(self):
        from agentic_chatbot_next.sandbox.docker_exec import DockerSandboxExecutor

        executor = DockerSandboxExecutor.__new__(DockerSandboxExecutor)
        executor.image = "agentic-chatbot-sandbox:py312"
        executor.timeout_seconds = 10
        executor.memory_limit = "256m"
        executor.network_disabled = True
        executor.work_dir = "/workspace"

        executor2 = _make_executor()
        mock_docker = MagicMock()
        mock_client = mock_docker.from_env.return_value
        container = _make_container_mock()
        mock_client.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            executor2.execute("print('ok')")

        _, kwargs = mock_client.containers.create.call_args
        assert kwargs.get("mem_limit") == "256m"


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------

class TestExecuteWithPackages:
    def test_run_script_does_not_install_packages(self):
        executor = _make_executor()
        script = executor._build_run_script("print('hi')")
        assert "pip install" not in script

    def test_supported_packages_are_allowed(self):
        executor = _make_executor()
        mock_docker = MagicMock()
        mock_client = mock_docker.from_env.return_value
        container = _make_container_mock(stdout=b"ok\n")
        mock_client.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = executor.execute("print('ok')", packages=["pandas", "numpy"])

        assert result.success is True

    def test_unsupported_packages_fail_fast(self):
        executor = _make_executor()

        with pytest.raises(SandboxUnavailableError, match="does not guarantee these additional package\\(s\\): scipy"):
            executor.execute("print('ok')", packages=["scipy"])


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestExecuteTimeout:
    def test_timeout_returns_exit_code_minus_one(self):
        executor = _make_executor()
        mock_docker = MagicMock()
        container = MagicMock()
        # Simulate ReadTimeout-like exception
        container.wait.side_effect = Exception("ReadTimeout")
        container.logs.return_value = b""
        mock_docker.from_env.return_value.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = executor.execute("import time; time.sleep(999)")

        assert result.exit_code == -1
        assert result.success is False
        assert "timed out" in result.stderr.lower()

    def test_container_killed_on_timeout(self):
        executor = _make_executor()
        mock_docker = MagicMock()
        container = MagicMock()
        container.wait.side_effect = Exception("ReadTimeout")
        container.logs.return_value = b""
        mock_docker.from_env.return_value.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            executor.execute("import time; time.sleep(999)")

        container.kill.assert_called_once()


# ---------------------------------------------------------------------------
# File copying
# ---------------------------------------------------------------------------

class TestExecuteWithFiles:
    def test_put_archive_called_for_each_file(self, tmp_path):
        executor = _make_executor()
        mock_docker = MagicMock()
        container = _make_container_mock(stdout=b"ok\n")
        mock_docker.from_env.return_value.containers.create.return_value = container

        # Create real temp files
        f1 = tmp_path / "data.csv"
        f1.write_text("a,b\n1,2\n")
        f2 = tmp_path / "other.xlsx"
        f2.write_bytes(b"fake excel bytes")

        files = {
            "/workspace/data.csv": str(f1),
            "/workspace/other.xlsx": str(f2),
        }

        with patch.dict("sys.modules", {"docker": mock_docker}):
            executor.execute("print('ok')", files=files)

        assert container.put_archive.call_count == 2

    def test_missing_host_file_skipped_gracefully(self, tmp_path):
        executor = _make_executor()
        mock_docker = MagicMock()
        container = _make_container_mock(stdout=b"ok\n")
        mock_docker.from_env.return_value.containers.create.return_value = container

        files = {"/workspace/missing.csv": str(tmp_path / "does_not_exist.csv")}

        with patch.dict("sys.modules", {"docker": mock_docker}):
            # Should not raise
            result = executor.execute("print('ok')", files=files)

        # put_archive should NOT be called for missing files
        container.put_archive.assert_not_called()


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

class TestOutputTruncation:
    def test_large_stdout_is_truncated(self):
        from agentic_chatbot_next.sandbox.docker_exec import _MAX_OUTPUT_BYTES
        executor = _make_executor()
        mock_docker = MagicMock()

        big_stdout = b"x" * (_MAX_OUTPUT_BYTES + 1000)
        container = _make_container_mock(stdout=big_stdout, stderr=b"")
        mock_docker.from_env.return_value.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = executor.execute("print('big')")

        assert result.truncated is True
        assert len(result.stdout.encode()) <= _MAX_OUTPUT_BYTES + 10  # allow for decode overhead

    def test_small_stdout_not_truncated(self):
        executor = _make_executor()
        mock_docker = MagicMock()
        container = _make_container_mock(stdout=b"small output\n", stderr=b"")
        mock_docker.from_env.return_value.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = executor.execute("print('small')")

        assert result.truncated is False


# ---------------------------------------------------------------------------
# Error exit code
# ---------------------------------------------------------------------------

class TestNonZeroExitCode:
    def test_error_exit_code_captured(self):
        executor = _make_executor()
        mock_docker = MagicMock()
        container = _make_container_mock(
            exit_code=1,
            stdout=b"",
            stderr=b"NameError: name 'x' is not defined\n",
        )
        mock_docker.from_env.return_value.containers.create.return_value = container

        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = executor.execute("print(x)")

        assert result.exit_code == 1
        assert result.success is False
        assert "NameError" in result.stderr


# ---------------------------------------------------------------------------
# _build_run_script
# ---------------------------------------------------------------------------

class TestBuildRunScript:
    def test_script_contains_code(self):
        from agentic_chatbot_next.sandbox.docker_exec import DockerSandboxExecutor
        executor = _make_executor()
        code = "import pandas as pd\nprint(pd.__version__)"
        script = executor._build_run_script(code)
        assert "import pandas as pd" in script
        assert "print(pd.__version__)" in script

    def test_script_runs_python(self):
        from agentic_chatbot_next.sandbox.docker_exec import DockerSandboxExecutor
        executor = _make_executor()
        script = executor._build_run_script("print('hi')")
        assert "python" in script

    def test_script_uses_workspace_dir(self):
        from agentic_chatbot_next.sandbox.docker_exec import DockerSandboxExecutor
        executor = _make_executor()
        script = executor._build_run_script("print('hi')")
        assert "/workspace" in script
