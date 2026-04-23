"""Docker-based sandbox executor for the next runtime."""
from __future__ import annotations

import io
import logging
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from agentic_chatbot_next.sandbox.exceptions import SandboxUnavailableError
from agentic_chatbot_next.sandbox.images import DEFAULT_SANDBOX_IMAGE, SUPPORTED_SANDBOX_PACKAGES, unsupported_sandbox_packages

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 50 * 1024


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    execution_time_seconds: float
    truncated: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class DockerSandboxExecutor:
    def __init__(
        self,
        image: str = DEFAULT_SANDBOX_IMAGE,
        timeout_seconds: int = 60,
        memory_limit: str = "512m",
        network_disabled: bool = True,
        work_dir: str = "/workspace",
    ) -> None:
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.network_disabled = network_disabled
        self.work_dir = work_dir

    def execute(
        self,
        code: str,
        files: Optional[Dict[str, str]] = None,
        packages: Optional[List[str]] = None,
        workspace_path: Optional[Path] = None,
    ) -> SandboxResult:
        unsupported = unsupported_sandbox_packages(packages)
        if unsupported:
            supported = ", ".join(SUPPORTED_SANDBOX_PACKAGES)
            requested = ", ".join(unsupported)
            raise SandboxUnavailableError(
                "The configured sandbox image "
                f"'{self.image}' does not guarantee these additional package(s): {requested}. "
                "Rebuild the analyst sandbox image with those dependencies or point SANDBOX_DOCKER_IMAGE "
                f"to a compatible image. Supported preinstalled packages: {supported}."
            )

        try:
            import docker  # noqa: PLC0415
        except ImportError as exc:
            raise SandboxUnavailableError(
                "The 'docker' package is not installed. Run: pip install docker>=7.0.0"
            ) from exc

        try:
            client = docker.from_env()
        except Exception as exc:
            raise SandboxUnavailableError(
                f"Docker is not available: {exc}. Ensure Docker is running."
            ) from exc

        run_script = self._build_run_script(code)
        volumes: Optional[Dict[str, Dict[str, str]]] = None
        if workspace_path is not None:
            workspace_path = workspace_path.resolve()
            workspace_path.mkdir(parents=True, exist_ok=True)
            volumes = {str(workspace_path): {"bind": self.work_dir, "mode": "rw"}}
            logger.debug("Sandbox: bind-mounting workspace %s -> %s", workspace_path, self.work_dir)

        container = None
        start_time = time.monotonic()
        try:
            container = client.containers.create(
                self.image,
                command=["/bin/sh", "-lc", run_script],
                network_disabled=self.network_disabled,
                mem_limit=self.memory_limit,
                working_dir=self.work_dir,
                detach=True,
                auto_remove=False,
                volumes=volumes,
            )
            if files and workspace_path is None:
                self._copy_files_to_container(container, files)

            container.start()
            try:
                container.wait(timeout=self.timeout_seconds)
            except Exception as exc:
                err_msg = str(exc)
                if "timeout" in err_msg.lower() or "ReadTimeout" in type(exc).__name__:
                    logger.warning("Sandbox container timed out after %ds", self.timeout_seconds)
                    try:
                        container.kill()
                    except Exception:
                        pass
                    elapsed = time.monotonic() - start_time
                    return SandboxResult(
                        stdout="",
                        stderr=f"Execution timed out after {self.timeout_seconds}s.",
                        exit_code=-1,
                        execution_time_seconds=elapsed,
                    )
                raise

            elapsed = time.monotonic() - start_time
            return self._extract_output(container, elapsed)
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _build_run_script(self, code: str) -> str:
        return (
            f"mkdir -p {self.work_dir}\n"
            f"cat << 'PYTHON_EOF' > {self.work_dir}/script.py\n"
            f"{code}\n"
            "PYTHON_EOF\n"
            f"python {self.work_dir}/script.py\n"
        )

    def _copy_files_to_container(self, container, files: Dict[str, str]) -> None:
        for container_path, host_path in files.items():
            host_p = Path(host_path)
            if not host_p.exists():
                logger.warning("Sandbox file not found, skipping: %s", host_path)
                continue

            container_p = Path(container_path)
            container_dir = str(container_p.parent)
            filename = container_p.name
            file_data = host_p.read_bytes()
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tf:
                info = tarfile.TarInfo(name=filename)
                info.size = len(file_data)
                tf.addfile(info, io.BytesIO(file_data))
            buf.seek(0)
            try:
                container.put_archive(container_dir, buf.read())
            except Exception as exc:
                logger.warning("Failed to copy %s into container: %s", host_path, exc)

    def _extract_output(self, container, elapsed: float) -> SandboxResult:
        truncated = False
        try:
            raw_stdout = container.logs(stdout=True, stderr=False)
            raw_stderr = container.logs(stdout=False, stderr=True)
        except Exception as exc:
            logger.warning("Could not read container logs: %s", exc)
            raw_stdout = b""
            raw_stderr = b""

        stdout_bytes = raw_stdout if isinstance(raw_stdout, bytes) else raw_stdout.encode()
        stderr_bytes = raw_stderr if isinstance(raw_stderr, bytes) else raw_stderr.encode()

        if len(stdout_bytes) > _MAX_OUTPUT_BYTES:
            stdout_bytes = stdout_bytes[:_MAX_OUTPUT_BYTES]
            truncated = True
        if len(stderr_bytes) > _MAX_OUTPUT_BYTES:
            stderr_bytes = stderr_bytes[:_MAX_OUTPUT_BYTES]
            truncated = True

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        try:
            inspect = container.wait(timeout=5)
            exit_code = inspect.get("StatusCode", 0)
        except Exception:
            exit_code = 0

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time_seconds=elapsed,
            truncated=truncated,
        )
