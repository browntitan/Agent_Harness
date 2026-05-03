from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_SANDBOX_IMAGE = "agentic-chatbot-sandbox:py312"
DEFAULT_SANDBOX_DOCKERFILE = Path("docker/sandbox.Dockerfile")
SUPPORTED_SANDBOX_PACKAGES = (
    "pandas",
    "numpy",
    "openpyxl",
    "xlrd",
    "matplotlib",
    "pillow",
)
_SANDBOX_IMPORT_PROBE = (
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("openpyxl", "openpyxl"),
    ("xlrd", "xlrd"),
    ("matplotlib", "matplotlib"),
    ("pillow", "PIL"),
)


@dataclass(frozen=True)
class DockerAvailabilityResult:
    ok: bool
    detail: str
    remediation: str = ""


@dataclass(frozen=True)
class SandboxImageProbeResult:
    ok: bool
    image: str
    detail: str
    remediation: str = ""
    command: str = ""


@dataclass(frozen=True)
class SandboxImageBuildResult:
    ok: bool
    image: str
    detail: str
    command: str
    remediation: str = ""


def check_docker_availability(*, timeout_seconds: float = 8.0) -> DockerAvailabilityResult:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return DockerAvailabilityResult(
            ok=False,
            detail="docker binary not found in PATH",
            remediation="Install Docker Desktop or place the docker CLI on PATH before using the data analyst sandbox.",
        )

    try:
        proc = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:
        return DockerAvailabilityResult(
            ok=False,
            detail=f"docker info failed ({exc})",
            remediation="Start Docker Desktop or the Docker daemon before using the data analyst sandbox.",
        )

    if proc.returncode == 0:
        return DockerAvailabilityResult(ok=True, detail="docker daemon reachable")

    stderr = (proc.stderr or proc.stdout or "").strip()[:400]
    return DockerAvailabilityResult(
        ok=False,
        detail=f"docker daemon unavailable ({stderr})",
        remediation="Start Docker Desktop or the Docker daemon before using the data analyst sandbox.",
    )


def probe_sandbox_image(
    image: str,
    *,
    timeout_seconds: float = 30.0,
) -> SandboxImageProbeResult:
    probe_timeout = max(0.1, float(timeout_seconds or 30.0))
    docker_check = check_docker_availability(timeout_seconds=min(8.0, probe_timeout))
    if not docker_check.ok:
        return SandboxImageProbeResult(
            ok=False,
            image=image,
            detail=docker_check.detail,
            remediation=docker_check.remediation,
        )

    docker_bin = shutil.which("docker") or "docker"
    inspect_command = [docker_bin, "image", "inspect", image]
    inspect_text = _command_text(inspect_command)
    try:
        inspect_proc = subprocess.run(
            inspect_command,
            capture_output=True,
            text=True,
            timeout=min(10.0, probe_timeout),
            check=False,
        )
    except Exception as exc:
        return SandboxImageProbeResult(
            ok=False,
            image=image,
            detail=f"Could not inspect sandbox image {image!r} ({exc}).",
            remediation=_build_remediation(image),
            command=inspect_text,
        )

    if inspect_proc.returncode != 0:
        stderr = (inspect_proc.stderr or inspect_proc.stdout or "").strip()[:500]
        return SandboxImageProbeResult(
            ok=False,
            image=image,
            detail=f"Sandbox image {image!r} is not available locally. {stderr}".strip(),
            remediation=_build_remediation(image),
            command=inspect_text,
        )

    probe_script = _build_import_probe_script()
    probe_command = [
        docker_bin,
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "python",
        image,
        "-c",
        probe_script,
    ]
    probe_text = _command_text(probe_command)
    try:
        probe_proc = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            timeout=probe_timeout,
            check=False,
        )
    except Exception as exc:
        return SandboxImageProbeResult(
            ok=False,
            image=image,
            detail=f"Sandbox image import probe failed for {image!r} ({exc}).",
            remediation=_build_remediation(image),
            command=probe_text,
        )

    if probe_proc.returncode != 0:
        details = (probe_proc.stderr or probe_proc.stdout or "").strip()[:700]
        return SandboxImageProbeResult(
            ok=False,
            image=image,
            detail=f"Sandbox image {image!r} failed the offline import probe: {details}",
            remediation=_build_remediation(image),
            command=probe_text,
        )

    imports_text = ", ".join(label for label, _module in _SANDBOX_IMPORT_PROBE)
    return SandboxImageProbeResult(
        ok=True,
        image=image,
        detail=f"Sandbox image {image!r} is present locally and imports {imports_text} with --network none.",
        command=probe_text,
    )


def build_sandbox_image(
    repo_root: Path,
    *,
    image: str = DEFAULT_SANDBOX_IMAGE,
    dockerfile_path: Path | None = None,
    timeout_seconds: float = 900.0,
    verify_timeout_seconds: float = 30.0,
) -> SandboxImageBuildResult:
    docker_check = check_docker_availability()
    if not docker_check.ok:
        return SandboxImageBuildResult(
            ok=False,
            image=image,
            detail=docker_check.detail,
            remediation=docker_check.remediation,
            command="docker build",
        )

    docker_bin = shutil.which("docker") or "docker"
    dockerfile = (dockerfile_path or DEFAULT_SANDBOX_DOCKERFILE)
    dockerfile_abs = dockerfile if dockerfile.is_absolute() else (repo_root / dockerfile)
    if not dockerfile_abs.exists():
        return SandboxImageBuildResult(
            ok=False,
            image=image,
            detail=f"Sandbox Dockerfile not found at {dockerfile_abs}.",
            remediation="Restore the repo-tracked sandbox Dockerfile before building the analyst sandbox image.",
            command="docker build",
        )

    build_command = [
        docker_bin,
        "build",
        "-f",
        str(dockerfile_abs),
        "-t",
        image,
        ".",
    ]
    command_text = _command_text(build_command)
    try:
        proc = subprocess.run(
            build_command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:
        return SandboxImageBuildResult(
            ok=False,
            image=image,
            detail=f"Sandbox image build failed before completion ({exc}).",
            remediation=_build_remediation(image),
            command=command_text,
        )

    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()[-900:]
        return SandboxImageBuildResult(
            ok=False,
            image=image,
            detail=f"Sandbox image build exited with code {proc.returncode}: {details}",
            remediation=_build_remediation(image),
            command=command_text,
        )

    probe = probe_sandbox_image(image, timeout_seconds=verify_timeout_seconds)
    if not probe.ok:
        return SandboxImageBuildResult(
            ok=False,
            image=image,
            detail=f"Sandbox image built but did not pass readiness checks. {probe.detail}",
            remediation=probe.remediation or _build_remediation(image),
            command=command_text,
        )

    return SandboxImageBuildResult(
        ok=True,
        image=image,
        detail=f"Built sandbox image {image!r} and verified analyst imports with networking disabled.",
        command=command_text,
    )


def unsupported_sandbox_packages(packages: Iterable[str] | None) -> list[str]:
    normalized = []
    for raw in packages or ():
        package = str(raw or "").strip().lower()
        if package and package not in normalized:
            normalized.append(package)
    return [package for package in normalized if package not in SUPPORTED_SANDBOX_PACKAGES]


def _build_import_probe_script() -> str:
    lines = [
        "import importlib",
        "import sys",
        f"required = {[(label, module) for label, module in _SANDBOX_IMPORT_PROBE]!r}",
        "missing = []",
        "for label, module in required:",
        "    try:",
        "        importlib.import_module(module)",
        "    except Exception as exc:",
        "        missing.append(f\"{label}: {exc.__class__.__name__}: {exc}\")",
        "if missing:",
        "    print('; '.join(missing))",
        "    raise SystemExit(1)",
        "print('sandbox-ready')",
    ]
    return "\n".join(lines)


def _build_remediation(image: str) -> str:
    return (
        f"Run `python run.py build-sandbox-image --image {image}` or update SANDBOX_DOCKER_IMAGE "
        "to a compatible prebuilt analyst sandbox image."
    )


def _command_text(command: list[str]) -> str:
    return " ".join(command)
