from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx

_LEGACY_SANDBOX_IMAGES = {
    "python:3.12-slim",
    "python:3.12",
}


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True
    hint: str = ""


@dataclass(frozen=True)
class BootstrapAction:
    name: str
    ok: bool
    command: str
    detail: str

    def to_row(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "command": self.command,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PreflightReport:
    checks: List[PreflightCheck] = field(default_factory=list)
    resolved_settings: Dict[str, object] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return all(check.ok or not check.required for check in self.checks)

    def blocking_checks(self) -> List[PreflightCheck]:
        return [check for check in self.checks if check.required and not check.ok]

    def failure_summary(self) -> str:
        blockers = self.blocking_checks()
        if not blockers:
            return "All blocking preflight checks passed."
        parts: List[str] = []
        for check in blockers:
            detail = " ".join(str(check.detail or "").split())
            if check.hint:
                parts.append(f"{check.name}: {detail} Hint: {check.hint}")
            else:
                parts.append(f"{check.name}: {detail}")
        return " | ".join(parts)

    def to_rows(self) -> List[Dict[str, object]]:
        return [
            {
                "name": check.name,
                "ok": check.ok,
                "required": check.required,
                "detail": check.detail,
                "hint": check.hint,
            }
            for check in self.checks
        ]


def run_preflight(
    *,
    repo_root: Path,
    runtime_root: Path,
    workspace_root: Path,
    memory_root: Path,
) -> PreflightReport:
    settings = _load_runtime_settings(repo_root)
    checks: List[PreflightCheck] = []
    checks.extend(
        [
            _check_directory(runtime_root, "runtime_root"),
            _check_directory(workspace_root, "workspace_root"),
            _check_directory(memory_root, "memory_root"),
            _check_database(settings, repo_root=repo_root),
            _check_docker(),
            _check_sandbox_image(settings),
        ]
    )
    checks.extend(_check_providers(settings, repo_root=repo_root))
    return PreflightReport(checks=checks, resolved_settings=_resolved_settings_summary(settings))


def bootstrap_local_dependencies(*, repo_root: Path, report: PreflightReport) -> List[BootstrapAction]:
    if report.ready:
        return []

    settings = _load_runtime_settings(repo_root)
    rows = {check.name: check for check in report.checks}
    docker_ok = rows.get("docker").ok if rows.get("docker") else False
    if not docker_ok:
        return []

    actions: List[BootstrapAction] = []

    database_check = rows.get("database")
    if database_check and not database_check.ok and _is_local_database(settings.pg_dsn):
        actions.append(_bootstrap_service(repo_root, service="rag-postgres", wait=_wait_for_database, settings=settings))

    ollama_failed = any(
        not rows.get(role_name, PreflightCheck(name=role_name, ok=True, detail="")).ok
        for role_name, provider in {
            "chat_provider": settings.llm_provider,
            "embeddings_provider": settings.embeddings_provider,
            "judge_provider": settings.judge_provider,
        }.items()
        if provider == "ollama"
    )
    if ollama_failed and _is_local_http_url(settings.ollama_base_url):
        actions.append(
            _bootstrap_service(
                repo_root,
                service="ollama",
                compose_args=["--profile", "ollama", "up", "-d", "ollama"],
                wait=_wait_for_ollama,
                settings=settings,
            )
        )

    sandbox_check = rows.get("sandbox_image")
    if sandbox_check and not sandbox_check.ok:
        configured_image = str(getattr(settings, "sandbox_docker_image", "") or "").strip()
        if _is_legacy_sandbox_image(configured_image):
            supported_image = _supported_sandbox_image()
            actions.append(
                BootstrapAction(
                    name="sandbox_image",
                    ok=False,
                    command=f"Update .env: SANDBOX_DOCKER_IMAGE={supported_image}",
                    detail=(
                        f"Configured sandbox image {configured_image!r} is a legacy/stale value. "
                        f"Update .env to SANDBOX_DOCKER_IMAGE={supported_image} and then run "
                        "`python run.py build-sandbox-image`. Notebook bootstrap will not "
                        "silently build or accept the legacy image."
                    ),
                )
            )
        else:
            _ensure_repo_import_roots()
            from agentic_chatbot_next.sandbox import build_sandbox_image

            build_result = build_sandbox_image(repo_root, image=settings.sandbox_docker_image)
            actions.append(
                BootstrapAction(
                    name="sandbox_image",
                    ok=build_result.ok,
                    command=build_result.command,
                    detail=build_result.detail,
                )
            )

    return actions


def _check_directory(path: Path, name: str) -> PreflightCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return PreflightCheck(name=name, ok=True, detail=str(path.resolve()))
    except Exception as exc:
        return PreflightCheck(name=name, ok=False, detail=f"{path}: {exc}")


def _check_database(settings: Any, *, repo_root: Path) -> PreflightCheck:
    parsed = urlparse(settings.pg_dsn)
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 5432)
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return PreflightCheck(
                name="database",
                ok=True,
                detail=f"Configured via PG_DSN={settings.pg_dsn} -> {host}:{port} reachable",
            )
    except Exception as exc:
        detail = f"Configured via PG_DSN={settings.pg_dsn} -> {host}:{port} unreachable ({exc})"
        if _is_local_host(host):
            service_hint = _docker_compose_service_status(repo_root, "rag-postgres")
            if service_hint:
                detail = f"{detail}; {service_hint}"
            langfuse_hint = _langfuse_postgres_hint()
            if langfuse_hint:
                detail = f"{detail}; {langfuse_hint}"
        return PreflightCheck(
            name="database",
            ok=False,
            detail=detail,
            hint="Start the app database with `docker compose up -d rag-postgres` or point PG_DSN at a reachable Postgres instance.",
        )


def _check_docker() -> PreflightCheck:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return PreflightCheck(
            name="docker",
            ok=False,
            detail="docker binary not found in PATH",
            hint="Install Docker Desktop or place the docker CLI on PATH before running the notebook.",
        )
    try:
        proc = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return PreflightCheck(
            name="docker",
            ok=False,
            detail=f"docker info failed ({exc})",
            hint="Start Docker Desktop or the Docker daemon before running the notebook.",
        )
    if proc.returncode == 0:
        return PreflightCheck(name="docker", ok=True, detail="docker daemon reachable")
    stderr = (proc.stderr or proc.stdout or "").strip()[:240]
    return PreflightCheck(
        name="docker",
        ok=False,
        detail=f"docker daemon unavailable ({stderr})",
        hint="Start Docker Desktop or the Docker daemon before running the notebook.",
    )


def _check_sandbox_image(settings: Any) -> PreflightCheck:
    configured_image = str(getattr(settings, "sandbox_docker_image", "") or "").strip()
    if _is_legacy_sandbox_image(configured_image):
        supported_image = _supported_sandbox_image()
        return PreflightCheck(
            name="sandbox_image",
            ok=False,
            detail=(
                f"Configured SANDBOX_DOCKER_IMAGE={configured_image!r} is a legacy/stale sandbox "
                f"image setting. The current notebook/runtime expects the prebuilt offline analyst "
                f"image {supported_image!r}, which already contains pandas, numpy, openpyxl, xlrd, "
                "matplotlib, and pillow."
            ),
            hint=(
                f"Update .env to SANDBOX_DOCKER_IMAGE={supported_image} and run "
                "`python run.py build-sandbox-image`, then rerun the notebook preflight."
            ),
        )

    _ensure_repo_import_roots()
    from agentic_chatbot_next.sandbox import probe_sandbox_image

    probe = probe_sandbox_image(configured_image)
    return PreflightCheck(
        name="sandbox_image",
        ok=probe.ok,
        detail=probe.detail,
        hint=probe.remediation,
    )


def _check_providers(settings: Any, *, repo_root: Path) -> List[PreflightCheck]:
    providers = {
        "chat_provider": settings.llm_provider,
        "embeddings_provider": settings.embeddings_provider,
        "judge_provider": settings.judge_provider,
    }
    checks: List[PreflightCheck] = []
    checks.extend(_check_provider_role(name, provider, settings=settings, repo_root=repo_root) for name, provider in providers.items())
    if "ollama" in providers.values():
        checks.extend(_check_ollama_models(providers, settings=settings))
    return checks


def _check_provider_role(role_name: str, provider: str, *, settings: Any, repo_root: Path) -> PreflightCheck:
    if provider == "ollama":
        base_url = settings.ollama_base_url
        try:
            response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
            if response.status_code == 200:
                return PreflightCheck(name=role_name, ok=True, detail=f"{provider} reachable at {base_url}")
            return PreflightCheck(
                name=role_name,
                ok=False,
                detail=f"{provider} returned HTTP {response.status_code} from {base_url}",
                hint="Ensure Ollama is reachable and the notebook is using the same OLLAMA_BASE_URL as the runtime.",
            )
        except Exception as exc:
            detail = f"{provider} unreachable at {base_url} ({exc})"
            if _is_local_http_url(base_url):
                service_hint = _docker_compose_service_status(repo_root, "ollama")
                if service_hint:
                    detail = f"{detail}; {service_hint}"
            return PreflightCheck(
                name=role_name,
                ok=False,
                detail=detail,
                hint="Start local Ollama or run `docker compose --profile ollama up -d ollama`.",
            )

    if provider == "azure":
        required = [
            ("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key),
            ("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint),
        ]
    elif provider == "nvidia":
        required = [
            ("NVIDIA_API_TOKEN", settings.nvidia_api_token),
            ("NVIDIA_OPENAI_ENDPOINT", settings.nvidia_openai_endpoint),
        ]
    else:
        return PreflightCheck(name=role_name, ok=False, detail=f"Unsupported provider {provider!r} for notebook preflight")

    missing = [name for name, value in required if not value]
    if missing:
        return PreflightCheck(
            name=role_name,
            ok=False,
            detail=f"{provider} missing env vars: {', '.join(missing)}",
            hint=f"Set the required {provider} environment variables before running the notebook.",
        )
    return PreflightCheck(name=role_name, ok=True, detail=f"{provider} configuration present")


def _check_ollama_models(providers: Dict[str, str], *, settings: Any) -> List[PreflightCheck]:
    base_url = settings.ollama_base_url.rstrip("/")
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        payload = dict(response.json())
        models = {
            str(item.get("name") or "")
            for item in (payload.get("models") or [])
            if isinstance(item, dict)
        }
    except Exception as exc:
        return [
            PreflightCheck(
                name="ollama_models",
                ok=False,
                detail=f"could not inspect ollama models ({exc})",
                hint="Confirm the Ollama API is reachable before relying on local model checks.",
            )
        ]

    checks: List[PreflightCheck] = []
    required_model_map = {
        "chat_provider": settings.ollama_chat_model,
        "judge_provider": settings.ollama_judge_model,
        "embeddings_provider": settings.ollama_embed_model,
    }
    for role_name, provider in providers.items():
        if provider != "ollama":
            continue
        model_name = required_model_map[role_name]
        model_available = _ollama_model_present(model_name, models)
        checks.append(
            PreflightCheck(
                name=f"{role_name}_model",
                ok=model_available,
                detail=f"{model_name} {'available' if model_available else 'missing'} at {base_url}",
                hint="" if model_available else f"Pull the model with `ollama pull {model_name}` or update the runtime model setting.",
            )
        )
    return checks


def _ollama_model_present(model_name: str, available_models: set[str]) -> bool:
    return any(candidate in available_models for candidate in _ollama_model_aliases(model_name))


def _ollama_model_aliases(model_name: str) -> set[str]:
    normalized = model_name.strip()
    if not normalized:
        return set()
    aliases = {normalized}
    if ":" in normalized:
        base_name, tag = normalized.rsplit(":", 1)
        if tag == "latest":
            aliases.add(base_name)
    else:
        aliases.add(f"{normalized}:latest")
    return aliases


def _ensure_repo_import_roots() -> None:
    package_root = Path(__file__).resolve().parents[2]
    src_root = package_root / "src"
    for candidate in (str(package_root), str(src_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _load_runtime_settings(repo_root: Path) -> Any:
    _ensure_repo_import_roots()
    from dotenv import dotenv_values
    from agentic_chatbot_next.config import load_settings

    dotenv_path = repo_root / ".env"
    overlay_dir = repo_root / "data" / "control_panel" / "overlays"
    scoped_env = {
        "DATA_DIR": str(repo_root / "data"),
        "CONTROL_PANEL_OVERLAY_DIR": str(overlay_dir),
        "CONTROL_PANEL_RUNTIME_ENV_PATH": str(overlay_dir / "runtime.env"),
    }
    dotenv_overrides = {
        str(name): str(value)
        for name, value in dotenv_values(dotenv_path).items()
        if name and value is not None
    }
    original_env = dict(os.environ)
    try:
        for name, value in scoped_env.items():
            os.environ[name] = value
        return load_settings(
            dotenv_path=str(dotenv_path),
            env_overrides={**dotenv_overrides, **scoped_env},
        )
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def _resolved_settings_summary(settings: Any) -> Dict[str, object]:
    return {
        "pg_dsn": settings.pg_dsn,
        "llm_provider": settings.llm_provider,
        "embeddings_provider": settings.embeddings_provider,
        "judge_provider": settings.judge_provider,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_chat_model": settings.ollama_chat_model,
        "ollama_embed_model": settings.ollama_embed_model,
        "ollama_judge_model": settings.ollama_judge_model,
        "memory_enabled": bool(getattr(settings, "memory_enabled", True)),
        "sandbox_docker_image": settings.sandbox_docker_image,
    }


def _is_legacy_sandbox_image(image: str) -> bool:
    return str(image or "").strip().lower() in _LEGACY_SANDBOX_IMAGES


def _supported_sandbox_image() -> str:
    _ensure_repo_import_roots()
    try:
        from agentic_chatbot_next.sandbox.images import DEFAULT_SANDBOX_IMAGE

        return str(DEFAULT_SANDBOX_IMAGE)
    except Exception:
        return "agentic-chatbot-sandbox:py312"


def _is_local_host(host: str) -> bool:
    return host.strip().lower() in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _is_local_database(dsn: str) -> bool:
    parsed = urlparse(dsn)
    return _is_local_host(parsed.hostname or "localhost")


def _is_local_http_url(raw_url: str) -> bool:
    parsed = urlparse(raw_url)
    return _is_local_host(parsed.hostname or "localhost")


def _docker_compose_service_status(repo_root: Path, service: str) -> str:
    docker_bin = shutil.which("docker")
    compose_file = repo_root / "docker-compose.yml"
    if not docker_bin or not compose_file.exists():
        return ""
    try:
        proc = subprocess.run(
            [docker_bin, "compose", "ps", "--services", "--status", "running", service],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return ""

    running = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    if service in running:
        return f"docker compose service `{service}` is already running"
    return f"docker compose service `{service}` is not running"


def _langfuse_postgres_hint() -> str:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return ""
    try:
        proc = subprocess.run(
            [docker_bin, "ps", "--format", "{{.Names}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return ""

    for line in proc.stdout.splitlines():
        if "langfuse-postgres" in line and ":5433->5432" in line:
            return "langfuse-postgres is listening on localhost:5433, but that is the observability database rather than the app DB"
    return ""


def _bootstrap_service(
    repo_root: Path,
    *,
    service: str,
    settings: Any,
    wait,
    compose_args: List[str] | None = None,
) -> BootstrapAction:
    docker_bin = shutil.which("docker")
    command_parts = [docker_bin or "docker", "compose", *(compose_args or ["up", "-d", service])]
    command_text = " ".join(command_parts)
    if not docker_bin:
        return BootstrapAction(
            name=service,
            ok=False,
            command=command_text,
            detail="Docker CLI is not available, so the notebook could not bootstrap this dependency.",
        )

    try:
        proc = subprocess.run(
            command_parts,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:
        return BootstrapAction(
            name=service,
            ok=False,
            command=command_text,
            detail=f"Bootstrap command failed before completion ({exc}).",
        )

    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()[:600]
        return BootstrapAction(
            name=service,
            ok=False,
            command=command_text,
            detail=f"Bootstrap command exited with code {proc.returncode}: {stderr}",
        )

    wait_ok, wait_detail = wait(settings)
    return BootstrapAction(name=service, ok=wait_ok, command=command_text, detail=wait_detail)


def _wait_for_database(settings: Any, *, timeout_seconds: float = 60.0) -> tuple[bool, str]:
    parsed = urlparse(settings.pg_dsn)
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 5432)
    return _wait_for_socket(host=host, port=port, timeout_seconds=timeout_seconds)


def _wait_for_ollama(settings: Any, *, timeout_seconds: float = 60.0) -> tuple[bool, str]:
    request_timeout = httpx.Timeout(5.0)
    base_url = settings.ollama_base_url.rstrip("/")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=request_timeout)
            if response.status_code == 200:
                return True, f"Ollama became reachable at {base_url}"
        except Exception:
            pass
        time.sleep(1.0)
    return False, f"Ollama did not become reachable at {base_url} within {int(timeout_seconds)} seconds"


def _wait_for_socket(*, host: str, port: int, timeout_seconds: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True, f"{host}:{port} became reachable"
        except Exception:
            time.sleep(1.0)
    return False, f"{host}:{port} did not become reachable within {int(timeout_seconds)} seconds"
