from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


OPENWEBUI_BASE_URL = os.environ.get("OPENWEBUI_BASE_URL", "http://openwebui:8080").rstrip("/")
ADMIN_EMAIL = os.environ.get("OPENWEBUI_ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.environ.get("OPENWEBUI_ADMIN_PASSWORD", "change-me-openwebui-admin")
ADMIN_NAME = os.environ.get("OPENWEBUI_ADMIN_NAME", "Open WebUI Admin")
PIPE_FILE = Path(os.environ.get("OPENWEBUI_PIPE_FILE", "/config/enterprise_agent_pipe.py"))
PIPE_ID = os.environ.get("OPENWEBUI_PIPE_ID", "enterprise_agent_pipe")
PIPE_NAME = os.environ.get("OPENWEBUI_PIPE_NAME", "Enterprise Agent Pipe")
PIPE_SUBMODEL_ID = os.environ.get("OPENWEBUI_PIPE_SUBMODEL_ID", "owui_enterprise_agent")
PIPE_MODEL_NAME = os.environ.get("OPENWEBUI_PIPE_MODEL_NAME", "Enterprise Agent")
BOOTSTRAP_TIMEOUT_SECONDS = int(os.environ.get("OPENWEBUI_BOOTSTRAP_TIMEOUT_SECONDS", "180") or "180")
OPENWEBUI_THIN_MODE = str(os.environ.get("OPENWEBUI_THIN_MODE", "true")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: int = 30,
) -> tuple[int, dict[str, Any], str]:
    url = f"{OPENWEBUI_BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return response.status, parsed, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {}
        return exc.code, parsed, body


def _wait_for_openwebui() -> None:
    deadline = time.time() + BOOTSTRAP_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            request = urllib.request.Request(f"{OPENWEBUI_BASE_URL}/", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status < 500:
                    return
        except Exception:
            time.sleep(2)
            continue
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for Open WebUI at {OPENWEBUI_BASE_URL}")


def _signin() -> str | None:
    status, payload, _ = _request(
        "POST",
        "/api/v1/auths/signin",
        payload={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
    )
    if status == 200:
        return str(payload.get("token") or "")
    return None


def _signup() -> str | None:
    status, payload, _ = _request(
        "POST",
        "/api/v1/auths/signup",
        payload={"name": ADMIN_NAME, "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
    )
    if status == 200:
        return str(payload.get("token") or "")
    return None


def _ensure_admin_token() -> str:
    token = _signin()
    if token:
        return token

    token = _signup()
    if token:
        return token

    token = _signin()
    if token:
        return token

    raise RuntimeError("Unable to sign in or sign up the Open WebUI admin user.")


def _read_pipe_content() -> str:
    if not PIPE_FILE.exists():
        raise FileNotFoundError(f"Pipe file not found: {PIPE_FILE}")
    return PIPE_FILE.read_text(encoding="utf-8")


def _upsert_function(token: str, content: str) -> None:
    status, existing, _ = _request(
        "GET",
        f"/api/v1/functions/id/{urllib.parse.quote(PIPE_ID, safe='')}",
        token=token,
    )
    payload = {
        "id": PIPE_ID,
        "name": PIPE_NAME,
        "content": content,
        "meta": {"description": "Enterprise Agent bridge for Open WebUI file uploads and chat."},
    }
    if status == 200:
        _request(
            "POST",
            f"/api/v1/functions/id/{urllib.parse.quote(PIPE_ID, safe='')}/update",
            payload=payload,
            token=token,
        )
        current = existing
    else:
        status, current, body = _request("POST", "/api/v1/functions/create", payload=payload, token=token)
        if status not in (200, 201):
            raise RuntimeError(f"Failed to create function {PIPE_ID}: {body}")

    if not bool(current.get("is_global")):
        _request(
            "POST",
            f"/api/v1/functions/id/{urllib.parse.quote(PIPE_ID, safe='')}/toggle/global",
            token=token,
        )
    if not bool(current.get("is_active")):
        _request(
            "POST",
            f"/api/v1/functions/id/{urllib.parse.quote(PIPE_ID, safe='')}/toggle",
            token=token,
        )


def _model_payload(full_model_id: str) -> dict[str, Any]:
    capabilities = {
        "file_upload": True,
        "file_context": False,
        "vision": False,
        "web_search": False,
        "code_interpreter": False,
        "tools": False,
        "builtin_tools": False,
        "citations": False,
        "knowledge": False,
    }
    params = {
        "file_context": False,
        "web_search": False,
        "code_interpreter": False,
        "tools": False,
        "builtin_tools": False,
    }
    return {
        "id": full_model_id,
        "base_model_id": full_model_id,
        "name": PIPE_MODEL_NAME,
        "meta": {
            "description": "Open WebUI model wrapper for the Enterprise Agent bridge.",
            "capabilities": capabilities,
            "openwebui_thin_mode": OPENWEBUI_THIN_MODE,
            "document_source_policy": "agent_repository_only",
        },
        "params": params,
        "is_active": True,
    }


def _disable_native_document_rag(token: str) -> None:
    if not OPENWEBUI_THIN_MODE:
        return
    attempted = [
        *_disable_task_helpers(token),
        *_disable_retrieval_config(token),
    ]
    failures = [item for item in attempted if not item.endswith(":200") and not item.endswith(":201")]
    if failures:
        print(
            "Open WebUI native RAG/helper disable was only partially accepted "
            f"(attempted {', '.join(attempted)}). Model capabilities and pipe stripping remain enabled.",
            file=sys.stderr,
        )


def _disable_task_helpers(token: str) -> list[str]:
    attempted: list[str] = []
    status, config, body = _request("GET", "/api/v1/tasks/config", token=token)
    attempted.append(f"/api/v1/tasks/config:{status}")
    if status != 200:
        print(f"Open WebUI task config read failed: {body}", file=sys.stderr)
        return attempted
    payload = dict(config or {})
    defaults: dict[str, Any] = {
        "TASK_MODEL": "",
        "TASK_MODEL_EXTERNAL": "",
        "ENABLE_TITLE_GENERATION": False,
        "TITLE_GENERATION_PROMPT_TEMPLATE": "",
        "IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE": "",
        "ENABLE_AUTOCOMPLETE_GENERATION": False,
        "AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH": 0,
        "TAGS_GENERATION_PROMPT_TEMPLATE": "",
        "FOLLOW_UP_GENERATION_PROMPT_TEMPLATE": "",
        "ENABLE_FOLLOW_UP_GENERATION": False,
        "ENABLE_TAGS_GENERATION": False,
        "ENABLE_SEARCH_QUERY_GENERATION": False,
        "ENABLE_RETRIEVAL_QUERY_GENERATION": False,
        "QUERY_GENERATION_PROMPT_TEMPLATE": "",
        "TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE": "",
        "VOICE_MODE_PROMPT_TEMPLATE": "",
    }
    for key, value in defaults.items():
        payload.setdefault(key, value)
    payload.update(
        {
            "ENABLE_TITLE_GENERATION": False,
            "ENABLE_AUTOCOMPLETE_GENERATION": False,
            "ENABLE_FOLLOW_UP_GENERATION": False,
            "ENABLE_TAGS_GENERATION": False,
            "ENABLE_SEARCH_QUERY_GENERATION": False,
            "ENABLE_RETRIEVAL_QUERY_GENERATION": False,
            "QUERY_GENERATION_PROMPT_TEMPLATE": "",
        }
    )
    status, _, body = _request("POST", "/api/v1/tasks/config/update", payload=payload, token=token)
    attempted.append(f"/api/v1/tasks/config/update:{status}")
    if status not in (200, 201):
        print(f"Open WebUI task helper disable failed: {body}", file=sys.stderr)
    return attempted


def _disable_retrieval_config(token: str) -> list[str]:
    attempted: list[str] = []
    status, config, body = _request("GET", "/api/v1/retrieval/config", token=token)
    attempted.append(f"/api/v1/retrieval/config:{status}")
    if status != 200:
        print(f"Open WebUI retrieval config read failed: {body}", file=sys.stderr)
        return attempted
    payload = dict(config or {})
    payload.update(
        {
            "BYPASS_EMBEDDING_AND_RETRIEVAL": True,
            "RAG_FULL_CONTEXT": False,
        }
    )
    status, _, body = _request("POST", "/api/v1/retrieval/config/update", payload=payload, token=token)
    attempted.append(f"/api/v1/retrieval/config/update:{status}")
    if status not in (200, 201):
        print(f"Open WebUI retrieval bypass disable failed: {body}", file=sys.stderr)
    return attempted


def _upsert_model(token: str) -> None:
    full_model_id = f"{PIPE_ID}.{PIPE_SUBMODEL_ID}"
    payload = _model_payload(full_model_id)
    status, _, _ = _request(
        "GET",
        f"/api/v1/models/model?id={urllib.parse.quote(full_model_id, safe='')}",
        token=token,
    )
    if status == 200:
        status, _, body = _request("POST", "/api/v1/models/model/update", payload=payload, token=token)
        if status not in (200, 201):
            raise RuntimeError(f"Failed to update model {full_model_id}: {body}")
        return

    status, _, body = _request("POST", "/api/v1/models/create", payload=payload, token=token)
    if status not in (200, 201):
        raise RuntimeError(f"Failed to create model {full_model_id}: {body}")


def main() -> int:
    try:
        _wait_for_openwebui()
        token = _ensure_admin_token()
        content = _read_pipe_content()
        _upsert_function(token, content)
        _upsert_model(token)
        _disable_native_document_rag(token)
        print(f"Open WebUI bootstrap completed for function={PIPE_ID}")
        return 0
    except Exception as exc:
        print(f"Open WebUI bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
