from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_bootstrap_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "deployment" / "openwebui" / "bootstrap_openwebui.py"
    spec = importlib.util.spec_from_file_location("bootstrap_openwebui_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_openwebui_model_payload_disables_native_rag_capabilities() -> None:
    module = _load_bootstrap_module()

    payload = module._model_payload("enterprise_agent_pipe.owui_enterprise_agent")
    capabilities = payload["meta"]["capabilities"]

    assert capabilities["file_upload"] is True
    assert capabilities["file_context"] is False
    assert capabilities["web_search"] is False
    assert capabilities["code_interpreter"] is False
    assert capabilities["tools"] is False
    assert capabilities["builtin_tools"] is False
    assert payload["meta"]["document_source_policy"] == "agent_repository_only"
    assert payload["params"]["file_context"] is False


def test_openwebui_bootstrap_attempts_native_rag_disable(monkeypatch) -> None:
    module = _load_bootstrap_module()
    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_request(method, path, *, payload=None, token=None, timeout=30):
        del token, timeout
        calls.append((method, path, payload or {}))
        if method == "GET" and path == "/api/v1/tasks/config":
            return 200, {
                "TASK_MODEL": "",
                "TASK_MODEL_EXTERNAL": "",
                "ENABLE_TITLE_GENERATION": True,
                "TITLE_GENERATION_PROMPT_TEMPLATE": "title",
                "IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE": "",
                "ENABLE_AUTOCOMPLETE_GENERATION": True,
                "AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH": 64,
                "TAGS_GENERATION_PROMPT_TEMPLATE": "tags",
                "FOLLOW_UP_GENERATION_PROMPT_TEMPLATE": "followups",
                "ENABLE_FOLLOW_UP_GENERATION": True,
                "ENABLE_TAGS_GENERATION": True,
                "ENABLE_SEARCH_QUERY_GENERATION": True,
                "ENABLE_RETRIEVAL_QUERY_GENERATION": True,
                "QUERY_GENERATION_PROMPT_TEMPLATE": "queries",
                "TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE": "",
                "VOICE_MODE_PROMPT_TEMPLATE": "",
            }, "{}"
        if method == "GET" and path == "/api/v1/retrieval/config":
            return 200, {
                "BYPASS_EMBEDDING_AND_RETRIEVAL": False,
                "RAG_FULL_CONTEXT": True,
                "TOP_K": 5,
            }, "{}"
        return 200, {}, "{}"

    monkeypatch.setattr(module, "_request", fake_request)

    module._disable_native_document_rag("admin-token")

    posted = {(method, path): payload for method, path, payload in calls if method == "POST"}
    task_payload = posted[("POST", "/api/v1/tasks/config/update")]
    retrieval_payload = posted[("POST", "/api/v1/retrieval/config/update")]

    assert task_payload["ENABLE_SEARCH_QUERY_GENERATION"] is False
    assert task_payload["ENABLE_RETRIEVAL_QUERY_GENERATION"] is False
    assert task_payload["ENABLE_FOLLOW_UP_GENERATION"] is False
    assert task_payload["ENABLE_TAGS_GENERATION"] is False
    assert task_payload["QUERY_GENERATION_PROMPT_TEMPLATE"] == ""
    assert retrieval_payload["BYPASS_EMBEDDING_AND_RETRIEVAL"] is True
    assert retrieval_payload["RAG_FULL_CONTEXT"] is False
