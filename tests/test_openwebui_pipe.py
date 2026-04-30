from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agentic_chatbot_next.api.status_tracker import (
    PHASE_GRAPH_CATALOG,
    PHASE_SEARCHING,
    PHASE_SYNTHESIZING,
    TurnStatusTracker,
    infer_phase_from_progress,
)


def _load_pipe_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "deployment" / "openwebui" / "enterprise_agent_pipe.py"
    spec = importlib.util.spec_from_file_location("enterprise_agent_pipe_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pipe_valves_default_and_env_override(monkeypatch) -> None:
    module = _load_pipe_module()

    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    pipe = module.Pipe()
    assert pipe.valves.AGENT_BASE_URL == "http://app:8000/v1"

    monkeypatch.setenv("AGENT_BASE_URL", "http://example.test:9000/v1")
    overridden_pipe = module.Pipe()
    assert overridden_pipe.valves.AGENT_BASE_URL == "http://example.test:9000/v1"


def test_status_tracker_marks_clarification_completion_as_needs_input() -> None:
    tracker = TurnStatusTracker(turn_started_at=0.0)

    snapshots = tracker.completion_snapshots(
        1.0,
        metadata={
            "turn_outcome": "clarification_request",
            "clarification": {"question": "What output format should I use?"},
        },
    )

    assert snapshots[-1]["description"].startswith("Needs input")
    assert snapshots[-1]["agentic_status"]["title"] == "Needs input"
    assert snapshots[-1]["agentic_status"]["subtitle"] == "What output format should I use?"


def test_pipe_short_circuits_helper_tasks_by_default(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    captured: dict[str, object] = {}
    emitted: list[dict[str, object]] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "Architecture Overview"}}],
                "artifacts": [],
            }

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def post(self, url, *, headers=None, json=None, **kwargs):
            del url, headers, kwargs
            captured["json"] = json
            captured["stream"] = bool(json and json.get("stream"))
            return _Response()

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)

    async def emit(event):
        emitted.append(event)

    result = asyncio.run(
        pipe.pipe(
            {
                "messages": [
                    {"role": "user", "content": "Generate a concise, 3-5 word title for this chat."},
                ]
            },
            __event_emitter__=emit,
        )
    )

    assert result == "Enterprise Agent"
    assert captured == {}
    assert emitted == []


def test_pipe_forwards_user_email_in_headers_and_metadata(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    captured: dict[str, object] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "ok"}}],
                "artifacts": [],
            }

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def post(self, url, *, headers=None, json=None, **kwargs):
            del url, kwargs
            captured["headers"] = headers or {}
            captured["json"] = json or {}
            return _Response()

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)

    result = asyncio.run(
        pipe.pipe(
            {
                "messages": [{"role": "user", "content": "Summarize this collection."}],
            },
            __user__={"id": "user-1", "email": "Analyst@Example.com"},
        )
    )

    assert result == "ok"
    assert captured["headers"]["X-User-Email"] == "analyst@example.com"
    assert captured["json"]["metadata"]["user_email"] == "analyst@example.com"


def test_pipe_upload_roundtrip_downloads_then_uploads_and_streams_completion(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []
    captured: dict[str, object] = {}

    class _DownloadResponse:
        def __init__(self):
            self.content = b"reviews\nGreat service\n"
            self.headers = {"content-type": "text/csv"}

        def raise_for_status(self) -> None:
            return None

    class _UploadResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "object": "upload.result",
                "doc_ids": ["UPLOAD_file_1"],
                "active_uploaded_doc_ids": ["UPLOAD_file_1"],
                "upload_manifest": {"collection_id": "owui-chat-chat-123"},
            }

    class _StreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            lines = [
                'event: progress',
                'data: {"label":"Searching knowledge base","detail":"Uploaded files ready"}',
                "",
                'data: {"choices":[{"delta":{"content":"Analysis ready."},"finish_reason":null}]}',
                "",
                "data: [DONE]",
                "",
            ]
            for item in lines:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def get(self, url, *, headers=None, cookies=None, **kwargs):
            del kwargs
            captured["download"] = {
                "url": url,
                "headers": headers or {},
                "cookies": cookies or {},
            }
            return _DownloadResponse()

        async def post(self, *args, **kwargs):
            del args, kwargs
            # httpx 0.28.x builds a sync multipart stream for files=... on
            # AsyncClient, so /v1/upload must stay on the sync client path.
            raise AssertionError("Async multipart upload should not be used for /v1/upload")

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del kwargs
            captured["stream"] = {
                "method": method,
                "url": url,
                "headers": headers or {},
                "json": json or {},
            }
            return _StreamContext()

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)

    class _SyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def post(self, url, *, params=None, headers=None, files=None, data=None, **kwargs):
            del kwargs
            captured["upload"] = {
                "url": url,
                "params": params or {},
                "headers": headers or {},
                "files": files or {},
                "data": data,
            }
            return _UploadResponse()

    monkeypatch.setattr(module.httpx, "Client", _SyncClient)

    result = asyncio.run(
        pipe.pipe(
            {
                "messages": [
                    {"role": "user", "content": "Provide sentiment analysis of all of the reviews in the reviews column."}
                ]
            },
            __user__={"id": "user-1", "email": "analyst@example.com"},
            __metadata__={"chat_id": "chat-123", "message_id": "msg-123"},
            __files__=[{"id": "file-1", "filename": "customer_reviews_100.csv"}],
            __request__=SimpleNamespace(
                headers={"authorization": "Bearer owui-token"},
                cookies={"session": "cookie-1"},
            ),
            __event_emitter__=emit,
        )
    )

    assert result == "Analysis ready."
    assert captured["download"]["url"] == "http://127.0.0.1:8080/api/v1/files/file-1/content"
    assert captured["download"]["headers"]["Authorization"] == "Bearer owui-token"
    assert captured["download"]["cookies"] == {"session": "cookie-1"}
    assert captured["upload"]["params"] == {
        "source_type": "upload",
        "collection_id": "owui-chat-chat-123",
    }
    upload_tuple = captured["upload"]["files"]["files"]
    assert upload_tuple[0] == "customer_reviews_100.csv"
    assert upload_tuple[1] == b"reviews\nGreat service\n"
    assert upload_tuple[2] == "text/csv"
    assert captured["upload"]["headers"]["X-Upload-Source-Ids"] == json.dumps(["file-1"])
    assert captured["upload"]["data"] is None
    assert captured["stream"]["method"] == "POST"
    assert captured["stream"]["url"].endswith("/chat/completions")
    assert captured["stream"]["json"]["stream"] is True
    assert captured["stream"]["json"]["metadata"]["upload_collection_id"] == "owui-chat-chat-123"
    assert captured["stream"]["json"]["metadata"]["kb_collection_id"] == "default"
    assert captured["stream"]["json"]["metadata"]["document_source_policy"] == "agent_repository_only"
    assert captured["stream"]["json"]["metadata"]["openwebui_thin_mode"] is True
    assert captured["stream"]["json"]["metadata"]["uploaded_doc_ids"] == ["UPLOAD_file_1"]
    assert captured["stream"]["json"]["metadata"]["upload_manifest"]["uploaded_file_count"] == 1
    assert "Attempted to send an sync request with an AsyncClient instance." not in result
    assert emitted[0]["type"] == "status"
    assert emitted[-1]["data"]["done"] is True


def test_pipe_helper_tasks_with_files_stay_local_by_default(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    captured_payloads: list[dict[str, object]] = []
    prompts = [
        ("Generate a concise, 3-5 word title for this chat.", "title"),
        ("Generate 1-3 broad tags for this chat.", "tags"),
        ("Suggest 3-5 relevant follow-up questions the user could ask.", "follow_ups"),
        (
            (
                "### Task:\n"
                "Analyze the chat history to determine the necessity of generating search queries.\n"
                "Prioritize generating 1-3 broad and relevant search queries.\n"
                "Respond **exclusively** with a JSON object like {\"queries\":[]}."
            ),
            "search_queries",
        ),
    ]

    class _Response:
        def __init__(self, helper_type: str):
            self._helper_type = helper_type

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": f"ok:{self._helper_type}"}}],
                "artifacts": [],
            }

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def get(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("Helper tasks should not download attached files")

        async def post(self, url, *, headers=None, json=None, **kwargs):
            del url, headers, kwargs
            captured_payloads.append(dict(json or {}))
            helper_type = str(((json or {}).get("metadata") or {}).get("openwebui_helper_task_type") or "")
            return _Response(helper_type)

        def stream(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("Helper tasks should not use the streaming path")

    class _SyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("Helper tasks should not create a sync upload client")

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(module.httpx, "Client", _SyncClient)

    for prompt, helper_type in prompts:
        result = asyncio.run(
            pipe.pipe(
                {"messages": [{"role": "user", "content": prompt}]},
                __metadata__={"chat_id": "helper-chat", "message_id": f"{helper_type}-msg"},
                __files__=[{"id": "file-1", "filename": "customer_reviews_100.csv"}],
                __request__=SimpleNamespace(
                    headers={"authorization": "Bearer owui-token"},
                    cookies={"session": "cookie-1"},
                ),
            )
        )
        if helper_type == "title":
            assert result == "Enterprise Agent"
        elif helper_type == "tags":
            assert result == "enterprise-agent"
        elif helper_type == "search_queries":
            assert result == '{"queries":[]}'
        else:
            assert result == "[]"

    assert captured_payloads == []


def test_sync_upload_request_shape_stays_valid_multipart() -> None:
    request = httpx.Client().build_request(
        "POST",
        "http://example.test/v1/upload",
        files={"files": ("customer_reviews_100.csv", b"reviews\nGreat service\n", "text/csv")},
        headers={"X-Upload-Source-Ids": json.dumps(["file-1"])},
    )

    assert request.headers["X-Upload-Source-Ids"] == json.dumps(["file-1"])
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert type(request.stream).__name__ == "MultipartStream"
    assert all(isinstance(chunk, (bytes, bytearray)) for chunk in request.stream)


def test_pipe_thin_mode_strips_openwebui_context_and_non_chat_roles(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    captured: dict[str, object] = {}

    class _StreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            for item in [
                'data: {"choices":[{"delta":{"content":"Done."}}]}',
                "",
                "data: [DONE]",
                "",
            ]:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del method, url, headers, kwargs
            captured["json"] = dict(json or {})
            return _StreamContext()

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)

    result = asyncio.run(
        pipe.pipe(
            {
                "messages": [
                    {"role": "system", "content": "Use OpenWebUI retrieved context."},
                    {"role": "tool", "content": "Tool-only OpenWebUI content."},
                    {
                        "role": "user",
                        "content": (
                            "<context>Injected source snippet that must not reach the backend.</context>\n"
                            "### User Query:\n"
                            "extract all requirements/shall statements from the uploaded document\n"
                            "Sources:\n"
                            "- [1] OpenWebUI citation"
                        ),
                    },
                ],
                "sources": [{"content": "OpenWebUI source"}],
                "files": [{"id": "file-1"}],
            }
        )
    )

    assert result == "Done."
    assert captured["json"]["messages"] == [
        {"role": "user", "content": "extract all requirements/shall statements from the uploaded document"}
    ]
    assert "sources" not in captured["json"]
    assert "files" not in captured["json"]
    assert captured["json"]["metadata"]["document_source_policy"] == "agent_repository_only"


def test_pipe_thin_mode_unwraps_openwebui_rag_task_prompt(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    captured: dict[str, object] = {}

    class _StreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            for item in [
                'data: {"choices":[{"delta":{"content":"Done."}}]}',
                "",
                "data: [DONE]",
                "",
            ]:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del method, url, headers, kwargs
            captured["json"] = dict(json or {})
            return _StreamContext()

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)

    result = asyncio.run(
        pipe.pipe(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "### Task:\n"
                            "Respond to the user query using the provided context.\n\n"
                            "### Guidelines:\n"
                            "- If you don't know the answer, say so.\n"
                            "- Don't present information that's not present in the context.\n\n"
                            "<context>OpenWebUI snippet that must not become evidence.</context>\n\n"
                            "### Output:\n"
                            "Provide a clear and direct response.\n"
                            "extract all requirements/ shall statements from the uploaded document\n"
                        ),
                    }
                ]
            }
        )
    )

    assert result == "Done."
    assert captured["json"]["messages"] == [
        {"role": "user", "content": "extract all requirements/ shall statements from the uploaded document"}
    ]


def test_pipe_streams_progress_as_status_events(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []
    monotonic_values = [100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0]

    def fake_now() -> float:
        if len(monotonic_values) > 1:
            return monotonic_values.pop(0)
        return monotonic_values[0]

    class _StreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            lines = [
                'event: progress',
                'data: {"label":"Searching knowledge base","detail":"2 documents"}',
                "",
                'event: progress',
                'data: {"label":"Reviewing candidate documents","detail":"Architecture docs"}',
                "",
                'event: progress',
                'data: {"label":"Synthesizing answer","detail":"Grounding final response"}',
                "",
                'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
                "",
                'data: {"choices":[{"delta":{"content":"Runtime details."},"finish_reason":null}]}',
                "",
                'event: artifacts',
                'data: [{"label":"report.md","download_url":"/v1/files/abc"}]',
                "",
                'data: [DONE]',
                "",
            ]
            for item in lines:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del method, url, headers, kwargs
            assert json["stream"] is True
            return _StreamContext()

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(pipe, "_now", fake_now)

    result = asyncio.run(
        pipe.pipe(
            {
                "messages": [{"role": "user", "content": "What are the key implementation details in the architecture docs?"}]
            },
            __event_emitter__=emit,
        )
    )

    assert "Runtime details." in result
    assert "[report.md](http://localhost:18000/v1/files/abc)" in result
    assert emitted[0]["type"] == "status"
    assert emitted[0]["data"]["description"] == "Starting • 00:00 elapsed"
    assert any(item["data"]["phase"] == "searching_knowledge_base" for item in emitted)
    assert any(item["data"]["phase"] == "synthesizing_answer" for item in emitted)
    assert emitted[-1]["data"]["description"].startswith("Answer ready")
    assert emitted[-1]["data"]["done"] is True
    assert emitted[-1]["data"]["hidden"] is False
    for item in emitted:
        data = item["data"]
        assert data["agentic_status"]["version"] == 1
        assert "agent" in data
        assert "selected_agent" in data
        assert "phase" in data
        assert "phase_label" in data
        assert "phase_elapsed_ms" in data
        assert "source_event_type" in data
        assert "timestamp" in data
        assert data["agentic_status"]["timing"]["snapshot_timestamp_ms"] == data["timestamp"]
    assert emitted[0]["data"]["agentic_status"]["title"] == "Thinking"
    assert emitted[0]["data"]["agentic_status"]["timing"]["kind"] == "stage"
    assert emitted[0]["data"]["agentic_status"]["timing"]["live"] is True
    assert any(item["data"]["agentic_status"]["title"] == "Researching evidence" for item in emitted)
    assert emitted[-1]["data"]["agentic_status"]["title"] == "Answer ready"
    assert emitted[-1]["data"]["agentic_status"]["timing"]["kind"] == "total"
    assert emitted[-1]["data"]["agentic_status"]["timing"]["live"] is False


def test_pipe_emits_agent_status_updates_from_progress_events(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []
    monotonic_values = [300.0, 300.0, 300.0, 301.0, 302.0, 304.0, 309.0]

    def fake_now() -> float:
        if len(monotonic_values) > 1:
            return monotonic_values.pop(0)
        return monotonic_values[0]

    class _StreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            lines = [
                'event: progress',
                'data: {"type":"route_decision","label":"Routed to general","agent":"general"}',
                "",
                'event: progress',
                'data: {"type":"agent_selected","label":"Running general","agent":"general"}',
                "",
                'event: progress',
                'data: {"type":"worker_start","label":"Search KB set","agent":"rag_worker"}',
                "",
                'event: progress',
                'data: {"type":"worker_end","label":"Search KB set","agent":"rag_worker"}',
                "",
                'event: progress',
                'data: {"label":"Synthesizing answer","detail":"Grounding final response"}',
                "",
                'data: {"choices":[{"delta":{"content":"Inventory response."},"finish_reason":null}]}',
                "",
                'data: [DONE]',
                "",
            ]
            for item in lines:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del method, url, headers, kwargs
            assert json["stream"] is True
            return _StreamContext()

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(pipe, "_now", fake_now)

    result = asyncio.run(
        pipe.pipe(
            {
                "messages": [{"role": "user", "content": "What documents do we have access to?"}]
            },
            __event_emitter__=emit,
        )
    )

    payloads = [item["data"] for item in emitted]

    assert "Inventory response." in result
    assert payloads[0]["description"] == "Starting • 00:00 elapsed"
    assert any(item["selected_agent"] == "general" and item["agent"] == "general" for item in payloads)
    assert any(item["selected_agent"] == "general" and item["agent"] == "rag_worker" for item in payloads)
    assert any("rag_worker" in item["description"] for item in payloads if item["agent"] == "rag_worker")
    assert any(item["agentic_status"]["title"] == "Routing request" for item in payloads)
    assert any(item["agentic_status"]["title"] == "Running General Agent" for item in payloads)
    assert any(item["agentic_status"]["kind"] == "worker" for item in payloads if item["agent"] == "rag_worker")
    assert any(item["agentic_status"]["timing"]["kind"] == "stage" for item in payloads[:-1])
    assert any(item["agentic_status"]["timing"]["live"] is True for item in payloads[:-1])
    assert any(
        item["phase"] == "searching_knowledge_base"
        and item["status"] == "complete"
        and item["agentic_status"]["timing"]["live"] is False
        for item in payloads[:-1]
    )
    assert payloads[-1]["description"].startswith("Answer ready")
    assert payloads[-1]["selected_agent"] == "general"
    assert payloads[-1]["agent"] == "rag_worker"
    assert payloads[-1]["agentic_status"]["title"] == "Answer ready"
    assert payloads[-1]["agentic_status"]["timing"]["kind"] == "total"
    assert payloads[-1]["agentic_status"]["timing"]["live"] is False


def test_pipe_prefers_backend_status_snapshots_when_available(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []

    def fake_now() -> float:
        return 400.0

    def status_payload(
        *,
        description: str,
        phase: str,
        elapsed_ms: int,
        status_id: str | None = None,
        status_key: str | None = None,
        status_seq: int | None = None,
        status_elapsed_ms: int | None = None,
        agent: str = "",
        selected_agent: str = "",
        done: bool = False,
        status: str = "in_progress",
        source_event_type: str = "status",
        agentic_status: dict[str, object] | None = None,
    ) -> str:
        phase_elapsed_ms = elapsed_ms if phase != "answer_ready" else 0
        payload = {
            "status_id": status_id or f"{phase}-{agent or selected_agent or 'none'}-{elapsed_ms}",
            "status_key": status_key or f"{phase}\u241f{agent or selected_agent}\u241f{'done' if done else 'active'}",
            "status_seq": status_seq if status_seq is not None else max(1, int((elapsed_ms / 1000) + 1)),
            "description": description,
            "status": status,
            "done": done,
            "hidden": False,
            "elapsed_ms": elapsed_ms,
            "delta_ms": None,
            "status_elapsed_ms": status_elapsed_ms if status_elapsed_ms is not None else phase_elapsed_ms,
            "agent": agent,
            "selected_agent": selected_agent,
            "phase": phase,
            "phase_label": phase.replace("_", " ").title(),
            "phase_elapsed_ms": phase_elapsed_ms,
            "source_event_type": source_event_type,
            "label": phase.replace("_", " ").title(),
            "detail": "",
            "job_id": "",
            "task_id": "",
            "why": "",
            "waiting_on": "",
            "timestamp": 1713139200000,
        }
        if agentic_status is not None:
            payload["agentic_status"] = agentic_status
        return json.dumps(payload)

    class _StreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            lines = [
                "event: status",
                f'data: {status_payload(description="Starting • 00:00 elapsed", phase="starting", elapsed_ms=0)}',
                "",
                "event: status",
                (
                    'data: '
                    + status_payload(
                        description="Searching knowledge base • general • 00:01 elapsed",
                        phase="searching_knowledge_base",
                        elapsed_ms=1000,
                        agent="general",
                        selected_agent="general",
                        source_event_type="route_decision",
                        agentic_status={
                            "version": 1,
                            "state": "active",
                            "kind": "routing",
                            "title": "Routing request",
                            "subtitle": "Preserved from backend.",
                            "chips": ["Research", "General Agent"],
                            "timing": {
                                "kind": "stage",
                                "live": True,
                                "elapsed_ms": 1000,
                                "status_elapsed_ms": 1000,
                                "phase_elapsed_ms": 1000,
                                "total_elapsed_ms": 1000,
                                "snapshot_timestamp_ms": 1713139200000,
                            },
                        },
                    )
                ),
                "",
                'event: progress',
                'data: {"type":"route_decision","label":"Routed to general","agent":"general"}',
                "",
                "event: status",
                (
                    'data: '
                    + status_payload(
                        description="Searching knowledge base • rag_worker • 00:02 elapsed",
                        phase="searching_knowledge_base",
                        elapsed_ms=2000,
                        agent="rag_worker",
                        selected_agent="general",
                        source_event_type="worker_start",
                    )
                ),
                "",
                'event: progress',
                'data: {"type":"worker_start","label":"Search KB set","agent":"rag_worker"}',
                "",
                "event: status",
                (
                    'data: '
                    + status_payload(
                        description="Synthesizing answer • rag_worker • 00:03 elapsed",
                        phase="synthesizing_answer",
                        elapsed_ms=3000,
                        agent="rag_worker",
                        selected_agent="general",
                        source_event_type="content_delta",
                    )
                ),
                "",
                'data: {"choices":[{"delta":{"content":"Grounded answer."},"finish_reason":null}]}',
                "",
                "event: status",
                (
                    'data: '
                    + status_payload(
                        description="Answer ready • rag_worker • Completed in 00:04",
                        phase="answer_ready",
                        elapsed_ms=4000,
                        agent="rag_worker",
                        selected_agent="general",
                        done=True,
                        status="complete",
                        source_event_type="turn_completed",
                    )
                ),
                "",
                'data: [DONE]',
                "",
            ]
            for item in lines:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del method, url, headers, kwargs
            assert json["metadata"]["openwebui_client"] is True
            assert json["stream"] is True
            return _StreamContext()

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(pipe, "_now", fake_now)

    result = asyncio.run(
        pipe.pipe(
            {
                "messages": [{"role": "user", "content": "Explain the architecture deeply and keep it grounded."}]
            },
            __event_emitter__=emit,
        )
    )

    assert result == "Grounded answer."
    assert [item["data"]["description"] for item in emitted] == [
        "Starting • 00:00 elapsed",
        "Searching knowledge base • 00:00 elapsed",
        "Searching knowledge base • general • 00:01 elapsed",
        "Searching knowledge base • rag_worker • 00:02 elapsed",
        "Synthesizing answer • rag_worker • 00:03 elapsed",
        "Answer ready • rag_worker • Completed in 00:04",
    ]
    assert [item["data"]["agent"] for item in emitted[-4:]] == [
        "general",
        "rag_worker",
        "rag_worker",
        "rag_worker",
    ]
    assert emitted[2]["data"]["agentic_status"]["subtitle"] == "Preserved from backend."
    assert emitted[2]["data"]["agentic_status"]["timing"] == {
        "kind": "stage",
        "live": True,
        "elapsed_ms": 1000,
        "status_elapsed_ms": 1000,
        "phase_elapsed_ms": 1000,
        "total_elapsed_ms": 1000,
        "snapshot_timestamp_ms": 1713139200000,
    }
    assert emitted[2]["data"]["status_id"] != emitted[3]["data"]["status_id"]
    assert emitted[3]["data"]["status_seq"] > emitted[2]["data"]["status_seq"]
    assert emitted[-1]["data"]["agentic_status"]["timing"]["kind"] == "total"


def test_pipe_forwards_backend_tool_call_status_cards(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []

    def tool_status(status: str, *, output: dict[str, object] | None = None) -> str:
        done = status != "running"
        payload = {
            "type": "tool_trace",
            "status_id": "tool-call-1",
            "status_key": "tool-call-1",
            "description": f"search_indexed_docs {'completed' if done else 'running'}",
            "status": "complete" if done else "in_progress",
            "done": done,
            "hidden": False,
            "elapsed_ms": 120 if done else 0,
            "agent": "general",
            "selected_agent": "general",
            "phase": "searching_knowledge_base",
            "phase_label": "Searching Knowledge Base",
            "phase_elapsed_ms": 120 if done else 0,
            "status_elapsed_ms": 120 if done else 0,
            "source_event_type": "tool_end" if done else "tool_start",
            "label": f"search_indexed_docs {'completed' if done else 'running'}",
            "detail": '{"hits":1}' if done else '{"query":"agent trace"}',
            "job_id": "",
            "task_id": "",
            "why": "A runtime agent invoked a tool.",
            "waiting_on": "",
            "timestamp": 1713139200000 + (120 if done else 0),
            "agentic_tool_call": {
                "version": 1,
                "tool_call_id": "call-1",
                "tool_name": "search_indexed_docs",
                "agent_name": "general",
                "job_id": "",
                "status": status,
                "started_at": "2026-04-23T10:00:00Z",
                "completed_at": "2026-04-23T10:00:01Z" if done else "",
                "duration_ms": 120 if done else None,
                "input_preview": '{"query":"agent trace"}',
                "output_preview": '{"hits":1}' if done else "",
                "input": {"query": "agent trace"},
                "output": output,
                "error": None,
                "truncated": False,
                "truncated_fields": [],
                "source_event_id": "evt_end" if done else "evt_start",
            },
        }
        return json.dumps(payload)

    class _StreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            lines = [
                "event: status",
                f"data: {tool_status('running')}",
                "",
                "event: status",
                f"data: {tool_status('completed', output={'hits': 1})}",
                "",
                'data: {"choices":[{"delta":{"content":"Tool-backed answer."},"finish_reason":null}]}',
                "",
                "event: status",
                (
                    'data: {"status_id":"status-ready","status_key":"answer_ready\\u241fgeneral\\u241fdone",'
                    '"description":"Answer ready • general • Completed in 00:01","status":"complete",'
                    '"done":true,"hidden":false,"elapsed_ms":1000,"delta_ms":null,'
                    '"status_elapsed_ms":1000,"agent":"general","selected_agent":"general",'
                    '"phase":"answer_ready","phase_label":"Answer Ready","phase_elapsed_ms":1000,'
                    '"source_event_type":"turn_completed","label":"Answer ready","detail":"",'
                    '"job_id":"","task_id":"","why":"","waiting_on":"","timestamp":1713139201000}'
                ),
                "",
                "data: [DONE]",
                "",
            ]
            for item in lines:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del method, url, headers, json, kwargs
            return _StreamContext()

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)

    result = asyncio.run(
        pipe.pipe(
            {"messages": [{"role": "user", "content": "Use a tool and show the trace."}]},
            __event_emitter__=emit,
        )
    )

    tool_events = [item["data"] for item in emitted if item["data"].get("agentic_tool_call")]

    assert result == "Tool-backed answer."
    assert len(tool_events) == 2
    assert {item["status_id"] for item in tool_events} == {"tool-call-1"}
    assert tool_events[0]["agentic_tool_call"]["status"] == "running"
    assert tool_events[0]["agentic_tool_call"]["input"] == {"query": "agent trace"}
    assert tool_events[1]["done"] is True
    assert tool_events[1]["agentic_tool_call"]["status"] == "completed"
    assert tool_events[1]["agentic_tool_call"]["output"] == {"hits": 1}
    assert emitted[-1]["data"]["done"] is True
    assert "agentic_tool_call" not in emitted[-1]["data"]


def test_pipe_forwards_backend_agent_and_parallel_audit_status_cards(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []

    agent_status = {
        "type": "worker_start",
        "status_id": "agent-worker-T2",
        "status_key": "agent-worker-T2",
        "description": "RAG Worker is searching evidence for task T2",
        "status": "in_progress",
        "done": False,
        "hidden": False,
        "elapsed_ms": 0,
        "agent": "rag_worker",
        "selected_agent": "coordinator",
        "phase": "searching_knowledge_base",
        "phase_label": "Searching Knowledge Base",
        "phase_elapsed_ms": 0,
        "status_elapsed_ms": 0,
        "source_event_type": "worker_agent_started",
        "label": "RAG Worker is searching evidence for task T2",
        "detail": "Searching evidence for task T2.",
        "job_id": "job-2",
        "task_id": "T2",
        "why": "",
        "waiting_on": "",
        "timestamp": 1713139200000,
        "agentic_status": {"version": 1, "state": "active", "kind": "agent", "title": "RAG Worker", "subtitle": ""},
        "agentic_agent_activity": {
            "version": 1,
            "activity_id": "agent-worker-rag_worker-T2-job-2",
            "agent_name": "rag_worker",
            "role": "worker",
            "status": "running",
            "title": "RAG Worker is searching evidence for task T2",
            "description": "Searching evidence for task T2.",
            "parent_agent": "coordinator",
            "task_id": "T2",
            "job_id": "job-2",
            "parallel_group_id": "worker-batch-T1-T2",
            "started_at": "2026-04-23T10:00:00Z",
            "completed_at": "",
            "duration_ms": None,
        },
    }
    parallel_status = {
        "type": "parallel_group_trace",
        "status_id": "group-worker-batch-T1-T2",
        "status_key": "group-worker-batch-T1-T2",
        "description": "Parallel worker batch: 2 running",
        "status": "in_progress",
        "done": False,
        "hidden": False,
        "elapsed_ms": 0,
        "agent": "coordinator",
        "selected_agent": "coordinator",
        "phase": "searching_knowledge_base",
        "phase_label": "Searching Knowledge Base",
        "phase_elapsed_ms": 0,
        "status_elapsed_ms": 0,
        "source_event_type": "coordinator_worker_batch_started",
        "label": "Parallel worker batch: 2 running",
        "detail": "Coordinator dispatched this worker batch in parallel.",
        "job_id": "",
        "task_id": "",
        "why": "",
        "waiting_on": "",
        "timestamp": 1713139200001,
        "agentic_status": {
            "version": 1,
            "state": "active",
            "kind": "parallel_group",
            "title": "Parallel worker batch",
            "subtitle": "",
        },
        "agentic_parallel_group": {
            "version": 1,
            "group_id": "worker-batch-T1-T2",
            "group_kind": "worker_batch",
            "status": "running",
            "execution_mode": "parallel",
            "size": 2,
            "members": [
                {"agent_name": "rag_worker", "task_id": "T1", "job_id": "job-1"},
                {"agent_name": "table_worker", "task_id": "T2", "job_id": "job-2"},
            ],
            "reason": "Coordinator dispatched this worker batch in parallel.",
            "started_at": "2026-04-23T10:00:00Z",
            "completed_at": "",
            "duration_ms": None,
        },
    }

    class _StreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            lines = [
                "event: status",
                f"data: {json.dumps(parallel_status)}",
                "",
                "event: status",
                f"data: {json.dumps(agent_status)}",
                "",
                'data: {"choices":[{"delta":{"content":"Auditable answer."},"finish_reason":null}]}',
                "",
                "data: [DONE]",
                "",
            ]
            for item in lines:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del method, url, headers, json, kwargs
            return _StreamContext()

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)

    result = asyncio.run(
        pipe.pipe(
            {"messages": [{"role": "user", "content": "Show audit status."}]},
            __event_emitter__=emit,
        )
    )

    agent_events = [item["data"] for item in emitted if item["data"].get("agentic_agent_activity")]
    parallel_events = [item["data"] for item in emitted if item["data"].get("agentic_parallel_group")]

    assert result == "Auditable answer."
    assert len(parallel_events) == 1
    assert parallel_events[0]["agentic_parallel_group"]["execution_mode"] == "parallel"
    assert len(agent_events) == 1
    assert agent_events[0]["agentic_agent_activity"]["parallel_group_id"] == "worker-batch-T1-T2"


def test_pipe_ignores_stale_backend_status_updates(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []

    def fake_now() -> float:
        return 500.0

    class _StreamResponse:
        def raise_for_status(self) -> None:
            return None

        async def aiter_lines(self):
            lines = [
                "event: status",
                'data: {"status_id":"status-1","status_key":"searching_knowledge_base\\u241frag_worker\\u241factive","status_seq":3,"status_elapsed_ms":3000,"description":"Searching knowledge base • rag_worker • 00:03 elapsed","status":"in_progress","done":false,"hidden":false,"elapsed_ms":3000,"delta_ms":1000,"agent":"rag_worker","selected_agent":"general","phase":"searching_knowledge_base","phase_label":"Searching Knowledge Base","phase_elapsed_ms":3000,"source_event_type":"heartbeat","label":"Searching knowledge base","detail":"","job_id":"","task_id":"","why":"","waiting_on":"","timestamp":1713139203000}',
                "",
                "event: status",
                'data: {"status_id":"status-1","status_key":"searching_knowledge_base\\u241frag_worker\\u241factive","status_seq":2,"status_elapsed_ms":2000,"description":"Searching knowledge base • rag_worker • 00:02 elapsed","status":"in_progress","done":false,"hidden":false,"elapsed_ms":2000,"delta_ms":1000,"agent":"rag_worker","selected_agent":"general","phase":"searching_knowledge_base","phase_label":"Searching Knowledge Base","phase_elapsed_ms":2000,"source_event_type":"heartbeat","label":"Searching knowledge base","detail":"","job_id":"","task_id":"","why":"","waiting_on":"","timestamp":1713139202000}',
                "",
                'data: {"choices":[{"delta":{"content":"Grounded answer."},"finish_reason":null}]}',
                "",
                'data: [DONE]',
                "",
            ]
            for item in lines:
                yield item

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del method, url, headers, json, kwargs
            return _StreamContext()

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(pipe, "_now", fake_now)

    result = asyncio.run(
        pipe.pipe(
            {"messages": [{"role": "user", "content": "Explain the architecture deeply and keep it grounded."}]},
            __event_emitter__=emit,
        )
    )

    assert result == "Grounded answer."
    descriptions = [item["data"]["description"] for item in emitted]
    assert descriptions[-1] == "Searching knowledge base • rag_worker • 00:03 elapsed"
    assert "Searching knowledge base • rag_worker • 00:02 elapsed" not in descriptions
    assert emitted[-1]["data"]["agentic_status"]["chips"][-1] == "Live"
    assert emitted[-1]["data"]["agentic_status"]["timing"]["kind"] == "stage"
    assert emitted[-1]["data"]["agentic_status"]["timing"]["live"] is True
    assert emitted[-1]["data"]["agentic_status"]["timing"]["elapsed_ms"] == 3000


def test_pipe_emits_failed_duration_when_stream_and_fallback_fail(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []
    monotonic_values = [200.0, 200.0, 202.0, 207.0]

    def fake_now() -> float:
        if len(monotonic_values) > 1:
            return monotonic_values.pop(0)
        return monotonic_values[0]

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, *, headers=None, json=None, **kwargs):
            del method, url, headers, json, kwargs

            class _StreamContext:
                async def __aenter__(self):
                    raise RuntimeError("stream failed")

                async def __aexit__(self, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return None

            return _StreamContext()

        async def post(self, url, *, headers=None, json=None, **kwargs):
            del url, headers, json, kwargs
            raise RuntimeError("fallback failed")

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(pipe, "_now", fake_now)

    try:
        asyncio.run(
            pipe.pipe(
                {
                    "messages": [
                        {"role": "user", "content": "Tell me about the architecture docs."},
                    ]
                },
                __event_emitter__=emit,
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "fallback failed"
    else:
        raise AssertionError("Expected RuntimeError")

    assert emitted[0]["data"]["description"] == "Starting • 00:00 elapsed"
    assert emitted[1]["data"]["description"] == "Searching knowledge base • 00:02 elapsed"
    assert emitted[-1]["data"]["description"] == "Failed during Searching knowledge base • +00:05 (total 00:07)"
    assert emitted[-1]["data"]["done"] is True
    assert emitted[-1]["data"]["phase"] == "failed"
    assert emitted[-1]["data"]["status"] == "error"
    assert emitted[-1]["data"]["agentic_status"]["title"] == "Run failed"
    assert emitted[-1]["data"]["agentic_status"]["state"] == "error"
    assert emitted[-1]["data"]["agentic_status"]["timing"]["kind"] == "total"
    assert emitted[-1]["data"]["agentic_status"]["timing"]["live"] is False
    assert emitted[-1]["data"]["agentic_status"]["timing"]["elapsed_ms"] == 7000


def test_pipe_upload_handoff_failure_emits_failed_status_and_friendly_error(monkeypatch, caplog) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []
    monotonic_values = [50.0, 50.0, 50.0, 51.0]
    captured_uploads: list[dict[str, object]] = []

    def fake_now() -> float:
        if len(monotonic_values) > 1:
            return monotonic_values.pop(0)
        return monotonic_values[0]

    class _DownloadResponse:
        def __init__(self):
            self.content = b"reviews\nGreat service\n"
            self.headers = {"content-type": "text/csv"}

        def raise_for_status(self) -> None:
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def get(self, *args, **kwargs):
            del args, kwargs
            return _DownloadResponse()

        async def post(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("Async multipart upload should not be used for /v1/upload")

        def stream(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("Chat stream should not start after upload handoff failure")

    class _SyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def post(self, url, *, params=None, headers=None, files=None, data=None, **kwargs):
            del kwargs
            captured_uploads.append(
                {
                    "url": url,
                    "params": params or {},
                    "headers": headers or {},
                    "files": files or {},
                    "data": data,
                }
            )

            class _ErrorResponse:
                status_code = 422
                text = "invalid upload"
                is_error = True

                def raise_for_status(self) -> None:
                    request = httpx.Request("POST", url)
                    response = httpx.Response(status_code=422, request=request, text=self.text)
                    raise httpx.HTTPStatusError("upload rejected", request=request, response=response)

            return _ErrorResponse()

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(module.httpx, "Client", _SyncClient)
    monkeypatch.setattr(pipe, "_now", fake_now)
    caplog.set_level(logging.ERROR)

    result = asyncio.run(
        pipe.pipe(
            {"messages": [{"role": "user", "content": "Provide sentiment analysis of all of the reviews in the reviews column."}]},
            __metadata__={"chat_id": "chat-123", "message_id": "msg-123"},
            __files__=[{"id": "file-1", "filename": "customer_reviews_100.csv"}],
            __request__=SimpleNamespace(
                headers={"authorization": "Bearer owui-token"},
                cookies={"session": "cookie-1"},
            ),
            __event_emitter__=emit,
        )
    )

    assert "backend rejected the upload request" in result
    assert "invalid upload" in result
    assert emitted[0]["data"]["phase"] == "starting"
    assert emitted[1]["data"]["phase"] == "uploading_inputs"
    assert emitted[-1]["data"]["phase"] == "failed"
    assert emitted[-1]["data"]["status"] == "error"
    assert emitted[-1]["data"]["agentic_status"]["title"] == "Run failed"
    assert all(item["data"]["phase"] != "searching_knowledge_base" for item in emitted)
    assert captured_uploads[0]["headers"]["X-Upload-Source-Ids"] == json.dumps(["file-1"])
    assert captured_uploads[0]["data"] is None
    assert any("backend_upload_failed_response" in record.getMessage() for record in caplog.records)
    assert any("status_code=422" in record.getMessage() for record in caplog.records)
    assert any("backend_upload_failed" in record.getMessage() for record in caplog.records)


def test_backend_upload_failure_message_names_runtime_registry_tools() -> None:
    module = _load_pipe_module()

    class _Response:
        status_code = 503
        text = json.dumps(
            {
                "detail": {
                    "error_code": "runtime_registry_invalid",
                    "missing_tools": [
                        {"agent": "coordinator", "tool": "list_worker_requests"},
                        {"agent": "coordinator", "tool": "respond_worker_request"},
                    ],
                    "affected_agents": ["coordinator"],
                    "remediation": "Rebuild/recreate the app image.",
                }
            }
        )

        def json(self):
            return json.loads(self.text)

    message = module._backend_upload_failure_message(_Response())

    assert "backend runtime registry is invalid" in message
    assert "coordinator" in message
    assert "list_worker_requests" in message
    assert "respond_worker_request" in message
    assert "Rebuild/recreate the app image" in message


def test_backend_completion_failure_message_names_runtime_registry_tools() -> None:
    module = _load_pipe_module()

    class _Response:
        status_code = 503
        text = json.dumps(
            {
                "error_code": "runtime_registry_invalid",
                "missing_tools": [
                    {"agent": "general", "tool": "mcp__*"},
                    {"agent": "coordinator", "tool": "create_team_channel"},
                ],
                "affected_agents": ["coordinator", "general"],
                "remediation": "Rebuild/recreate the app image.",
            }
        )

        def json(self):
            return json.loads(self.text)

    message = module._backend_completion_failure_message(_Response())

    assert "Agent request failed because the backend runtime registry is invalid" in message
    assert "coordinator" in message
    assert "general" in message
    assert "mcp__*" in message
    assert "create_team_channel" in message
    assert "Rebuild/recreate the app image" in message


def test_pipe_returns_backend_503_message_from_stream(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []

    class _StreamResponse:
        status_code = 503
        text = json.dumps(
            {
                "error_code": "runtime_registry_invalid",
                "missing_tools": [{"agent": "coordinator", "tool": "create_team_channel"}],
                "affected_agents": ["coordinator"],
                "remediation": "Rebuild/recreate the app image.",
            }
        )

        async def aread(self):
            return self.text.encode("utf-8")

        def json(self):
            return json.loads(self.text)

        def raise_for_status(self) -> None:
            raise AssertionError("status handling should parse backend 503s before raise_for_status")

    class _StreamContext:
        async def __aenter__(self):
            return _StreamResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def post(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("backend 503 stream responses should not retry as non-stream")

        def stream(self, *args, **kwargs):
            del args, kwargs
            return _StreamContext()

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)

    async def emit(event):
        emitted.append(event)

    result = asyncio.run(
        pipe.pipe(
            {"messages": [{"role": "user", "content": "What documents are indexed?"}]},
            __metadata__={"chat_id": "chat-503", "message_id": "msg-503"},
            __event_emitter__=emit,
        )
    )

    assert "backend runtime registry is invalid" in result
    assert "create_team_channel" in result
    assert "Rebuild/recreate the app image" in result
    assert emitted[-1]["data"]["phase"] == "failed"
    assert emitted[-1]["data"]["status"] == "error"


def test_pipe_returns_backend_503_message_from_non_stream_fallback(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    captured: dict[str, object] = {}

    class _PostResponse:
        status_code = 503
        text = json.dumps(
            {
                "detail": {
                    "error_code": "runtime_registry_invalid",
                    "missing_tools": [{"agent": "general", "tool": "mcp__*"}],
                    "affected_agents": ["general"],
                    "remediation": "Rebuild/recreate the app image.",
                }
            }
        )

        def json(self):
            return json.loads(self.text)

        def raise_for_status(self) -> None:
            raise AssertionError("status handling should parse backend 503s before raise_for_status")

    class _BrokenStreamContext:
        async def __aenter__(self):
            request = httpx.Request("POST", "http://app:8000/v1/chat/completions")
            raise httpx.ConnectError("stream connection failed", request=request)

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def post(self, url, *, headers=None, json=None, **kwargs):
            del url, headers, kwargs
            captured["fallback_payload"] = dict(json or {})
            return _PostResponse()

        def stream(self, *args, **kwargs):
            del args, kwargs
            return _BrokenStreamContext()

    monkeypatch.setattr(module.httpx, "AsyncClient", _AsyncClient)

    result = asyncio.run(
        pipe.pipe(
            {"messages": [{"role": "user", "content": "Summarize the KB."}]},
            __metadata__={"chat_id": "chat-fallback-503", "message_id": "msg-fallback-503"},
        )
    )

    assert "backend runtime registry is invalid" in result
    assert "mcp__*" in result
    assert "Rebuild/recreate the app image" in result
    assert captured["fallback_payload"]["stream"] is False


def test_active_phase_live_status_updates_every_second(monkeypatch) -> None:
    module = _load_pipe_module()
    pipe = module.Pipe()
    emitted: list[dict[str, object]] = []
    monotonic_values = [300.0, 300.0, 301.0, 301.4, 302.0]

    def fake_now() -> float:
        if len(monotonic_values) > 1:
            return monotonic_values.pop(0)
        return monotonic_values[0]

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(pipe, "_now", fake_now)
    state = module._TurnStatusState(tracker=module.TurnStatusTracker(turn_started_at=300.0))

    async def scenario():
        await pipe._emit_status_snapshots(emit, state, state.tracker.start_snapshots(pipe._now()))
        await pipe._transition_phase(
            emit,
            state,
            module.PHASE_SEARCHING,
            source_event_type="test_phase",
            label="Searching knowledge base",
            detail="Waiting on retrieval",
        )
        for _ in range(3):
            now = pipe._now()
            async with state.emit_lock:
                due_in = state.tracker.seconds_until_next_heartbeat(
                    now,
                    interval_seconds=module.STATUS_HEARTBEAT_SECONDS,
                )
                heartbeat = (
                    state.tracker.heartbeat_snapshot(now)
                    if due_in == 0.0
                    else None
                )
            await pipe._emit_status_snapshots(emit, state, [heartbeat] if heartbeat else [])

    asyncio.run(scenario())

    assert [item["data"]["description"] for item in emitted] == [
        "Starting • 00:00 elapsed",
        "Searching knowledge base • 00:00 elapsed",
        "Searching knowledge base • 00:01 elapsed",
        "Searching knowledge base • 00:02 elapsed",
    ]
    assert all(item["data"]["phase"] == "searching_knowledge_base" for item in emitted[1:])
    assert emitted[1]["data"]["status_id"] == emitted[2]["data"]["status_id"] == emitted[3]["data"]["status_id"]
    assert emitted[1]["data"]["status_seq"] < emitted[2]["data"]["status_seq"] < emitted[3]["data"]["status_seq"]
    assert emitted[3]["data"]["agentic_status"]["chips"][-1] == "Live"
    assert emitted[1]["data"]["agentic_status"]["timing"]["elapsed_ms"] == 0
    assert emitted[2]["data"]["agentic_status"]["timing"]["elapsed_ms"] == 1000
    assert emitted[3]["data"]["agentic_status"]["timing"]["elapsed_ms"] == 2000


def test_phase_mapping_collapses_low_level_progress_labels() -> None:
    assert infer_phase_from_progress({"label": "Reading document", "detail": "KB_123"}) == PHASE_SEARCHING
    assert infer_phase_from_progress({"label": "Spawning 3 evidence workers", "detail": "Parallel evidence gathering"}) == PHASE_SEARCHING
    assert infer_phase_from_progress({"tool": "list_graph_indexes", "label": "Preparing list_graph_indexes"}) == PHASE_GRAPH_CATALOG
    assert infer_phase_from_progress({"label": "Synthesizing answer", "detail": "Grounding final response"}) == PHASE_SYNTHESIZING
    assert infer_phase_from_progress({"label": "Something else", "detail": "Noisy detail"}) == ""
