from __future__ import annotations

import json

import httpx
import pytest

from agentic_chatbot_next.mcp_servers.web_research import (
    WebResearchClient,
    WebResearchConfigError,
    WebResearchProviderError,
    WebResearchUrlError,
    validate_public_url,
)


def _json_body(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))


def test_validate_public_url_blocks_non_public_targets() -> None:
    assert validate_public_url("https://example.com/path?q=1") == "https://example.com/path?q=1"

    blocked = [
        "ftp://example.com/file",
        "https://user:pass@example.com/secret",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]:8000",
    ]
    for url in blocked:
        with pytest.raises(WebResearchUrlError):
            validate_public_url(url)

    assert validate_public_url("http://127.0.0.1:8000", allow_private_network=True) == "http://127.0.0.1:8000"


@pytest.mark.asyncio
async def test_web_search_calls_tavily_and_normalizes_results() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = _json_body(request)
        return httpx.Response(
            200,
            json={
                "answer": "Short answer",
                "results": [
                    {
                        "title": "Example result",
                        "url": "https://example.com/a",
                        "content": "Useful snippet",
                        "published_date": "2026-05-01",
                        "score": 0.9,
                    }
                ],
                "response_time": 0.12,
                "request_id": "req-1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = WebResearchClient(tavily_api_key="tvly-secret", http_client=http_client)
        result = await client.web_search(
            query="agentic search",
            max_results=3,
            include_domains=["example.com"],
            country="united states",
            include_answer=True,
        )

    assert captured["authorization"] == "Bearer tvly-secret"
    assert captured["payload"] == {
        "query": "agentic search",
        "topic": "general",
        "search_depth": "basic",
        "max_results": 3,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
        "include_favicon": False,
        "include_domains": ["example.com"],
        "country": "united states",
    }
    assert result["answer"] == "Short answer"
    assert result["results"][0]["url"] == "https://example.com/a"
    assert result["results"][0]["source_provider"] == "tavily"
    assert result["usage"]["request_id"] == "req-1"


@pytest.mark.asyncio
async def test_read_url_uses_tavily_extract_first() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _json_body(request)
        assert request.headers["authorization"] == "Bearer tvly-secret"
        assert payload["urls"] == ["https://example.com/article"]
        assert payload["format"] == "markdown"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/article",
                        "raw_content": "# Article\n\nBody text",
                    }
                ],
                "failed_results": [],
                "request_id": "extract-1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = WebResearchClient(tavily_api_key="tvly-secret", http_client=http_client)
        result = await client.read_url(url="https://example.com/article", max_chars=9)

    assert result["provider"] == "tavily"
    assert result["content"] == "# Article"
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_read_url_auto_falls_back_to_jina_when_tavily_is_unconfigured() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        assert request.url.host == "r.jina.ai"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "url": "https://example.com/article",
                "title": "Jina title",
                "content": "Jina extracted content",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = WebResearchClient(tavily_api_key="", http_client=http_client)
        result = await client.read_url(url="https://example.com/article")

    assert len(requests) == 1
    assert result["provider"] == "jina"
    assert result["title"] == "Jina title"
    assert "Tavily extraction unavailable" in result["warnings"][0]


@pytest.mark.asyncio
async def test_batch_read_urls_uses_jina_for_tavily_failed_urls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.tavily.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"url": "https://example.com/one", "raw_content": "One content"},
                    ],
                    "failed_results": [
                        {"url": "https://example.com/two", "error": "blocked upstream"},
                    ],
                },
            )
        assert request.url.host == "r.jina.ai"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"url": "https://example.com/two", "title": "Two", "content": "Two content"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = WebResearchClient(tavily_api_key="tvly-secret", http_client=http_client)
        result = await client.batch_read_urls(urls=["https://example.com/one", "https://example.com/two"])

    assert [item["provider"] for item in result["results"]] == ["tavily", "jina"]
    assert result["failed_results"] == []


@pytest.mark.asyncio
async def test_search_and_read_searches_then_extracts_top_urls() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "One", "url": "https://example.com/one", "content": "One snippet"},
                        {"title": "Two", "url": "https://example.com/two", "content": "Two snippet"},
                    ]
                },
            )
        payload = _json_body(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": url, "raw_content": f"Content for {url}"}
                    for url in payload["urls"]
                ],
                "failed_results": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = WebResearchClient(tavily_api_key="tvly-secret", http_client=http_client)
        result = await client.search_and_read(query="agentic search", max_results=2, read_top_n=2)

    assert seen_paths == ["/search", "/extract"]
    assert len(result["search"]["results"]) == 2
    assert len(result["pages"]) == 2


@pytest.mark.asyncio
async def test_tavily_missing_key_and_error_messages_are_safe() -> None:
    client = WebResearchClient(tavily_api_key="")
    try:
        with pytest.raises(WebResearchConfigError):
            await client.web_search(query="latest AI")
    finally:
        await client.close()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"bad key {request.headers['authorization']}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        keyed_client = WebResearchClient(tavily_api_key="tvly-secret", http_client=http_client)
        with pytest.raises(WebResearchProviderError) as exc_info:
            await keyed_client.web_search(query="latest AI")

    assert "tvly-secret" not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)


def test_server_module_exposes_streamable_http_read_only_tools() -> None:
    from agentic_chatbot_next.mcp_servers.web_research import mcp

    tool_names = {tool.name for tool in mcp._tool_manager.list_tools()}  # type: ignore[attr-defined]

    assert {"web_search", "read_url", "batch_read_urls", "search_and_read"} <= tool_names
    raw_tools = [tool for tool in mcp._tool_manager.list_tools() if tool.name == "web_search"]  # type: ignore[attr-defined]
    assert json.loads(raw_tools[0].annotations.model_dump_json())["readOnlyHint"] is True
