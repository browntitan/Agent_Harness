from __future__ import annotations

import ipaddress
import logging
import os
import re
from typing import Annotated, Any, Mapping
from urllib.parse import urlsplit

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
JINA_READER_BASE_URL = "https://r.jina.ai/"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_CONTENT_CHARS = 20_000
MAX_CONTENT_CHARS = 50_000
MAX_SEARCH_RESULTS = 20
MAX_BATCH_URLS = 10
MAX_SEARCH_AND_READ_RESULTS = 10
MAX_READ_TOP_N = 5
VALID_TOPICS = {"general", "news", "finance"}
VALID_READ_PROVIDERS = {"auto", "tavily", "jina"}
VALID_FORMATS = {"markdown", "text"}
LOCALHOST_NAMES = {"localhost", "localhost.localdomain"}

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class WebResearchConfigError(RuntimeError):
    """Raised when the web research MCP server is missing required runtime config."""


class WebResearchProviderError(RuntimeError):
    """Raised when a web research provider returns an error response."""


class WebResearchUrlError(ValueError):
    """Raised when a URL is outside the allowed public-page fetch policy."""


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    clean = str(value).strip().lower()
    if not clean:
        return default
    return clean in {"1", "true", "yes", "y", "on"}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list | tuple | set) else str(value).split(",")
    seen: set[str] = set()
    items: list[str] = []
    for item in list(raw_items or []):
        clean = _clean_string(item)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        items.append(clean)
    return items


def _redact_secret(text: str, *secrets: str) -> str:
    clean = str(text or "")
    for secret in secrets:
        if secret:
            clean = clean.replace(secret, "[redacted]")
    return clean


def _bounded_limit(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _bounded_max_chars(value: Any, default: int) -> int:
    return _bounded_limit(value, default=default, minimum=1, maximum=MAX_CONTENT_CHARS)


def _safe_topic(value: str) -> str:
    topic = _clean_string(value).lower() or "general"
    if topic not in VALID_TOPICS:
        raise ValueError(f"topic must be one of {sorted(VALID_TOPICS)}.")
    return topic


def _safe_provider(value: str) -> str:
    provider = _clean_string(value).lower() or "auto"
    if provider not in VALID_READ_PROVIDERS:
        raise ValueError(f"provider must be one of {sorted(VALID_READ_PROVIDERS)}.")
    return provider


def _safe_format(value: str) -> str:
    output_format = _clean_string(value).lower() or "markdown"
    if output_format not in VALID_FORMATS:
        raise ValueError(f"format must be one of {sorted(VALID_FORMATS)}.")
    return output_format


def _host_is_blocked(hostname: str) -> bool:
    clean_host = hostname.strip().strip("[]").lower().rstrip(".")
    if not clean_host:
        return True
    if clean_host in LOCALHOST_NAMES or clean_host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(clean_host)
    except ValueError:
        return False
    return any(
        (
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_reserved,
            address.is_multicast,
            address.is_unspecified,
        )
    )


def validate_public_url(url: str, *, allow_private_network: bool = False) -> str:
    clean_url = _clean_string(url)
    if not clean_url:
        raise WebResearchUrlError("url is required.")
    parts = urlsplit(clean_url)
    if parts.scheme.lower() not in {"http", "https"}:
        raise WebResearchUrlError("Only http and https URLs can be fetched.")
    if not parts.netloc or not parts.hostname:
        raise WebResearchUrlError("URL must include a host.")
    if parts.username or parts.password:
        raise WebResearchUrlError("Credentialed URLs are not allowed.")
    if not allow_private_network and _host_is_blocked(parts.hostname):
        raise WebResearchUrlError("Private, localhost, link-local, and reserved network URLs are blocked.")
    return clean_url


def _provider_error_message(response: httpx.Response, provider: str, *secrets: str) -> str:
    status = response.status_code
    if status in {401, 403}:
        return f"{provider} rejected the configured API key. Regenerate the key and update the MCP server environment."
    if status == 429:
        return f"{provider} rate limit was exceeded. Try again later or use a higher-limit key."
    if 500 <= status <= 599:
        return f"{provider} returned HTTP {status}. Try again later."
    body = _redact_secret(response.text[:500], *secrets)
    return f"{provider} returned HTTP {status}: {body}"


def _truncate_content(content: str, *, max_chars: int) -> tuple[str, bool]:
    clean = str(content or "")
    if len(clean) <= max_chars:
        return clean, False
    return clean[:max_chars].rstrip(), True


def _title_from_content(content: str) -> str:
    for line in str(content or "").splitlines():
        clean = line.strip().strip("#").strip()
        if not clean:
            continue
        match = re.match(r"^title\s*:\s*(.+)$", clean, flags=re.IGNORECASE)
        return (match.group(1) if match else clean)[:300].strip()
    return ""


def _content_from_tavily_result(raw: Mapping[str, Any]) -> str:
    for key in ("raw_content", "content", "text"):
        content = _clean_string(raw.get(key))
        if content:
            return content
    return ""


def _page_result(
    *,
    source_url: str,
    final_url: str,
    title: str,
    content: str,
    output_format: str,
    provider: str,
    max_chars: int,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    clipped, truncated = _truncate_content(content, max_chars=max_chars)
    return {
        "object": "web_research.page",
        "title": _clean_string(title) or _title_from_content(clipped),
        "url": source_url,
        "final_url": final_url or source_url,
        "content": clipped,
        "format": output_format,
        "content_chars": len(clipped),
        "truncated": truncated,
        "provider": provider,
        "warnings": list(warnings or []),
    }


class WebResearchClient:
    def __init__(
        self,
        *,
        tavily_api_key: str | None = None,
        jina_api_key: str | None = None,
        tavily_search_url: str | None = None,
        tavily_extract_url: str | None = None,
        jina_reader_base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float | None = None,
        max_content_chars: int | None = None,
        allow_private_network: bool | None = None,
    ) -> None:
        self.tavily_api_key = _clean_string(tavily_api_key if tavily_api_key is not None else os.getenv("TAVILY_API_KEY"))
        self.jina_api_key = _clean_string(jina_api_key if jina_api_key is not None else os.getenv("JINA_API_KEY"))
        self.tavily_search_url = _clean_string(tavily_search_url or os.getenv("TAVILY_SEARCH_URL")) or TAVILY_SEARCH_URL
        self.tavily_extract_url = _clean_string(tavily_extract_url or os.getenv("TAVILY_EXTRACT_URL")) or TAVILY_EXTRACT_URL
        self.jina_reader_base_url = _clean_string(jina_reader_base_url or os.getenv("JINA_READER_BASE_URL")) or JINA_READER_BASE_URL
        self.timeout_seconds = float(timeout_seconds or os.getenv("WEB_RESEARCH_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS)
        self.max_content_chars = _bounded_max_chars(max_content_chars or os.getenv("WEB_RESEARCH_MAX_CONTENT_CHARS"), DEFAULT_MAX_CONTENT_CHARS)
        self.allow_private_network = (
            bool(allow_private_network)
            if allow_private_network is not None
            else _as_bool(os.getenv("WEB_RESEARCH_ALLOW_PRIVATE_NETWORK"), default=False)
        )
        self._owned_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True)

    async def close(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    def _validated_url(self, url: str) -> str:
        return validate_public_url(url, allow_private_network=self.allow_private_network)

    def _tavily_key(self) -> str:
        if not self.tavily_api_key:
            raise WebResearchConfigError(
                "TAVILY_API_KEY is not configured for the Web Research MCP server. "
                "Create a Tavily API key and set it only in the MCP server environment."
            )
        return self.tavily_api_key

    async def _post_tavily_json(self, url: str, *, payload: Mapping[str, Any]) -> dict[str, Any]:
        api_key = self._tavily_key()
        try:
            response = await self._client.post(
                url,
                json=dict(payload),
                headers={"Authorization": f"Bearer {api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise WebResearchProviderError("Tavily request timed out. Try again later.") from exc
        except httpx.HTTPError as exc:
            raise WebResearchProviderError(f"Tavily request failed: {_redact_secret(str(exc), api_key)}") from exc
        if response.status_code >= 400:
            raise WebResearchProviderError(_provider_error_message(response, "Tavily", api_key))
        try:
            data = response.json()
        except ValueError as exc:
            raise WebResearchProviderError("Tavily returned a non-JSON response.") from exc
        if not isinstance(data, dict):
            raise WebResearchProviderError("Tavily returned an unexpected response payload.")
        return data

    async def web_search(
        self,
        *,
        query: str,
        max_results: int = 5,
        topic: str = "general",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        country: str = "",
        include_answer: bool = False,
    ) -> dict[str, Any]:
        clean_query = _clean_string(query)
        if not clean_query:
            raise ValueError("query is required.")
        clean_topic = _safe_topic(topic)
        clean_max_results = _bounded_limit(max_results, default=5, minimum=1, maximum=MAX_SEARCH_RESULTS)
        domains_to_include = _string_list(include_domains)
        domains_to_exclude = _string_list(exclude_domains)
        clean_country = _clean_string(country).lower()
        payload: dict[str, Any] = {
            "query": clean_query,
            "topic": clean_topic,
            "search_depth": "basic",
            "max_results": clean_max_results,
            "include_answer": bool(include_answer),
            "include_raw_content": False,
            "include_images": False,
            "include_favicon": False,
        }
        if domains_to_include:
            payload["include_domains"] = domains_to_include
        if domains_to_exclude:
            payload["exclude_domains"] = domains_to_exclude
        warnings: list[str] = []
        if clean_country:
            if clean_topic == "general":
                payload["country"] = clean_country
            else:
                warnings.append("country was ignored because Tavily only supports country boosts for general search.")
        data = await self._post_tavily_json(self.tavily_search_url, payload=payload)
        results: list[dict[str, Any]] = []
        for raw in list(data.get("results") or []):
            if not isinstance(raw, Mapping):
                continue
            result_url = _clean_string(raw.get("url"))
            try:
                safe_url = self._validated_url(result_url)
            except WebResearchUrlError as exc:
                warnings.append(f"Dropped unsafe search result URL: {exc}")
                continue
            results.append(
                {
                    "title": _clean_string(raw.get("title")),
                    "url": safe_url,
                    "snippet": _clean_string(raw.get("content") or raw.get("snippet")),
                    "published_date": _clean_string(raw.get("published_date")),
                    "score": raw.get("score"),
                    "source_provider": "tavily",
                }
            )
        return {
            "object": "web_research.search",
            "query": {
                "query": clean_query,
                "max_results": clean_max_results,
                "topic": clean_topic,
                "include_domains": domains_to_include,
                "exclude_domains": domains_to_exclude,
                "country": clean_country,
                "include_answer": bool(include_answer),
            },
            "answer": _clean_string(data.get("answer")) if include_answer else "",
            "results": results,
            "returned_results": len(results),
            "provider": "tavily",
            "usage": {
                "response_time": data.get("response_time"),
                "request_id": _clean_string(data.get("request_id")),
                "raw_usage": data.get("usage") if isinstance(data.get("usage"), Mapping) else {},
            },
            "warnings": warnings,
        }

    async def _tavily_extract_pages(
        self,
        urls: list[str],
        *,
        output_format: str,
        max_chars: int,
    ) -> dict[str, Any]:
        clean_urls = [self._validated_url(url) for url in urls]
        payload = {
            "urls": clean_urls,
            "include_images": False,
            "include_favicon": False,
            "extract_depth": "basic",
            "format": output_format,
            "timeout": max(1.0, min(float(self.timeout_seconds), 60.0)),
        }
        data = await self._post_tavily_json(self.tavily_extract_url, payload=payload)
        results: list[dict[str, Any]] = []
        for raw in list(data.get("results") or []):
            if not isinstance(raw, Mapping):
                continue
            source_url = _clean_string(raw.get("url"))
            try:
                final_url = self._validated_url(source_url)
            except WebResearchUrlError as exc:
                results.append({"url": source_url, "error": str(exc), "provider": "tavily"})
                continue
            content = _content_from_tavily_result(raw)
            results.append(
                _page_result(
                    source_url=source_url,
                    final_url=final_url,
                    title=_clean_string(raw.get("title")),
                    content=content,
                    output_format=output_format,
                    provider="tavily",
                    max_chars=max_chars,
                    warnings=[] if content else ["Tavily returned empty extracted content."],
                )
            )
        failed_results = []
        for raw in list(data.get("failed_results") or []):
            failed_results.append(
                {
                    "url": _clean_string(raw.get("url") if isinstance(raw, Mapping) else raw),
                    "error": _clean_string(raw.get("error") if isinstance(raw, Mapping) else "Tavily extraction failed."),
                    "provider": "tavily",
                }
            )
        return {
            "object": "web_research.tavily_extract",
            "results": [item for item in results if item.get("object") == "web_research.page"],
            "failed_results": failed_results + [item for item in results if item.get("object") != "web_research.page"],
            "usage": {
                "response_time": data.get("response_time"),
                "request_id": _clean_string(data.get("request_id")),
            },
        }

    async def _jina_read_page(self, url: str, *, output_format: str, max_chars: int) -> dict[str, Any]:
        safe_url = self._validated_url(url)
        reader_url = f"{self.jina_reader_base_url.rstrip('/')}/{safe_url}"
        headers = {
            "Accept": "application/json",
            "X-Respond-With": "text" if output_format == "text" else "markdown",
        }
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"
        try:
            response = await self._client.get(reader_url, headers=headers, follow_redirects=True)
        except httpx.TimeoutException as exc:
            raise WebResearchProviderError("Jina Reader request timed out. Try again later.") from exc
        except httpx.HTTPError as exc:
            raise WebResearchProviderError(f"Jina Reader request failed: {_redact_secret(str(exc), self.jina_api_key)}") from exc
        if response.status_code >= 400:
            raise WebResearchProviderError(_provider_error_message(response, "Jina Reader", self.jina_api_key))
        content_type = response.headers.get("content-type", "")
        payload: dict[str, Any] = {}
        if "json" in content_type:
            try:
                parsed = response.json()
            except ValueError:
                parsed = {}
            if isinstance(parsed, Mapping):
                payload = dict(parsed)
        content = _clean_string(payload.get("content")) if payload else response.text
        final_url = _clean_string(payload.get("url")) or safe_url
        final_url = self._validated_url(final_url)
        return _page_result(
            source_url=safe_url,
            final_url=final_url,
            title=_clean_string(payload.get("title")),
            content=content,
            output_format=output_format,
            provider="jina",
            max_chars=max_chars,
            warnings=[],
        )

    async def read_url(
        self,
        *,
        url: str,
        provider: str = "auto",
        format: str = "markdown",
        max_chars: int | None = None,
        query: str = "",
    ) -> dict[str, Any]:
        safe_url = self._validated_url(url)
        clean_provider = _safe_provider(provider)
        output_format = _safe_format(format)
        clean_max_chars = _bounded_max_chars(max_chars, self.max_content_chars)
        warnings: list[str] = []
        if _clean_string(query):
            warnings.append("query is accepted for caller context but is not sent to the v1 extraction providers.")
        if clean_provider in {"auto", "tavily"}:
            try:
                extracted = await self._tavily_extract_pages([safe_url], output_format=output_format, max_chars=clean_max_chars)
                if extracted["results"]:
                    page = dict(extracted["results"][0])
                    page["warnings"] = warnings + list(page.get("warnings") or [])
                    return page
                tavily_failure = extracted["failed_results"][0]["error"] if extracted["failed_results"] else "No extracted content returned."
                if clean_provider == "tavily":
                    raise WebResearchProviderError(f"Tavily extraction failed: {tavily_failure}")
                warnings.append(f"Tavily extraction failed; using Jina Reader fallback: {tavily_failure}")
            except (WebResearchConfigError, WebResearchProviderError) as exc:
                if clean_provider == "tavily":
                    raise
                warnings.append(f"Tavily extraction unavailable; using Jina Reader fallback: {exc}")
        page = await self._jina_read_page(safe_url, output_format=output_format, max_chars=clean_max_chars)
        page["warnings"] = warnings + list(page.get("warnings") or [])
        return page

    async def batch_read_urls(
        self,
        *,
        urls: list[str],
        provider: str = "auto",
        format: str = "markdown",
        max_chars: int | None = None,
        query: str = "",
    ) -> dict[str, Any]:
        clean_provider = _safe_provider(provider)
        output_format = _safe_format(format)
        clean_max_chars = _bounded_max_chars(max_chars, self.max_content_chars)
        input_urls = [_clean_string(url) for url in list(urls or []) if _clean_string(url)]
        input_urls = input_urls[:MAX_BATCH_URLS]
        safe_urls: list[str] = []
        failed_results: list[dict[str, Any]] = []
        for input_url in input_urls:
            try:
                safe_urls.append(self._validated_url(input_url))
            except WebResearchUrlError as exc:
                failed_results.append({"url": input_url, "error": str(exc), "provider": clean_provider})
        warnings = []
        if _clean_string(query):
            warnings.append("query is accepted for caller context but is not sent to the v1 extraction providers.")
        results: list[dict[str, Any]] = []
        pending_fallback_urls = list(safe_urls)
        if clean_provider in {"auto", "tavily"} and safe_urls:
            try:
                extracted = await self._tavily_extract_pages(safe_urls, output_format=output_format, max_chars=clean_max_chars)
                results.extend(extracted["results"])
                successful_urls = {str(item.get("url") or item.get("final_url") or "") for item in extracted["results"]}
                pending_fallback_urls = [url for url in safe_urls if url not in successful_urls]
                if clean_provider == "tavily":
                    failed_results.extend(extracted["failed_results"])
                    pending_fallback_urls = []
                elif extracted["failed_results"]:
                    failed_results.extend(extracted["failed_results"])
            except (WebResearchConfigError, WebResearchProviderError) as exc:
                if clean_provider == "tavily":
                    raise
                warnings.append(f"Tavily batch extraction unavailable; using Jina Reader fallback: {exc}")
        if clean_provider in {"auto", "jina"}:
            for pending_url in pending_fallback_urls:
                try:
                    results.append(await self._jina_read_page(pending_url, output_format=output_format, max_chars=clean_max_chars))
                    failed_results = [item for item in failed_results if item.get("url") != pending_url]
                except Exception as exc:
                    failed_results.append({"url": pending_url, "error": str(exc), "provider": "jina"})
        return {
            "object": "web_research.batch_page_read",
            "query": {
                "url_count": len(input_urls),
                "provider": clean_provider,
                "format": output_format,
                "max_chars": clean_max_chars,
            },
            "results": results,
            "failed_results": failed_results,
            "warnings": warnings,
        }

    async def search_and_read(
        self,
        *,
        query: str,
        max_results: int = 5,
        read_top_n: int = 3,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        country: str = "",
        format: str = "markdown",
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        clean_max_results = _bounded_limit(max_results, default=5, minimum=1, maximum=MAX_SEARCH_AND_READ_RESULTS)
        clean_read_top_n = _bounded_limit(read_top_n, default=3, minimum=0, maximum=MAX_READ_TOP_N)
        search = await self.web_search(
            query=query,
            max_results=clean_max_results,
            topic="general",
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            country=country,
            include_answer=False,
        )
        urls = [str(item.get("url") or "") for item in search["results"][:clean_read_top_n]]
        pages = (
            await self.batch_read_urls(urls=urls, provider="auto", format=format, max_chars=max_chars, query=query)
            if urls
            else {"results": [], "failed_results": [], "warnings": []}
        )
        return {
            "object": "web_research.search_and_read",
            "query": {
                "query": _clean_string(query),
                "max_results": clean_max_results,
                "read_top_n": clean_read_top_n,
                "include_domains": _string_list(include_domains),
                "exclude_domains": _string_list(exclude_domains),
                "country": _clean_string(country).lower(),
            },
            "search": search,
            "pages": pages.get("results", []),
            "failed_pages": pages.get("failed_results", []),
            "warnings": list(search.get("warnings") or []) + list(pages.get("warnings") or []),
        }


_DEFAULT_CLIENT: WebResearchClient | None = None


def _default_client() -> WebResearchClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = WebResearchClient()
    return _DEFAULT_CLIENT


def build_mcp_server() -> FastMCP:
    server = FastMCP(
        "Web Research",
        instructions=(
            "Search the public web and parse public website content. "
            "Provider API keys are configured only in the MCP server environment."
        ),
        host=os.getenv("WEB_RESEARCH_MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("WEB_RESEARCH_MCP_PORT", "8000")),
        streamable_http_path=os.getenv("WEB_RESEARCH_MCP_PATH", "/mcp"),
    )
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)

    @server.tool(
        name="web_search",
        description="Search the public web with Tavily and return normalized result metadata, snippets, optional answer, and provider usage details.",
        annotations=read_only,
    )
    async def web_search_tool(
        query: Annotated[str, Field(description="Search query to execute.")],
        max_results: Annotated[int, Field(description="Maximum results to return, 1-20.", ge=1, le=20)] = 5,
        topic: Annotated[str, Field(description="Search topic: general, news, or finance.")] = "general",
        include_domains: Annotated[list[str] | None, Field(description="Optional domains to include in search results.")] = None,
        exclude_domains: Annotated[list[str] | None, Field(description="Optional domains to exclude from search results.")] = None,
        country: Annotated[str, Field(description="Optional country boost for general searches, for example united states.")] = "",
        include_answer: Annotated[bool, Field(description="When true, include Tavily's generated answer when available.")] = False,
    ) -> dict[str, Any]:
        return await _default_client().web_search(
            query=query,
            max_results=max_results,
            topic=topic,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            country=country,
            include_answer=include_answer,
        )

    @server.tool(
        name="read_url",
        description="Fetch and parse one public URL into bounded markdown or text, using Tavily first and Jina Reader as fallback in auto mode.",
        annotations=read_only,
    )
    async def read_url_tool(
        url: Annotated[str, Field(description="Public http or https URL to parse.")],
        provider: Annotated[str, Field(description="Provider selection: auto, tavily, or jina.")] = "auto",
        format: Annotated[str, Field(description="Output format: markdown or text.")] = "markdown",
        max_chars: Annotated[int | None, Field(description="Maximum content characters to return, 1-50000.", ge=1, le=50000)] = None,
        query: Annotated[str, Field(description="Optional caller context for the extraction request.")] = "",
    ) -> dict[str, Any]:
        return await _default_client().read_url(url=url, provider=provider, format=format, max_chars=max_chars, query=query)

    @server.tool(
        name="batch_read_urls",
        description="Fetch and parse up to 10 public URLs into bounded markdown or text with per-URL success and failure details.",
        annotations=read_only,
    )
    async def batch_read_urls_tool(
        urls: Annotated[list[str], Field(description="Public http or https URLs to parse. Maximum 10.")],
        provider: Annotated[str, Field(description="Provider selection: auto, tavily, or jina.")] = "auto",
        format: Annotated[str, Field(description="Output format: markdown or text.")] = "markdown",
        max_chars: Annotated[int | None, Field(description="Maximum content characters per URL, 1-50000.", ge=1, le=50000)] = None,
        query: Annotated[str, Field(description="Optional caller context for the extraction request.")] = "",
    ) -> dict[str, Any]:
        return await _default_client().batch_read_urls(urls=urls, provider=provider, format=format, max_chars=max_chars, query=query)

    @server.tool(
        name="search_and_read",
        description="Search the public web with Tavily, then parse the top public result URLs into bounded page content.",
        annotations=read_only,
    )
    async def search_and_read_tool(
        query: Annotated[str, Field(description="Search query to execute before reading top result URLs.")],
        max_results: Annotated[int, Field(description="Maximum search results to return, 1-10.", ge=1, le=10)] = 5,
        read_top_n: Annotated[int, Field(description="How many top result URLs to parse, 0-5.", ge=0, le=5)] = 3,
        include_domains: Annotated[list[str] | None, Field(description="Optional domains to include in search results.")] = None,
        exclude_domains: Annotated[list[str] | None, Field(description="Optional domains to exclude from search results.")] = None,
        country: Annotated[str, Field(description="Optional country boost for general searches, for example united states.")] = "",
        format: Annotated[str, Field(description="Output page format: markdown or text.")] = "markdown",
        max_chars: Annotated[int | None, Field(description="Maximum content characters per parsed URL, 1-50000.", ge=1, le=50000)] = None,
    ) -> dict[str, Any]:
        return await _default_client().search_and_read(
            query=query,
            max_results=max_results,
            read_top_n=read_top_n,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            country=country,
            format=format,
            max_chars=max_chars,
        )

    return server


mcp = build_mcp_server()


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
