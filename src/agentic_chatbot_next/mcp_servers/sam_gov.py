from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from html.parser import HTMLParser
from typing import Annotated, Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field


SAM_OPPORTUNITIES_URL = "https://api.sam.gov/opportunities/v2/search"
SAM_DESCRIPTION_URL = "https://api.sam.gov/prod/opportunities/v1/noticedesc"
SAM_DATE_FORMAT = "%m/%d/%Y"
MAX_SAM_DATE_RANGE_DAYS = 366

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class SamGovConfigError(RuntimeError):
    """Raised when the SAM.gov MCP server is missing required runtime config."""


class SamGovApiError(RuntimeError):
    """Raised when SAM.gov returns an error response."""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        clean = str(data or "").strip()
        if clean:
            self.parts.append(clean)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _optional_param(value: Any) -> Any | None:
    clean = _clean_string(value)
    return clean or None


def _parse_sam_date(value: str, *, field_name: str) -> datetime:
    clean = _clean_string(value)
    if not clean:
        raise ValueError(f"{field_name} is required and must use MM/DD/YYYY.")
    try:
        return datetime.strptime(clean, SAM_DATE_FORMAT)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use MM/DD/YYYY.") from exc


def validate_posted_date_range(posted_from: str, posted_to: str) -> tuple[str, str]:
    start = _parse_sam_date(posted_from, field_name="posted_from")
    end = _parse_sam_date(posted_to, field_name="posted_to")
    if end < start:
        raise ValueError("posted_to must be on or after posted_from.")
    if (end - start).days > MAX_SAM_DATE_RANGE_DAYS:
        raise ValueError("SAM.gov posted date range cannot exceed one year.")
    return start.strftime(SAM_DATE_FORMAT), end.strftime(SAM_DATE_FORMAT)


def _api_key_from_env() -> str:
    key = _clean_string(os.getenv("SAM_API_KEY"))
    if not key:
        raise SamGovConfigError(
            "SAM_API_KEY is not configured for the SAM.gov MCP server. "
            "Create a SAM.gov API key and set it only in the MCP server environment."
        )
    return key


def _url_without_api_key(url: str) -> str:
    parts = urlsplit(str(url or ""))
    query = urlencode([(key, value) for key, value in parse_qsl(parts.query) if key.lower() != "api_key"])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _redact_secret(text: str, secret: str) -> str:
    clean = str(text or "")
    return clean.replace(secret, "[redacted]") if secret else clean


def _error_message(response: httpx.Response, api_key: str) -> str:
    status = response.status_code
    if status in {401, 403}:
        return "SAM.gov rejected the API key. Regenerate the key in SAM.gov account details and update the MCP server environment."
    if status == 429:
        return "SAM.gov rate limit was exceeded for this API key. Try again later or use a higher-limit SAM.gov key."
    if 500 <= status <= 599:
        return f"SAM.gov returned HTTP {status}. Try again later."
    body = _redact_secret(response.text[:500], api_key)
    return f"SAM.gov returned HTTP {status}: {body}"


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(str(html or ""))
    text = parser.text()
    return text or re.sub(r"\s+", " ", str(html or "")).strip()


def summarize_service_need(text: str, *, max_chars: int = 320) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    clean = re.sub(r"^(?:description|scope|requirement|requirements|summary)\s*[:\-]\s*", "", clean, flags=re.IGNORECASE)
    if not clean:
        return ""
    max_len = max(80, min(int(max_chars or 320), 1200))
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    summary = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{summary} {sentence}".strip()
        if len(candidate) > max_len:
            break
        summary = candidate
        if len(summary) >= 120:
            break
    if not summary:
        summary = clean[:max_len]
    if len(summary) > max_len:
        summary = summary[: max_len - 3].rstrip() + "..."
    return summary


def _bid_ready_type(value: Any) -> bool:
    clean = _clean_string(value).casefold()
    return clean in {"solicitation", "combined synopsis/solicitation", "combined synopsis solicitation"}


def normalize_opportunity(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(raw or {})
    agency = (
        _clean_string(data.get("fullParentPathName"))
        or _clean_string(data.get("department"))
        or _clean_string(data.get("subTier"))
        or _clean_string(data.get("office"))
    )
    set_aside = _clean_string(data.get("typeOfSetAsideDescription")) or _clean_string(data.get("setAside"))
    set_aside_code = _clean_string(data.get("typeOfSetAside")) or _clean_string(data.get("setAsideCode"))
    active_raw = data.get("active")
    active = active_raw if isinstance(active_raw, bool) else _clean_string(active_raw).lower() in {"yes", "true", "active", "1"}
    return {
        "notice_id": _clean_string(data.get("noticeId") or data.get("noticeID") or data.get("noticeid")),
        "title": _clean_string(data.get("title")),
        "solicitation_number": _clean_string(data.get("solicitationNumber")),
        "agency": agency,
        "posted_date": _clean_string(data.get("postedDate")),
        "response_deadline": _clean_string(data.get("responseDeadLine") or data.get("responseDeadline")),
        "naics_code": _clean_string(data.get("naicsCode")),
        "classification_code": _clean_string(data.get("classificationCode")),
        "set_aside": set_aside,
        "set_aside_code": set_aside_code,
        "type": _clean_string(data.get("type")),
        "active": bool(active),
        "service_summary": "",
        "ui_link": _clean_string(data.get("uiLink")),
        "description_link": _url_without_api_key(_clean_string(data.get("description"))),
        "resource_links": [str(item) for item in list(data.get("resourceLinks") or []) if str(item).strip()],
    }


def _normalize_links(raw_links: Any) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for item in list(raw_links or []):
        if not isinstance(item, Mapping):
            continue
        links.append(
            {
                "rel": _clean_string(item.get("rel")),
                "href": _url_without_api_key(_clean_string(item.get("href"))),
            }
        )
    return links


class SamGovClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        opportunities_url: str | None = None,
        description_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = _clean_string(api_key) or _api_key_from_env()
        self.opportunities_url = _clean_string(opportunities_url or os.getenv("SAM_OPPORTUNITIES_URL")) or SAM_OPPORTUNITIES_URL
        self.description_url = _clean_string(description_url or os.getenv("SAM_DESCRIPTION_URL")) or SAM_DESCRIPTION_URL
        self._owned_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=float(timeout_seconds or os.getenv("SAM_GOV_TIMEOUT_SECONDS") or 30))

    async def close(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def _get_json(self, url: str, *, params: Mapping[str, Any]) -> dict[str, Any]:
        response = await self._client.get(url, params=dict(params))
        if response.status_code >= 400:
            raise SamGovApiError(_error_message(response, self.api_key))
        try:
            data = response.json()
        except ValueError as exc:
            raise SamGovApiError("SAM.gov returned a non-JSON response for the opportunity search.") from exc
        if not isinstance(data, dict):
            raise SamGovApiError("SAM.gov returned an unexpected opportunity search payload.")
        return data

    async def _get_text(self, url: str, *, params: Mapping[str, Any]) -> tuple[str, str]:
        response = await self._client.get(url, params=dict(params))
        if response.status_code == 404:
            return "not_found", ""
        if response.status_code >= 400:
            raise SamGovApiError(_error_message(response, self.api_key))
        return "ok", response.text

    async def search_open_contracts(
        self,
        *,
        posted_from: str,
        posted_to: str,
        keyword: str = "",
        procurement_type: str = "",
        naics_code: str = "",
        set_aside_code: str = "",
        state: str = "",
        response_deadline_from: str = "",
        response_deadline_to: str = "",
        agency_keyword: str = "",
        limit: int = 10,
        offset: int = 0,
        active_only: bool = True,
        solicitation_only: bool = False,
        include_descriptions: bool = False,
        summary_max_chars: int = 320,
    ) -> dict[str, Any]:
        clean_posted_from, clean_posted_to = validate_posted_date_range(posted_from, posted_to)
        clean_limit = max(1, min(int(limit or 10), 1000))
        clean_offset = max(0, int(offset or 0))
        params = {
            "api_key": self.api_key,
            "postedFrom": clean_posted_from,
            "postedTo": clean_posted_to,
            "title": _optional_param(keyword),
            "ptype": _optional_param(procurement_type),
            "ncode": _optional_param(naics_code),
            "typeOfSetAside": _optional_param(set_aside_code),
            "state": _optional_param(state),
            "rdlfrom": _optional_param(response_deadline_from),
            "rdlto": _optional_param(response_deadline_to),
            "limit": clean_limit,
            "offset": clean_offset,
        }
        payload = await self._get_json(self.opportunities_url, params={key: value for key, value in params.items() if value is not None})
        records = [item for item in list(payload.get("opportunitiesData") or []) if isinstance(item, Mapping)]
        normalized = [normalize_opportunity(item) for item in records]
        if active_only:
            normalized = [item for item in normalized if item["active"]]
        if solicitation_only:
            normalized = [item for item in normalized if _bid_ready_type(item.get("type"))]
        agency_filter = _clean_string(agency_keyword).lower()
        if agency_filter:
            normalized = [item for item in normalized if agency_filter in str(item.get("agency") or "").lower()]
        description_fetch_count = 0
        if include_descriptions:
            for item in normalized[: min(len(normalized), 10)]:
                notice_id = _clean_string(item.get("notice_id"))
                if not notice_id:
                    continue
                description = await self.get_opportunity_description(notice_id=notice_id)
                description_fetch_count += 1
                if description.get("status") == "ok":
                    item["service_summary"] = summarize_service_need(
                        str(description.get("description_text") or ""),
                        max_chars=summary_max_chars,
                    )
                if not item.get("service_summary"):
                    item["service_summary"] = summarize_service_need(str(item.get("title") or ""), max_chars=summary_max_chars)
        return {
            "object": "sam_gov.open_contract_search",
            "query": {
                "posted_from": clean_posted_from,
                "posted_to": clean_posted_to,
                "keyword": _clean_string(keyword),
                "procurement_type": _clean_string(procurement_type),
                "naics_code": _clean_string(naics_code),
                "set_aside_code": _clean_string(set_aside_code),
                "state": _clean_string(state),
                "response_deadline_from": _clean_string(response_deadline_from),
                "response_deadline_to": _clean_string(response_deadline_to),
                "agency_keyword": _clean_string(agency_keyword),
                "active_only": bool(active_only),
                "solicitation_only": bool(solicitation_only),
                "include_descriptions": bool(include_descriptions),
                "summary_max_chars": max(80, min(int(summary_max_chars or 320), 1200)),
            },
            "total_records": int(payload.get("totalRecords") or 0),
            "returned_records": len(normalized),
            "description_fetch_count": description_fetch_count,
            "limit": clean_limit,
            "offset": clean_offset,
            "opportunities": normalized,
            "links": _normalize_links(payload.get("links")),
        }

    async def get_opportunity_description(self, *, notice_id: str, include_html: bool = False) -> dict[str, Any]:
        clean_notice_id = _clean_string(notice_id)
        if not clean_notice_id:
            raise ValueError("notice_id is required.")
        status, raw_text = await self._get_text(
            self.description_url,
            params={"api_key": self.api_key, "noticeid": clean_notice_id},
        )
        if status == "not_found" or "description not found" in raw_text[:250].lower():
            return {
                "object": "sam_gov.opportunity_description",
                "notice_id": clean_notice_id,
                "status": "not_found",
                "description_text": "",
                "description_html": "" if include_html else "",
                "source_url": _url_without_api_key(f"{self.description_url}?noticeid={clean_notice_id}"),
            }
        description_text = _html_to_text(raw_text)
        return {
            "object": "sam_gov.opportunity_description",
            "notice_id": clean_notice_id,
            "status": "ok",
            "description_text": description_text,
            "description_html": raw_text if include_html else "",
            "source_url": _url_without_api_key(f"{self.description_url}?noticeid={clean_notice_id}"),
        }


_DEFAULT_CLIENT: SamGovClient | None = None


def _default_client() -> SamGovClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = SamGovClient()
    return _DEFAULT_CLIENT


def build_mcp_server() -> FastMCP:
    server = FastMCP(
        "SAM.gov Open Contracts",
        instructions=(
            "Search SAM.gov contract opportunities and retrieve opportunity descriptions. "
            "The SAM.gov API key is configured only in the MCP server environment."
        ),
        host=os.getenv("SAM_GOV_MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("SAM_GOV_MCP_PORT", "8000")),
        streamable_http_path=os.getenv("SAM_GOV_MCP_PATH", "/mcp"),
    )
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)

    @server.tool(
        name="search_open_contracts",
        description="Search active SAM.gov contract opportunities with date, NAICS, set-aside, procurement type, state, and response-deadline filters.",
        annotations=read_only,
    )
    async def search_open_contracts_tool(
        posted_from: Annotated[str, Field(description="Posted date from, required, MM/DD/YYYY.")],
        posted_to: Annotated[str, Field(description="Posted date to, required, MM/DD/YYYY. The range cannot exceed one year.")],
        keyword: Annotated[str, Field(description="Optional title keyword search, for example IT services.")] = "",
        procurement_type: Annotated[str, Field(description="Optional SAM.gov ptype code, for example o=Solicitation, k=Combined Synopsis/Solicitation, r=Sources Sought.")] = "",
        naics_code: Annotated[str, Field(description="Optional NAICS code, maximum 6 digits.")] = "",
        set_aside_code: Annotated[str, Field(description="Optional set-aside code, for example SDVOSBC, SBA, 8A, WOSB.")] = "",
        state: Annotated[str, Field(description="Optional two-letter place-of-performance state.")] = "",
        response_deadline_from: Annotated[str, Field(description="Optional response deadline from date, MM/DD/YYYY.")] = "",
        response_deadline_to: Annotated[str, Field(description="Optional response deadline to date, MM/DD/YYYY.")] = "",
        agency_keyword: Annotated[str, Field(description="Optional agency substring post-filter against fullParentPathName/department/subtier/office.")] = "",
        limit: Annotated[int, Field(description="Maximum results to request from SAM.gov, 1-1000.", ge=1, le=1000)] = 10,
        offset: Annotated[int, Field(description="SAM.gov page offset.", ge=0)] = 0,
        active_only: Annotated[bool, Field(description="When true, keep only opportunities marked active by SAM.gov.")] = True,
        solicitation_only: Annotated[bool, Field(description="When true, keep only opportunity types that are actively soliciting bids: Solicitation or Combined Synopsis/Solicitation.")] = False,
        include_descriptions: Annotated[bool, Field(description="When true, fetch descriptions for up to the first 10 returned opportunities and add service_summary. Use when the user asks for service summaries or a description column.")] = False,
        summary_max_chars: Annotated[int, Field(description="Maximum service_summary length, 80-1200 characters.", ge=80, le=1200)] = 320,
    ) -> dict[str, Any]:
        return await _default_client().search_open_contracts(
            posted_from=posted_from,
            posted_to=posted_to,
            keyword=keyword,
            procurement_type=procurement_type,
            naics_code=naics_code,
            set_aside_code=set_aside_code,
            state=state,
            response_deadline_from=response_deadline_from,
            response_deadline_to=response_deadline_to,
            agency_keyword=agency_keyword,
            limit=limit,
            offset=offset,
            active_only=active_only,
            solicitation_only=solicitation_only,
            include_descriptions=include_descriptions,
            summary_max_chars=summary_max_chars,
        )

    @server.tool(
        name="get_opportunity_description",
        description="Fetch and text-normalize the SAM.gov HTML description for a notice ID.",
        annotations=read_only,
    )
    async def get_opportunity_description_tool(
        notice_id: Annotated[str, Field(description="SAM.gov notice ID.")],
        include_html: Annotated[bool, Field(description="Include raw HTML in addition to normalized text.")] = False,
    ) -> dict[str, Any]:
        return await _default_client().get_opportunity_description(notice_id=notice_id, include_html=include_html)

    return server


mcp = build_mcp_server()


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
