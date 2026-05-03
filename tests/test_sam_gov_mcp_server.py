from __future__ import annotations

import json

import httpx
import pytest

from agentic_chatbot_next.mcp_servers.sam_gov import (
    SamGovApiError,
    SamGovClient,
    normalize_opportunity,
    summarize_service_need,
    validate_posted_date_range,
)


def test_validate_posted_date_range_enforces_sam_format_and_one_year_limit() -> None:
    assert validate_posted_date_range("04/02/2026", "05/02/2026") == ("04/02/2026", "05/02/2026")

    with pytest.raises(ValueError, match="MM/DD/YYYY"):
        validate_posted_date_range("2026-04-02", "05/02/2026")
    with pytest.raises(ValueError, match="on or after"):
        validate_posted_date_range("05/02/2026", "04/02/2026")
    with pytest.raises(ValueError, match="one year"):
        validate_posted_date_range("01/01/2024", "01/02/2025")


def test_normalize_opportunity_returns_compact_safe_shape_without_api_keys() -> None:
    normalized = normalize_opportunity(
        {
            "noticeId": "abc123",
            "title": "Cloud modernization",
            "solicitationNumber": "SOL-1",
            "fullParentPathName": "GENERAL SERVICES ADMINISTRATION.FAS",
            "postedDate": "2026-04-15",
            "responseDeadLine": "05/15/2026",
            "naicsCode": "541512",
            "classificationCode": "DA01",
            "typeOfSetAsideDescription": "Service-Disabled Veteran-Owned Small Business",
            "typeOfSetAside": "SDVOSBC",
            "type": "Solicitation",
            "active": "Yes",
            "uiLink": "https://sam.gov/opp/abc123/view",
            "description": "https://api.sam.gov/prod/opportunities/v1/noticedesc?noticeid=abc123&api_key=secret",
            "resourceLinks": ["https://sam.gov/file.pdf"],
        }
    )

    assert normalized == {
        "notice_id": "abc123",
        "title": "Cloud modernization",
        "solicitation_number": "SOL-1",
        "agency": "GENERAL SERVICES ADMINISTRATION.FAS",
        "posted_date": "2026-04-15",
        "response_deadline": "05/15/2026",
        "naics_code": "541512",
        "classification_code": "DA01",
        "set_aside": "Service-Disabled Veteran-Owned Small Business",
        "set_aside_code": "SDVOSBC",
        "type": "Solicitation",
        "active": True,
        "service_summary": "",
        "ui_link": "https://sam.gov/opp/abc123/view",
        "description_link": "https://api.sam.gov/prod/opportunities/v1/noticedesc?noticeid=abc123",
        "resource_links": ["https://sam.gov/file.pdf"],
    }


@pytest.mark.asyncio
async def test_search_open_contracts_calls_sam_with_expected_filters_and_normalizes_results() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        payload = {
            "totalRecords": 2,
            "opportunitiesData": [
                {
                    "noticeId": "active-1",
                    "title": "IT services",
                    "fullParentPathName": "DEPARTMENT OF VETERANS AFFAIRS",
                    "active": "Yes",
                    "naicsCode": "541512",
                    "typeOfSetAside": "SDVOSBC",
                    "type": "Solicitation",
                },
                {
                    "noticeId": "archived-1",
                    "title": "Old IT services",
                    "fullParentPathName": "DEPARTMENT OF VETERANS AFFAIRS",
                    "active": "No",
                    "type": "Solicitation",
                },
            ],
            "links": [{"rel": "self", "href": "https://api.sam.gov/opportunities/v2/search?api_key=test-key"}],
        }
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SamGovClient(api_key="test-key", http_client=http_client)
        result = await client.search_open_contracts(
            posted_from="04/02/2026",
            posted_to="05/02/2026",
            keyword="IT services",
            naics_code="541512",
            set_aside_code="SDVOSBC",
            limit=5,
        )

    assert captured["api_key"] == "test-key"
    assert captured["postedFrom"] == "04/02/2026"
    assert captured["postedTo"] == "05/02/2026"
    assert captured["title"] == "IT services"
    assert captured["ncode"] == "541512"
    assert captured["typeOfSetAside"] == "SDVOSBC"
    assert captured["limit"] == "5"
    assert result["returned_records"] == 1
    assert result["opportunities"][0]["notice_id"] == "active-1"
    assert result["links"][0]["href"] == "https://api.sam.gov/opportunities/v2/search"


@pytest.mark.asyncio
async def test_search_open_contracts_can_add_service_summaries_and_bid_ready_filter() -> None:
    captured_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_paths.append(request.url.path)
        if "noticedesc" in request.url.path:
            assert request.url.params["noticeid"] == "bid-1"
            return httpx.Response(
                200,
                text=(
                    "<p>The Government requires machine learning model development, MLOps deployment, "
                    "and analytics support for mission data pipelines. Additional boilerplate follows.</p>"
                ),
            )
        payload = {
            "totalRecords": 2,
            "opportunitiesData": [
                {
                    "noticeId": "bid-1",
                    "title": "AI services",
                    "fullParentPathName": "DEPARTMENT OF DEFENSE",
                    "active": "Yes",
                    "type": "Solicitation",
                },
                {
                    "noticeId": "rfi-1",
                    "title": "AI RFI",
                    "fullParentPathName": "DEPARTMENT OF DEFENSE",
                    "active": "Yes",
                    "type": "Sources Sought",
                },
            ],
        }
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SamGovClient(api_key="test-key", http_client=http_client)
        result = await client.search_open_contracts(
            posted_from="04/02/2026",
            posted_to="05/02/2026",
            keyword="artificial intelligence",
            solicitation_only=True,
            include_descriptions=True,
            summary_max_chars=160,
        )

    assert result["returned_records"] == 1
    assert result["description_fetch_count"] == 1
    assert result["query"]["solicitation_only"] is True
    assert result["query"]["include_descriptions"] is True
    assert result["opportunities"][0]["notice_id"] == "bid-1"
    assert "machine learning model development" in result["opportunities"][0]["service_summary"]
    assert any("noticedesc" in path for path in captured_paths)


@pytest.mark.asyncio
async def test_get_opportunity_description_text_normalizes_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "test-key"
        assert request.url.params["noticeid"] == "notice-1"
        return httpx.Response(200, text="<html><body><h1>Work</h1><p>Submit by email.</p></body></html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SamGovClient(api_key="test-key", http_client=http_client)
        result = await client.get_opportunity_description(notice_id="notice-1")

    assert result["status"] == "ok"
    assert result["description_text"] == "Work Submit by email."
    assert result["description_html"] == ""
    assert result["source_url"].endswith("noticeid=notice-1")


@pytest.mark.asyncio
async def test_sam_error_messages_do_not_leak_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"bad key {request.url.params['api_key']}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = SamGovClient(api_key="secret-key", http_client=http_client)
        with pytest.raises(SamGovApiError) as exc:
            await client.search_open_contracts(posted_from="04/02/2026", posted_to="05/02/2026")

    assert "secret-key" not in str(exc.value)
    assert "[redacted]" in str(exc.value)


def test_service_summary_prefers_short_sentence_window() -> None:
    summary = summarize_service_need(
        "Description: The contractor shall provide AI model development and data science support. "
        "The contractor shall also maintain dashboards. This sentence should be clipped.",
        max_chars=140,
    )

    assert summary.startswith("The contractor shall provide AI model development")
    assert len(summary) <= 140


def test_server_module_exposes_streamable_http_tools() -> None:
    from agentic_chatbot_next.mcp_servers.sam_gov import mcp

    tool_names = {tool.name for tool in mcp._tool_manager.list_tools()}  # type: ignore[attr-defined]

    assert {"search_open_contracts", "get_opportunity_description"} <= tool_names
    raw_tools = [tool for tool in mcp._tool_manager.list_tools() if tool.name == "search_open_contracts"]  # type: ignore[attr-defined]
    assert json.loads(raw_tools[0].annotations.model_dump_json())["readOnlyHint"] is True
