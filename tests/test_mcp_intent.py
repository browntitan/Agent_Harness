from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from agentic_chatbot_next.persistence.postgres.mcp import McpConnectionRecord, McpToolCatalogRecord
from agentic_chatbot_next.router.mcp_intent import detect_mcp_intent
from agentic_chatbot_next.runtime.query_loop import mcp_intent_prompt_block
from agentic_chatbot_next.skills.pack_loader import load_skill_pack_from_file
from agentic_chatbot_next.tools.discovery import ToolDiscoveryService
from agentic_chatbot_next.tools.mcp_registry import build_mcp_tool_definitions
from agentic_chatbot_next.contracts.agents import AgentDefinition


def _settings(**overrides):
    payload = {
        "mcp_tool_plane_enabled": True,
        "deferred_tool_discovery_enabled": True,
        "deferred_tool_discovery_top_k": 8,
        "deferred_tool_discovery_require_search": True,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


class _FakeSamMcpStore:
    def __init__(self, *, connection_status: str = "active", allowed_agents=None, tool_enabled: bool = True) -> None:
        self.connection = McpConnectionRecord(
            connection_id="mcp_conn_sam",
            tenant_id="tenant-1",
            owner_user_id="user-1",
            display_name="SAM.gov",
            connection_slug="sam_gov",
            server_url="https://sam.example.test/mcp",
            allowed_agents=list(allowed_agents if allowed_agents is not None else ["general"]),
            visibility="tenant",
            status=connection_status,
        )
        self.tool = McpToolCatalogRecord(
            tool_id="mcp_tool_sam_search",
            connection_id="mcp_conn_sam",
            tenant_id="tenant-1",
            owner_user_id="user-1",
            raw_tool_name="search_open_contracts",
            registry_name="mcp__sam_gov__search_open_contracts",
            tool_slug="search_open_contracts",
            description="Search active SAM.gov contract opportunities and solicitations.",
            input_schema={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Keyword query for opportunities."},
                    "posted_from": {"type": "string", "description": "Start date in MM/dd/yyyy."},
                    "posted_to": {"type": "string", "description": "End date in MM/dd/yyyy."},
                    "naics_code": {"type": "string", "description": "NAICS code."},
                },
            },
            read_only=True,
            should_defer=True,
            search_hint="sam.gov open contracts opportunities solicitations artificial intelligence machine learning",
            enabled=tool_enabled,
            status="active" if tool_enabled else "disabled",
        )

    def list_connections(self, **kwargs):
        if not kwargs.get("include_disabled", True) and self.connection.status != "active":
            return []
        return [self.connection]

    def list_tool_catalog(self, **kwargs):
        if not kwargs.get("include_disabled", False) and (not self.tool.enabled or self.tool.status != "active"):
            return []
        return [self.tool]


def _detect(text: str, store: object | None = None):
    return detect_mcp_intent(
        text,
        settings=_settings(),
        stores=SimpleNamespace(mcp_connection_store=store or _FakeSamMcpStore()),
        tenant_id="tenant-1",
        user_id="user-1",
    )


def test_explicit_mcp_request_detects_intent_and_candidates() -> None:
    payload = _detect("Using MCP tooling, find 10 active SAM.gov opportunities for AI and machine learning.")

    assert payload["detected"] is True
    assert payload["trigger"] == "explicit_mcp"
    assert payload["matched_connections"][0]["connection_slug"] == "sam_gov"
    assert payload["matched_tools"][0]["registry_name"] == "mcp__sam_gov__search_open_contracts"


def test_registry_connection_match_detects_sam_without_mcp_wording() -> None:
    payload = _detect("Find SAM.gov opportunities posted from 04/02/2026 to 05/02/2026.")

    assert payload["detected"] is True
    assert payload["trigger"] == "connection_match"
    assert payload["matched_connections"][0]["display_name"] == "SAM.gov"


def test_conceptual_mcp_question_does_not_force_tool_discovery() -> None:
    payload = _detect("What is MCP?")
    block = mcp_intent_prompt_block({"route_context": {"mcp_intent": payload}}, user_text="What is MCP?")

    assert payload["detected"] is True
    assert payload["trigger"] == "explicit_mcp"
    assert payload["tool_discovery_required"] is False
    assert "Answer directly" in block
    assert "discover_tools" not in block


def test_registry_tool_match_detects_open_contract_request_without_service_name() -> None:
    payload = _detect(
        "Find open contract opportunities for artificial intelligence machine learning software solicitations."
    )

    assert payload["detected"] is True
    assert payload["trigger"] == "tool_match"
    assert payload["matched_tools"][0]["registry_name"] == "mcp__sam_gov__search_open_contracts"


def test_disabled_or_non_general_mcp_catalog_rows_are_not_matched() -> None:
    disabled_payload = _detect(
        "Find SAM.gov opportunities.",
        store=_FakeSamMcpStore(tool_enabled=False),
    )
    non_allowed_payload = _detect(
        "Find SAM.gov opportunities.",
        store=_FakeSamMcpStore(allowed_agents=["rag_worker"]),
    )

    assert disabled_payload["detected"] is False
    assert non_allowed_payload["detected"] is False


def test_prompt_block_guides_general_to_search_skills_and_discover_mcp_tools() -> None:
    block = mcp_intent_prompt_block(
        {
            "route_context": {
                "mcp_intent": {
                    "detected": True,
                    "trigger": "explicit_mcp",
                    "discover_query": "Find SAM.gov AI opportunities",
                    "matched_connections": [{"display_name": "SAM.gov", "connection_slug": "sam_gov"}],
                    "matched_tools": [{"registry_name": "mcp__sam_gov__search_open_contracts"}],
                }
            }
        }
    )

    assert "search_skills" in block
    assert "discover_tools" in block
    assert "group=\"mcp\"" in block
    assert "call_deferred_tool" in block
    assert "mcp__sam_gov__search_open_contracts" in block


def test_mcp_skill_pack_is_retrievable_for_general_agent() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / "data" / "skill_packs"
    pack = load_skill_pack_from_file(root / "general" / "mcp_tooling_workflow.md", root=root)

    assert pack.agent_scope == "general"
    assert "discover_tools" in pack.tool_tags
    assert "call_deferred_tool" in pack.tool_tags
    assert any("MCP tools as the primary evidence path" in chunk for chunk in pack.chunks)


def test_discovery_returns_fake_sam_mcp_tool_and_invocation_requires_prior_discovery() -> None:
    ctx = SimpleNamespace(
        settings=_settings(),
        stores=SimpleNamespace(mcp_connection_store=_FakeSamMcpStore()),
        session=SimpleNamespace(tenant_id="tenant-1", user_id="user-1", session_id="session-1", metadata={}),
        metadata={},
        active_agent="general",
        kernel=None,
        workspace_root=None,
    )
    agent = AgentDefinition(name="general", mode="react", allowed_tools=["mcp__*"])
    definition = build_mcp_tool_definitions(ctx)["mcp__sam_gov__search_open_contracts"]
    service = ToolDiscoveryService(
        agent=agent,
        tool_context=ctx,
        definitions={"mcp__sam_gov__search_open_contracts": definition},
    )

    denied = service.invoke("mcp__sam_gov__search_open_contracts", {"q": "AI"})
    discovered = service.search("SAM.gov open contracts artificial intelligence", group="mcp")

    assert denied["status"] == "error"
    assert "discover_tools" in denied["error"]
    assert [match["name"] for match in discovered["matches"]] == ["mcp__sam_gov__search_open_contracts"]
    assert ctx.metadata["deferred_tool_discovered_targets"] == ["mcp__sam_gov__search_open_contracts"]
