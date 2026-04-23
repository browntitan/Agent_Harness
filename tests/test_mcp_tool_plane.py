from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.tools import ToolDefinition
from agentic_chatbot_next.mcp.security import (
    decrypt_mcp_secret,
    encrypt_mcp_secret,
    normalize_mcp_registry_name,
    validate_mcp_server_url,
)
from agentic_chatbot_next.persistence.postgres.mcp import McpConnectionRecord, McpToolCatalogRecord
from agentic_chatbot_next.tools.discovery import ToolDiscoveryService
from agentic_chatbot_next.tools.mcp_registry import build_mcp_tool_definitions
from agentic_chatbot_next.tools.policy import ToolPolicyService


def _settings(**overrides):
    payload = {
        "mcp_tool_plane_enabled": True,
        "mcp_require_https": True,
        "mcp_allow_private_network": False,
        "mcp_secret_encryption_key": "test-secret-key",
        "deferred_tool_discovery_enabled": True,
        "deferred_tool_discovery_top_k": 8,
        "deferred_tool_discovery_require_search": True,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_mcp_secret_encryption_requires_key_and_never_round_trips_plaintext() -> None:
    settings = _settings(mcp_secret_encryption_key="abc123")
    encrypted = encrypt_mcp_secret(settings, "token-value")

    assert encrypted.startswith("fernet:v1:")
    assert "token-value" not in encrypted
    assert decrypt_mcp_secret(settings, encrypted) == "token-value"

    with pytest.raises(ValueError):
        encrypt_mcp_secret(_settings(mcp_secret_encryption_key=""), "token-value")


def test_mcp_url_validation_enforces_https_and_private_network_policy() -> None:
    assert validate_mcp_server_url(_settings(), "https://mcp.example.com/mcp") == "https://mcp.example.com/mcp"

    with pytest.raises(ValueError):
        validate_mcp_server_url(_settings(), "http://mcp.example.com/mcp")
    with pytest.raises(ValueError):
        validate_mcp_server_url(_settings(mcp_require_https=False), "http://127.0.0.1:3000/mcp")

    allowed = validate_mcp_server_url(
        _settings(mcp_require_https=False, mcp_allow_private_network=True),
        "http://127.0.0.1:3000/mcp",
    )
    assert allowed == "http://127.0.0.1:3000/mcp"


def test_mcp_registry_name_normalization_is_stable() -> None:
    assert normalize_mcp_registry_name("GitHub Tools", "Search Issues!") == "mcp__github_tools__search_issues"


def test_mcp_policy_supports_wildcard_selectors_and_owner_visibility() -> None:
    agent = AgentDefinition(name="general", mode="react", allowed_tools=["mcp__*"])
    definition = ToolDefinition(
        name="mcp__github__search",
        group="mcp",
        description="Search remote issues.",
        args_schema={"type": "object", "properties": {}},
        when_to_use="Use for issue lookup.",
        output_description="Returns remote data.",
        read_only=True,
        background_safe=True,
        metadata={
            "tenant_id": "tenant-1",
            "owner_user_id": "user-1",
            "visibility": "private",
            "allowed_agents": ["general"],
        },
    )
    ctx = SimpleNamespace(
        settings=_settings(),
        metadata={},
        session=SimpleNamespace(tenant_id="tenant-1", user_id="user-1", metadata={}),
        workspace_root=None,
    )

    assert ToolPolicyService().is_allowed(agent, definition, ctx)

    ctx.session.user_id = "someone-else"
    assert not ToolPolicyService().is_allowed(agent, definition, ctx)


class _FakeMcpStore:
    def __init__(self) -> None:
        self.connection = McpConnectionRecord(
            connection_id="mcp_conn_1",
            tenant_id="tenant-1",
            owner_user_id="user-1",
            display_name="GitHub",
            connection_slug="github",
            server_url="https://mcp.example.com/mcp",
            allowed_agents=["general"],
            visibility="private",
        )
        self.tool = McpToolCatalogRecord(
            tool_id="mcp_tool_1",
            connection_id="mcp_conn_1",
            tenant_id="tenant-1",
            owner_user_id="user-1",
            raw_tool_name="search_issues",
            registry_name="mcp__github__search_issues",
            tool_slug="search_issues",
            description="Search GitHub issues.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Issue query."}},
                "required": ["query"],
            },
            read_only=True,
            destructive=False,
            background_safe=False,
            should_defer=True,
            search_hint="github issues search",
        )

    def list_connections(self, **kwargs):
        return [self.connection]

    def list_tool_catalog(self, **kwargs):
        return [self.tool]


def test_mcp_catalog_provider_builds_deferred_tool_definitions_without_network() -> None:
    ctx = SimpleNamespace(
        settings=_settings(),
        stores=SimpleNamespace(mcp_connection_store=_FakeMcpStore()),
        session=SimpleNamespace(tenant_id="tenant-1", user_id="user-1"),
    )

    definitions = build_mcp_tool_definitions(ctx)

    definition = definitions["mcp__github__search_issues"]
    assert definition.group == "mcp"
    assert definition.should_defer is True
    assert definition.read_only is True
    assert definition.background_safe is False
    assert definition.metadata["connection_id"] == "mcp_conn_1"


def test_deferred_discovery_finds_mcp_targets_through_wildcard_allowed_tools() -> None:
    ctx = SimpleNamespace(
        settings=_settings(),
        stores=SimpleNamespace(mcp_connection_store=_FakeMcpStore()),
        session=SimpleNamespace(tenant_id="tenant-1", user_id="user-1", session_id="session-1", metadata={}),
        metadata={},
        active_agent="general",
        kernel=None,
        workspace_root=None,
    )
    agent = AgentDefinition(name="general", mode="react", allowed_tools=["mcp__*"])
    definition = build_mcp_tool_definitions(ctx)["mcp__github__search_issues"]
    service = ToolDiscoveryService(
        agent=agent,
        tool_context=ctx,
        definitions={"mcp__github__search_issues": definition},
    )

    result = service.search("github issues")

    assert [match["name"] for match in result["matches"]] == ["mcp__github__search_issues"]
    assert ctx.metadata["deferred_tool_discovered_targets"] == ["mcp__github__search_issues"]
