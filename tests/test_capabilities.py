from __future__ import annotations

from types import SimpleNamespace

from agentic_chatbot_next.capabilities import (
    CapabilityProfile,
    EffectiveCapabilities,
    resolve_effective_capabilities,
)


def test_effective_capabilities_hide_unavailable_agents_and_collections() -> None:
    effective = EffectiveCapabilities.from_profile(
        CapabilityProfile(
            enabled_agents=["coordinator", "missing_agent"],
            enabled_collections=["policy", "secret"],
        ),
        tenant_available_agents=["coordinator", "general"],
        tenant_available_collections=["policy"],
    )

    assert effective.enabled_agents == ["coordinator"]
    assert effective.enabled_collections == ["policy"]
    assert "missing_agent" in effective.hidden_unavailable
    assert "secret" in effective.hidden_unavailable


def test_capability_tool_policy_helpers() -> None:
    effective = EffectiveCapabilities(
        enabled_tools=["calculator"],
        disabled_tools=["danger_*"],
        enabled_tool_groups=["utility"],
        permission_mode="restricted",
    )

    assert effective.allows_tool("calculator", group="utility", read_only=True)
    assert not effective.allows_tool("calculator", group="utility", read_only=False)
    assert not effective.allows_tool("danger_delete", group="utility", read_only=True)
    assert not effective.allows_tool("read_indexed_doc", group="rag_gateway", read_only=True)


def test_authz_collection_denial_overrides_user_enablement() -> None:
    effective = EffectiveCapabilities.from_profile(
        CapabilityProfile(enabled_collections=["allowed", "denied"]),
        access_summary={
            "authz_enabled": True,
            "resources": {
                "collection": {"use": ["allowed"], "manage": [], "use_all": False, "manage_all": False},
            },
        },
    )

    assert effective.enabled_collections == ["allowed"]
    assert not effective.allows_collection("denied")


def test_authz_empty_collection_grant_denies_unlisted_collections() -> None:
    effective = EffectiveCapabilities.from_profile(
        CapabilityProfile(),
        access_summary={
            "authz_enabled": True,
            "resources": {
                "collection": {"use": [], "manage": [], "use_all": False, "manage_all": False},
            },
        },
    )

    assert effective.collection_access_limited is True
    assert not effective.allows_collection("policy")


def test_resolve_effective_capabilities_uses_session_metadata_profile() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        user_id="user",
        metadata={"capability_profile": {"disabled_skill_pack_ids": ["legal_review"]}},
        access_summary={},
    )

    effective = resolve_effective_capabilities(
        settings=SimpleNamespace(default_tenant_id="tenant", default_user_id="user"),
        stores=SimpleNamespace(collection_store=None),
        session=session,
    )

    assert not effective.allows_skill("legal_review")
