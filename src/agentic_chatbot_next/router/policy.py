from __future__ import annotations

from typing import Optional

from agentic_chatbot_next.router.router import build_router_targets
from agentic_chatbot_next.router.router import RouterDecision


def choose_agent_name(settings: object, decision: RouterDecision, *, registry: object | None = None) -> Optional[str]:
    targets = build_router_targets(registry)
    if bool(getattr(settings, "enable_coordinator_mode", False)):
        return targets.coordinator_agent
    suggested = str(getattr(decision, "suggested_agent", "") or "").strip().lower()
    if suggested and suggested in set(targets.suggested_agents):
        return suggested
    return targets.default_agent or None


__all__ = ["RouterDecision", "choose_agent_name"]
