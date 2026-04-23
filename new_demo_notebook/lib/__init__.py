"""Helper library for the API-driven orchestration showcase notebook."""

from new_demo_notebook.lib.client import GatewayChatResponse, GatewayClient
from new_demo_notebook.lib.preflight import PreflightCheck, PreflightReport, run_preflight
from new_demo_notebook.lib.scenario_runner import (
    REQUIRED_AGENT_COVERAGE,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioRunner,
    load_scenarios,
    validate_agent_coverage,
)
from new_demo_notebook.lib.server import BackendServerManager
from new_demo_notebook.lib.trace_reader import (
    TraceBundle,
    cleanup_conversation_artifacts,
    collect_trace_bundle,
    extract_latest_router_decision,
    extract_observed_agents,
    extract_observed_event_types,
    extract_observed_route,
)

__all__ = [
    "BackendServerManager",
    "GatewayChatResponse",
    "GatewayClient",
    "PreflightCheck",
    "PreflightReport",
    "REQUIRED_AGENT_COVERAGE",
    "ScenarioDefinition",
    "ScenarioResult",
    "ScenarioRunner",
    "TraceBundle",
    "cleanup_conversation_artifacts",
    "collect_trace_bundle",
    "extract_latest_router_decision",
    "extract_observed_agents",
    "extract_observed_event_types",
    "extract_observed_route",
    "load_scenarios",
    "run_preflight",
    "validate_agent_coverage",
]
