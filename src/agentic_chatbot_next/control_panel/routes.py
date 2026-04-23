from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from agentic_chatbot_next.authz import normalize_user_email
from agentic_chatbot_next.agents.loader import load_agent_markdown_text
from agentic_chatbot_next.agents.registry import AgentRegistry
from agentic_chatbot_next.config import runtime_settings_diagnostics
from agentic_chatbot_next.contracts.messages import utc_now_iso
from agentic_chatbot_next.context import build_local_context
from agentic_chatbot_next.control_panel.auth import require_admin_token
from agentic_chatbot_next.control_panel.config_catalog import build_config_catalog
from agentic_chatbot_next.control_panel.overlay_store import OverlayStore
from agentic_chatbot_next.control_panel.runtime_manager import RuntimeManager, get_runtime_manager
from agentic_chatbot_next.graph.prompt_tuning import GraphPromptTuningService
from agentic_chatbot_next.graph.service import GraphService
from agentic_chatbot_next.mcp.client import McpClientError
from agentic_chatbot_next.mcp.service import McpCatalogService
from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.persistence.postgres import (
    AuthPrincipalMembershipRecord,
    AuthPrincipalRecord,
    AuthRoleBindingRecord,
    AuthRolePermissionRecord,
    AuthRoleRecord,
    get_graph_indexes_embedding_dim,
    get_table_embedding_dim,
)
from agentic_chatbot_next.persistence.postgres.collections import (
    COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES,
    COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,
)
from agentic_chatbot_next.persistence.postgres.skills import SkillPackRecord
from agentic_chatbot_next.rag.ingest import (
    build_collection_health_report,
    build_kb_health_report,
    get_collection_readiness_status,
    get_kb_coverage_status,
    ingest_paths,
    iter_kb_source_paths,
    repair_collection_documents,
    repair_kb_collection,
)
from agentic_chatbot_next.rag.ocr import IMAGE_SUFFIXES
from agentic_chatbot_next.router.feedback_loop import summarize_router_outcomes
from agentic_chatbot_next.router.router import build_router_targets
from agentic_chatbot_next.runtime.context import filesystem_key
from agentic_chatbot_next.sandbox.workspace import SessionWorkspace
from agentic_chatbot_next.skills.pack_loader import load_skill_pack_from_text
from agentic_chatbot_next.tools.registry import build_tool_definitions


TEXT_SOURCE_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sql",
    ".text",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_RAW_SOURCE_CHARS = 200_000
MAX_EXTRACTED_CONTENT_CHARS = 240_000
CONTROL_PANEL_CAPABILITIES_SCHEMA_VERSION = "1"
CONTROL_PANEL_CONTRACT_VERSION = "control-panel-v1"
COLLECTION_IGNORED_FILENAMES = {".ds_store", "thumbs.db"}
COLLECTION_SUPPORTED_SUFFIXES = TEXT_SOURCE_SUFFIXES | {".docx", ".pdf", ".xls", ".xlsx"} | set(IMAGE_SUFFIXES)
UPLOAD_STREAM_CHUNK_SIZE = 1024 * 1024
MAX_COLLECTION_UPLOAD_FILES = 2_000
MAX_COLLECTION_UPLOAD_BYTES = 512 * 1024 * 1024
CONTROL_PANEL_REQUIRED_ROUTES: Dict[str, List[str]] = {
    "dashboard": ["/v1/admin/overview"],
    "architecture": ["/v1/admin/architecture", "/v1/admin/architecture/activity"],
    "config": ["/v1/admin/config/schema", "/v1/admin/config/effective"],
    "agents": ["/v1/admin/agents"],
    "prompts": ["/v1/admin/prompts"],
    "collections": ["/v1/admin/collections"],
    "graphs": ["/v1/admin/graphs", "/v1/admin/graphs/{graph_id}"],
    "skills": ["/v1/skills"],
    "access": ["/v1/admin/access/principals", "/v1/admin/access/roles", "/v1/admin/access/effective-access"],
    "mcp": ["/v1/admin/mcp/connections"],
    "operations": ["/v1/admin/operations"],
}
COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ConfigChangeRequest(BaseModel):
    changes: Dict[str, Any] = Field(default_factory=dict)
    actor: str = "control-panel"


class PromptUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1)
    actor: str = "control-panel"


class AgentUpdateRequest(BaseModel):
    description: Optional[str] = None
    prompt_file: Optional[str] = None
    skill_scope: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    allowed_worker_agents: Optional[List[str]] = None
    preload_skill_packs: Optional[List[str]] = None
    memory_scopes: Optional[List[str]] = None
    max_steps: Optional[int] = None
    max_tool_calls: Optional[int] = None
    allow_background_jobs: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    body: Optional[str] = None
    actor: str = "control-panel"


class PathIngestRequest(BaseModel):
    paths: List[str] = Field(default_factory=list)
    source_type: str = "host_path"
    conversation_id: Optional[str] = None
    actor: str = "control-panel"


class CollectionCreateRequest(BaseModel):
    collection_id: str = Field(..., min_length=1)
    actor: str = "control-panel"


@dataclass(frozen=True)
class CollectionIngestCandidate:
    absolute_path: Path
    source_display_path: str
    source_type: str
    collection_id: str
    source_identity: str


class GraphAdminUpsertRequest(BaseModel):
    graph_id: Optional[str] = None
    display_name: Optional[str] = None
    collection_id: Optional[str] = None
    source_doc_ids: List[str] = Field(default_factory=list)
    source_paths: List[str] = Field(default_factory=list)
    backend: Optional[str] = None
    visibility: str = "tenant"
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    prompt_overrides: Dict[str, Any] = Field(default_factory=dict)
    graph_skill_ids: List[str] = Field(default_factory=list)
    actor: str = "control-panel"


class GraphLifecycleRequest(BaseModel):
    actor: str = "control-panel"


class GraphPromptUpdateRequest(BaseModel):
    prompt_overrides: Dict[str, Any] = Field(default_factory=dict)
    actor: str = "control-panel"


class GraphResearchTuneRequest(BaseModel):
    guidance: str = ""
    target_prompt_files: List[str] = Field(default_factory=list)
    actor: str = "control-panel"


class GraphResearchTuneApplyRequest(BaseModel):
    prompt_files: List[str] = Field(default_factory=list)
    actor: str = "control-panel"


class GraphSkillUpdateRequest(BaseModel):
    skill_ids: List[str] = Field(default_factory=list)
    overlay_markdown: str = ""
    overlay_skill_name: str = ""
    actor: str = "control-panel"


class AccessPrincipalUpsertRequest(BaseModel):
    principal_id: Optional[str] = None
    principal_type: str = "user"
    provider: str = "email"
    external_id: str = ""
    email_normalized: str = ""
    display_name: str = ""
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    active: bool = True
    actor: str = "control-panel"


class AccessMembershipUpsertRequest(BaseModel):
    membership_id: Optional[str] = None
    parent_principal_id: str = Field(..., min_length=1)
    child_principal_id: str = Field(..., min_length=1)
    actor: str = "control-panel"


class AccessRoleUpsertRequest(BaseModel):
    role_id: Optional[str] = None
    name: str = Field(..., min_length=1)
    description: str = ""
    actor: str = "control-panel"


class AccessRoleBindingUpsertRequest(BaseModel):
    binding_id: Optional[str] = None
    role_id: str = Field(..., min_length=1)
    principal_id: str = Field(..., min_length=1)
    disabled: bool = False
    actor: str = "control-panel"


class AccessRolePermissionUpsertRequest(BaseModel):
    permission_id: Optional[str] = None
    role_id: str = Field(..., min_length=1)
    resource_type: str = Field(..., min_length=1)
    action: str = "use"
    resource_selector: str = "*"
    actor: str = "control-panel"


class McpAdminConnectionCreateRequest(BaseModel):
    display_name: str = Field(..., min_length=1)
    server_url: str = Field(..., min_length=1)
    auth_type: str = "none"
    secret: str = ""
    allowed_agents: List[str] = Field(default_factory=lambda: ["general"])
    visibility: str = "private"
    owner_user_id: Optional[str] = None
    actor: str = "control-panel"
    metadata_json: Dict[str, Any] = Field(default_factory=dict)


class McpAdminConnectionUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    server_url: Optional[str] = None
    auth_type: Optional[str] = None
    secret: Optional[str] = None
    allowed_agents: Optional[List[str]] = None
    visibility: Optional[str] = None
    status: Optional[str] = None
    actor: str = "control-panel"
    metadata_json: Optional[Dict[str, Any]] = None


class McpAdminToolUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    read_only: Optional[bool] = None
    destructive: Optional[bool] = None
    background_safe: Optional[bool] = None
    should_defer: Optional[bool] = None
    search_hint: Optional[str] = None
    defer_priority: Optional[int] = None
    status: Optional[str] = None
    actor: str = "control-panel"


class ArchitectureNodeModel(BaseModel):
    id: str
    label: str
    kind: str
    layer: str
    description: str = ""
    status: str = "configured"
    mode: str = ""
    role_kind: str = ""
    entry_path: str = ""
    prompt_file: str = ""
    overlay_active: bool = False
    allowed_tools: List[str] = Field(default_factory=list)
    allowed_worker_agents: List[str] = Field(default_factory=list)
    preload_skill_packs: List[str] = Field(default_factory=list)
    memory_scopes: List[str] = Field(default_factory=list)
    badges: List[str] = Field(default_factory=list)


class ArchitectureEdgeModel(BaseModel):
    id: str
    source: str
    target: str
    kind: str
    label: str = ""
    emphasis: str = "normal"


class CanonicalPathModel(BaseModel):
    id: str
    label: str
    route: str
    summary: str
    when: str = ""
    target_agent: str = ""
    badges: List[str] = Field(default_factory=list)
    node_ids: List[str] = Field(default_factory=list)
    edge_ids: List[str] = Field(default_factory=list)


class ArchitectureSnapshotModel(BaseModel):
    generated_at: str
    system: Dict[str, Any] = Field(default_factory=dict)
    router: Dict[str, Any] = Field(default_factory=dict)
    nodes: List[ArchitectureNodeModel] = Field(default_factory=list)
    edges: List[ArchitectureEdgeModel] = Field(default_factory=list)
    canonical_paths: List[CanonicalPathModel] = Field(default_factory=list)


class ArchitectureActivityModel(BaseModel):
    route_counts: Dict[str, int] = Field(default_factory=dict)
    router_method_counts: Dict[str, int] = Field(default_factory=dict)
    start_agent_counts: Dict[str, int] = Field(default_factory=dict)
    delegation_counts: Dict[str, int] = Field(default_factory=dict)
    outcome_counts: Dict[str, int] = Field(default_factory=dict)
    negative_rate_by_route: Dict[str, float] = Field(default_factory=dict)
    negative_rate_by_router_method: Dict[str, float] = Field(default_factory=dict)
    recent_mispicks: List[Dict[str, Any]] = Field(default_factory=list)
    review_backlog: Dict[str, Any] = Field(default_factory=dict)
    last_retrain_report: Dict[str, Any] = Field(default_factory=dict)
    recent_flows: List[Dict[str, Any]] = Field(default_factory=list)
    updated_at: str = ""


class CapabilitiesSectionModel(BaseModel):
    supported: bool = False
    required_routes: List[str] = Field(default_factory=list)
    missing_routes: List[str] = Field(default_factory=list)
    reason: str = ""


class CapabilitiesModel(BaseModel):
    schema_version: str
    contract_version: str
    compatible: bool
    generated_at: str
    sections: Dict[str, CapabilitiesSectionModel] = Field(default_factory=dict)


def _admin_manager(
    manager: RuntimeManager = Depends(get_runtime_manager),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> RuntimeManager:
    require_admin_token(manager.get_settings(), x_admin_token=x_admin_token)
    return manager


def _snapshot_or_503(manager: RuntimeManager) -> Any:
    try:
        return manager.get_snapshot()
    except Exception as exc:
        detail = str(exc).strip() or "Runtime is not available."
        raise HTTPException(status_code=503, detail=detail) from exc


def _mcp_store_or_503(runtime: Any) -> Any:
    store = getattr(getattr(getattr(runtime, "bot", None), "ctx", None), "stores", None)
    store = getattr(store, "mcp_connection_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="MCP connection store is not configured.")
    return store


def _mcp_service(runtime: Any) -> McpCatalogService:
    return McpCatalogService(runtime.settings, _mcp_store_or_503(runtime))


def _serialize_mcp_record(record: Any) -> Dict[str, Any]:
    return record.to_dict() if hasattr(record, "to_dict") else dict(record or {})


def _request_context(
    settings: Any,
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
    conversation_id: str = "control-panel",
) -> Any:
    return build_local_context(
        settings,
        tenant_id=tenant_id or getattr(settings, "default_tenant_id", "local-dev"),
        user_id=user_id or getattr(settings, "default_user_id", "local-cli"),
        conversation_id=conversation_id,
    )


def _serialize_tool_catalog() -> List[Dict[str, Any]]:
    definitions = build_tool_definitions(None)
    return [
        {
            "name": getattr(definition, "name", ""),
            "group": getattr(definition, "group", ""),
            "description": getattr(definition, "description", ""),
            "tool_card": (
                definition.render_tool_card()
                if callable(getattr(definition, "render_tool_card", None))
                else str(getattr(definition, "description", "") or "").strip()
            ),
            "args_schema": dict(getattr(definition, "args_schema", {}) or {}),
            "when_to_use": getattr(definition, "when_to_use", ""),
            "avoid_when": getattr(definition, "avoid_when", ""),
            "output_description": getattr(definition, "output_description", ""),
            "examples": list(getattr(definition, "examples", []) or []),
            "keywords": list(getattr(definition, "keywords", []) or []),
            "read_only": bool(getattr(definition, "read_only", False)),
            "destructive": bool(getattr(definition, "destructive", False)),
            "background_safe": bool(getattr(definition, "background_safe", False)),
            "requires_workspace": bool(getattr(definition, "requires_workspace", False)),
            "concurrency_key": getattr(definition, "concurrency_key", ""),
            "serializer": getattr(definition, "serializer", "default"),
            "should_defer": bool(getattr(definition, "should_defer", False)),
            "search_hint": getattr(definition, "search_hint", ""),
            "defer_reason": getattr(definition, "defer_reason", ""),
            "defer_priority": int(getattr(definition, "defer_priority", 50) or 50),
            "eager_for_agents": list(getattr(definition, "eager_for_agents", []) or []),
            "metadata": dict(getattr(definition, "metadata", {}) or {}),
        }
        for definition in sorted(definitions.values(), key=lambda item: (item.group, item.name))
    ]


def _registered_route_paths(request: Request) -> set[str]:
    return {
        path
        for route in getattr(request.app, "routes", [])
        if (path := getattr(route, "path", ""))
    }


def _build_capabilities_payload(request: Request) -> Dict[str, Any]:
    registered_paths = _registered_route_paths(request)
    sections: Dict[str, Dict[str, Any]] = {}
    supported_sections = 0
    for section_name, required_routes in CONTROL_PANEL_REQUIRED_ROUTES.items():
        missing_routes = [route for route in required_routes if route not in registered_paths]
        supported = len(missing_routes) == 0
        if supported:
            supported_sections += 1
        sections[section_name] = {
            "supported": supported,
            "required_routes": list(required_routes),
            "missing_routes": missing_routes,
            "reason": "" if supported else "Running backend is missing one or more required routes for this section.",
        }
    return {
        "schema_version": CONTROL_PANEL_CAPABILITIES_SCHEMA_VERSION,
        "contract_version": CONTROL_PANEL_CONTRACT_VERSION,
        "compatible": supported_sections == len(CONTROL_PANEL_REQUIRED_ROUTES),
        "generated_at": utc_now_iso(),
        "sections": sections,
    }


def _agent_overlay_active(overlay_store: OverlayStore, agent_name: str) -> bool:
    return overlay_store.agent_overlay_path(agent_name).exists()


def _prompt_overlay_active(overlay_store: OverlayStore, prompt_file: str) -> bool:
    return overlay_store.prompt_overlay_path(prompt_file).exists()


def _serialize_skill_reference(record: Any) -> Dict[str, Any]:
    return {
        "skill_id": record.skill_id,
        "name": record.name,
        "agent_scope": record.agent_scope,
        "graph_id": getattr(record, "graph_id", ""),
        "version": record.version,
        "enabled": bool(record.enabled),
        "status": record.status,
        "visibility": record.visibility,
        "version_parent": record.version_parent or record.skill_id,
        "kind": getattr(record, "kind", "retrievable"),
        "execution_config": dict(getattr(record, "execution_config", {}) or {}),
    }


def _access_store(runtime: Any) -> Any:
    store = getattr(getattr(runtime.bot, "ctx", None), "stores", None)
    store = getattr(store, "access_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Access control store is not configured.")
    return store


def _access_authorization_service(runtime: Any) -> Any:
    service = getattr(getattr(runtime.bot, "ctx", None), "stores", None)
    service = getattr(service, "authorization_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Authorization service is not configured.")
    return service


def _serialize_access_principal(record: Any) -> Dict[str, Any]:
    return {
        "principal_id": str(getattr(record, "principal_id", "") or ""),
        "tenant_id": str(getattr(record, "tenant_id", "") or ""),
        "principal_type": str(getattr(record, "principal_type", "") or ""),
        "provider": str(getattr(record, "provider", "") or ""),
        "external_id": str(getattr(record, "external_id", "") or ""),
        "email_normalized": str(getattr(record, "email_normalized", "") or ""),
        "display_name": str(getattr(record, "display_name", "") or ""),
        "metadata_json": dict(getattr(record, "metadata_json", {}) or {}),
        "active": bool(getattr(record, "active", True)),
        "created_at": str(getattr(record, "created_at", "") or ""),
        "updated_at": str(getattr(record, "updated_at", "") or ""),
    }


def _serialize_access_membership(record: Any) -> Dict[str, Any]:
    return {
        "membership_id": str(getattr(record, "membership_id", "") or ""),
        "tenant_id": str(getattr(record, "tenant_id", "") or ""),
        "parent_principal_id": str(getattr(record, "parent_principal_id", "") or ""),
        "child_principal_id": str(getattr(record, "child_principal_id", "") or ""),
        "created_at": str(getattr(record, "created_at", "") or ""),
    }


def _serialize_access_role(record: Any) -> Dict[str, Any]:
    return {
        "role_id": str(getattr(record, "role_id", "") or ""),
        "tenant_id": str(getattr(record, "tenant_id", "") or ""),
        "name": str(getattr(record, "name", "") or ""),
        "description": str(getattr(record, "description", "") or ""),
        "created_at": str(getattr(record, "created_at", "") or ""),
        "updated_at": str(getattr(record, "updated_at", "") or ""),
    }


def _serialize_access_binding(record: Any) -> Dict[str, Any]:
    return {
        "binding_id": str(getattr(record, "binding_id", "") or ""),
        "tenant_id": str(getattr(record, "tenant_id", "") or ""),
        "role_id": str(getattr(record, "role_id", "") or ""),
        "principal_id": str(getattr(record, "principal_id", "") or ""),
        "created_at": str(getattr(record, "created_at", "") or ""),
        "disabled_at": str(getattr(record, "disabled_at", "") or ""),
        "disabled": bool(str(getattr(record, "disabled_at", "") or "").strip()),
    }


def _serialize_access_permission(record: Any) -> Dict[str, Any]:
    return {
        "permission_id": str(getattr(record, "permission_id", "") or ""),
        "tenant_id": str(getattr(record, "tenant_id", "") or ""),
        "role_id": str(getattr(record, "role_id", "") or ""),
        "resource_type": str(getattr(record, "resource_type", "") or ""),
        "action": str(getattr(record, "action", "") or ""),
        "resource_selector": str(getattr(record, "resource_selector", "") or ""),
        "created_at": str(getattr(record, "created_at", "") or ""),
    }


def _serialize_job_summary(job: Any, job_manager: Any | None = None) -> Dict[str, Any]:
    payload = {
        "job_id": getattr(job, "job_id", ""),
        "session_id": getattr(job, "session_id", ""),
        "agent_name": getattr(job, "agent_name", ""),
        "status": getattr(job, "status", ""),
        "scheduler_state": getattr(job, "scheduler_state", ""),
        "priority": getattr(job, "priority", ""),
        "queue_class": getattr(job, "queue_class", ""),
        "tenant_id": getattr(job, "tenant_id", ""),
        "user_id": getattr(job, "user_id", ""),
        "description": getattr(job, "description", ""),
        "estimated_token_cost": int(getattr(job, "estimated_token_cost", 0) or 0),
        "actual_token_cost": int(getattr(job, "actual_token_cost", 0) or 0),
        "budget_block_reason": getattr(job, "budget_block_reason", ""),
        "enqueued_at": getattr(job, "enqueued_at", ""),
        "started_at": getattr(job, "started_at", ""),
        "updated_at": getattr(job, "updated_at", ""),
        "output_path": getattr(job, "output_path", ""),
    }
    if job_manager is not None and hasattr(job_manager, "mailbox_summary"):
        payload["mailbox"] = job_manager.mailbox_summary(str(getattr(job, "job_id", "") or ""))
    return payload


def _graph_service(
    runtime: Any,
    *,
    tenant_id: str,
    user_id: str,
    conversation_id: str = "control-panel-graphs",
) -> GraphService:
    return GraphService(
        runtime.settings,
        runtime.bot.ctx.stores,
        session=_request_context(
            runtime.settings,
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        ),
    )


def _graph_tuning_service(
    runtime: Any,
    *,
    tenant_id: str,
    user_id: str,
    conversation_id: str = "control-panel-graphs",
) -> GraphPromptTuningService:
    return GraphPromptTuningService(
        runtime.settings,
        runtime.bot.ctx.stores,
        session=_request_context(
            runtime.settings,
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        ),
    )


def _graph_skill_records(runtime: Any, graph_payload: Dict[str, Any], *, owner_user_id: str) -> List[Dict[str, Any]]:
    skill_store = getattr(runtime.bot.ctx.stores, "skill_store", None)
    graph = dict(graph_payload.get("graph") or {})
    graph_id = str(graph.get("graph_id") or "")
    tenant_id = str(graph.get("tenant_id") or runtime.settings.default_tenant_id or "local-dev")
    if skill_store is None or not graph_id:
        return []

    bound_records: List[Any] = []
    seen_skill_ids: set[str] = set()
    for skill_id in [str(item) for item in (graph.get("graph_skill_ids") or []) if str(item).strip()]:
        record = skill_store.get_skill_pack(
            skill_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        if record is None or record.skill_id in seen_skill_ids:
            continue
        seen_skill_ids.add(record.skill_id)
        bound_records.append(record)

    try:
        graph_scoped = skill_store.list_skill_packs(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            graph_id=graph_id,
        )
    except TypeError:
        graph_scoped = []
    for record in graph_scoped:
        if record.skill_id in seen_skill_ids:
            continue
        seen_skill_ids.add(record.skill_id)
        bound_records.append(record)
    return [_serialize_skill_reference(record) for record in bound_records]


def _save_graph_overlay_skill(
    runtime: Any,
    *,
    tenant_id: str,
    owner_user_id: str,
    graph_id: str,
    overlay_markdown: str,
    overlay_skill_name: str = "",
) -> str:
    skill_store = getattr(runtime.bot.ctx.stores, "skill_store", None)
    if skill_store is None:
        raise HTTPException(status_code=503, detail="Skill store is not configured.")
    text = str(overlay_markdown or "").strip()
    if not text:
        return ""

    skill_id = f"graph-{filesystem_key(graph_id)}-overlay"
    source_path = f"api://admin/graphs/{graph_id}/overlay-skill.md"
    parsed = load_skill_pack_from_text(
        text,
        source_path=source_path,
        metadata_defaults={
            "name": str(overlay_skill_name or f"{graph_id} graph overlay").strip() or f"{graph_id} graph overlay",
            "agent_scope": "rag",
            "description": f"Graph-scoped overlay guidance for {graph_id}.",
            "version": "1",
            "enabled": True,
            "tool_tags": ["search_graph_index", "inspect_graph_index"],
            "task_tags": ["graph-research", "relationship-analysis"],
        },
    )
    parsed.skill_id = skill_id
    parsed.version_parent = skill_id
    parsed.graph_id = graph_id
    parsed.name = str(overlay_skill_name or parsed.name or f"{graph_id} graph overlay").strip() or f"{graph_id} graph overlay"
    parsed.agent_scope = str(parsed.agent_scope or "rag").strip() or "rag"
    parsed.enabled = True
    parsed.owner_user_id = owner_user_id
    parsed.visibility = "tenant"
    parsed.status = "active"
    parsed.source_path = source_path
    skill_store.upsert_skill_pack(
        SkillPackRecord(
            skill_id=parsed.skill_id,
            tenant_id=tenant_id,
            graph_id=parsed.graph_id,
            name=parsed.name,
            agent_scope=parsed.agent_scope,
            checksum=parsed.checksum,
            tool_tags=list(parsed.tool_tags),
            task_tags=list(parsed.task_tags),
            version=parsed.version,
            enabled=parsed.enabled,
            source_path=parsed.source_path,
            description=parsed.description,
            retrieval_profile=parsed.retrieval_profile,
            controller_hints=dict(parsed.controller_hints),
            coverage_goal=parsed.coverage_goal,
            result_mode=parsed.result_mode,
            body_markdown=parsed.body_markdown,
            owner_user_id=parsed.owner_user_id,
            visibility=parsed.visibility,
            status=parsed.status,
            version_parent=parsed.version_parent or parsed.skill_id,
            kind=parsed.kind,
            execution_config=dict(parsed.execution_config),
        ),
        parsed.chunks,
    )
    return skill_id


def _enrich_graph_payload(runtime: Any, payload: Dict[str, Any], *, owner_user_id: str) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched["skills"] = _graph_skill_records(runtime, enriched, owner_user_id=owner_user_id)
    return enriched


def _serialize_agent(
    runtime: Any,
    overlay_store: OverlayStore,
    *,
    agent_name: str,
    include_body: bool,
    owner_user_id: str = "",
) -> Dict[str, Any]:
    loaded = runtime.bot.kernel.registry.get_loaded_file(agent_name)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Agent not found.")
    definition = loaded.definition
    skill_store = getattr(runtime.bot.ctx.stores, "skill_store", None)
    pinned_skills = []
    if skill_store is not None and definition.preload_skill_packs:
        pinned_skills = [
            _serialize_skill_reference(record)
            for record in skill_store.get_skill_packs_by_ids(
                list(definition.preload_skill_packs),
                tenant_id=runtime.settings.default_tenant_id,
                owner_user_id=owner_user_id,
            )
        ]
    payload = {
        "name": definition.name,
        "mode": definition.mode,
        "description": definition.description,
        "prompt_file": definition.prompt_file,
        "skill_scope": definition.skill_scope,
        "allowed_tools": list(definition.allowed_tools),
        "allowed_worker_agents": list(definition.allowed_worker_agents),
        "preload_skill_packs": list(definition.preload_skill_packs),
        "memory_scopes": list(definition.memory_scopes),
        "max_steps": definition.max_steps,
        "max_tool_calls": definition.max_tool_calls,
        "allow_background_jobs": bool(definition.allow_background_jobs),
        "metadata": dict(definition.metadata or {}),
        "source_path": str(loaded.source_path),
        "overlay_active": _agent_overlay_active(overlay_store, definition.name),
        "pinned_skills": pinned_skills,
    }
    if include_body:
        payload["body"] = loaded.body
        payload["overlay_markdown"] = overlay_store.read_agent_overlay(definition.name)
    return payload


def _normalize_collection_id(value: str) -> str:
    collection_id = str(value or "").strip()
    if not collection_id:
        raise HTTPException(status_code=400, detail="Collection ID is required.")
    if not COLLECTION_ID_RE.match(collection_id):
        raise HTTPException(
            status_code=400,
            detail="Collection IDs may contain letters, numbers, dots, underscores, and hyphens only.",
        )
    return collection_id


def _embedding_model_for_settings(settings: Any) -> str:
    provider = str(getattr(settings, "embeddings_provider", "") or "").lower()
    if provider == "ollama":
        return str(getattr(settings, "ollama_embed_model", "") or "")
    if provider == "azure":
        return str(getattr(settings, "azure_openai_embed_deployment", "") or "")
    return ""


def _collection_graph_indexes(runtime: Any, tenant_id: str) -> List[Any]:
    graph_index_store = getattr(runtime.bot.ctx.stores, "graph_index_store", None)
    if graph_index_store is None:
        return []
    return list(graph_index_store.list_indexes(tenant_id=tenant_id, limit=500))


def _empty_collection_health_payload(*, tenant_id: str, collection_id: str) -> Dict[str, Any]:
    return {
        "status": "ready",
        "reason": "empty_collection",
        "tenant_id": tenant_id,
        "collection_id": collection_id,
        "maintenance_policy": COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,
        "configured_source_count": 0,
        "indexed_doc_count": 0,
        "active_doc_count": 0,
        "missing_sources": [],
        "duplicate_group_count": 0,
        "content_drift_count": 0,
        "duplicate_groups": [],
        "drifted_groups": [],
        "source_groups": [],
        "sync_error": "",
        "suggested_fix": "",
    }


def _collection_readiness(runtime: Any, tenant_id: str, collection_id: str) -> Any:
    return get_collection_readiness_status(
        runtime.settings,
        runtime.bot.ctx.stores,
        tenant_id=tenant_id,
        collection_id=collection_id,
    )


def _collection_health_payload(runtime: Any, tenant_id: str, collection_id: str) -> Dict[str, Any]:
    readiness = _collection_readiness(runtime, tenant_id, collection_id)
    if readiness.reason == "empty_collection":
        payload = _empty_collection_health_payload(tenant_id=tenant_id, collection_id=collection_id)
        payload["maintenance_policy"] = readiness.maintenance_policy
        payload["reason"] = readiness.reason
        payload["suggested_fix"] = str(readiness.suggested_fix or "")
        return payload
    if readiness.maintenance_policy == COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES:
        payload = build_kb_health_report(
            runtime.settings,
            runtime.bot.ctx.stores,
            tenant_id=tenant_id,
            collection_id=collection_id,
        ).to_dict()
        payload["maintenance_policy"] = readiness.maintenance_policy
        return payload
    payload = build_collection_health_report(
        runtime.settings,
        runtime.bot.ctx.stores,
        tenant_id=tenant_id,
        collection_id=collection_id,
        maintenance_policy=readiness.maintenance_policy,
    ).to_dict()
    payload["maintenance_policy"] = readiness.maintenance_policy
    return payload


def _collection_storage_profile(runtime: Any, collection_graphs: List[Any]) -> Dict[str, Any]:
    configured_embedding_dim = int(getattr(runtime.settings, "embedding_dim", 0) or 0)
    actual_embedding_dims: Dict[str, int] = {}
    try:
        chunks_dim = get_table_embedding_dim("chunks")
        if chunks_dim is not None:
            actual_embedding_dims["chunks"] = chunks_dim
        if collection_graphs:
            graph_indexes_dim = get_graph_indexes_embedding_dim()
            if graph_indexes_dim is not None:
                actual_embedding_dims["graph_indexes"] = graph_indexes_dim
    except Exception:
        actual_embedding_dims = {}
    mismatch_warnings = [
        f"{table_name} uses vector({actual_dim}) while EMBEDDING_DIM is {configured_embedding_dim}."
        for table_name, actual_dim in actual_embedding_dims.items()
        if configured_embedding_dim and actual_dim != configured_embedding_dim
    ]
    tables = ["documents", "chunks"]
    if collection_graphs:
        tables.extend(["graph_indexes", "graph_index_sources", "graph_index_runs"])
    return {
        "vector_store_backend": str(getattr(runtime.settings, "vector_store_backend", "") or ""),
        "tables": tables,
        "embeddings_provider": str(getattr(runtime.settings, "embeddings_provider", "") or ""),
        "embedding_model": _embedding_model_for_settings(runtime.settings),
        "graph_embedding_model": (
            str(getattr(runtime.settings, "graphrag_embed_model", "") or "")
            if collection_graphs
            else ""
        ),
        "configured_embedding_dim": configured_embedding_dim,
        "actual_embedding_dims": actual_embedding_dims,
        "mismatch_warnings": mismatch_warnings,
    }


def _build_collection_payload(
    runtime: Any,
    tenant_id: str,
    collection_id: str,
    *,
    collection_record: Any = None,
    doc_summary: Dict[str, Any] | None = None,
    collection_graphs: List[Any] | None = None,
) -> Dict[str, Any]:
    graphs = list(collection_graphs or [])
    summary = dict(doc_summary or {})
    collection_store = getattr(runtime.bot.ctx.stores, "collection_store", None)
    record = collection_record
    if record is None and collection_store is not None:
        record = collection_store.get_collection(collection_id, tenant_id=tenant_id)
        if record is None and (summary or graphs):
            record = collection_store.ensure_collection(tenant_id=tenant_id, collection_id=collection_id)
    latest_ingested_at = str(summary.get("latest_ingested_at") or "")
    created_at = str(getattr(record, "created_at", "") or "")
    updated_at = str(getattr(record, "updated_at", "") or latest_ingested_at or created_at)
    readiness = _collection_readiness(runtime, tenant_id, collection_id)
    return {
        "collection_id": collection_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "maintenance_policy": str(getattr(readiness, "maintenance_policy", "") or ""),
        "document_count": int(summary.get("document_count") or 0),
        "source_type_counts": dict(summary.get("source_type_counts") or {}),
        "latest_ingested_at": latest_ingested_at,
        "graph_count": len(graphs),
        "graph_ids": [str(getattr(graph, "graph_id", "") or "") for graph in graphs],
        "storage_profile": _collection_storage_profile(runtime, graphs),
        "status": _serialize_collection_status(
            runtime,
            tenant_id,
            collection_id,
            doc_summary=summary,
            graph_count=len(graphs),
        ),
    }


def _list_collection_payloads(runtime: Any, tenant_id: str) -> List[Dict[str, Any]]:
    collection_store = getattr(runtime.bot.ctx.stores, "collection_store", None)
    catalog_records = {
        str(record.collection_id or ""): record
        for record in (collection_store.list_collections(tenant_id=tenant_id) if collection_store is not None else [])
    }
    doc_summaries = {
        str(item.get("collection_id") or ""): dict(item)
        for item in runtime.bot.ctx.stores.doc_store.list_collections(tenant_id=tenant_id)
    }
    graphs_by_collection: Dict[str, List[Any]] = {}
    for graph in _collection_graph_indexes(runtime, tenant_id):
        collection_id = str(getattr(graph, "collection_id", "") or "")
        if not collection_id:
            continue
        graphs_by_collection.setdefault(collection_id, []).append(graph)

    collection_ids = sorted(set(catalog_records) | set(doc_summaries) | set(graphs_by_collection))
    return [
        _build_collection_payload(
            runtime,
            tenant_id,
            collection_id,
            collection_record=catalog_records.get(collection_id),
            doc_summary=doc_summaries.get(collection_id),
            collection_graphs=graphs_by_collection.get(collection_id, []),
        )
        for collection_id in collection_ids
        if collection_id
    ]


def _serialize_collection_status(
    runtime: Any,
    tenant_id: str,
    collection_id: str,
    *,
    doc_summary: Dict[str, Any] | None = None,
    graph_count: int = 0,
) -> Dict[str, Any]:
    del doc_summary, graph_count
    readiness = _collection_readiness(runtime, tenant_id, collection_id)
    return {
        "ready": bool(readiness.ready),
        "reason": str(readiness.reason or ""),
        "collection_id": readiness.collection_id,
        "missing_sources": list(readiness.missing_source_paths),
        "indexed_doc_count": int(readiness.document_count),
        "active_doc_count": int(readiness.document_count),
        "duplicate_group_count": 0,
        "content_drift_count": 0,
        "suggested_fix": str(readiness.suggested_fix or ""),
    }


def _serialize_collection_health(runtime: Any, tenant_id: str, collection_id: str) -> Dict[str, Any]:
    return _collection_health_payload(runtime, tenant_id, collection_id)


def _read_raw_source(record: Any) -> Dict[str, Any] | None:
    path = Path(str(getattr(record, "source_path", "") or ""))
    if not path.exists() or path.suffix.lower() not in TEXT_SOURCE_SUFFIXES:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    truncated = len(text) > MAX_RAW_SOURCE_CHARS
    return {
        "path": str(path),
        "content": text[:MAX_RAW_SOURCE_CHARS],
        "truncated": truncated,
    }


def _reconstruct_document_content(chunks: List[Any]) -> Dict[str, Any]:
    sections: List[str] = []
    for chunk in chunks:
        labels = [f"Chunk {int(getattr(chunk, 'chunk_index', 0))}"]
        if getattr(chunk, "page_number", None) is not None:
            labels.append(f"page {chunk.page_number}")
        if getattr(chunk, "section_title", None):
            labels.append(str(chunk.section_title))
        header = " | ".join(labels)
        sections.append(f"## {header}\n{str(getattr(chunk, 'content', '') or '').strip()}")
    text = "\n\n".join(section for section in sections if section.strip())
    truncated = len(text) > MAX_EXTRACTED_CONTENT_CHARS
    return {
        "content": text[:MAX_EXTRACTED_CONTENT_CHARS],
        "truncated": truncated,
        "chunk_count": len(chunks),
    }


def _workspace_copy(
    settings: Any,
    *,
    tenant_id: str,
    user_id: str,
    conversation_id: str,
    paths: List[Path],
    display_paths: Dict[str, str] | None = None,
) -> List[str]:
    workspace = SessionWorkspace.for_session(
        f"{tenant_id}:{user_id}:{conversation_id}",
        Path(getattr(settings, "workspace_dir")),
    )
    workspace.open()
    copied: List[str] = []
    resolved_display_paths = {
        str(Path(raw_path).expanduser().resolve()): str(relative_path or "")
        for raw_path, relative_path in (display_paths or {}).items()
    }
    for path in paths:
        try:
            relative_path = resolved_display_paths.get(str(path.resolve())) or path.name
            destination = workspace.root / PurePosixPath(relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(relative_path)
        except Exception:
            continue
    return copied


def _sanitize_relative_upload_path(raw_value: str, fallback_name: str) -> str:
    cleaned = str(raw_value or "").replace("\\", "/").strip()
    safe_parts = [
        part
        for part in PurePosixPath(cleaned).parts
        if part not in {"", ".", ".."}
    ]
    candidate = "/".join(safe_parts)
    if not candidate:
        candidate = str(fallback_name or "").strip()
    return candidate or f"upload_{uuid.uuid4().hex}"


def _dedupe_relative_upload_path(relative_path: str, used_paths: set[str]) -> str:
    if relative_path not in used_paths:
        used_paths.add(relative_path)
        return relative_path

    path = PurePosixPath(relative_path)
    stem = path.stem or path.name or "upload"
    suffix = path.suffix
    parent = "." if str(path.parent) == "." else str(path.parent)
    index = 2
    while True:
        candidate_name = f"{stem}_{index}{suffix}"
        candidate = candidate_name if parent == "." else f"{parent}/{candidate_name}"
        if candidate not in used_paths:
            used_paths.add(candidate)
            return candidate
        index += 1


def _ensure_collection_catalog_entry(
    runtime: Any,
    tenant_id: str,
    collection_id: str,
    *,
    maintenance_policy: str = "",
) -> None:
    collection_store = getattr(runtime.bot.ctx.stores, "collection_store", None)
    if collection_store is not None:
        collection_store.ensure_collection(
            tenant_id=tenant_id,
            collection_id=collection_id,
            maintenance_policy=maintenance_policy,
        )


def _collection_candidate_skip_reason(path: Path) -> str:
    normalized_name = path.name.lower()
    if normalized_name in COLLECTION_IGNORED_FILENAMES or normalized_name.startswith("._"):
        return "Ignored system file."
    if any(part.startswith(".") for part in path.parts if part not in {".", ".."}):
        return "Ignored hidden file."
    suffix = path.suffix.lower()
    if suffix and suffix in COLLECTION_SUPPORTED_SUFFIXES:
        return ""
    return "Unsupported file type."


def _collection_file_result(
    *,
    candidate: CollectionIngestCandidate | None = None,
    display_path: str = "",
    source_type: str = "",
    outcome: str,
    error: str = "",
    doc_ids: List[str] | None = None,
) -> Dict[str, Any]:
    resolved_display_path = display_path or (candidate.source_display_path if candidate else "")
    resolved_path = str(candidate.absolute_path) if candidate is not None else ""
    return {
        "display_path": resolved_display_path,
        "filename": Path(resolved_display_path or resolved_path or "unknown").name,
        "source_type": source_type or (candidate.source_type if candidate else "upload"),
        "source_path": resolved_path,
        "outcome": outcome,
        "error": error,
        "doc_ids": list(doc_ids or []),
    }


async def _stream_upload_to_path(
    file: UploadFile,
    dest: Path,
    *,
    max_bytes: int = MAX_COLLECTION_UPLOAD_BYTES,
) -> int:
    written = 0
    with dest.open("wb") as handle:
        while True:
            chunk = await file.read(UPLOAD_STREAM_CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                handle.close()
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"{file.filename or dest.name} exceeds the upload limit of {max_bytes} bytes")
            handle.write(chunk)
    return written


def _collection_operation_payload(
    *,
    collection_id: str,
    file_results: List[Dict[str, Any]],
    missing_paths: List[str] | None = None,
    workspace_copies: List[str] | None = None,
) -> Dict[str, Any]:
    missing = list(missing_paths or [])
    doc_ids = [doc_id for item in file_results for doc_id in item.get("doc_ids", [])]
    ingested_count = len(doc_ids)
    already_indexed_count = sum(1 for item in file_results if str(item.get("outcome") or "") == "already_indexed")
    skipped_count = sum(1 for item in file_results if str(item.get("outcome") or "") == "skipped")
    failed_count = sum(1 for item in file_results if str(item.get("outcome") or "") == "failed")
    resolved_count = len(file_results)
    if failed_count and ingested_count == 0:
        status = "failed"
    elif failed_count or skipped_count or missing:
        status = "partial" if ingested_count > 0 else "failed"
    else:
        status = "success"
    errors = [str(item.get("error") or "") for item in file_results if str(item.get("error") or "").strip()]
    return {
        "collection_id": collection_id,
        "status": status,
        "summary": {
            "resolved_count": resolved_count,
            "ingested_count": ingested_count,
            "already_indexed_count": already_indexed_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "missing_count": len(missing),
        },
        "resolved_count": resolved_count,
        "ingested_count": ingested_count,
        "already_indexed_count": already_indexed_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "doc_ids": doc_ids,
        "missing_paths": missing,
        "errors": errors,
        "files": file_results,
        "filenames": [str(item.get("filename") or "") for item in file_results],
        "display_paths": [str(item.get("display_path") or "") for item in file_results],
        "workspace_copies": list(workspace_copies or []),
    }


def _ingest_collection_candidates(
    runtime: Any,
    *,
    tenant_id: str,
    collection_id: str,
    candidates: List[CollectionIngestCandidate],
    user_id: str,
    conversation_id: str,
) -> Dict[str, Any]:
    file_results: List[Dict[str, Any]] = []
    copied_paths: List[Path] = []
    copied_display_paths: Dict[str, str] = {}
    for candidate in candidates:
        try:
            doc_ids = ingest_paths(
                runtime.settings,
                runtime.bot.ctx.stores,
                [candidate.absolute_path],
                source_type=candidate.source_type,
                tenant_id=tenant_id,
                collection_id=collection_id,
                source_display_paths={str(candidate.absolute_path): candidate.source_display_path},
                source_identities={str(candidate.absolute_path): candidate.source_identity},
            )
            outcome = "ingested" if doc_ids else "already_indexed"
            message = "" if doc_ids else "Already indexed or no extractable content was found."
            file_results.append(
                _collection_file_result(
                    candidate=candidate,
                    outcome=outcome,
                    error=message,
                    doc_ids=doc_ids,
                )
            )
            copied_paths.append(candidate.absolute_path)
            copied_display_paths[str(candidate.absolute_path)] = candidate.source_display_path
        except Exception as exc:
            file_results.append(
                _collection_file_result(
                    candidate=candidate,
                    outcome="failed",
                    error=str(exc),
                )
            )
    workspace_copies = _workspace_copy(
        runtime.settings,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        paths=copied_paths,
        display_paths=copied_display_paths,
    )
    return _collection_operation_payload(
        collection_id=collection_id,
        file_results=file_results,
        workspace_copies=workspace_copies,
    )


def _render_agent_overlay_markdown(existing: Any, request: AgentUpdateRequest) -> str:
    definition = existing.definition
    payload = {
        "name": definition.name,
        "mode": definition.mode,
        "description": request.description if request.description is not None else definition.description,
        "prompt_file": request.prompt_file if request.prompt_file is not None else definition.prompt_file,
        "skill_scope": request.skill_scope if request.skill_scope is not None else definition.skill_scope,
        "allowed_tools": request.allowed_tools if request.allowed_tools is not None else list(definition.allowed_tools),
        "allowed_worker_agents": request.allowed_worker_agents if request.allowed_worker_agents is not None else list(definition.allowed_worker_agents),
        "preload_skill_packs": request.preload_skill_packs if request.preload_skill_packs is not None else list(definition.preload_skill_packs),
        "memory_scopes": request.memory_scopes if request.memory_scopes is not None else list(definition.memory_scopes),
        "max_steps": request.max_steps if request.max_steps is not None else definition.max_steps,
        "max_tool_calls": request.max_tool_calls if request.max_tool_calls is not None else definition.max_tool_calls,
        "allow_background_jobs": request.allow_background_jobs if request.allow_background_jobs is not None else definition.allow_background_jobs,
        "metadata": request.metadata if request.metadata is not None else dict(definition.metadata or {}),
    }
    body = request.body if request.body is not None else existing.body
    lines = ["---"]
    for key, value in payload.items():
        if isinstance(value, (list, dict)):
            rendered = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.extend(["---", str(body or "").rstrip(), ""])
    return "\n".join(lines)


def _catalog_for_settings(settings: Any) -> Any:
    try:
        registry = AgentRegistry(
            Path(getattr(settings, "agents_dir")),
            overlay_dir=Path(getattr(settings, "control_panel_agent_overlays_dir")),
        )
    except Exception:
        # Config repair should remain available even if an overlay draft is invalid.
        registry = AgentRegistry(Path(getattr(settings, "agents_dir")))
    return build_config_catalog(agent_names=[agent.name for agent in registry.list()])


def _preview_config_change(manager: RuntimeManager, request: ConfigChangeRequest) -> Dict[str, Any]:
    settings = manager.get_settings()
    catalog = _catalog_for_settings(settings)
    validation = catalog.validate_changes(request.changes)
    if not validation["valid"]:
        return validation

    overlay_store = manager.get_overlay_store()
    merged_env = overlay_store.apply_runtime_env_changes(validation["normalized_changes"])
    try:
        preview = manager.preview_snapshot(env_overrides=merged_env)
    except Exception as exc:
        return {
            **validation,
            "valid": False,
            "errors": {"runtime": str(exc)},
        }

    before = catalog.effective_values(settings, masked=True)
    after = catalog.effective_values(preview.settings, masked=True)
    diff = {
        key: {"before": before.get(key, ""), "after": after.get(key, "")}
        for key in sorted(set(before) | set(after))
        if before.get(key, "") != after.get(key, "")
    }
    return {
        **validation,
        "_merged_overlay": merged_env,
        "preview_diff": diff,
        "reload_scope": "runtime_swap",
        "runtime_diagnostics": runtime_settings_diagnostics(preview.settings),
    }


def _role_kind(agent: Any) -> str:
    return str(getattr(agent, "metadata", {}).get("role_kind") or "").strip().lower()


def _entry_path(agent: Any) -> str:
    return str(getattr(agent, "metadata", {}).get("entry_path") or "").strip().lower()


def _expected_output(agent: Any) -> str:
    return str(getattr(agent, "metadata", {}).get("expected_output") or "").strip().lower()


def _router_mode_label(settings: Any) -> str:
    if not bool(getattr(settings, "llm_router_enabled", True)):
        return "Deterministic only"
    mode = str(getattr(settings, "llm_router_mode", "hybrid") or "hybrid").strip().lower()
    if mode == "llm_only":
        return "LLM primary"
    if mode == "hybrid":
        return "Hybrid"
    return "Deterministic only"


def _event_payload_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _architecture_service_names(agent: Any) -> set[str]:
    tools = {str(item).strip() for item in getattr(agent, "allowed_tools", []) if str(item).strip()}
    lower_tools = {tool.lower() for tool in tools}
    services: set[str] = set()
    if (
        getattr(agent, "mode", "") == "rag"
        or _expected_output(agent) == "rag_contract"
        or {
            "search_indexed_docs",
            "list_indexed_docs",
            "read_indexed_document",
            "read_document",
            "resolve_indexed_docs",
            "read_indexed_doc",
            "compare_indexed_docs",
        } & lower_tools
    ):
        services.add("Knowledge Base")
    if (
        bool(getattr(agent, "allowed_worker_agents", []))
        or bool(getattr(agent, "allow_background_jobs", False))
        or _role_kind(agent) == "manager"
        or getattr(agent, "mode", "") == "coordinator"
    ):
        services.add("Job Manager")
    if bool(getattr(agent, "preload_skill_packs", [])) or {"search_skills", "preview_skills"} & lower_tools:
        services.add("Skill Store")
    if bool(getattr(agent, "memory_scopes", [])) or any("memory" in tool for tool in lower_tools):
        services.add("Memory Store")
    if {"execute_code", "load_dataset", "return_file", "analyze_dataframe"} & lower_tools:
        services.add("Python Sandbox")
    return services


def _service_badges(service_name: str) -> List[str]:
    if service_name == "Knowledge Base":
        return ["Grounded", "Collections"]
    if service_name == "Job Manager":
        return ["Background jobs", "Delegation"]
    if service_name == "Skill Store":
        return ["Pinned skills", "Similarity search"]
    if service_name == "Memory Store":
        return ["Conversation memory"]
    if service_name == "Python Sandbox":
        return ["Code execution", "Data analysis"]
    return []


def _iter_recent_runtime_events(runtime_root: Path, *, max_session_files: int = 40, max_lines_per_file: int = 160) -> List[RuntimeEvent]:
    sessions_root = runtime_root / "sessions"
    if not sessions_root.exists():
        return []
    try:
        event_paths = sorted(
            sessions_root.glob("*/events.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:max_session_files]
    except OSError:
        return []
    events: List[RuntimeEvent] = []
    for path in event_paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines_per_file:]
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(RuntimeEvent.from_dict(raw))
    return sorted(events, key=lambda event: event.created_at)


def _build_architecture_snapshot(runtime: Any, overlay_store: OverlayStore) -> Dict[str, Any]:
    registry = runtime.bot.kernel.registry
    targets = build_router_targets(registry)
    settings = runtime.settings
    worker_names = {
        str(worker_name)
        for definition in registry.list()
        for worker_name in getattr(definition, "allowed_worker_agents", []) or []
        if str(worker_name).strip()
    }

    nodes: List[Dict[str, Any]] = [
        {
            "id": "entry-user",
            "label": "User",
            "kind": "entry",
            "layer": "entry",
            "description": "The starting point for every chat request entering the runtime.",
            "status": "active",
            "badges": ["Requests"],
        },
        {
            "id": "entry-api-gateway",
            "label": "API Gateway",
            "kind": "gateway",
            "layer": "entry",
            "description": f"HTTP entry point using gateway model {getattr(settings, 'gateway_model_id', 'local')}.",
            "status": "configured",
            "badges": [str(getattr(settings, "gateway_model_id", "local") or "local")],
        },
        {
            "id": "router-core",
            "label": "Router",
            "kind": "router",
            "layer": "routing",
            "description": "Chooses BASIC vs AGENT and may suggest a specialist starting agent.",
            "status": "configured",
            "badges": [
                _router_mode_label(settings),
                f"Threshold {float(getattr(settings, 'llm_router_confidence_threshold', 0.70)):.2f}",
            ],
        },
    ]
    edges: List[Dict[str, Any]] = [
        {
            "id": "edge-entry-user-api",
            "source": "entry-user",
            "target": "entry-api-gateway",
            "kind": "request_flow",
            "label": "send request",
            "emphasis": "high",
        },
        {
            "id": "edge-entry-api-router",
            "source": "entry-api-gateway",
            "target": "router-core",
            "kind": "request_flow",
            "label": "route turn",
            "emphasis": "high",
        },
    ]
    known_services: set[str] = set()
    agent_ids: Dict[str, str] = {}

    for definition in sorted(registry.list(), key=lambda item: item.name):
        entry_path = _entry_path(definition)
        role_kind = _role_kind(definition)
        node_id = f"agent-{definition.name}"
        layer = "top_level" if definition.mode == "basic" or registry.is_routable(definition) else "workers"
        if definition.name in worker_names and layer != "top_level":
            layer = "workers"
        badges: List[str] = []
        if definition.mode == "basic":
            badges.append("Basic")
        if definition.mode == "rag" or _expected_output(definition) == "rag_contract":
            badges.append("RAG")
        if role_kind == "manager" or definition.mode == "coordinator":
            badges.append("Manager")
        if getattr(definition, "allow_background_jobs", False):
            badges.append("Background jobs")
        if getattr(definition, "allowed_worker_agents", []):
            badges.append("Worker-capable")
        if _agent_overlay_active(overlay_store, definition.name):
            badges.append("Overlay active")
        nodes.append(
            {
                "id": node_id,
                "label": definition.name,
                "kind": "agent",
                "layer": layer,
                "description": definition.description,
                "status": "overlay active" if _agent_overlay_active(overlay_store, definition.name) else "configured",
                "mode": definition.mode,
                "role_kind": role_kind,
                "entry_path": entry_path,
                "prompt_file": definition.prompt_file,
                "overlay_active": _agent_overlay_active(overlay_store, definition.name),
                "allowed_tools": list(definition.allowed_tools),
                "allowed_worker_agents": list(definition.allowed_worker_agents),
                "preload_skill_packs": list(definition.preload_skill_packs),
                "memory_scopes": list(definition.memory_scopes),
                "badges": badges,
            }
        )
        agent_ids[definition.name] = node_id

        for worker_name in getattr(definition, "allowed_worker_agents", []) or []:
            if worker_name not in agent_ids:
                agent_ids[worker_name] = f"agent-{worker_name}"
            edges.append(
                {
                    "id": f"edge-delegate-{definition.name}-{worker_name}",
                    "source": node_id,
                    "target": agent_ids[worker_name],
                    "kind": "delegation",
                    "label": "delegates work",
                    "emphasis": "normal",
                }
            )

        for service_name in sorted(_architecture_service_names(definition)):
            known_services.add(service_name)
            service_id = f"service-{filesystem_key(service_name)}"
            edges.append(
                {
                    "id": f"edge-service-{definition.name}-{filesystem_key(service_name)}",
                    "source": node_id,
                    "target": service_id,
                    "kind": "service_dependency",
                    "label": service_name,
                    "emphasis": "normal",
                }
            )

    for service_name in sorted(known_services):
        nodes.append(
            {
                "id": f"service-{filesystem_key(service_name)}",
                "label": service_name,
                "kind": "service",
                "layer": "services",
                "description": f"Runtime service used by one or more agents through live tools or capabilities.",
                "status": "active",
                "badges": _service_badges(service_name),
            }
        )

    route_edges: List[tuple[str, str, str, str]] = [
        ("basic", targets.basic_agent, "BASIC", "high"),
        ("default", targets.default_agent, "Default AGENT", "high"),
        ("coordinator", targets.coordinator_agent, "Coordinator", "normal"),
        ("data-analyst", targets.data_analyst_agent, "Data analyst", "normal"),
        ("rag", targets.rag_agent, "Grounded lookup", "normal"),
    ]
    for route_key, agent_name, label, emphasis in route_edges:
        if not agent_name or agent_name not in agent_ids:
            continue
        edges.append(
            {
                "id": f"edge-router-{route_key}-{agent_name}",
                "source": "router-core",
                "target": agent_ids[agent_name],
                "kind": "routing_path",
                "label": label,
                "emphasis": emphasis,
            }
        )

    def _path_node_ids(target_agent: str) -> List[str]:
        base = ["entry-user", "entry-api-gateway", "router-core"]
        if target_agent in agent_ids:
            base.append(agent_ids[target_agent])
        return base

    def _path_edge_ids(route_key: str, target_agent: str) -> List[str]:
        edge_ids = ["edge-entry-user-api", "edge-entry-api-router"]
        edge_ids.append(f"edge-router-{route_key}-{target_agent}")
        return edge_ids

    canonical_paths = [
        {
            "id": "basic",
            "label": "Basic response",
            "route": "BASIC",
            "summary": "General questions and small talk stay on the lightweight chat path.",
            "when": "The router finds no tool, grounding, or multi-step signal.",
            "target_agent": targets.basic_agent,
            "badges": ["Fast path", "Low overhead"],
            "node_ids": _path_node_ids(targets.basic_agent),
            "edge_ids": _path_edge_ids("basic", targets.basic_agent),
        },
        {
            "id": "default-agent",
            "label": "Default agent",
            "route": "AGENT",
            "summary": "Standard tool-capable route when the system needs agent reasoning but no specialist hint is required.",
            "when": "The router selects AGENT and the default top-level agent remains the best starting role.",
            "target_agent": targets.default_agent,
            "badges": ["Default start", "Top-level agent"],
            "node_ids": _path_node_ids(targets.default_agent),
            "edge_ids": _path_edge_ids("default", targets.default_agent),
        },
        {
            "id": "grounded-lookup",
            "label": "Grounded lookup",
            "route": "AGENT",
            "summary": "Focused citation and retrieval requests start in the grounded document specialist.",
            "when": "The router sees citation, grounding, or uploaded-document lookup signals.",
            "target_agent": targets.rag_agent,
            "badges": ["RAG", "Documents"],
            "node_ids": _path_node_ids(targets.rag_agent),
            "edge_ids": _path_edge_ids("rag", targets.rag_agent),
        },
        {
            "id": "data-analysis",
            "label": "Data analysis",
            "route": "AGENT",
            "summary": "Spreadsheet, CSV, workbook, and charting requests start in the data-analysis specialist.",
            "when": "The router sees tabular analysis intent or uploaded spreadsheet-like inputs.",
            "target_agent": targets.data_analyst_agent,
            "badges": ["Python", "Tabular data"],
            "node_ids": _path_node_ids(targets.data_analyst_agent),
            "edge_ids": _path_edge_ids("data-analyst", targets.data_analyst_agent),
        },
        {
            "id": "coordinator-campaign",
            "label": "Coordinator campaign",
            "route": "AGENT",
            "summary": "Broad multi-document campaigns can start in the coordinator so it can fan work out to workers.",
            "when": "The router detects corpus-wide research, orchestration, or manager-style campaign work.",
            "target_agent": targets.coordinator_agent,
            "badges": ["Manager", "Delegation"],
            "node_ids": _path_node_ids(targets.coordinator_agent),
            "edge_ids": _path_edge_ids("coordinator", targets.coordinator_agent),
        },
    ]

    return {
        "generated_at": utc_now_iso(),
        "system": {
            "gateway_model_id": getattr(settings, "gateway_model_id", "local"),
            "providers": {
                "llm_provider": getattr(settings, "llm_provider", ""),
                "judge_provider": getattr(settings, "judge_provider", ""),
                "embeddings_provider": getattr(settings, "embeddings_provider", ""),
            },
            "counts": {
                "agents": len([node for node in nodes if node["kind"] == "agent"]),
                "services": len([node for node in nodes if node["kind"] == "service"]),
                "edges": len(edges),
                "overlays": sum(1 for node in nodes if node.get("overlay_active")),
            },
        },
        "router": {
            "mode_label": _router_mode_label(settings),
            "llm_router_enabled": bool(getattr(settings, "llm_router_enabled", True)),
            "llm_router_mode": str(getattr(settings, "llm_router_mode", "hybrid") or "hybrid"),
            "confidence_threshold": float(getattr(settings, "llm_router_confidence_threshold", 0.70)),
            "enable_coordinator_mode": bool(getattr(settings, "enable_coordinator_mode", False)),
            "default_agent": targets.default_agent,
            "basic_agent": targets.basic_agent,
            "coordinator_agent": targets.coordinator_agent,
            "data_analyst_agent": targets.data_analyst_agent,
            "rag_agent": targets.rag_agent,
            "suggested_agents": list(targets.suggested_agents),
        },
        "nodes": nodes,
        "edges": edges,
        "canonical_paths": canonical_paths,
    }


def _build_architecture_activity(runtime: Any) -> Dict[str, Any]:
    kernel = runtime.bot.kernel
    runtime_root = Path(getattr(getattr(kernel, "paths", None), "runtime_root", Path("data") / "runtime"))
    events = _iter_recent_runtime_events(runtime_root)
    route_counts: Counter[str] = Counter()
    router_method_counts: Counter[str] = Counter()
    start_agent_counts: Counter[str] = Counter()
    delegation_counts: Counter[str] = Counter()
    flow_index: Dict[str, Dict[str, Any]] = {}

    for event in events:
        payload = dict(event.payload or {})
        flow = flow_index.setdefault(
            event.session_id,
            {
                "session_id": event.session_id,
                "conversation_id": str(payload.get("conversation_id") or ""),
                "route": "",
                "router_method": "",
                "start_agent": "",
                "suggested_agent": "",
                "reasons": [],
                "worker_agents": [],
                "degraded": False,
                "degraded_events": [],
                "updated_at": event.created_at,
            },
        )
        flow["updated_at"] = event.created_at
        if payload.get("conversation_id"):
            flow["conversation_id"] = str(payload.get("conversation_id") or "")

        if event.event_type == "router_decision":
            flow["route"] = str(payload.get("route") or flow["route"])
            flow["router_method"] = str(payload.get("router_method") or flow["router_method"])
            flow["suggested_agent"] = str(payload.get("suggested_agent") or flow["suggested_agent"])
            flow["reasons"] = _event_payload_list(payload.get("reasons")) or flow["reasons"]
            continue

        if event.event_type in {"basic_turn_started", "agent_turn_started"}:
            route = str(payload.get("route") or ("BASIC" if event.event_type == "basic_turn_started" else "AGENT"))
            router_method = str(payload.get("router_method") or flow["router_method"] or "deterministic")
            reasons = _event_payload_list(payload.get("router_reasons")) or flow["reasons"]
            route_counts[route] += 1
            router_method_counts[router_method] += 1
            if event.agent_name:
                start_agent_counts[event.agent_name] += 1
            flow.update(
                {
                    "route": route,
                    "router_method": router_method,
                    "start_agent": str(event.agent_name or flow["start_agent"]),
                    "suggested_agent": str(payload.get("suggested_agent") or flow["suggested_agent"]),
                    "reasons": reasons,
                    "started_at": event.created_at,
                }
            )
            continue

        if event.event_type in {"worker_agent_started", "worker_agent_completed"}:
            worker_name = str(event.agent_name or "")
            if worker_name:
                delegation_counts[worker_name] += 1 if event.event_type == "worker_agent_started" else 0
                if worker_name not in flow["worker_agents"]:
                    flow["worker_agents"].append(worker_name)
            continue

        if event.event_type in {"router_degraded_to_deterministic", "agent_downgraded_to_basic", "degraded_response_returned"}:
            flow["degraded"] = True
            if event.event_type not in flow["degraded_events"]:
                flow["degraded_events"].append(event.event_type)

    recent_flows = sorted(flow_index.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)[:12]
    for flow in recent_flows:
        flow["worker_agents"] = list(flow.get("worker_agents") or [])
        flow["reasons"] = list(flow.get("reasons") or [])
        flow["degraded_events"] = list(flow.get("degraded_events") or [])

    router_feedback = getattr(kernel, "router_feedback", None)
    recent_outcomes: List[Dict[str, Any]] = []
    review_samples: List[Dict[str, Any]] = []
    last_retrain_report: Dict[str, Any] = {}
    outcome_summary = {
        "outcome_counts": {},
        "negative_rate_by_route": {},
        "negative_rate_by_router_method": {},
    }
    if router_feedback is not None:
        try:
            router_feedback.finalize_stale_decisions()
            recent_outcomes = [item.to_dict() for item in router_feedback.list_recent_outcomes(limit=200)]
            review_samples = [item.to_dict() for item in router_feedback.list_review_samples(limit=200)]
            last_retrain_report = dict(router_feedback.get_last_retrain_report_metadata() or {})
            outcome_summary = summarize_router_outcomes(recent_outcomes)
        except Exception:
            recent_outcomes = []
            review_samples = []
            last_retrain_report = {}
    recent_mispicks = review_samples[:8]
    review_backlog = {
        "pending": sum(1 for sample in review_samples if str(sample.get("review_status") or "") == "pending"),
        "total_samples": len(review_samples),
        "negative_samples": sum(1 for sample in review_samples if str(sample.get("outcome_label") or "") == "negative"),
        "neutral_samples": sum(1 for sample in review_samples if str(sample.get("outcome_label") or "") == "neutral"),
    }
    updated_at_candidates = [recent_flows[0]["updated_at"]] if recent_flows else []
    if recent_outcomes:
        updated_at_candidates.append(str(recent_outcomes[0].get("scored_at") or ""))
    updated_at = next((value for value in updated_at_candidates if str(value).strip()), utc_now_iso())
    return {
        "route_counts": dict(route_counts),
        "router_method_counts": dict(router_method_counts),
        "start_agent_counts": dict(start_agent_counts),
        "delegation_counts": {key: int(value) for key, value in delegation_counts.items() if value},
        "outcome_counts": dict(outcome_summary.get("outcome_counts") or {}),
        "negative_rate_by_route": dict(outcome_summary.get("negative_rate_by_route") or {}),
        "negative_rate_by_router_method": dict(outcome_summary.get("negative_rate_by_router_method") or {}),
        "recent_mispicks": recent_mispicks,
        "review_backlog": review_backlog,
        "last_retrain_report": last_retrain_report,
        "recent_flows": recent_flows,
        "updated_at": updated_at,
    }


router = APIRouter(prefix="/v1/admin", tags=["control_panel"])


@router.get("/overview")
def get_overview(
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    overlay_store = manager.get_overlay_store()
    collections = _list_collection_payloads(runtime, tenant_id)
    jobs = runtime.bot.kernel.job_manager.list_jobs()[:20]
    skills = runtime.bot.ctx.stores.skill_store.list_skill_packs(
        tenant_id=tenant_id,
        owner_user_id=user_id,
    )
    return {
        "status": "ok",
        "gateway_model_id": runtime.settings.gateway_model_id,
        "providers": {
            "llm_provider": runtime.settings.llm_provider,
            "judge_provider": runtime.settings.judge_provider,
            "embeddings_provider": runtime.settings.embeddings_provider,
        },
        "models": {
            "ollama_chat_model": runtime.settings.ollama_chat_model,
            "ollama_judge_model": runtime.settings.ollama_judge_model,
            "ollama_embed_model": runtime.settings.ollama_embed_model,
            "graphrag_chat_model": runtime.settings.graphrag_chat_model,
            "graphrag_index_chat_model": getattr(runtime.settings, "graphrag_index_chat_model", ""),
            "graphrag_community_report_mode": getattr(runtime.settings, "graphrag_community_report_mode", ""),
            "graphrag_community_report_chat_model": getattr(runtime.settings, "graphrag_community_report_chat_model", ""),
        },
        "runtime_diagnostics": runtime_settings_diagnostics(runtime.settings),
        "counts": {
            "collections": len(collections),
            "agents": len(runtime.bot.kernel.registry.list()),
            "skills": len(skills),
            "tools": len(build_tool_definitions(None)),
            "jobs": len(jobs),
            "mcp_connections": (
                len(
                    _mcp_store_or_503(runtime).list_connections(
                        tenant_id=tenant_id,
                        owner_user_id=user_id,
                        include_disabled=True,
                    )
                )
                if bool(getattr(runtime.settings, "mcp_tool_plane_enabled", False))
                else 0
            ),
        },
        "collections": collections,
        "agents": [
            {
                "name": agent.name,
                "mode": agent.mode,
                "prompt_file": agent.prompt_file,
                "overlay_active": _agent_overlay_active(overlay_store, agent.name),
            }
            for agent in runtime.bot.kernel.registry.list()
        ],
        "jobs": [
            _serialize_job_summary(job, runtime.bot.kernel.job_manager)
            for job in jobs
        ],
        "last_reload": manager.last_reload_summary(),
        "audit_events": overlay_store.read_audit_events(limit=20),
    }


@router.get("/mcp/connections")
def admin_list_mcp_connections(
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    include_disabled: bool = True,
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    if not bool(getattr(runtime.settings, "mcp_tool_plane_enabled", False)):
        return {"enabled": False, "connections": []}
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or ""
    store = _mcp_store_or_503(runtime)
    rows = store.list_connections(
        tenant_id=tenant_id,
        owner_user_id=user_id or None,
        include_disabled=include_disabled,
    )
    tools_by_connection: Dict[str, List[Dict[str, Any]]] = {}
    for record in store.list_tool_catalog(
        tenant_id=tenant_id,
        owner_user_id=user_id or None,
        include_disabled=True,
    ):
        tools_by_connection.setdefault(record.connection_id, []).append(_serialize_mcp_record(record))
    return {
        "enabled": True,
        "connections": [
            {
                **_serialize_mcp_record(record),
                "tools": tools_by_connection.get(record.connection_id, []),
            }
            for record in rows
        ],
    }


@router.post("/mcp/connections")
def admin_create_mcp_connection(
    request: McpAdminConnectionCreateRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    if not bool(getattr(runtime.settings, "mcp_tool_plane_enabled", False)):
        raise HTTPException(status_code=404, detail="MCP tool plane is disabled.")
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    owner_user_id = request.owner_user_id or runtime.settings.default_user_id
    try:
        record = _mcp_service(runtime).create_connection(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            display_name=request.display_name,
            server_url=request.server_url,
            auth_type=request.auth_type,
            secret=request.secret,
            allowed_agents=request.allowed_agents,
            visibility=request.visibility,
            metadata_json=request.metadata_json,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manager.get_overlay_store().append_audit_event(
        action="mcp_connection_create",
        actor=request.actor,
        details={"connection_id": record.connection_id, "owner_user_id": owner_user_id},
    )
    return {"connection": _serialize_mcp_record(record)}


@router.patch("/mcp/connections/{connection_id}")
def admin_update_mcp_connection(
    connection_id: str,
    request: McpAdminConnectionUpdateRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    if not bool(getattr(runtime.settings, "mcp_tool_plane_enabled", False)):
        raise HTTPException(status_code=404, detail="MCP tool plane is disabled.")
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    try:
        record = _mcp_service(runtime).update_connection(
            connection_id,
            tenant_id=tenant_id,
            owner_user_id=x_user_id or None,
            display_name=request.display_name,
            server_url=request.server_url,
            auth_type=request.auth_type,
            secret=request.secret,
            allowed_agents=request.allowed_agents,
            visibility=request.visibility,
            status=request.status,
            metadata_json=request.metadata_json,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="MCP connection not found.")
    manager.get_overlay_store().append_audit_event(
        action="mcp_connection_update",
        actor=request.actor,
        details={"connection_id": connection_id},
    )
    return {"connection": _serialize_mcp_record(record)}


@router.post("/mcp/connections/{connection_id}/disable")
def admin_disable_mcp_connection(
    connection_id: str,
    request: ConfigChangeRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    record = _mcp_store_or_503(runtime).update_connection(
        connection_id,
        tenant_id=tenant_id,
        owner_user_id=None,
        status="disabled",
    )
    if record is None:
        raise HTTPException(status_code=404, detail="MCP connection not found.")
    manager.get_overlay_store().append_audit_event(
        action="mcp_connection_disable",
        actor=request.actor,
        details={"connection_id": connection_id},
    )
    return {"connection": _serialize_mcp_record(record)}


@router.post("/mcp/connections/{connection_id}/test")
def admin_test_mcp_connection(
    connection_id: str,
    request: ConfigChangeRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    connection = _mcp_store_or_503(runtime).get_connection(connection_id, tenant_id=tenant_id, owner_user_id=x_user_id or None)
    if connection is None:
        raise HTTPException(status_code=404, detail="MCP connection not found.")
    try:
        health = _mcp_service(runtime).test_connection(connection)
    except McpClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    manager.get_overlay_store().append_audit_event(
        action="mcp_connection_test",
        actor=request.actor,
        details={"connection_id": connection_id, "health": health},
    )
    return {"health": health}


@router.post("/mcp/connections/{connection_id}/refresh-tools")
def admin_refresh_mcp_tools(
    connection_id: str,
    request: ConfigChangeRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    try:
        tools = _mcp_service(runtime).refresh_tools(
            connection_id,
            tenant_id=tenant_id,
            owner_user_id=x_user_id or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except McpClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    manager.get_overlay_store().append_audit_event(
        action="mcp_tools_refresh",
        actor=request.actor,
        details={"connection_id": connection_id, "tool_count": len(tools)},
    )
    return {"tools": [_serialize_mcp_record(record) for record in tools]}


@router.patch("/mcp/connections/{connection_id}/tools/{tool_id}")
def admin_update_mcp_tool(
    connection_id: str,
    tool_id: str,
    request: McpAdminToolUpdateRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    store = _mcp_store_or_503(runtime)
    existing_tool_ids = {
        item.tool_id
        for item in store.list_tool_catalog(
            tenant_id=tenant_id,
            owner_user_id=x_user_id or None,
            connection_id=connection_id,
            include_disabled=True,
        )
    }
    if tool_id not in existing_tool_ids:
        raise HTTPException(status_code=404, detail="MCP tool not found.")
    record = store.update_tool_catalog(
        tool_id,
        tenant_id=tenant_id,
        owner_user_id=x_user_id or None,
        enabled=request.enabled,
        read_only=request.read_only,
        destructive=request.destructive,
        background_safe=request.background_safe,
        should_defer=request.should_defer,
        search_hint=request.search_hint,
        defer_priority=request.defer_priority,
        status=request.status,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="MCP tool not found.")
    manager.get_overlay_store().append_audit_event(
        action="mcp_tool_update",
        actor=request.actor,
        details={"connection_id": connection_id, "tool_id": tool_id},
    )
    return {"tool": _serialize_mcp_record(record)}


@router.get("/operations")
def get_operations(manager: RuntimeManager = Depends(_admin_manager)) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    overlay_store = manager.get_overlay_store()
    jobs = runtime.bot.kernel.job_manager.list_jobs()[:50]
    scheduler_snapshot = {}
    if hasattr(runtime.bot.kernel.job_manager, "scheduler_snapshot"):
        try:
            scheduler_snapshot = dict(runtime.bot.kernel.job_manager.scheduler_snapshot() or {})
        except Exception:
            scheduler_snapshot = {}
    return {
        "last_reload": manager.last_reload_summary(),
        "scheduler": scheduler_snapshot,
        "jobs": [_serialize_job_summary(job, runtime.bot.kernel.job_manager) for job in jobs],
        "audit_events": overlay_store.read_audit_events(limit=100),
    }


@router.get("/access/principals")
def list_access_principals(
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    principal_type: str = "",
    query: str = "",
    limit: int = 200,
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    principals = store.list_principals(
        tenant_id=tenant_id,
        principal_type=principal_type,
        query=query,
        limit=max(1, min(limit, 500)),
    )
    return {"principals": [_serialize_access_principal(record) for record in principals]}


@router.post("/access/principals")
def upsert_access_principal(
    request: AccessPrincipalUpsertRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_email = normalize_user_email(request.email_normalized)
    principal_record: AuthPrincipalRecord
    if request.provider == "email" and request.principal_type == "user" and normalized_email:
        ensured = store.ensure_email_principal(
            tenant_id=tenant_id,
            email_normalized=normalized_email,
            display_name=request.display_name or normalized_email,
        )
        principal_record = store.upsert_principal(
            AuthPrincipalRecord(
                principal_id=ensured.principal_id,
                tenant_id=tenant_id,
                principal_type=request.principal_type,
                provider=request.provider,
                external_id=request.external_id,
                email_normalized=normalized_email,
                display_name=request.display_name or ensured.display_name,
                metadata_json=dict(request.metadata_json or {}),
                active=bool(request.active),
            )
        )
    else:
        principal_record = store.upsert_principal(
            AuthPrincipalRecord(
                principal_id=str(request.principal_id or ""),
                tenant_id=tenant_id,
                principal_type=request.principal_type,
                provider=request.provider,
                external_id=request.external_id,
                email_normalized=normalized_email,
                display_name=request.display_name,
                metadata_json=dict(request.metadata_json or {}),
                active=bool(request.active),
            )
        )
    manager.get_overlay_store().append_audit_event(
        action="access_principal_upsert",
        actor=request.actor,
        details={
            "tenant_id": tenant_id,
            "principal_id": principal_record.principal_id,
            "principal_type": principal_record.principal_type,
            "provider": principal_record.provider,
            "email_normalized": principal_record.email_normalized,
            "display_name": principal_record.display_name,
            "active": bool(principal_record.active),
        },
    )
    return {"principal": _serialize_access_principal(principal_record)}


@router.get("/access/memberships")
def list_access_memberships(
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    parent_principal_id: str = "",
    child_principal_id: str = "",
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    memberships = store.list_memberships(
        tenant_id=tenant_id,
        parent_principal_id=parent_principal_id,
        child_principal_id=child_principal_id,
    )
    return {"memberships": [_serialize_access_membership(record) for record in memberships]}


@router.post("/access/memberships")
def upsert_access_membership(
    request: AccessMembershipUpsertRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    if request.parent_principal_id == request.child_principal_id:
        raise HTTPException(status_code=400, detail="A principal cannot be a member of itself.")
    parent = store.get_principal(request.parent_principal_id, tenant_id=tenant_id)
    child = store.get_principal(request.child_principal_id, tenant_id=tenant_id)
    if parent is None or child is None:
        raise HTTPException(status_code=404, detail="Parent or child principal was not found.")
    membership = store.upsert_membership(
        AuthPrincipalMembershipRecord(
            membership_id=str(request.membership_id or ""),
            tenant_id=tenant_id,
            parent_principal_id=request.parent_principal_id,
            child_principal_id=request.child_principal_id,
        )
    )
    manager.get_overlay_store().append_audit_event(
        action="access_membership_upsert",
        actor=request.actor,
        details={
            "tenant_id": tenant_id,
            "membership_id": membership.membership_id,
            "parent_principal_id": membership.parent_principal_id,
            "child_principal_id": membership.child_principal_id,
        },
    )
    return {"membership": _serialize_access_membership(membership)}


@router.delete("/access/memberships/{membership_id}")
def delete_access_membership(
    membership_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    deleted = store.delete_membership(membership_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Membership not found.")
    manager.get_overlay_store().append_audit_event(
        action="access_membership_delete",
        actor="control-panel",
        details={"tenant_id": tenant_id, "membership_id": membership_id},
    )
    return {"deleted": True, "membership_id": membership_id}


@router.get("/access/roles")
def list_access_roles(
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    roles = store.list_roles(tenant_id=tenant_id)
    return {"roles": [_serialize_access_role(record) for record in roles]}


@router.post("/access/roles")
def upsert_access_role(
    request: AccessRoleUpsertRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    role = store.upsert_role(
        AuthRoleRecord(
            role_id=str(request.role_id or ""),
            tenant_id=tenant_id,
            name=request.name,
            description=request.description,
        )
    )
    manager.get_overlay_store().append_audit_event(
        action="access_role_upsert",
        actor=request.actor,
        details={"tenant_id": tenant_id, "role_id": role.role_id, "name": role.name},
    )
    return {"role": _serialize_access_role(role)}


@router.delete("/access/roles/{role_id}")
def delete_access_role(
    role_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    deleted = store.delete_role(role_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Role not found.")
    manager.get_overlay_store().append_audit_event(
        action="access_role_delete",
        actor="control-panel",
        details={"tenant_id": tenant_id, "role_id": role_id},
    )
    return {"deleted": True, "role_id": role_id}


@router.get("/access/bindings")
def list_access_bindings(
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    role_id: str = "",
    principal_id: str = "",
    include_disabled: bool = False,
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    bindings = store.list_role_bindings(
        tenant_id=tenant_id,
        role_id=role_id,
        principal_id=principal_id,
        include_disabled=include_disabled,
    )
    return {"bindings": [_serialize_access_binding(record) for record in bindings]}


@router.post("/access/bindings")
def upsert_access_binding(
    request: AccessRoleBindingUpsertRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    if store.get_role(request.role_id, tenant_id=tenant_id) is None:
        raise HTTPException(status_code=404, detail="Role not found.")
    if store.get_principal(request.principal_id, tenant_id=tenant_id) is None:
        raise HTTPException(status_code=404, detail="Principal not found.")
    binding = store.upsert_role_binding(
        AuthRoleBindingRecord(
            binding_id=str(request.binding_id or ""),
            tenant_id=tenant_id,
            role_id=request.role_id,
            principal_id=request.principal_id,
            disabled_at=utc_now_iso() if request.disabled else "",
        )
    )
    manager.get_overlay_store().append_audit_event(
        action="access_binding_upsert",
        actor=request.actor,
        details={
            "tenant_id": tenant_id,
            "binding_id": binding.binding_id,
            "role_id": binding.role_id,
            "principal_id": binding.principal_id,
            "disabled": bool(binding.disabled_at),
        },
    )
    return {"binding": _serialize_access_binding(binding)}


@router.delete("/access/bindings/{binding_id}")
def delete_access_binding(
    binding_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    deleted = store.delete_role_binding(binding_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Binding not found.")
    manager.get_overlay_store().append_audit_event(
        action="access_binding_delete",
        actor="control-panel",
        details={"tenant_id": tenant_id, "binding_id": binding_id},
    )
    return {"deleted": True, "binding_id": binding_id}


@router.get("/access/permissions")
def list_access_permissions(
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    role_id: str = "",
    resource_type: str = "",
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    try:
        permissions = store.list_role_permissions(
            tenant_id=tenant_id,
            role_id=role_id,
            resource_type=resource_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"permissions": [_serialize_access_permission(record) for record in permissions]}


@router.post("/access/permissions")
def upsert_access_permission(
    request: AccessRolePermissionUpsertRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    if store.get_role(request.role_id, tenant_id=tenant_id) is None:
        raise HTTPException(status_code=404, detail="Role not found.")
    try:
        permission = store.upsert_role_permission(
            AuthRolePermissionRecord(
                permission_id=str(request.permission_id or ""),
                tenant_id=tenant_id,
                role_id=request.role_id,
                resource_type=request.resource_type,
                action=request.action,
                resource_selector=request.resource_selector,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manager.get_overlay_store().append_audit_event(
        action="access_permission_upsert",
        actor=request.actor,
        details={
            "tenant_id": tenant_id,
            "permission_id": permission.permission_id,
            "role_id": permission.role_id,
            "resource_type": permission.resource_type,
            "action": permission.action,
            "resource_selector": permission.resource_selector,
        },
    )
    return {"permission": _serialize_access_permission(permission)}


@router.delete("/access/permissions/{permission_id}")
def delete_access_permission(
    permission_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    store = _access_store(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    deleted = store.delete_role_permission(permission_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Permission not found.")
    manager.get_overlay_store().append_audit_event(
        action="access_permission_delete",
        actor="control-panel",
        details={"tenant_id": tenant_id, "permission_id": permission_id},
    )
    return {"deleted": True, "permission_id": permission_id}


@router.get("/access/effective-access")
def get_effective_access(
    email: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    authorization_service = _access_authorization_service(runtime)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_email = normalize_user_email(email)
    return {
        "email": normalized_email,
        "access": authorization_service.preview_effective_access(
            tenant_id=tenant_id,
            email=normalized_email,
        ),
    }


@router.get("/capabilities", response_model=CapabilitiesModel)
def get_capabilities(
    request: Request,
    manager: RuntimeManager = Depends(_admin_manager),
) -> Dict[str, Any]:
    del manager
    return _build_capabilities_payload(request)


@router.get("/architecture", response_model=ArchitectureSnapshotModel)
def get_architecture(manager: RuntimeManager = Depends(_admin_manager)) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    return _build_architecture_snapshot(runtime, manager.get_overlay_store())


@router.get("/architecture/activity", response_model=ArchitectureActivityModel)
def get_architecture_activity(manager: RuntimeManager = Depends(_admin_manager)) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    return _build_architecture_activity(runtime)


@router.get("/config/schema")
def get_config_schema(manager: RuntimeManager = Depends(_admin_manager)) -> Dict[str, Any]:
    settings = manager.get_settings()
    catalog = _catalog_for_settings(settings)
    return {"fields": catalog.schema(settings)}


@router.get("/config/effective")
def get_effective_config(manager: RuntimeManager = Depends(_admin_manager)) -> Dict[str, Any]:
    settings = manager.get_settings()
    catalog = _catalog_for_settings(settings)
    overlay_store = manager.get_overlay_store()
    overlay_values = overlay_store.read_runtime_env()
    return {
        "values": catalog.effective_values(settings, masked=True),
        "overlay_values": {
            field.env_name: ("configured" if overlay_values.get(field.env_name) and field.secret else overlay_values.get(field.env_name, ""))
            for field in catalog.fields
            if field.env_name in overlay_values
        },
    }


@router.post("/config/validate")
def validate_config(
    request: ConfigChangeRequest,
    manager: RuntimeManager = Depends(_admin_manager),
) -> Dict[str, Any]:
    preview = _preview_config_change(manager, request)
    preview.pop("_merged_overlay", None)
    return preview


@router.post("/config/apply")
def apply_config(
    request: ConfigChangeRequest,
    manager: RuntimeManager = Depends(_admin_manager),
) -> Dict[str, Any]:
    validation = _preview_config_change(manager, request)
    if not validation.get("valid", False):
        validation.pop("_merged_overlay", None)
        return validation

    overlay_store = manager.get_overlay_store()
    previous_overlay = overlay_store.read_runtime_env()
    merged_overlay = dict(validation.get("_merged_overlay") or {})
    overlay_store.write_runtime_env(merged_overlay)
    reload_summary = manager.reload_runtime(
        reason="config_apply",
        actor=request.actor,
        changed_keys=sorted((validation.get("preview_diff") or {}).keys()),
    )
    if reload_summary.get("status") != "success":
        overlay_store.write_runtime_env(previous_overlay)
        validation.pop("_merged_overlay", None)
        return {
            **validation,
            "valid": False,
            "errors": {"runtime": reload_summary.get("error") or "Runtime reload failed."},
            "reload": reload_summary,
        }
    overlay_store.append_audit_event(
        action="config_apply",
        actor=request.actor,
        details={"changed_keys": sorted((validation.get("preview_diff") or {}).keys())},
    )
    validation.pop("_merged_overlay", None)
    return {
        **validation,
        "applied": True,
        "reload": reload_summary,
    }


@router.get("/agents")
def list_agents(
    manager: RuntimeManager = Depends(_admin_manager),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    overlay_store = manager.get_overlay_store()
    owner_user_id = x_user_id or runtime.settings.default_user_id
    return {
        "agents": [
            _serialize_agent(
                runtime,
                overlay_store,
                agent_name=agent.name,
                include_body=False,
                owner_user_id=owner_user_id,
            )
            for agent in runtime.bot.kernel.registry.list()
        ],
        "tools": _serialize_tool_catalog(),
    }


@router.get("/agents/{agent_name}")
def get_agent(
    agent_name: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    overlay_store = manager.get_overlay_store()
    return _serialize_agent(
        runtime,
        overlay_store,
        agent_name=agent_name,
        include_body=True,
        owner_user_id=x_user_id or runtime.settings.default_user_id,
    )


@router.put("/agents/{agent_name}")
def update_agent(
    agent_name: str,
    request: AgentUpdateRequest,
    manager: RuntimeManager = Depends(_admin_manager),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    overlay_store = manager.get_overlay_store()
    existing = runtime.bot.kernel.registry.get_loaded_file(agent_name)
    if existing is None:
        raise HTTPException(status_code=404, detail="Agent not found.")

    markdown = _render_agent_overlay_markdown(existing, request)
    overlay_path = overlay_store.agent_overlay_path(agent_name)
    load_agent_markdown_text(markdown, source_path=overlay_path)
    overlay_store.write_agent_overlay(agent_name, markdown)
    overlay_store.append_audit_event(
        action="agent_overlay_write",
        actor=request.actor,
        details={"agent_name": agent_name},
    )
    return {
        "saved": True,
        "pending_reload": True,
        "overlay_path": str(overlay_path),
        "agent": get_agent(agent_name, manager, None),
    }


@router.delete("/agents/{agent_name}")
def reset_agent_overlay(
    agent_name: str,
    manager: RuntimeManager = Depends(_admin_manager),
) -> Dict[str, Any]:
    overlay_store = manager.get_overlay_store()
    removed = overlay_store.delete_agent_overlay(agent_name)
    return {"removed": removed, "pending_reload": removed}


@router.post("/agents/reload")
def reload_agents(
    request: ConfigChangeRequest,
    manager: RuntimeManager = Depends(_admin_manager),
) -> Dict[str, Any]:
    summary = manager.reload_agents(actor=request.actor, changed_keys=list(request.changes.keys()))
    if summary.get("status") == "success":
        manager.get_overlay_store().append_audit_event(
            action="agent_reload",
            actor=request.actor,
            details={"changed_keys": list(request.changes.keys())},
        )
    return summary


@router.get("/prompts")
def list_prompts(manager: RuntimeManager = Depends(_admin_manager)) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    overlay_store = manager.get_overlay_store()
    prompt_files = {
        *(path.name for path in Path(runtime.settings.skills_dir).glob("*.md")),
        *(path.name for path in Path(runtime.settings.prompts_dir).glob("*") if path.is_file()),
        *(agent.prompt_file for agent in runtime.bot.kernel.registry.list() if agent.prompt_file),
        *overlay_store.list_prompt_overlays(),
    }
    return {
        "prompts": [
            {
                "prompt_file": prompt_file,
                "overlay_active": _prompt_overlay_active(overlay_store, prompt_file),
            }
            for prompt_file in sorted(prompt_files)
        ]
    }


@router.get("/prompts/{prompt_file}")
def get_prompt(prompt_file: str, manager: RuntimeManager = Depends(_admin_manager)) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    overlay_store = manager.get_overlay_store()
    base_path = Path(runtime.settings.skills_dir) / prompt_file
    prompt_kind = "agent_prompt"
    if not base_path.exists():
        alt_path = Path(runtime.settings.prompts_dir) / prompt_file
        if alt_path.exists():
            base_path = alt_path
            prompt_kind = "template_prompt"
    base_content = base_path.read_text(encoding="utf-8") if base_path.exists() else ""
    overlay_content = overlay_store.read_prompt_overlay(prompt_file)
    if prompt_kind == "agent_prompt" and (base_content or overlay_content):
        effective_content = runtime.bot.kernel.prompt_builder.load_prompt(prompt_file)
    else:
        effective_content = overlay_content or base_content
    return {
        "prompt_file": prompt_file,
        "kind": prompt_kind,
        "base_content": base_content,
        "overlay_content": overlay_content,
        "effective_content": effective_content,
        "overlay_active": bool(overlay_content),
    }


@router.put("/prompts/{prompt_file}")
def update_prompt(
    prompt_file: str,
    request: PromptUpdateRequest,
    manager: RuntimeManager = Depends(_admin_manager),
) -> Dict[str, Any]:
    overlay_store = manager.get_overlay_store()
    path = overlay_store.write_prompt_overlay(prompt_file, request.content)
    overlay_store.append_audit_event(
        action="prompt_overlay_write",
        actor=request.actor,
        details={"prompt_file": prompt_file},
    )
    return {"saved": True, "overlay_path": str(path), "prompt": get_prompt(prompt_file, manager)}


@router.delete("/prompts/{prompt_file}")
def reset_prompt(prompt_file: str, manager: RuntimeManager = Depends(_admin_manager)) -> Dict[str, Any]:
    overlay_store = manager.get_overlay_store()
    removed = overlay_store.delete_prompt_overlay(prompt_file)
    return {"removed": removed}


@router.get("/graphs")
def list_graphs(
    manager: RuntimeManager = Depends(_admin_manager),
    collection_id: str = "",
    limit: int = 100,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_service(runtime, tenant_id=tenant_id, user_id=user_id)
    return {
        "graphs": service.list_indexes(collection_id=collection_id, limit=max(1, min(limit, 250))),
    }


@router.get("/graphs/{graph_id}")
def get_graph(
    graph_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_service(runtime, tenant_id=tenant_id, user_id=user_id)
    payload = service.inspect_index(graph_id)
    if payload.get("error"):
        raise HTTPException(status_code=404, detail=str(payload["error"]))
    return _enrich_graph_payload(runtime, payload, owner_user_id=user_id)


@router.post("/graphs")
def create_graph(
    request: GraphAdminUpsertRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    collection_id = _normalize_collection_id(str(request.collection_id or "")) if str(request.collection_id or "").strip() else ""
    collection_store = getattr(runtime.bot.ctx.stores, "collection_store", None)
    if collection_store is not None and collection_id:
        collection_store.ensure_collection(tenant_id=tenant_id, collection_id=collection_id)
    service = _graph_service(runtime, tenant_id=tenant_id, user_id=user_id)
    payload = service.create_admin_graph(
        graph_id=str(request.graph_id or ""),
        display_name=str(request.display_name or ""),
        collection_id=collection_id,
        source_doc_ids=list(request.source_doc_ids or []),
        source_paths=list(request.source_paths or []),
        backend=str(request.backend or ""),
        owner_admin_user_id=user_id,
        visibility=request.visibility,
        config_overrides=dict(request.config_overrides or {}),
        prompt_overrides=dict(request.prompt_overrides or {}),
        graph_skill_ids=list(request.graph_skill_ids or []),
    )
    if payload.get("error"):
        raise HTTPException(status_code=400, detail=str(payload["error"]))
    manager.get_overlay_store().append_audit_event(
        action="graph_create",
        actor=request.actor,
        details={"graph_id": payload.get("graph_id"), "collection_id": collection_id},
    )
    return payload


@router.post("/graphs/{graph_id}/validate")
def validate_graph(
    graph_id: str,
    request: GraphLifecycleRequest | None = None,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_service(runtime, tenant_id=tenant_id, user_id=user_id)
    payload = service.validate_admin_graph(graph_id)
    if payload.get("error"):
        raise HTTPException(status_code=404, detail=str(payload["error"]))
    manager.get_overlay_store().append_audit_event(
        action="graph_validate",
        actor=(request.actor if request is not None else "control-panel"),
        details={"graph_id": graph_id, "status": payload.get("status")},
    )
    return payload


@router.post("/graphs/{graph_id}/build")
def build_graph(
    graph_id: str,
    request: GraphLifecycleRequest | None = None,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_service(runtime, tenant_id=tenant_id, user_id=user_id)
    payload = service.build_admin_graph(graph_id, refresh=False)
    if payload.get("error"):
        raise HTTPException(status_code=400, detail=str(payload["error"]))
    manager.get_overlay_store().append_audit_event(
        action="graph_build",
        actor=(request.actor if request is not None else "control-panel"),
        details={"graph_id": graph_id, "status": payload.get("status")},
    )
    return _enrich_graph_payload(runtime, payload, owner_user_id=user_id)


@router.post("/graphs/{graph_id}/refresh")
def refresh_graph(
    graph_id: str,
    request: GraphLifecycleRequest | None = None,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_service(runtime, tenant_id=tenant_id, user_id=user_id)
    payload = service.refresh_admin_graph(graph_id)
    if payload.get("error"):
        raise HTTPException(status_code=400, detail=str(payload["error"]))
    manager.get_overlay_store().append_audit_event(
        action="graph_refresh",
        actor=(request.actor if request is not None else "control-panel"),
        details={"graph_id": graph_id, "status": payload.get("status")},
    )
    return _enrich_graph_payload(runtime, payload, owner_user_id=user_id)


@router.get("/graphs/{graph_id}/runs")
def get_graph_runs(
    graph_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_service(runtime, tenant_id=tenant_id, user_id=user_id)
    return {"runs": service.list_graph_runs(graph_id)}


@router.post("/graphs/{graph_id}/research-tune")
def start_graph_research_tune(
    graph_id: str,
    request: GraphResearchTuneRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_tuning_service(runtime, tenant_id=tenant_id, user_id=user_id)
    payload = service.start_tuning_run(
        graph_id,
        guidance=request.guidance,
        target_prompt_files=list(request.target_prompt_files or []),
        actor=request.actor,
    )
    if payload.get("error"):
        raise HTTPException(status_code=400, detail=str(payload["error"]))
    manager.get_overlay_store().append_audit_event(
        action="graph_research_tune",
        actor=request.actor,
        details={
            "graph_id": graph_id,
            "run_id": payload.get("run_id"),
            "target_prompt_files": list(request.target_prompt_files or []),
            "status": payload.get("status"),
        },
    )
    return payload


@router.get("/graphs/{graph_id}/research-tune/{run_id}")
def get_graph_research_tune(
    graph_id: str,
    run_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_tuning_service(runtime, tenant_id=tenant_id, user_id=user_id)
    payload = service.get_tuning_run(graph_id, run_id)
    if payload.get("error"):
        raise HTTPException(status_code=404, detail=str(payload["error"]))
    return payload


@router.post("/graphs/{graph_id}/research-tune/{run_id}/apply")
def apply_graph_research_tune(
    graph_id: str,
    run_id: str,
    request: GraphResearchTuneApplyRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_tuning_service(runtime, tenant_id=tenant_id, user_id=user_id)
    payload = service.apply_tuning_run(
        graph_id,
        run_id,
        prompt_files=list(request.prompt_files or []),
        actor=request.actor,
    )
    if payload.get("error"):
        raise HTTPException(status_code=400, detail=str(payload["error"]))
    manager.get_overlay_store().append_audit_event(
        action="graph_research_tune_apply",
        actor=request.actor,
        details={
            "graph_id": graph_id,
            "run_id": run_id,
            "prompt_files": payload.get("applied_prompt_files", []),
        },
    )
    return _enrich_graph_payload(runtime, payload, owner_user_id=user_id)


@router.put("/graphs/{graph_id}/prompts")
def update_graph_prompts(
    graph_id: str,
    request: GraphPromptUpdateRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_service(runtime, tenant_id=tenant_id, user_id=user_id)
    payload = service.update_graph_prompts(
        graph_id,
        prompt_overrides=dict(request.prompt_overrides or {}),
        owner_admin_user_id=user_id,
    )
    if payload.get("error"):
        raise HTTPException(status_code=404, detail=str(payload["error"]))
    manager.get_overlay_store().append_audit_event(
        action="graph_prompts_update",
        actor=request.actor,
        details={"graph_id": graph_id, "prompt_files": sorted(request.prompt_overrides.keys())},
    )
    return _enrich_graph_payload(runtime, payload, owner_user_id=user_id)


@router.put("/graphs/{graph_id}/skills")
def update_graph_skills(
    graph_id: str,
    request: GraphSkillUpdateRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    user_id = x_user_id or runtime.settings.default_user_id
    service = _graph_service(runtime, tenant_id=tenant_id, user_id=user_id)
    skill_ids = [str(item) for item in request.skill_ids if str(item).strip()]
    overlay_skill_id = _save_graph_overlay_skill(
        runtime,
        tenant_id=tenant_id,
        owner_user_id=user_id,
        graph_id=graph_id,
        overlay_markdown=request.overlay_markdown,
        overlay_skill_name=request.overlay_skill_name,
    )
    if overlay_skill_id and overlay_skill_id not in skill_ids:
        skill_ids.append(overlay_skill_id)
    payload = service.update_graph_skills(
        graph_id,
        graph_skill_ids=skill_ids,
        owner_admin_user_id=user_id,
    )
    if payload.get("error"):
        raise HTTPException(status_code=404, detail=str(payload["error"]))
    manager.get_overlay_store().append_audit_event(
        action="graph_skills_update",
        actor=request.actor,
        details={"graph_id": graph_id, "skill_ids": skill_ids},
    )
    return _enrich_graph_payload(runtime, payload, owner_user_id=user_id)


@router.get("/collections")
def list_collections(
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    return {"collections": _list_collection_payloads(runtime, tenant_id)}


@router.post("/collections")
def create_collection(
    request: CollectionCreateRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    collection_id = _normalize_collection_id(request.collection_id)
    collection_store = getattr(runtime.bot.ctx.stores, "collection_store", None)
    if collection_store is None:
        raise HTTPException(status_code=503, detail="Collection catalog is not available.")
    existing = collection_store.get_collection(collection_id, tenant_id=tenant_id)
    record = collection_store.ensure_collection(
        tenant_id=tenant_id,
        collection_id=collection_id,
        maintenance_policy=COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,
    )
    manager.get_overlay_store().append_audit_event(
        action="collection_create",
        actor=request.actor,
        details={"collection_id": collection_id, "created": existing is None},
    )
    return {
        "created": existing is None,
        "collection": _build_collection_payload(
            runtime,
            tenant_id,
            collection_id,
            collection_record=record,
            doc_summary=runtime.bot.ctx.stores.doc_store.get_collection_summary(collection_id, tenant_id=tenant_id),
            collection_graphs=[
                graph
                for graph in _collection_graph_indexes(runtime, tenant_id)
                if str(getattr(graph, "collection_id", "") or "") == collection_id
            ],
        ),
    }


@router.get("/collections/{collection_id}")
def get_collection(
    collection_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_collection_id = _normalize_collection_id(collection_id)
    payload = next(
        (
            item
            for item in _list_collection_payloads(runtime, tenant_id)
            if str(item.get("collection_id") or "") == normalized_collection_id
        ),
        None,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Collection not found.")
    return {"collection": payload}


@router.delete("/collections/{collection_id}")
def delete_collection(
    collection_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_collection_id = _normalize_collection_id(collection_id)
    collection_store = getattr(runtime.bot.ctx.stores, "collection_store", None)
    if collection_store is None:
        raise HTTPException(status_code=503, detail="Collection catalog is not available.")
    summary = runtime.bot.ctx.stores.doc_store.get_collection_summary(normalized_collection_id, tenant_id=tenant_id) or {}
    collection_graphs = [
        graph
        for graph in _collection_graph_indexes(runtime, tenant_id)
        if str(getattr(graph, "collection_id", "") or "") == normalized_collection_id
    ]
    record = collection_store.get_collection(normalized_collection_id, tenant_id=tenant_id)
    if record is None and not summary and not collection_graphs:
        raise HTTPException(status_code=404, detail="Collection not found.")
    if int(summary.get("document_count") or 0) > 0:
        raise HTTPException(status_code=409, detail="Collection is not empty and cannot be deleted.")
    if collection_graphs:
        raise HTTPException(status_code=409, detail="Collection is referenced by one or more graphs and cannot be deleted.")
    deleted = collection_store.delete_collection(normalized_collection_id, tenant_id=tenant_id)
    manager.get_overlay_store().append_audit_event(
        action="collection_delete",
        actor="control-panel",
        details={"collection_id": normalized_collection_id, "deleted": deleted},
    )
    return {
        "deleted": deleted,
        "collection_id": normalized_collection_id,
    }


@router.post("/collections/{collection_id}/sync")
def sync_collection(
    collection_id: str,
    request: ConfigChangeRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_collection_id = _normalize_collection_id(collection_id)
    _ensure_collection_catalog_entry(
        runtime,
        tenant_id,
        normalized_collection_id,
        maintenance_policy=COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES,
    )
    candidates: List[CollectionIngestCandidate] = []
    file_results: List[Dict[str, Any]] = []
    for path in iter_kb_source_paths(runtime.settings):
        resolved = path.resolve()
        display_path = resolved.name
        skip_reason = _collection_candidate_skip_reason(resolved)
        if skip_reason:
            file_results.append(
                _collection_file_result(
                    display_path=display_path,
                    source_type="kb",
                    outcome="skipped",
                    error=skip_reason,
                )
            )
            continue
        candidates.append(
            CollectionIngestCandidate(
                absolute_path=resolved,
                source_display_path=display_path,
                source_type="kb",
                collection_id=normalized_collection_id,
                source_identity=f"path:{str(resolved)}",
            )
        )
    result = _ingest_collection_candidates(
        runtime,
        tenant_id=tenant_id,
        collection_id=normalized_collection_id,
        candidates=candidates,
        user_id=getattr(runtime.settings, "default_user_id", "local-cli"),
        conversation_id="control-panel-sync",
    )
    if file_results:
        result = _collection_operation_payload(
            collection_id=normalized_collection_id,
            file_results=[*result.get("files", []), *file_results],
            missing_paths=result.get("missing_paths", []),
            workspace_copies=result.get("workspace_copies", []),
        )
    manager.get_overlay_store().append_audit_event(
        action="collection_sync",
        actor=request.actor,
        details={"collection_id": normalized_collection_id, "count": int(result.get("ingested_count") or 0)},
    )
    result["status"] = result.get("status") or "success"
    result["collection_status"] = _serialize_collection_status(runtime, tenant_id, normalized_collection_id)
    return result


@router.post("/collections/{collection_id}/ingest-paths")
def ingest_collection_paths(
    collection_id: str,
    request: PathIngestRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    normalized_collection_id = _normalize_collection_id(collection_id)
    _ensure_collection_catalog_entry(
        runtime,
        x_tenant_id or runtime.settings.default_tenant_id,
        normalized_collection_id,
        maintenance_policy=COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,
    )
    ctx = _request_context(
        runtime.settings,
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        conversation_id=request.conversation_id or "control-panel-ingest",
    )
    candidates: List[CollectionIngestCandidate] = []
    file_results: List[Dict[str, Any]] = []
    missing_paths: List[str] = []
    seen: set[Path] = set()
    for raw in request.paths:
        path = Path(str(raw or "")).expanduser()
        if not path.exists():
            missing_paths.append(str(path))
            continue
        items_to_scan = [path] if path.is_file() else [item for item in sorted(path.rglob("*")) if item.is_file()]
        for candidate in items_to_scan:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.is_dir():
                display_path = candidate.relative_to(path).as_posix()
            else:
                display_path = resolved.name
            skip_reason = _collection_candidate_skip_reason(resolved)
            if skip_reason:
                file_results.append(
                    _collection_file_result(
                        display_path=display_path,
                        source_type=request.source_type,
                        outcome="skipped",
                        error=skip_reason,
                    )
                )
                continue
            candidates.append(
                CollectionIngestCandidate(
                    absolute_path=resolved,
                    source_display_path=display_path,
                    source_type=request.source_type,
                    collection_id=normalized_collection_id,
                    source_identity=f"path:{str(resolved)}",
                )
            )
    result = _ingest_collection_candidates(
        runtime,
        tenant_id=ctx.tenant_id,
        collection_id=normalized_collection_id,
        candidates=candidates,
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
    )
    if file_results or missing_paths:
        result = _collection_operation_payload(
            collection_id=normalized_collection_id,
            file_results=[*result.get("files", []), *file_results],
            missing_paths=missing_paths,
            workspace_copies=result.get("workspace_copies", []),
        )
    manager.get_overlay_store().append_audit_event(
        action="collection_path_ingest",
        actor=request.actor,
        details={"collection_id": normalized_collection_id, "count": int(result.get("ingested_count") or 0)},
    )
    return result


@router.post("/collections/{collection_id}/upload")
async def upload_collection_files(
    collection_id: str,
    files: List[UploadFile] = File(...),
    relative_paths: List[str] = Form([]),
    source_type: str = Form("upload"),
    conversation_id: str = Form("control-panel-upload"),
    actor: str = Form("control-panel"),
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    if len(files) > MAX_COLLECTION_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files in one upload. Limit is {MAX_COLLECTION_UPLOAD_FILES}.",
        )
    normalized_collection_id = _normalize_collection_id(collection_id)
    ctx = _request_context(
        runtime.settings,
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        conversation_id=conversation_id,
    )
    _ensure_collection_catalog_entry(
        runtime,
        ctx.tenant_id,
        normalized_collection_id,
        maintenance_policy=COLLECTION_MAINTENANCE_INDEXED_DOCUMENTS,
    )
    uploads_dir = Path(runtime.settings.uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    request_dir = uploads_dir / f"{normalized_collection_id}-{uuid.uuid4().hex}"
    request_dir.mkdir(parents=True, exist_ok=True)
    candidates: List[CollectionIngestCandidate] = []
    file_results: List[Dict[str, Any]] = []
    used_relative_paths: set[str] = set()
    for index, file in enumerate(files):
        raw_relative_path = relative_paths[index] if index < len(relative_paths) else (file.filename or "")
        relative_path = _dedupe_relative_upload_path(
            _sanitize_relative_upload_path(raw_relative_path, file.filename or ""),
            used_relative_paths,
        )
        try:
            dest = request_dir / PurePosixPath(relative_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            await _stream_upload_to_path(file, dest)
            resolved_dest = dest.resolve()
            skip_reason = _collection_candidate_skip_reason(resolved_dest)
            if skip_reason:
                file_results.append(
                    _collection_file_result(
                        display_path=relative_path,
                        source_type=source_type,
                        outcome="skipped",
                        error=skip_reason,
                    )
                )
                continue
            candidates.append(
                CollectionIngestCandidate(
                    absolute_path=resolved_dest,
                    source_display_path=relative_path,
                    source_type=source_type,
                    collection_id=normalized_collection_id,
                    source_identity=f"upload:{relative_path}",
                )
            )
        except Exception as exc:
            file_results.append(
                _collection_file_result(
                    display_path=relative_path,
                    source_type=source_type,
                    outcome="failed",
                    error=str(exc),
                )
            )
    result = _ingest_collection_candidates(
        runtime,
        tenant_id=ctx.tenant_id,
        collection_id=normalized_collection_id,
        candidates=candidates,
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
    )
    if file_results:
        result = _collection_operation_payload(
            collection_id=normalized_collection_id,
            file_results=[*result.get("files", []), *file_results],
            missing_paths=result.get("missing_paths", []),
            workspace_copies=result.get("workspace_copies", []),
        )
    manager.get_overlay_store().append_audit_event(
        action="collection_upload",
        actor=actor,
        details={"collection_id": normalized_collection_id, "count": int(result.get("ingested_count") or 0)},
    )
    return result


@router.get("/collections/{collection_id}/health")
def get_collection_health(
    collection_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_collection_id = _normalize_collection_id(collection_id)
    return _serialize_collection_health(runtime, tenant_id, normalized_collection_id)


@router.post("/collections/{collection_id}/repair")
def repair_collection(
    collection_id: str,
    request: ConfigChangeRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_collection_id = _normalize_collection_id(collection_id)
    readiness = _collection_readiness(runtime, tenant_id, normalized_collection_id)
    if readiness.maintenance_policy == COLLECTION_MAINTENANCE_CONFIGURED_KB_SOURCES:
        result = repair_kb_collection(
            runtime.settings,
            runtime.bot.ctx.stores,
            tenant_id=tenant_id,
            collection_id=normalized_collection_id,
        )
    else:
        result = repair_collection_documents(
            runtime.settings,
            runtime.bot.ctx.stores,
            tenant_id=tenant_id,
            collection_id=normalized_collection_id,
            maintenance_policy=readiness.maintenance_policy,
        )
    manager.get_overlay_store().append_audit_event(
        action="collection_repair",
        actor=request.actor,
        details={
            "collection_id": normalized_collection_id,
            "deleted_doc_count": len(result.deleted_doc_ids),
            "reindexed_doc_count": len(result.reindexed_doc_ids),
            "ingested_missing_count": len(result.ingested_missing_doc_ids),
        },
    )
    return result.to_dict()


@router.get("/collections/{collection_id}/documents")
def list_collection_documents(
    collection_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    title_contains: str = "",
    limit: int = 100,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_collection_id = _normalize_collection_id(collection_id)
    records = runtime.bot.ctx.stores.doc_store.search_by_metadata(
        tenant_id=tenant_id,
        collection_id=normalized_collection_id,
        title_contains=title_contains,
        limit=max(1, min(limit, 500)),
    )
    return {
        "documents": [
            {
                "doc_id": record.doc_id,
                "title": record.title,
                "source_type": record.source_type,
                "source_path": record.source_path,
                "source_display_path": record.source_display_path,
                "collection_id": record.collection_id,
                "num_chunks": record.num_chunks,
                "ingested_at": record.ingested_at,
                "file_type": record.file_type,
                "doc_structure_type": record.doc_structure_type,
            }
            for record in records
        ]
    }


@router.get("/collections/{collection_id}/documents/{doc_id}")
def get_collection_document(
    collection_id: str,
    doc_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_collection_id = _normalize_collection_id(collection_id)
    record = runtime.bot.ctx.stores.doc_store.get_document(doc_id, tenant_id=tenant_id)
    if record is None or record.collection_id != normalized_collection_id:
        raise HTTPException(status_code=404, detail="Document not found.")
    chunks = runtime.bot.ctx.stores.chunk_store.list_document_chunks(doc_id, tenant_id=tenant_id)
    extracted = _reconstruct_document_content(chunks)
    return {
        "document": {
            "doc_id": record.doc_id,
            "title": record.title,
            "source_type": record.source_type,
            "source_path": record.source_path,
            "source_display_path": record.source_display_path,
            "collection_id": record.collection_id,
            "num_chunks": record.num_chunks,
            "ingested_at": record.ingested_at,
            "file_type": record.file_type,
            "doc_structure_type": record.doc_structure_type,
        },
        "extracted_content": extracted,
        "raw_source": _read_raw_source(record),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "clause_number": chunk.clause_number,
                "sheet_name": chunk.sheet_name,
                "content": chunk.content,
            }
            for chunk in chunks[:200]
        ],
    }


@router.post("/collections/{collection_id}/documents/{doc_id}/reindex")
def reindex_collection_document(
    collection_id: str,
    doc_id: str,
    request: ConfigChangeRequest,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_collection_id = _normalize_collection_id(collection_id)
    record = runtime.bot.ctx.stores.doc_store.get_document(doc_id, tenant_id=tenant_id)
    if record is None or record.collection_id != normalized_collection_id:
        raise HTTPException(status_code=404, detail="Document not found.")
    source_path = Path(str(record.source_path or ""))
    if not source_path.exists():
        raise HTTPException(status_code=400, detail="Document source path is no longer available for reindex.")
    runtime.bot.ctx.stores.doc_store.delete_document(doc_id, tenant_id=tenant_id)
    doc_ids = ingest_paths(
        runtime.settings,
        runtime.bot.ctx.stores,
        [source_path],
        source_type=record.source_type,
        tenant_id=tenant_id,
        collection_id=normalized_collection_id,
        source_display_paths={str(source_path.resolve()): str(record.source_display_path or source_path.name)},
        source_identities={str(source_path.resolve()): str(record.source_identity or source_path.resolve())},
    )
    manager.get_overlay_store().append_audit_event(
        action="document_reindex",
        actor=request.actor,
        details={"collection_id": normalized_collection_id, "doc_id": doc_id},
    )
    return {
        "deleted_doc_id": doc_id,
        "ingested_doc_ids": doc_ids,
        "collection_id": normalized_collection_id,
    }


@router.delete("/collections/{collection_id}/documents/{doc_id}")
def delete_collection_document(
    collection_id: str,
    doc_id: str,
    manager: RuntimeManager = Depends(_admin_manager),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Dict[str, Any]:
    runtime = _snapshot_or_503(manager)
    tenant_id = x_tenant_id or runtime.settings.default_tenant_id
    normalized_collection_id = _normalize_collection_id(collection_id)
    record = runtime.bot.ctx.stores.doc_store.get_document(doc_id, tenant_id=tenant_id)
    if record is None or record.collection_id != normalized_collection_id:
        raise HTTPException(status_code=404, detail="Document not found.")
    runtime.bot.ctx.stores.doc_store.delete_document(doc_id, tenant_id=tenant_id)
    return {
        "deleted": True,
        "doc_id": doc_id,
        "collection_id": normalized_collection_id,
        "title": record.title,
    }
