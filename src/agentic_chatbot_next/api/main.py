from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
from io import BytesIO
import json
import logging
import mimetypes
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import unquote_to_bytes, urlparse

import httpx
from agentic_chatbot_next.capabilities import (
    CapabilityProfile,
    build_capability_catalog,
    resolve_effective_capabilities,
    save_capability_profile,
)
from agentic_chatbot_next.authz import (
    access_summary_allows,
    access_summary_allowed_ids,
    access_summary_authz_enabled,
    normalize_user_email,
)
from agentic_chatbot_next.api.progress_callback import ProgressCallback
from agentic_chatbot_next.api.connector_security import (
    ConnectorAuthResult,
    require_connector_bearer_auth,
)
from agentic_chatbot_next.api.live_progress import LiveProgressSink
from agentic_chatbot_next.api.gateway_security import (
    build_signed_download_url,
    is_authorized_bearer_token,
    verify_download_token,
)
from agentic_chatbot_next.api.status_tracker import (
    PHASE_GRAPH_CATALOG,
    PHASE_SEARCHING,
    PHASE_SYNTHESIZING,
    PHASE_UPLOADING,
    STATUS_HEARTBEAT_SECONDS,
    TurnStatusTracker,
)
from agentic_chatbot_next.control_panel.routes import router as control_panel_router
from agentic_chatbot_next.control_panel.runtime_manager import get_runtime_manager
from agentic_chatbot_next.control_panel.auth import require_admin_token

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic_chatbot_next.config import Settings, load_settings
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.graph.service import GraphService
from agentic_chatbot_next.mcp.client import McpClientError
from agentic_chatbot_next.mcp.service import McpCatalogService
from agentic_chatbot_next.sandbox import probe_sandbox_image
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_INDEXES,
    classify_inventory_query,
)
from agentic_chatbot_next.persistence.postgres.skills import SkillPackRecord
from agentic_chatbot_next.providers import (
    ProviderConfigurationError,
    ProviderDependencyError,
    build_providers,
)
from agentic_chatbot_next.app.service import RuntimeService
from agentic_chatbot_next.context import RequestContext, build_local_context
from agentic_chatbot_next.rag import ingest_paths, load_stores
from agentic_chatbot_next.rag.ingest import preview_path_index_metadata
from agentic_chatbot_next.rag.retrieval_scope import merge_scope_metadata, resolve_upload_collection_id
from agentic_chatbot_next.runtime.artifacts import latest_assistant_artifacts, normalize_artifact
from agentic_chatbot_next.runtime.context import RuntimePaths, filesystem_key
from agentic_chatbot_next.runtime.long_output import latest_assistant_metadata
from agentic_chatbot_next.runtime.openwebui_helpers import infer_openwebui_helper_task_type
from agentic_chatbot_next.runtime.registry_diagnostics import build_runtime_error_payload
from agentic_chatbot_next.runtime.transcript_store import RuntimeTranscriptStore
from agentic_chatbot_next.sandbox.workspace import SessionWorkspace
from agentic_chatbot_next.session import ChatSession
from agentic_chatbot_next.storage import blob_ref_from_record, build_blob_store
from agentic_chatbot_next.skills.dependency_graph import (
    build_dependency_error_payload,
    build_record_activation_validation,
    build_skill_dependency_graph,
    build_transition_validation,
)
from agentic_chatbot_next.utils.json_utils import make_json_compatible
from agentic_chatbot_next.skills.execution import (
    EXECUTABLE_SKILL_KINDS,
    SkillExecutionConfig,
    build_skill_execution_preview,
)
from agentic_chatbot_next.skills.pack_loader import SkillPackFile, load_skill_pack_from_text
from agentic_chatbot_next.skills.telemetry import compute_skill_health_by_family

logger = logging.getLogger(__name__)
UPLOAD_STREAM_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_FILES = 2_000
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_DEFAULT_UI_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:18000",
    "http://localhost:8000",
)
_SKILL_BUILDER_SYSTEM_PROMPT = """You are the Skill Pack Builder for this agent runtime.

Your job is to turn operator-provided context into a retrievable skill pack draft.

Output contract:
- Return exactly one JSON object and nothing else.
- Do not wrap the JSON in markdown fences.
- Do not include commentary, hidden reasoning, chain-of-thought, or explanations.
- Use this JSON shape:
  {
    "name": "short title",
    "description": "one sentence",
    "when_to_apply": "one sentence",
    "tool_tags": ["tag"],
    "task_tags": ["tag"],
    "workflow": ["imperative step"],
    "examples": ["example user request or situation"],
    "warnings": ["optional warning"]
  }

Skill rules:
- Build a retrievable skill only; never create executable or hybrid skill instructions.
- Preserve the requested agent_scope exactly if you mention it.
- Ground every workflow step and example in the supplied context and examples.
- If context is thin, produce a conservative reusable workflow from the given fields instead of inventing facts.
- Keep workflow steps concise, specific, and operational.
- Do not claim tools, APIs, files, or permissions that were not supplied.
- Do not include secrets, credentials, private tokens, policy bypasses, or unsafe instructions.
- Do not instruct agents to fabricate citations, evidence, or unsupported claims.
- Target this markdown shape indirectly through the JSON fields: frontmatter with name, agent_scope, tool_tags, task_tags, version: 1, enabled: true, description, when_to_apply, kind: retrievable; then # Name, ## Workflow, and optional ## Examples.
"""


async def _stream_upload_to_path(file: UploadFile, dest: Path, *, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
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


def _safe_blob_key_part(value: str) -> str:
    parts = [
        part
        for part in str(value or "").replace("\\", "/").split("/")
        if part and part not in {".", ".."}
    ]
    return "/".join(parts)


def _upload_object_key(*, ctx: RequestContext, collection_id: str, filename: str, index: int) -> str:
    safe_filename = Path(filename or f"upload_{index}").name or f"upload_{index}"
    return "/".join(
        part
        for part in [
            "tenants",
            _safe_blob_key_part(ctx.tenant_id) or "default",
            "collections",
            _safe_blob_key_part(collection_id) or "default",
            "conversations",
            _safe_blob_key_part(ctx.conversation_id) or "default",
            _safe_blob_key_part(ctx.request_id) or uuid.uuid4().hex,
            f"{index:04d}_{uuid.uuid4().hex}",
            safe_filename,
        ]
        if part
    )


class OpenAIMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    user: Optional[str] = None
    userEmail: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkerMailboxRespondRequest(BaseModel):
    response: str = ""
    decision: str = ""
    resume: bool = True


class TeamMailboxChannelRequest(BaseModel):
    name: str = Field(..., min_length=1)
    purpose: str = ""
    member_agents: List[str] = Field(default_factory=list)
    member_job_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TeamMailboxMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)
    message_type: str = "message"
    target_agents: List[str] = Field(default_factory=list)
    target_job_ids: List[str] = Field(default_factory=list)
    subject: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)


class TeamMailboxRespondRequest(BaseModel):
    response: str = ""
    decision: str = ""
    resolve: bool = True


class McpConnectionCreateRequest(BaseModel):
    display_name: str = Field(..., min_length=1)
    server_url: str = Field(..., min_length=1)
    auth_type: str = "none"
    secret: str = ""
    allowed_agents: List[str] = Field(default_factory=lambda: ["general"])
    visibility: str = "private"
    metadata_json: Dict[str, Any] = Field(default_factory=dict)


class McpConnectionUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    server_url: Optional[str] = None
    auth_type: Optional[str] = None
    secret: Optional[str] = None
    allowed_agents: Optional[List[str]] = None
    visibility: Optional[str] = None
    status: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None


class McpToolCatalogUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    read_only: Optional[bool] = None
    destructive: Optional[bool] = None
    background_safe: Optional[bool] = None
    should_defer: Optional[bool] = None
    search_hint: Optional[str] = None
    defer_priority: Optional[int] = None
    status: Optional[str] = None


class SessionCompactRequest(BaseModel):
    preview: bool = False
    reason: str = "manual"


class IngestDocumentsRequest(BaseModel):
    paths: List[str] = Field(default_factory=list)
    source_type: str = "upload"
    collection_id: Optional[str] = None
    source_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    source_display_paths: Dict[str, str] = Field(default_factory=dict)
    source_identities: Dict[str, str] = Field(default_factory=dict)
    metadata_profile: str = "auto"
    metadata_enrichment: str = "deterministic"
    index_preview: bool = False
    conversation_id: Optional[str] = Field(
        default=None,
        description=(
            "When provided (or when X-Conversation-ID header is set), files are also "
            "copied into the active session workspace so the data analyst sandbox can "
            "access them at /workspace/<filename> without a separate load_dataset call."
        ),
    )


def _index_metadata_summary_for_doc_ids(
    stores: object,
    tenant_id: str,
    doc_ids: List[str],
    *,
    metadata_profile: str = "auto",
) -> Dict[str, Any]:
    structure_counts: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}
    parser_counts: Dict[str, int] = {}
    warnings: List[str] = []
    chunk_count = 0
    doc_store = getattr(stores, "doc_store", None)
    for doc_id in doc_ids:
        if doc_store is None:
            continue
        try:
            record = doc_store.get_document(str(doc_id), tenant_id)
        except Exception:
            record = None
        if record is None:
            continue
        source_metadata = dict(getattr(record, "source_metadata", {}) or {})
        index_metadata = source_metadata.get("index_metadata")
        if not isinstance(index_metadata, dict):
            continue
        structure = str(index_metadata.get("doc_structure_type") or getattr(record, "doc_structure_type", "") or "general")
        structure_counts[structure] = int(structure_counts.get(structure, 0)) + 1
        chunk_count += int(getattr(record, "num_chunks", 0) or 0)
        for tag in list(index_metadata.get("tags") or []):
            text = str(tag or "")
            tag_counts[text] = int(tag_counts.get(text, 0)) + 1
        for parser in list(index_metadata.get("parser_chain") or []):
            text = str(parser or "")
            parser_counts[text] = int(parser_counts.get(text, 0)) + 1
        for warning in list(index_metadata.get("warnings") or []):
            text = str(warning or "").strip()
            if text and text not in warnings:
                warnings.append(text)
    return {
        "extractor_version": "document_index_metadata_v1",
        "metadata_profile": str(metadata_profile or "auto"),
        "document_count": sum(structure_counts.values()),
        "chunk_count": chunk_count,
        "structure_type_counts": dict(sorted(structure_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "parser_counts": dict(sorted(parser_counts.items())),
        "warnings": warnings[:20],
    }


def _merge_index_metadata_summaries(
    summaries: Iterable[Dict[str, Any]],
    *,
    metadata_profile: str = "auto",
) -> Dict[str, Any]:
    structure_counts: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}
    parser_counts: Dict[str, int] = {}
    warnings: List[str] = []
    document_count = 0
    chunk_count = 0
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        document_count += int(summary.get("document_count") or 0)
        chunk_count += int(summary.get("chunk_count") or 0)
        for key, target in (
            ("structure_type_counts", structure_counts),
            ("tag_counts", tag_counts),
            ("parser_counts", parser_counts),
        ):
            for item_key, item_value in dict(summary.get(key) or {}).items():
                text = str(item_key or "")
                if text:
                    target[text] = int(target.get(text, 0)) + int(item_value or 0)
        for warning in list(summary.get("warnings") or []):
            text = str(warning or "").strip()
            if text and text not in warnings:
                warnings.append(text)
    return {
        "extractor_version": "document_index_metadata_v1",
        "metadata_profile": str(metadata_profile or "auto"),
        "document_count": document_count,
        "chunk_count": chunk_count,
        "structure_type_counts": dict(sorted(structure_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "parser_counts": dict(sorted(parser_counts.items())),
        "warnings": warnings[:20],
    }


def _source_metadata_for_path(source_metadata: Dict[str, Dict[str, Any]], path: Path) -> Dict[str, Any]:
    candidates = [str(path), str(path.resolve())]
    for key in candidates:
        value = source_metadata.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _preview_index_metadata_for_paths(
    settings: Any,
    paths: Iterable[Path],
    *,
    metadata_profile: str = "auto",
    metadata_enrichment: str = "deterministic",
    providers: object | None = None,
    source_metadata_by_path: Dict[str, Dict[str, Any]] | None = None,
    source_display_paths: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    source_metadata = dict(source_metadata_by_path or {})
    source_display = dict(source_display_paths or {})
    files: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    errors: List[str] = []
    for path in paths:
        path_key = str(path.resolve())
        display_path = str(source_display.get(path_key) or source_display.get(str(path)) or path.name)
        try:
            preview = preview_path_index_metadata(
                settings,
                path,
                metadata_profile=metadata_profile,
                metadata_enrichment=metadata_enrichment,
                providers=providers,
                source_metadata=_source_metadata_for_path(source_metadata, path),
            )
            summary = dict(preview.get("metadata_summary") or {})
            summaries.append(summary)
            files.append(
                {
                    "display_path": display_path,
                    "filename": Path(display_path).name,
                    "source_path": path_key,
                    "outcome": "previewed",
                    "doc_ids": [],
                    "metadata_summary": summary,
                    "index_preview": preview,
                }
            )
        except Exception as exc:
            message = str(exc)
            errors.append(message)
            files.append(
                {
                    "display_path": display_path,
                    "filename": Path(display_path).name,
                    "source_path": path_key,
                    "outcome": "failed",
                    "doc_ids": [],
                    "error": message,
                }
            )
    return {
        "preview": True,
        "status": "preview" if not errors else "partial",
        "ingested_count": 0,
        "doc_ids": [],
        "files": files,
        "errors": errors,
        "metadata_summary": _merge_index_metadata_summaries(summaries, metadata_profile=metadata_profile),
    }


class SkillPackUpsertRequest(BaseModel):
    body_markdown: str = Field(..., min_length=1)
    skill_id: Optional[str] = None
    name: Optional[str] = None
    agent_scope: Optional[str] = None
    graph_id: Optional[str] = None
    collection_id: Optional[str] = None
    tool_tags: List[str] = Field(default_factory=list)
    task_tags: List[str] = Field(default_factory=list)
    version: Optional[str] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None
    retrieval_profile: Optional[str] = None
    controller_hints: Dict[str, Any] = Field(default_factory=dict)
    coverage_goal: Optional[str] = None
    result_mode: Optional[str] = None
    owner_user_id: Optional[str] = None
    visibility: Optional[str] = None
    status: Optional[str] = None
    version_parent: Optional[str] = None
    kind: Optional[str] = None
    execution_config: Dict[str, Any] = Field(default_factory=dict)


class SkillBuildDraftRequest(BaseModel):
    context: str = ""
    examples: str = ""
    name: str = ""
    agent_scope: str = "general"
    target_agent: str = ""
    tool_tags: List[str] = Field(default_factory=list)
    task_tags: List[str] = Field(default_factory=list)
    description: str = ""
    when_to_apply: str = ""


class SkillStatusRequest(BaseModel):
    status: Optional[str] = None
    enabled: Optional[bool] = None


class SkillRollbackRequest(BaseModel):
    target_skill_id: str = Field(..., min_length=1)


class SkillPreviewRequest(BaseModel):
    query: str = Field(..., min_length=1)
    agent_scope: str = ""
    top_k: int = 4
    tool_tags: List[str] = Field(default_factory=list)
    task_tags: List[str] = Field(default_factory=list)


class SkillExecutionPreviewRequest(BaseModel):
    input: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)


class GraphIndexRequest(BaseModel):
    graph_id: Optional[str] = None
    display_name: Optional[str] = None
    collection_id: Optional[str] = None
    source_doc_ids: List[str] = Field(default_factory=list)
    source_paths: List[str] = Field(default_factory=list)
    backend: Optional[str] = None
    refresh: bool = False


class GraphImportRequest(BaseModel):
    graph_id: Optional[str] = None
    display_name: Optional[str] = None
    collection_id: Optional[str] = None
    import_backend: str = "neo4j"
    artifact_path: str = ""
    source_doc_ids: List[str] = Field(default_factory=list)
    source_paths: List[str] = Field(default_factory=list)


class GraphQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    graph_id: Optional[str] = None
    collection_id: Optional[str] = None
    methods: List[str] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=20)
    top_k_graphs: int = Field(default=3, ge=1, le=8)


class CapabilityProfileRequest(BaseModel):
    enabled_tools: List[str] = Field(default_factory=list)
    disabled_tools: List[str] = Field(default_factory=list)
    enabled_tool_groups: List[str] = Field(default_factory=list)
    enabled_skill_pack_ids: List[str] = Field(default_factory=list)
    disabled_skill_pack_ids: List[str] = Field(default_factory=list)
    enabled_mcp_tool_ids: List[str] = Field(default_factory=list)
    enabled_agents: List[str] = Field(default_factory=list)
    enabled_collections: List[str] = Field(default_factory=list)
    enabled_plugins: List[str] = Field(default_factory=list)
    permission_mode: str = "default"
    fast_path_policy: str = "inventory_plus_simple"
    plugin_preferences: Dict[str, bool] = Field(default_factory=dict)


class Runtime:
    def __init__(
        self,
        settings: Settings,
        bot: RuntimeService,
        *,
        diagnostics: Optional[Dict[str, Any]] = None,
    ):
        self.settings = settings
        self.bot = bot
        self.diagnostics = dict(diagnostics or {})


def get_settings() -> Settings:
    return get_runtime_manager().get_settings()


def get_runtime() -> Runtime:
    snapshot = get_runtime_manager().get_snapshot()
    return Runtime(settings=snapshot.settings, bot=snapshot.bot)


def get_runtime_or_503() -> Runtime:
    try:
        return get_runtime()
    except (ProviderDependencyError, ProviderConfigurationError, Exception) as exc:
        payload = build_runtime_error_payload(exc)
        raise HTTPException(status_code=503, detail=payload) from exc


def get_runtime_readiness() -> Runtime | Dict[str, Any]:
    try:
        return get_runtime()
    except Exception as exc:
        return {
            "status": "not_ready",
            "registry_valid": False,
            **build_runtime_error_payload(exc),
        }


def _build_upload_only_runtime(startup_error: BaseException) -> Runtime:
    settings = get_runtime_manager().get_settings()
    providers = build_providers(settings)
    stores = load_stores(settings, providers.embeddings)
    transcript_store = RuntimeTranscriptStore(
        RuntimePaths.from_settings(settings),
        session_hydrate_window_messages=int(getattr(settings, "session_hydrate_window_messages", 40)),
        session_transcript_page_size=int(getattr(settings, "session_transcript_page_size", 100)),
    )
    bot = SimpleNamespace(
        ctx=SimpleNamespace(stores=stores),
        kernel=SimpleNamespace(transcript_store=transcript_store),
    )
    diagnostics = build_runtime_error_payload(startup_error)
    diagnostics["upload_runtime_mode"] = "repository_ingest_only"
    return Runtime(settings=settings, bot=bot, diagnostics=diagnostics)


def get_upload_runtime_or_503() -> Runtime:
    try:
        return get_runtime()
    except (ProviderDependencyError, ProviderConfigurationError, Exception) as exc:
        startup_payload = build_runtime_error_payload(exc)
        try:
            upload_runtime = _build_upload_only_runtime(exc)
        except Exception as upload_exc:
            payload = {
                **startup_payload,
                "upload_runtime_available": False,
                "upload_runtime_error": str(upload_exc),
            }
            raise HTTPException(status_code=503, detail=payload) from upload_exc
        upload_runtime.diagnostics = {
            **dict(getattr(upload_runtime, "diagnostics", {}) or {}),
            "upload_runtime_available": True,
        }
        logger.warning(
            "Using repository-ingest-only upload runtime because chat runtime startup failed: %s",
            startup_payload,
        )
        return upload_runtime


def _first_non_empty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _request_user_email(*values: Any) -> str:
    return normalize_user_email(_first_non_empty(*[str(value or "") for value in values]) or "")


def _authorization_service(runtime: Runtime) -> Any | None:
    stores = getattr(getattr(runtime.bot, "ctx", None), "stores", None)
    return getattr(stores, "authorization_service", None)


def _apply_request_access_snapshot(
    runtime: Runtime,
    session_or_state: Any,
    *,
    user_email: str = "",
    request_metadata: Dict[str, Any] | None = None,
    display_name: str = "",
) -> Dict[str, Any]:
    metadata = merge_scope_metadata(
        runtime.settings,
        {
            **dict(getattr(session_or_state, "metadata", {}) or {}),
            **dict(request_metadata or {}),
        },
    )
    session_or_state.metadata = metadata
    normalized_email = normalize_user_email(
        user_email
        or metadata.get("user_email")
        or getattr(session_or_state, "user_email", "")
    )
    authz_service = _authorization_service(runtime)
    if authz_service is not None:
        snapshot = authz_service.apply_access_snapshot(
            session_or_state,
            tenant_id=str(getattr(session_or_state, "tenant_id", runtime.settings.default_tenant_id) or runtime.settings.default_tenant_id),
            user_id=str(getattr(session_or_state, "user_id", runtime.settings.default_user_id) or runtime.settings.default_user_id),
            user_email=normalized_email,
            session_upload_collection_id=resolve_upload_collection_id(runtime.settings, session_or_state),
            display_name=display_name or str(getattr(session_or_state, "user_id", "") or normalized_email),
        )
        session_or_state.metadata = {
            **dict(getattr(session_or_state, "metadata", {}) or {}),
            "access_summary": snapshot.to_summary(),
            "role_ids": list(snapshot.role_ids),
            "user_email": snapshot.user_email,
            "auth_provider": snapshot.auth_provider,
            "principal_id": snapshot.principal_id,
        }
        return dict(snapshot.to_summary())

    session_or_state.user_email = normalized_email
    session_or_state.metadata = {
        **metadata,
        "user_email": normalized_email,
        "access_summary": dict(getattr(session_or_state, "access_summary", {}) or {}),
    }
    return dict(getattr(session_or_state, "access_summary", {}) or {})


def _require_collection_use_access(
    runtime: Runtime,
    session_or_state: Any,
    *,
    collection_id: str,
) -> None:
    normalized_collection_id = str(collection_id or "").strip()
    if not normalized_collection_id:
        return
    access_summary = dict((getattr(session_or_state, "metadata", {}) or {}).get("access_summary") or {})
    if not bool(access_summary.get("authz_enabled")):
        return
    implicit_upload_collection_id = str(
        access_summary.get("session_upload_collection_id")
        or resolve_upload_collection_id(runtime.settings, session_or_state)
        or ""
    ).strip()
    if access_summary_allows(
        access_summary,
        "collection",
        normalized_collection_id,
        action="use",
        implicit_resource_id=implicit_upload_collection_id,
    ):
        return
    raise HTTPException(
        status_code=403,
        detail=f"User is not allowed to use collection '{normalized_collection_id}'.",
    )


def _require_graph_use_access(
    runtime: Runtime,
    session_or_state: Any,
    *,
    graph_id: str,
) -> None:
    normalized_graph_id = str(graph_id or "").strip()
    if not normalized_graph_id:
        return
    access_summary = dict((getattr(session_or_state, "metadata", {}) or {}).get("access_summary") or {})
    if not bool(access_summary.get("authz_enabled")):
        return
    if not access_summary_allows(
        access_summary,
        "graph",
        normalized_graph_id,
        action="use",
    ):
        raise HTTPException(
            status_code=403,
            detail=f"User is not allowed to use graph '{normalized_graph_id}'.",
        )
    graph_store = getattr(getattr(getattr(runtime.bot, "ctx", None), "stores", None), "graph_index_store", None)
    if graph_store is None or not hasattr(graph_store, "get_index"):
        return
    record = graph_store.get_index(
        normalized_graph_id,
        tenant_id=str(getattr(session_or_state, "tenant_id", runtime.settings.default_tenant_id) or runtime.settings.default_tenant_id),
        user_id="*",
    )
    if record is not None:
        _require_collection_use_access(
            runtime,
            session_or_state,
            collection_id=str(getattr(record, "collection_id", "") or ""),
        )


def _require_gateway_bearer_auth(settings: Settings, authorization: Optional[str]) -> None:
    if is_authorized_bearer_token(authorization, settings.gateway_shared_bearer_token):
        return
    raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")


def _safe_document_source_path(settings: Settings, raw_path: str) -> Path:
    if not str(raw_path or "").strip():
        raise HTTPException(status_code=404, detail="Document source path is not available.")
    try:
        source_path = Path(raw_path).expanduser().resolve(strict=False)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Document source path is invalid.") from exc
    roots: List[Path] = []
    for candidate in [
        getattr(settings, "kb_dir", None),
        *list(getattr(settings, "kb_extra_dirs", ()) or ()),
        getattr(settings, "uploads_dir", None),
        getattr(settings, "data_dir", None),
    ]:
        if candidate is None:
            continue
        try:
            roots.append(Path(candidate).expanduser().resolve(strict=False))
        except Exception:
            continue
    if roots and not any(source_path == root or root in source_path.parents for root in roots):
        raise HTTPException(status_code=403, detail="Document source path is outside configured source roots.")
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=404, detail="Document source file is no longer available.")
    return source_path


def _document_source_media_type(record: Any, source_path: Path) -> str:
    source_metadata = dict(getattr(record, "source_metadata", {}) or {})
    explicit = str(source_metadata.get("mime_type") or source_metadata.get("content_type") or "").strip()
    if explicit:
        return explicit
    guessed, _ = mimetypes.guess_type(str(source_path))
    return guessed or "application/octet-stream"


def _serialize_mailbox_message(message: Any) -> Dict[str, Any]:
    return message.to_dict() if hasattr(message, "to_dict") else dict(message or {})


def _serialize_team_channel(channel: Any) -> Dict[str, Any]:
    return channel.to_dict() if hasattr(channel, "to_dict") else dict(channel or {})


def _serialize_team_message(message: Any) -> Dict[str, Any]:
    return message.to_dict() if hasattr(message, "to_dict") else dict(message or {})


def _require_team_mailbox_enabled(runtime: Runtime) -> None:
    if not bool(getattr(runtime.settings, "team_mailbox_enabled", False)):
        raise HTTPException(status_code=404, detail="Team mailbox is disabled.")


def _require_mcp_tool_plane_enabled(runtime: Runtime) -> None:
    if not bool(getattr(runtime.settings, "mcp_tool_plane_enabled", False)):
        raise HTTPException(status_code=404, detail="MCP tool plane is disabled.")


def _require_mcp_self_service_enabled(runtime: Runtime) -> None:
    if not bool(getattr(runtime.settings, "mcp_user_self_service_enabled", True)):
        raise HTTPException(status_code=403, detail="MCP user self-service is disabled.")


def _get_mcp_store(runtime: Runtime) -> Any:
    store = getattr(getattr(getattr(runtime.bot, "ctx", None), "stores", None), "mcp_connection_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="MCP connection store is not configured.")
    return store


def _get_mcp_service(runtime: Runtime) -> McpCatalogService:
    return McpCatalogService(runtime.settings, _get_mcp_store(runtime))


def _serialize_mcp_connection(record: Any) -> Dict[str, Any]:
    return record.to_dict() if hasattr(record, "to_dict") else dict(record or {})


def _serialize_mcp_tool(record: Any) -> Dict[str, Any]:
    return record.to_dict() if hasattr(record, "to_dict") else dict(record or {})


def _emit_mcp_api_event(runtime: Runtime, event_type: str, ctx: RequestContext, payload: Dict[str, Any]) -> None:
    kernel = getattr(runtime.bot, "kernel", None)
    if kernel is not None and hasattr(kernel, "_emit"):
        kernel._emit(event_type, ctx.session_id, agent_name="api", payload=payload)


def _validate_team_mailbox_job_ids(
    job_manager: Any,
    session_id: str,
    job_ids: Any,
    *,
    channel: Any = None,
) -> List[str]:
    clean_job_ids = [str(item).strip() for item in list(job_ids or []) if str(item).strip()]
    if not clean_job_ids:
        return []
    channel_job_ids = set(getattr(channel, "member_job_ids", []) or []) if channel is not None else set()
    if channel_job_ids:
        outside_channel = sorted(job_id for job_id in clean_job_ids if job_id not in channel_job_ids)
        if outside_channel:
            raise HTTPException(
                status_code=400,
                detail=f"Target job(s) are not members of this team mailbox channel: {', '.join(outside_channel)}",
            )
    get_job = getattr(job_manager, "get_job", None)
    if callable(get_job):
        wrong_session: List[str] = []
        for job_id in clean_job_ids:
            job = get_job(job_id)
            if job is None or str(getattr(job, "session_id", "") or "") != str(session_id or ""):
                wrong_session.append(job_id)
        if wrong_session:
            raise HTTPException(
                status_code=400,
                detail=f"Target job(s) are not in this session: {', '.join(sorted(wrong_session))}",
            )
    return clean_job_ids


def _worker_request_approval_allowed(ctx: RequestContext, *, job_id: str) -> bool:
    access_summary = dict(ctx.access_summary or {})
    if not access_summary_authz_enabled(access_summary):
        return False
    return access_summary_allows(access_summary, "worker_request", job_id, action="approve") or access_summary_allows(
        access_summary,
        "worker_request",
        "*",
        action="approve",
    )


def _get_transcript_store(runtime: Runtime) -> Any:
    return getattr(getattr(runtime.bot, "kernel", None), "transcript_store", None)


def _load_or_create_session_state(runtime: Runtime, ctx: RequestContext) -> SessionState:
    transcript_store = _get_transcript_store(runtime)
    if transcript_store is not None and hasattr(transcript_store, "load_session_state"):
        state = transcript_store.load_session_state(ctx.session_id)
        if isinstance(state, SessionState):
            if ctx.user_email:
                state.user_email = str(ctx.user_email or "").strip()
            if ctx.auth_provider:
                state.auth_provider = str(ctx.auth_provider or "").strip()
            if ctx.principal_id:
                state.principal_id = str(ctx.principal_id or "").strip()
            if ctx.access_summary:
                state.access_summary = dict(ctx.access_summary or {})
                state.metadata = {
                    **dict(state.metadata or {}),
                    "access_summary": dict(ctx.access_summary or {}),
                    "user_email": str(ctx.user_email or state.user_email or ""),
                    "auth_provider": str(ctx.auth_provider or state.auth_provider or ""),
                    "principal_id": str(ctx.principal_id or state.principal_id or ""),
                }
            return state
    return SessionState(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
        user_email=ctx.user_email,
        auth_provider=ctx.auth_provider,
        principal_id=ctx.principal_id,
        access_summary=dict(ctx.access_summary or {}),
        request_id=ctx.request_id,
        session_id=ctx.session_id,
        workspace_root=str(runtime.settings.workspace_dir / filesystem_key(ctx.session_id)),
    )


def _persist_session_state(runtime: Runtime, state: SessionState) -> None:
    transcript_store = _get_transcript_store(runtime)
    if transcript_store is not None and hasattr(transcript_store, "persist_session_state"):
        transcript_store.persist_session_state(state)


def _parse_source_upload_ids(
    source_ids: Optional[List[str]] = None,
    header_value: Optional[str] = None,
) -> List[str]:
    normalized = [str(item).strip() for item in list(source_ids or []) if str(item).strip()]
    raw_header = str(header_value or "").strip()
    if not raw_header:
        return normalized
    try:
        decoded = json.loads(raw_header)
        if isinstance(decoded, list):
            normalized.extend(str(item).strip() for item in decoded if str(item).strip())
            return normalized
    except json.JSONDecodeError:
        pass
    normalized.extend(part.strip() for part in raw_header.split(",") if part.strip())
    return normalized


def _artifact_scope(runtime: Runtime, artifact: Dict[str, Any]) -> tuple[str, str, str]:
    session_id = str(artifact.get("session_id") or "")
    if session_id.count(":") >= 2:
        tenant_id, user_id, conversation_id = session_id.split(":", 2)
        return tenant_id, user_id, conversation_id
    return (
        runtime.settings.default_tenant_id,
        runtime.settings.default_user_id,
        str(artifact.get("conversation_id") or runtime.settings.default_conversation_id),
    )


def _present_download_artifacts(runtime: Runtime, artifacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    presented: List[Dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        artifact = normalize_artifact(item)
        download_id = str(artifact.get("download_id") or "")
        if not download_id:
            presented.append(artifact)
            continue
        tenant_id, user_id, conversation_id = _artifact_scope(runtime, artifact)
        artifact["download_url"] = build_signed_download_url(
            download_id=download_id,
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            secret=runtime.settings.download_url_secret,
            ttl_seconds=runtime.settings.download_url_ttl_seconds,
            path=f"/v1/files/{download_id}",
        )
        presented.append(artifact)
    return presented


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _to_langchain_history(messages: List[OpenAIMessage]) -> tuple[List[Any], str]:
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    last = messages[-1]
    if last.role != "user":
        raise HTTPException(status_code=400, detail="last message must have role='user'")

    user_text = _coerce_content(last.content).strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="last user message content is empty")

    history: List[Any] = []
    for m in messages[:-1]:
        content = _coerce_content(m.content)
        if not content:
            continue
        role = (m.role or "").strip().lower()
        if role == "system":
            history.append(SystemMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
        elif role == "user":
            history.append(HumanMessage(content=content))
        else:
            # Ignore unsupported roles in v1 to preserve compatibility.
            continue

    return history, user_text


def _estimate_tokens(text: str) -> int:
    # Rough heuristic for compatibility payload only.
    return max(1, len(text) // 4) if text else 0


def _requested_agent_override(request: ChatCompletionsRequest, runtime: Runtime) -> str:
    requested_agent = str(request.metadata.get("requested_agent") or "").strip().lower()
    if not requested_agent:
        return ""
    list_overrides = getattr(runtime.bot, "list_requested_agent_overrides", None)
    if callable(list_overrides):
        allowed = list_overrides()
    else:
        registry = getattr(getattr(runtime.bot, "kernel", None), "registry", None)
        if registry is not None and hasattr(registry, "list_routable"):
            allowed = [
                str(agent.name).strip().lower()
                for agent in registry.list_routable()
                if str(getattr(agent, "mode", "") or "").strip().lower() != "basic"
            ]
        else:
            allowed = ["coordinator", "data_analyst", "general", "rag_worker"]
    if requested_agent not in set(allowed):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported requested_agent={requested_agent!r}. "
                f"Allowed values: {', '.join(sorted(allowed))}"
            ),
        )
    return requested_agent


def _runtime_agent_registry(runtime: Runtime) -> Any | None:
    registry = getattr(getattr(runtime.bot, "kernel", None), "registry", None)
    if registry is not None and hasattr(registry, "list"):
        return registry
    return None


def _agent_display_name(agent: Any) -> str:
    metadata = dict(getattr(agent, "metadata", {}) or {})
    display_name = str(
        metadata.get("display_name")
        or metadata.get("label")
        or getattr(agent, "name", "")
        or ""
    ).strip()
    return display_name or str(getattr(agent, "name", "") or "").strip()


def _serialize_agent_descriptor(agent: Any) -> Dict[str, Any]:
    name = str(getattr(agent, "name", "") or "").strip()
    return {
        "id": name,
        "display_name": _agent_display_name(agent) or name,
        "mode": str(getattr(agent, "mode", "") or "").strip(),
        "description": str(getattr(agent, "description", "") or "").strip(),
    }


def _list_available_agents(runtime: Runtime) -> List[Dict[str, Any]]:
    registry = _runtime_agent_registry(runtime)
    if registry is not None:
        return sorted(
            (_serialize_agent_descriptor(agent) for agent in registry.list()),
            key=lambda item: str(item.get("id") or ""),
        )

    list_overrides = getattr(runtime.bot, "list_requested_agent_overrides", None)
    if not callable(list_overrides):
        return []
    seen: set[str] = set()
    results: List[Dict[str, Any]] = []
    for raw_name in list_overrides():
        name = str(raw_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        results.append(
            {
                "id": name,
                "display_name": name,
                "mode": "",
                "description": "",
            }
        )
    return results


def _expand_request_paths(paths: Iterable[str]) -> tuple[List[Path], List[str]]:
    valid_files: List[Path] = []
    missing: List[str] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            missing.append(str(path))
            continue
        candidates = [path]
        if path.is_dir():
            candidates = [item for item in sorted(path.rglob("*")) if item.is_file()]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            valid_files.append(resolved)
    return valid_files, missing


def _build_openai_completion_payload(
    model: str,
    content: str,
    prompt_tokens: int,
    *,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    completion_tokens = _estimate_tokens(content)
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if artifacts:
        payload["artifacts"] = [normalize_artifact(item) for item in artifacts if isinstance(item, dict)]
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def _chunk_text(text: str, size: int = 180) -> Iterable[str]:
    if not text:
        yield ""
        return
    for i in range(0, len(text), size):
        yield text[i:i + size]


def _json_dumps(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(make_json_compatible(payload), ensure_ascii=False, indent=indent)


def _stream_chat_chunks(model: str, text: str) -> Iterable[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    first = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {_json_dumps(first)}\n\n"

    for part in _chunk_text(text):
        body = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        }
        yield f"data: {_json_dumps(body)}\n\n"

    end = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {_json_dumps(end)}\n\n"


def _stream_with_progress(
    model: str,
    session: Any,
    user_text: str,
    bot: Any,
    force_agent: bool,
    requested_agent: str,
    prompt_tokens: int,
    runtime: Runtime | None = None,
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Iterable[str]:
    """SSE generator that emits real-time progress events then text content.

    Runs process_turn() in a background thread. While it runs, reads typed
    progress events from the ProgressCallback queue and yields them as named
    SSE events. After completion yields the text content as chat.completion.chunk
    events.

    Named SSE event format:
        event: progress
        data: {"type": "agent_start", "node": "rag_agent", ...}

    Content chunks use the standard OpenAI format (no event: prefix):
        data: {"choices": [{"delta": {"content": "..."}}]}
    """
    helper_task_type = str((request_metadata or {}).get("openwebui_helper_task_type") or "").strip()
    tracker = None if helper_task_type else TurnStatusTracker(turn_started_at=time.monotonic())
    frontend_settings = getattr(runtime, "settings", None) or getattr(getattr(bot, "ctx", None), "settings", None)
    progress_sink = LiveProgressSink(settings=frontend_settings)
    frontend_policy = progress_sink.policy
    try:
        progress_cb = ProgressCallback(progress_sink)
        progress_source = progress_sink
    except TypeError:
        progress_cb = ProgressCallback()
        progress_source = progress_cb
    result_holder: Dict[str, Any] = {}
    exc_holder: Dict[str, Any] = {}

    def _run() -> None:
        try:
            attempts = [
                {"progress_sink": progress_sink, "request_metadata": request_metadata},
                {"progress_sink": progress_sink},
                {"request_metadata": request_metadata},
                {},
            ]
            last_type_error: TypeError | None = None
            answer = None
            for extra_kwargs in attempts:
                try:
                    answer = bot.process_turn(
                        session,
                        user_text=user_text,
                        upload_paths=[],
                        force_agent=force_agent,
                        requested_agent=requested_agent,
                        extra_callbacks=[progress_cb],
                        **extra_kwargs,
                    )
                    break
                except TypeError as exc:
                    message = str(exc)
                    if "progress_sink" not in message and "request_metadata" not in message:
                        raise
                    last_type_error = exc
                    continue
            if answer is None and last_type_error is not None:
                raise last_type_error
            result_holder["answer"] = answer
        except Exception as exc:
            exc_holder["error"] = exc
        finally:
            if hasattr(progress_source, "mark_done"):
                progress_source.mark_done()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    if tracker is not None:
        for snapshot in tracker.start_snapshots(time.monotonic()):
            if frontend_policy.allows_status_snapshot(snapshot):
                yield _named_sse_event("status", snapshot)

    # Stream progress events while process_turn is running
    while True:
        if tracker is not None:
            delay = tracker.seconds_until_next_heartbeat(
                time.monotonic(),
                interval_seconds=STATUS_HEARTBEAT_SECONDS,
            )
            if delay == 0.0:
                snapshot = tracker.heartbeat_snapshot(time.monotonic())
                if snapshot is not None:
                    if frontend_policy.allows_status_snapshot(snapshot):
                        yield _named_sse_event("status", snapshot)
                continue
            timeout = delay if delay is not None else 1.0
        else:
            timeout = 1.0
        try:
            event = progress_source.events.get(timeout=timeout)
        except queue.Empty:
            if tracker is not None:
                snapshot = tracker.heartbeat_snapshot(time.monotonic())
                if snapshot is not None:
                    if frontend_policy.allows_status_snapshot(snapshot):
                        yield _named_sse_event("status", snapshot)
                    continue
            # BASIC turns may emit no progress events for a while; keep waiting
            # while the worker thread is still computing the final answer.
            if thread.is_alive():
                continue
            break
        if event is None:  # sentinel: processing complete
            break
        if isinstance(event, dict) and any(
            isinstance(event.get(key), dict)
            for key in ("agentic_tool_call", "agentic_agent_activity", "agentic_parallel_group", "agentic_audit_item")
        ):
            if frontend_policy.allows_status_snapshot(event):
                yield _named_sse_event("status", event)
            if tracker is not None and not isinstance(event.get("agentic_tool_call"), dict):
                for snapshot in tracker.progress_snapshots(event, time.monotonic()):
                    if frontend_policy.allows_status_snapshot(snapshot):
                        yield _named_sse_event("status", snapshot)
            continue
        yield f"event: progress\ndata: {_json_dumps(event)}\n\n"
        if tracker is not None:
            for snapshot in tracker.progress_snapshots(event, time.monotonic()):
                if frontend_policy.allows_status_snapshot(snapshot):
                    yield _named_sse_event("status", snapshot)

    thread.join(timeout=1)

    # Check for errors
    if exc_holder.get("error"):
        if tracker is not None:
            failure = tracker.failure_snapshot(time.monotonic())
            if failure is not None:
                if frontend_policy.allows_status_snapshot(failure):
                    yield _named_sse_event("status", failure)
        err_text = f"Error: {str(exc_holder['error'])[:200]}"
        yield from _stream_chat_chunks(model, err_text)
        yield "data: [DONE]\n\n"
        return

    answer = result_holder.get("answer", "")
    if tracker is not None:
        for snapshot in tracker.transition_phase(
            PHASE_SYNTHESIZING,
            now=time.monotonic(),
            source_event_type="content_delta",
            label="Synthesizing answer",
            detail="Preparing final response",
        ):
            if frontend_policy.allows_status_snapshot(snapshot):
                yield _named_sse_event("status", snapshot)
    yield from _stream_chat_chunks(model, answer)
    artifacts = latest_assistant_artifacts(getattr(session, "messages", []) or [])
    if artifacts:
        if runtime is not None:
            artifacts = _present_download_artifacts(runtime, artifacts)
        yield f"event: artifacts\ndata: {_json_dumps(artifacts)}\n\n"
    metadata = latest_assistant_metadata(
        getattr(session, "messages", []) or [],
        keys=[
            "job_id",
            "long_output",
            "turn_outcome",
            "clarification",
            "rag_retrieval_summary",
            "retrieval_mode",
            "tool_calls_used",
        ],
    )
    if metadata:
        yield f"event: metadata\ndata: {_json_dumps(metadata)}\n\n"
    if tracker is not None:
        for snapshot in tracker.completion_snapshots(time.monotonic(), metadata=metadata):
            if frontend_policy.allows_status_snapshot(snapshot):
                yield _named_sse_event("status", snapshot)
    yield "data: [DONE]\n\n"


def get_request_context(
    runtime: Runtime,
    conversation_id: Optional[str],
    request_id: Optional[str],
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    auth_provider: str = "",
    principal_id: str = "",
    access_summary: Optional[Dict[str, Any]] = None,
) -> RequestContext:
    ctx = build_local_context(
        runtime.settings,
        tenant_id=tenant_id or runtime.settings.default_tenant_id,
        user_id=user_id or runtime.settings.default_user_id,
        conversation_id=conversation_id or runtime.settings.default_conversation_id,
        user_email=user_email or "",
        auth_provider=auth_provider,
        principal_id=principal_id,
        access_summary=access_summary,
        request_id=request_id or "",
    )
    authz_service = _authorization_service(runtime)
    if authz_service is None:
        return ctx
    snapshot = authz_service.resolve_access_snapshot(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        user_email=ctx.user_email,
        session_upload_collection_id="",
        display_name=ctx.user_id,
    )
    return build_local_context(
        runtime.settings,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
        user_email=snapshot.user_email,
        auth_provider=snapshot.auth_provider,
        principal_id=snapshot.principal_id,
        access_summary=snapshot.to_summary(),
        request_id=ctx.request_id,
    )


def _require_skill_store(runtime: Runtime) -> Any:
    skill_store = getattr(getattr(runtime.bot, "ctx", None), "stores", None)
    skill_store = getattr(skill_store, "skill_store", None)
    if skill_store is None:
        raise HTTPException(status_code=503, detail="Skill store is not configured.")
    return skill_store


def _skill_accessible_family_ids(ctx: RequestContext) -> List[str] | None:
    access_summary = dict(ctx.access_summary or {})
    if not access_summary_authz_enabled(access_summary):
        return None
    return list(access_summary_allowed_ids(access_summary, "skill_family", action="use"))


def _skill_management_allowed(ctx: RequestContext, *, version_parent: str = "") -> bool:
    access_summary = dict(ctx.access_summary or {})
    if not access_summary_authz_enabled(access_summary):
        return False
    candidate = str(version_parent or "").strip()
    if candidate and access_summary_allows(access_summary, "skill_family", candidate, action="manage"):
        return True
    return access_summary_allows(access_summary, "skill_family", "*", action="manage")


def _executable_skills_enabled(settings: Any) -> bool:
    return bool(getattr(settings, "executable_skills_enabled", False))


def _skill_kind(record_or_request: Any) -> str:
    return str(getattr(record_or_request, "kind", "") or "retrievable").strip().lower() or "retrievable"


def _list_visible_skill_records(
    store: Any,
    *,
    tenant_id: str,
    owner_user_id: str,
    accessible_skill_family_ids: List[str] | None = None,
) -> List[SkillPackRecord]:
    return list(
        store.list_skill_packs(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            accessible_skill_family_ids=accessible_skill_family_ids,
        )
    )


def _skill_health_map(store: Any, *, tenant_id: str) -> Dict[str, Dict[str, Any]]:
    list_events = getattr(store, "list_skill_telemetry_events", None)
    if not callable(list_events):
        return {}
    events = list_events(tenant_id=tenant_id, limit=2000)
    return {
        family_id: summary.to_dict()
        for family_id, summary in compute_skill_health_by_family(events).items()
    }


def _dependency_validation_for_record(
    record: SkillPackRecord,
    *,
    visible_records: List[SkillPackRecord],
) -> Dict[str, Any]:
    current_graph = build_skill_dependency_graph(visible_records)
    active_record = current_graph.active_record_for_identifier(record.skill_id)
    is_effective_active = (
        active_record is not None
        and active_record.skill_id == record.skill_id
        and bool(record.enabled)
        and str(record.status or "").strip().lower() == "active"
    )
    validation = (
        current_graph.dependency_validation_for_skill(record.skill_id)
        if is_effective_active
        else build_record_activation_validation(visible_records, skill_id=record.skill_id)
    )
    return validation.to_dict()


def _serialize_skill_pack(
    record: SkillPackRecord,
    *,
    visible_records: List[SkillPackRecord] | None = None,
    skill_health_by_family: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    payload = {
        "skill_id": record.skill_id,
        "name": record.name,
        "agent_scope": record.agent_scope,
        "graph_id": record.graph_id,
        "collection_id": record.collection_id,
        "tenant_id": record.tenant_id,
        "tool_tags": list(record.tool_tags),
        "task_tags": list(record.task_tags),
        "version": record.version,
        "enabled": bool(record.enabled),
        "source_path": record.source_path,
        "description": record.description,
        "retrieval_profile": record.retrieval_profile,
        "controller_hints": dict(record.controller_hints),
        "coverage_goal": record.coverage_goal,
        "result_mode": record.result_mode,
        "body_markdown": record.body_markdown,
        "owner_user_id": record.owner_user_id,
        "visibility": record.visibility,
        "status": record.status,
        "version_parent": record.version_parent or record.skill_id,
        "updated_at": record.updated_at,
        "kind": _skill_kind(record),
        "execution_config": (
            SkillExecutionConfig.from_raw(dict(getattr(record, "execution_config", {}) or {})).to_dict()
            if _skill_kind(record) in EXECUTABLE_SKILL_KINDS
            else {}
        ),
    }
    family_id = str(record.version_parent or record.skill_id)
    payload["dependency_validation"] = (
        _dependency_validation_for_record(record, visible_records=visible_records)
        if visible_records is not None
        else {}
    )
    payload["skill_health"] = dict((skill_health_by_family or {}).get(family_id) or {})
    return payload


def _simulate_skill_validation(
    record: SkillPackRecord,
    *,
    visible_records: List[SkillPackRecord],
) -> Dict[str, Any]:
    simulated_records = [
        existing
        for existing in visible_records
        if existing.skill_id != record.skill_id
    ] + [record]
    if bool(record.enabled) and str(record.status or "").strip().lower() == "active":
        validation = build_skill_dependency_graph(simulated_records).dependency_validation_for_skill(
            record.skill_id,
            evaluation_mode="activation_preview",
        )
    else:
        validation = build_record_activation_validation(simulated_records, skill_id=record.skill_id)
    return validation.to_dict()


def _raise_skill_dependency_conflict(*, action: str, message: str, validation: Dict[str, Any]) -> None:
    validation_proxy = _ValidationProxy(validation)
    raise HTTPException(
        status_code=409,
        detail=build_dependency_error_payload(
            message=message,
            action=action,
            validation=validation_proxy,
        ),
    )


class _ValidationProxy:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = dict(payload)

    @property
    def skill_id(self) -> str:
        return str(self.payload.get("skill_id") or "")

    @property
    def skill_family_id(self) -> str:
        return str(self.payload.get("skill_family_id") or "")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.payload)


def _default_skill_tool_tags(agent_scope: str, existing: SkillPackRecord | None = None) -> List[str]:
    if existing is not None and existing.tool_tags:
        return list(existing.tool_tags)
    scope = str(agent_scope or "").strip().lower()
    if scope == "data_analyst":
        return ["search_skills", "load_dataset"]
    if scope in {"rag", "rag_worker"}:
        return ["search_skills", "search_indexed_docs"]
    if scope == "graph_manager":
        return ["search_skills", "list_graph_indexes"]
    return ["search_skills"]


def _default_skill_task_tags(agent_scope: str, existing: SkillPackRecord | None = None) -> List[str]:
    if existing is not None and existing.task_tags:
        return list(existing.task_tags)
    scope = str(agent_scope or "").strip().lower()
    if scope == "data_analyst":
        return ["analysis"]
    if scope in {"rag", "rag_worker"}:
        return ["knowledge_retrieval"]
    if scope == "graph_manager":
        return ["graph_inventory"]
    return ["workflow"]


def _default_skill_description(
    *,
    request: SkillPackUpsertRequest,
    existing: SkillPackRecord | None,
    agent_scope: str,
) -> str:
    if request.description is not None and str(request.description).strip():
        return str(request.description).strip()
    if existing is not None and str(existing.description or "").strip():
        return str(existing.description).strip()
    skill_name = str(
        request.name
        or (existing.name if existing is not None else "")
        or request.skill_id
        or "Custom skill"
    ).strip() or "Custom skill"
    scope = str(agent_scope or "general").strip().replace("_", " ") or "general"
    return f"User-defined {scope} skill pack for {skill_name}."


def _skill_builder_clean_scalar(raw: Any, *, default: str = "", max_chars: int = 500) -> str:
    if isinstance(raw, (list, tuple)):
        text = " ".join(str(item or "").strip() for item in raw if str(item or "").strip())
    else:
        text = str(raw or "")
    text = re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()
    if not text:
        text = default
    return text[:max_chars].strip()


def _skill_builder_items(raw: Any) -> List[str]:
    if raw is None:
        values: List[Any] = []
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = re.split(r"[\n;]+", str(raw or ""))

    items: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _skill_builder_clean_scalar(value, max_chars=600)
        text = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", text).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        items.append(text)
    return items


def _skill_builder_tag_list(raw: Any, *, fallback: List[str]) -> List[str]:
    if isinstance(raw, str):
        values: List[Any] = re.split(r"[,\n]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = []
    tags: List[str] = []
    seen: set[str] = set()
    for value in values:
        tag = _skill_builder_clean_scalar(value, max_chars=80).replace(",", " ").strip()
        if not tag or tag.lower() in seen:
            continue
        seen.add(tag.lower())
        tags.append(tag)
    return tags or list(fallback)


def _skill_builder_markdown_list(raw: Any, *, fallback: str = "") -> str:
    items = _skill_builder_items(raw)
    if not items and fallback:
        items = _skill_builder_items(fallback)
    if not items:
        return ""
    return "\n".join(f"- {item}" for item in items)


def _render_skill_builder_markdown(
    *,
    name: str,
    agent_scope: str,
    description: str,
    tool_tags: List[str],
    task_tags: List[str],
    when_to_apply: str,
    workflow: str,
    examples: str,
) -> str:
    lines = [
        "---",
        f"name: {name}",
        f"agent_scope: {agent_scope}",
        f"tool_tags: {', '.join(tool_tags)}",
        f"task_tags: {', '.join(task_tags)}",
        "version: 1",
        "enabled: true",
        f"description: {description}",
        f"when_to_apply: {when_to_apply}",
        "kind: retrievable",
        "---",
        f"# {name}",
        "",
        "## Workflow",
        "",
        workflow,
        "",
    ]
    if examples.strip():
        lines.extend(["## Examples", "", examples.strip(), ""])
    return "\n".join(lines).strip() + "\n"


def _parse_skill_builder_json(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Skill builder LLM returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="Skill builder LLM returned JSON that was not an object.")
    return parsed


def _skill_builder_chat_model(runtime: Runtime) -> Any:
    kernel = getattr(runtime.bot, "kernel", None)
    providers = None
    resolve_base_providers = getattr(kernel, "resolve_base_providers", None)
    if callable(resolve_base_providers):
        providers = resolve_base_providers()
    if providers is None:
        providers = getattr(getattr(runtime.bot, "ctx", None), "providers", None)
    if providers is None:
        providers = getattr(kernel, "providers", None)
    chat = getattr(providers, "chat", None) if providers is not None else None
    if chat is None:
        raise HTTPException(status_code=503, detail="Skill builder requires a configured chat LLM.")
    return chat


def _skill_builder_prompt_payload(request: SkillBuildDraftRequest) -> Dict[str, Any]:
    agent_scope = _skill_builder_clean_scalar(request.agent_scope, default="general", max_chars=120) or "general"
    return {
        "context": str(request.context or "").strip(),
        "examples": str(request.examples or "").strip(),
        "current_name": str(request.name or "").strip(),
        "agent_scope": agent_scope,
        "target_agent": str(request.target_agent or "").strip(),
        "preferred_tool_tags": list(request.tool_tags or _default_skill_tool_tags(agent_scope)),
        "preferred_task_tags": list(request.task_tags or _default_skill_task_tags(agent_scope)),
        "current_description": str(request.description or "").strip(),
        "current_when_to_apply": str(request.when_to_apply or "").strip(),
    }


def _skill_builder_draft_from_output(request: SkillBuildDraftRequest, output: Dict[str, Any]) -> Dict[str, Any]:
    agent_scope = _skill_builder_clean_scalar(request.agent_scope, default="general", max_chars=120) or "general"
    default_name = _skill_builder_clean_scalar(request.name, default="New Skill", max_chars=120) or "New Skill"
    name = _skill_builder_clean_scalar(output.get("name"), default=default_name, max_chars=120) or default_name
    description = _skill_builder_clean_scalar(
        output.get("description"),
        default=_skill_builder_clean_scalar(
            request.description,
            default=f"User-defined {agent_scope.replace('_', ' ')} skill.",
            max_chars=220,
        ),
        max_chars=300,
    )
    when_to_apply = _skill_builder_clean_scalar(
        output.get("when_to_apply"),
        default=_skill_builder_clean_scalar(
            request.when_to_apply,
            default="Use when this reusable workflow fits the user task.",
            max_chars=220,
        ),
        max_chars=300,
    )
    tool_tags = _skill_builder_tag_list(
        output.get("tool_tags"),
        fallback=list(request.tool_tags or _default_skill_tool_tags(agent_scope)),
    )
    task_tags = _skill_builder_tag_list(
        output.get("task_tags"),
        fallback=list(request.task_tags or _default_skill_task_tags(agent_scope)),
    )
    workflow = _skill_builder_markdown_list(
        output.get("workflow"),
        fallback="- Clarify the user task and identify the relevant reusable workflow.\n- Apply the supplied guidance without inventing unsupported evidence.",
    )
    examples = _skill_builder_markdown_list(output.get("examples"), fallback=request.examples)
    warnings = _skill_builder_items(output.get("warnings"))
    body_markdown = _render_skill_builder_markdown(
        name=name,
        agent_scope=agent_scope,
        description=description,
        tool_tags=tool_tags,
        task_tags=task_tags,
        when_to_apply=when_to_apply,
        workflow=workflow,
        examples=examples,
    )
    try:
        parsed = load_skill_pack_from_text(
            body_markdown,
            source_path="api://skills/build-draft.md",
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Skill builder produced an invalid skill draft: {exc}") from exc
    return {
        "body_markdown": body_markdown,
        "name": parsed.name,
        "agent_scope": parsed.agent_scope,
        "description": parsed.description,
        "tool_tags": list(parsed.tool_tags),
        "task_tags": list(parsed.task_tags),
        "when_to_apply": parsed.when_to_apply,
        "workflow": workflow,
        "examples": examples,
        "warnings": warnings + list(parsed.warnings or []),
    }


def _materialize_skill_pack(
    request: SkillPackUpsertRequest,
    *,
    tenant_id: str,
    default_owner_user_id: str,
    source_path: str,
    existing: SkillPackRecord | None = None,
    force_skill_id: str | None = None,
    new_version: bool = False,
) -> SkillPackFile:
    del tenant_id
    default_agent_scope = request.agent_scope or (existing.agent_scope if existing is not None else "rag")
    parsed = load_skill_pack_from_text(
        request.body_markdown,
        source_path=source_path,
        metadata_defaults={
            "skill_id": force_skill_id or request.skill_id or (existing.skill_id if existing is not None else ""),
            "name": request.name or (existing.name if existing is not None else ""),
            "agent_scope": default_agent_scope,
            "graph_id": request.graph_id if request.graph_id is not None else (existing.graph_id if existing is not None else ""),
            "collection_id": request.collection_id if request.collection_id is not None else (existing.collection_id if existing is not None else ""),
            "tool_tags": list(request.tool_tags or _default_skill_tool_tags(default_agent_scope, existing)),
            "task_tags": list(request.task_tags or _default_skill_task_tags(default_agent_scope, existing)),
            "version": request.version or (existing.version if existing is not None else "1"),
            "enabled": True if request.enabled is None and existing is None else (
                request.enabled if request.enabled is not None else bool(existing.enabled)
            ),
            "description": _default_skill_description(
                request=request,
                existing=existing,
                agent_scope=str(default_agent_scope or "rag"),
            ),
            "retrieval_profile": request.retrieval_profile if request.retrieval_profile is not None else (existing.retrieval_profile if existing is not None else ""),
            "controller_hints": dict(request.controller_hints or (existing.controller_hints if existing is not None else {})),
            "coverage_goal": request.coverage_goal if request.coverage_goal is not None else (existing.coverage_goal if existing is not None else ""),
            "result_mode": request.result_mode if request.result_mode is not None else (existing.result_mode if existing is not None else ""),
            "version_parent": request.version_parent or (existing.version_parent if existing is not None else ""),
            "kind": request.kind or (existing.kind if existing is not None else "retrievable"),
            "execution_config": dict(
                request.execution_config
                or (existing.execution_config if existing is not None else {})
            ),
        },
    )
    family = str(
        request.version_parent
        or (existing.version_parent if existing is not None else "")
        or force_skill_id
        or request.skill_id
        or parsed.version_parent
        or parsed.skill_id
    ).strip()
    if not family:
        family = f"skill-{uuid.uuid4().hex[:10]}"
    skill_id = str(force_skill_id or request.skill_id or parsed.skill_id or family).strip() or family
    if new_version and skill_id == family:
        skill_id = f"{family}-v{uuid.uuid4().hex[:8]}"

    parsed.skill_id = skill_id
    fallback_name = parsed.name or (existing.name if existing is not None else "")
    parsed.name = str(request.name or fallback_name).strip() or parsed.name
    parsed.agent_scope = str(request.agent_scope or parsed.agent_scope or (existing.agent_scope if existing is not None else "rag")).strip() or "rag"
    parsed.graph_id = str(
        request.graph_id
        if request.graph_id is not None
        else (parsed.graph_id or (existing.graph_id if existing is not None else ""))
    ).strip()
    parsed.collection_id = str(
        request.collection_id
        if request.collection_id is not None
        else (parsed.collection_id or (existing.collection_id if existing is not None else ""))
    ).strip()
    parsed.tool_tags = list(request.tool_tags or parsed.tool_tags or (existing.tool_tags if existing is not None else []))
    parsed.task_tags = list(request.task_tags or parsed.task_tags or (existing.task_tags if existing is not None else []))
    parsed.version = str(request.version or parsed.version or (existing.version if existing is not None else "1")).strip() or "1"
    parsed.enabled = bool(parsed.enabled if request.enabled is None else request.enabled)
    parsed.source_path = source_path
    parsed.description = str(request.description if request.description is not None else (parsed.description or (existing.description if existing is not None else ""))).strip()
    parsed.retrieval_profile = str(request.retrieval_profile if request.retrieval_profile is not None else (parsed.retrieval_profile or (existing.retrieval_profile if existing is not None else ""))).strip()
    parsed.controller_hints = dict(request.controller_hints or parsed.controller_hints or (existing.controller_hints if existing is not None else {}))
    parsed.coverage_goal = str(request.coverage_goal if request.coverage_goal is not None else (parsed.coverage_goal or (existing.coverage_goal if existing is not None else ""))).strip()
    parsed.result_mode = str(request.result_mode if request.result_mode is not None else (parsed.result_mode or (existing.result_mode if existing is not None else ""))).strip()
    parsed.body_markdown = request.body_markdown
    parsed.kind = str(request.kind or parsed.kind or (existing.kind if existing is not None else "retrievable")).strip().lower() or "retrievable"
    parsed.execution_config = dict(
        request.execution_config
        or parsed.execution_config
        or (existing.execution_config if existing is not None else {})
    )
    parsed.owner_user_id = str(
        request.owner_user_id
        if request.owner_user_id is not None
        else (existing.owner_user_id if existing is not None else default_owner_user_id)
    ).strip()
    parsed.visibility = str(
        request.visibility
        if request.visibility is not None
        else (existing.visibility if existing is not None else "private")
    ).strip() or "private"
    parsed.status = str(
        request.status
        if request.status is not None
        else (existing.status if existing is not None else "draft")
    ).strip() or "draft"
    parsed.version_parent = family
    return parsed


def _skill_pack_to_record(pack: SkillPackFile, *, tenant_id: str) -> SkillPackRecord:
    return SkillPackRecord(
        skill_id=pack.skill_id,
        tenant_id=tenant_id,
        graph_id=pack.graph_id,
        collection_id=pack.collection_id,
        name=pack.name,
        agent_scope=pack.agent_scope,
        checksum=pack.checksum,
        tool_tags=list(pack.tool_tags),
        task_tags=list(pack.task_tags),
        version=pack.version,
        enabled=pack.enabled,
        source_path=pack.source_path,
        description=pack.description,
        retrieval_profile=pack.retrieval_profile,
        controller_hints=dict(pack.controller_hints),
        coverage_goal=pack.coverage_goal,
        result_mode=pack.result_mode,
        body_markdown=pack.body_markdown,
        owner_user_id=pack.owner_user_id,
        visibility=pack.visibility,
        status=pack.status,
        version_parent=pack.version_parent or pack.skill_id,
        kind=pack.kind,
        execution_config=dict(pack.execution_config),
    )


def _merge_cors_origins(*groups: Iterable[str]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            origin = str(item or "").strip()
            if not origin or origin in seen:
                continue
            seen.add(origin)
            merged.append(origin)
    return merged


def _internal_gateway_auth_headers(settings: Settings) -> Dict[str, str]:
    token = str(settings.gateway_shared_bearer_token or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _build_internal_scope_headers(
    settings: Settings,
    *,
    conversation_id: str,
    request_id: str,
    tenant_id: str,
    user_id: str,
    user_email: str = "",
    collection_id: str = "",
) -> Dict[str, str]:
    headers = {
        **_internal_gateway_auth_headers(settings),
        "X-Conversation-ID": conversation_id,
        "X-Request-ID": request_id,
        "X-Tenant-ID": tenant_id,
        "X-User-ID": user_id,
    }
    if str(user_email or "").strip():
        headers["X-User-Email"] = str(user_email or "").strip()
    if collection_id:
        headers["X-Collection-ID"] = collection_id
    return headers


def _extract_connector_payload_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="connector payload must include a non-empty messages array.")
    normalized: List[Dict[str, Any]] = []
    for item in messages:
        if isinstance(item, dict):
            normalized.append(item)
    if not normalized:
        raise HTTPException(status_code=400, detail="connector messages must be JSON objects.")
    return normalized


def _message_filename_from_url(url: str) -> str:
    parsed = urlparse(str(url or ""))
    name = Path(parsed.path).name
    return name or ""


def _message_text_parts(message: Dict[str, Any]) -> List[Dict[str, str]]:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        parts: List[Dict[str, str]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append({"type": "text", "text": item["text"]})
            elif isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append({"type": "text", "text": item["content"]})
            elif isinstance(item, str) and item.strip():
                parts.append({"type": "text", "text": item})
        return parts

    parts = message.get("parts")
    extracted: List[Dict[str, str]] = []
    if isinstance(parts, list):
        for index, item in enumerate(parts):
            if not isinstance(item, dict):
                continue
            part_type = str(item.get("type") or "").strip().lower()
            if part_type == "text" and isinstance(item.get("text"), str) and item.get("text"):
                extracted.append({"type": "text", "text": item["text"]})
                continue
            if part_type == "file":
                label = str(
                    item.get("filename")
                    or item.get("name")
                    or _message_filename_from_url(str(item.get("url") or ""))
                    or f"attachment_{index + 1}"
                ).strip()
                extracted.append({"type": "text", "text": f"[Attached file: {label}]"})
    return extracted


def _connector_messages_to_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue
        parts = _message_text_parts(message)
        if not parts:
            continue
        converted.append({"role": role, "content": parts})
    return converted


def _connector_payload_metadata(
    payload: Dict[str, Any],
    *,
    auth_result: ConnectorAuthResult,
    upload_collection_id: str,
    kb_collection_id: str,
    user_email: str = "",
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in ("body", "metadata", "requestMetadata"):
        value = payload.get(key)
        if isinstance(value, dict):
            metadata.update(value)
    if "forceAgent" in payload:
        metadata["force_agent"] = bool(payload.get("forceAgent"))
    if str(payload.get("requestedAgent") or "").strip():
        metadata["requested_agent"] = str(payload["requestedAgent"]).strip()
    metadata.setdefault("collection_id", upload_collection_id)
    metadata.setdefault("upload_collection_id", upload_collection_id)
    metadata.setdefault("kb_collection_id", kb_collection_id)
    metadata.setdefault("connector_client", True)
    metadata.setdefault("connector_token_type", auth_result.token_type)
    if normalize_user_email(user_email):
        metadata["user_email"] = normalize_user_email(user_email)
    return metadata


def _connector_last_message_id(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        message_id = str(message.get("id") or "").strip()
        if message_id:
            return message_id
    return ""


def _decode_data_url(url: str) -> tuple[bytes, str]:
    if not str(url or "").startswith("data:"):
        raise ValueError("Expected a data URL.")
    header, separator, raw_data = str(url).partition(",")
    if not separator:
        raise ValueError("Malformed data URL.")
    meta = header[5:]
    media_type = "application/octet-stream"
    is_base64 = False
    if meta:
        meta_parts = [part for part in meta.split(";") if part]
        if meta_parts:
            if "/" in meta_parts[0]:
                media_type = meta_parts[0]
                meta_parts = meta_parts[1:]
            is_base64 = "base64" in meta_parts
    if is_base64:
        try:
            return base64.b64decode(raw_data), media_type
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Malformed base64 data URL.") from exc
    return unquote_to_bytes(raw_data), media_type


async def _connector_upload_from_file_part(
    part: Dict[str, Any],
    *,
    source_id: str,
) -> Dict[str, Any]:
    url = str(part.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="File parts must include a url.")
    if url.startswith("blob:"):
        raise HTTPException(
            status_code=400,
            detail=(
                "blob: file parts cannot be resolved server-side. "
                "Use the helper package or send data URLs/multipart uploads."
            ),
        )
    filename = str(
        part.get("filename")
        or part.get("name")
        or _message_filename_from_url(url)
        or f"attachment_{uuid.uuid4().hex[:8]}"
    ).strip()
    media_type = str(part.get("mediaType") or part.get("contentType") or "").strip()
    if url.startswith("data:"):
        content, inferred_media_type = _decode_data_url(url)
        return {
            "filename": filename,
            "media_type": media_type or inferred_media_type,
            "content": content,
            "source_id": source_id,
        }
    if url.startswith("http://") or url.startswith("https://"):
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            remote_media_type = response.headers.get("content-type") or "application/octet-stream"
            return {
                "filename": filename,
                "media_type": media_type or remote_media_type,
                "content": response.content,
                "source_id": source_id,
            }
    raise HTTPException(
        status_code=400,
        detail="Unsupported file part url. Expected data:, http:, https:, or multipart upload.",
    )


async def _connector_message_uploads(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uploads: List[Dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        message_id = str(message.get("id") or f"user-{message_index}").strip()
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            if str(part.get("type") or "").strip().lower() != "file":
                continue
            source_id = str(
                part.get("sourceId")
                or part.get("id")
                or f"aisdk:{message_id}:file:{part_index}"
            ).strip()
            uploads.append(await _connector_upload_from_file_part(part, source_id=source_id))
    return uploads


def _connector_stream_sse(payload: Dict[str, Any]) -> str:
    return f"data: {_json_dumps(payload)}\n\n"


def _connector_message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and str(item.get("type") or "").strip().lower() == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    return ""


def _initial_connector_status(query_text: str) -> tuple[str, str]:
    if classify_inventory_query(query_text) == INVENTORY_QUERY_GRAPH_INDEXES:
        return PHASE_GRAPH_CATALOG, "Inspecting graph catalog"
    return PHASE_SEARCHING, "Searching knowledge base"


def _named_sse_event(event_name: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {_json_dumps(payload)}\n\n"


def _connector_status_part(status_id: str, payload: Dict[str, Any]) -> str:
    return _connector_stream_sse(
        {
            "type": "data-status",
            "id": status_id,
            "data": dict(payload or {}),
            "transient": True,
        }
    )


async def _aiter_sse_events(response: httpx.Response):
    event_name = ""
    data_lines: List[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = ""
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.partition(":")[2].lstrip())
    if data_lines:
        yield event_name, "\n".join(data_lines)


def _text_deltas_from_chunk(payload: Dict[str, Any]) -> tuple[List[str], bool]:
    deltas: List[str] = []
    saw_finish = False
    for choice in list(payload.get("choices") or []):
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            content = str(delta["content"])
            if content:
                deltas.append(content)
        if choice.get("finish_reason") is not None:
            saw_finish = True
    return deltas, saw_finish


def _artifact_parts_from_gateway_payload(payload: Any) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for item in list(payload or []):
        if not isinstance(item, dict):
            continue
        artifact = normalize_artifact(item)
        download_id = str(artifact.get("download_id") or artifact.get("artifact_ref") or uuid.uuid4().hex)
        filename = str(artifact.get("filename") or artifact.get("label") or download_id).strip() or download_id
        media_type = str(
            artifact.get("content_type")
            or artifact.get("mime_type")
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )
        download_url = str(artifact.get("download_url") or "").strip()
        parts.append(
            {
                "type": "data-artifact",
                "id": download_id,
                "data": {
                    "label": str(artifact.get("label") or filename).strip() or filename,
                    "download_url": download_url,
                    "mime_type": media_type,
                    "filename": filename,
                },
            }
        )
        if download_url:
            parts.append({"type": "file", "url": download_url, "mediaType": media_type})
    return parts


app = FastAPI(title="Agentic Gateway", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware

_static_settings = load_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_merge_cors_origins(
        _DEFAULT_UI_CORS_ORIGINS,
        getattr(_static_settings, "connector_allowed_origins", ()),
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(control_panel_router)

if bool(getattr(_static_settings, "control_panel_enabled", False)):
    control_panel_static_dir = Path(getattr(_static_settings, "control_panel_static_dir"))
    if control_panel_static_dir.exists():
        app.mount(
            "/control-panel",
            StaticFiles(directory=control_panel_static_dir, html=True),
            name="control-panel",
        )


@app.get("/health/live")
def health_live() -> Dict[str, str]:
    return {"status": "ok"}


def _runtime_capability_status(runtime: Runtime, *, include_diagnostics: bool = False) -> Dict[str, Any]:
    settings = runtime.settings
    kernel = getattr(runtime.bot, "kernel", None)
    stores = getattr(getattr(runtime.bot, "ctx", None), "stores", None)
    graph_store = getattr(stores, "graph_index_store", None)
    graph_count = 0
    graph_query_ready_count = 0
    if graph_store is not None and hasattr(graph_store, "list_indexes"):
        try:
            graphs = list(
                graph_store.list_indexes(
                    tenant_id=str(getattr(settings, "default_tenant_id", "local-dev") or "local-dev"),
                    limit=100,
                )
            )
            graph_count = len(graphs)
            graph_query_ready_count = sum(1 for item in graphs if bool(getattr(item, "query_ready", False)))
        except Exception:
            graph_count = 0
            graph_query_ready_count = 0

    sandbox_status: Dict[str, Any] = {
        "configured": bool(str(getattr(settings, "sandbox_docker_image", "") or "").strip()),
        "image": str(getattr(settings, "sandbox_docker_image", "") or ""),
        "ready": None,
        "probe_skipped": not include_diagnostics,
    }
    if include_diagnostics and sandbox_status["configured"]:
        probe = probe_sandbox_image(
            str(getattr(settings, "sandbox_docker_image", "") or ""),
            timeout_seconds=3.0,
        )
        sandbox_status.update(
            {
                "ready": bool(probe.ok),
                "detail": str(probe.detail or ""),
                "remediation": str(probe.remediation or ""),
                "probe_skipped": False,
            }
        )

    memory_configured = bool(getattr(settings, "memory_enabled", True))
    memory_store = getattr(kernel, "memory_store", None)
    file_memory_store = getattr(kernel, "file_memory_store", None)
    return {
        "memory": {
            "configured": memory_configured,
            "ready": bool(memory_configured and (memory_store is not None or file_memory_store is not None)),
            "memory_store_ready": memory_store is not None,
            "file_memory_store_ready": file_memory_store is not None,
        },
        "analyst_sandbox": sandbox_status,
        "graph": {
            "catalog_available": graph_store is not None,
            "index_count": graph_count,
            "query_ready_count": graph_query_ready_count,
            "query_ready": graph_query_ready_count > 0,
        },
        "uploads": {
            "document_store_available": getattr(stores, "doc_store", None) is not None,
            "workspace_root": str(getattr(settings, "workspace_dir", "") or ""),
        },
    }


@app.get("/health/ready")
def health_ready(
    runtime_or_error: Runtime | Dict[str, Any] = Depends(get_runtime_readiness),
    include_diagnostics: bool = False,
) -> JSONResponse:
    if isinstance(runtime_or_error, dict):
        return JSONResponse(status_code=503, content=runtime_or_error)

    runtime = runtime_or_error
    capability_status = _runtime_capability_status(runtime, include_diagnostics=include_diagnostics)
    kb_status = runtime.bot.get_kb_status(
        runtime.settings.default_tenant_id,
        refresh=True,
        attempt_sync=False,
    )
    if kb_status is not None and not kb_status.ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": kb_status.reason,
                "model": runtime.settings.gateway_model_id,
                "collection_id": kb_status.collection_id,
                "missing_sources": list(kb_status.missing_source_paths),
                "sync_attempted": kb_status.sync_attempted,
                "sync_error": kb_status.sync_error,
                "suggested_fix": kb_status.suggested_fix,
                "capability_status": capability_status,
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "model": runtime.settings.gateway_model_id,
            "capability_status": capability_status,
        },
    )


@app.get("/v1/admin/runtime/diagnostics")
def runtime_diagnostics(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")) -> JSONResponse:
    settings = get_settings()
    require_admin_token(settings, x_admin_token=x_admin_token)
    try:
        runtime = get_runtime()
        validate = getattr(getattr(runtime.bot, "kernel", None), "validate_registry", None)
        if callable(validate):
            validate()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "model": runtime.settings.gateway_model_id,
                "registry_valid": True,
                "upload_runtime_available": True,
            },
        )
    except Exception as exc:
        payload = build_runtime_error_payload(exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "registry_valid": False,
                **payload,
            },
        )


@app.get("/v1/models")
def list_models(
    settings: Settings = Depends(get_settings),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(settings, authorization)
    model_id = settings.gateway_model_id
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "agentic-chatbot",
            }
        ],
    }


@app.get("/v1/agents")
def list_agents(
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    return {
        "object": "list",
        "data": _list_available_agents(runtime),
    }


@app.get("/v1/skills")
def list_skills(
    runtime: Runtime = Depends(get_runtime_or_503),
    agent_scope: str = "",
    status: str = "",
    visibility: str = "",
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    store = _require_skill_store(runtime)
    accessible_skill_family_ids = _skill_accessible_family_ids(ctx)
    visible_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    skill_health_by_family = _skill_health_map(store, tenant_id=ctx.tenant_id)
    skills = store.list_skill_packs(
        tenant_id=ctx.tenant_id,
        agent_scope=agent_scope,
        owner_user_id=ctx.user_id,
        status=status,
        visibility=visibility,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    return {
        "object": "list",
        "data": [
            _serialize_skill_pack(
                item,
                visible_records=visible_records,
                skill_health_by_family=skill_health_by_family,
            )
            for item in skills
        ],
    }


@app.post("/v1/skills/build-draft")
def build_skill_draft(
    request: SkillBuildDraftRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> Dict[str, Any]:
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    if x_admin_token is None:
        if not _skill_management_allowed(ctx):
            raise HTTPException(status_code=403, detail="Skill mutations require an admin token or manage permission.")
    else:
        require_admin_token(runtime.settings, x_admin_token=x_admin_token)

    chat = _skill_builder_chat_model(runtime)
    prompt_payload = _skill_builder_prompt_payload(request)
    user_prompt = (
        "Build a retrievable skill draft from this JSON input. "
        "Return only the JSON object required by the system instructions.\n\n"
        f"{json.dumps(prompt_payload, indent=2, sort_keys=True)}"
    )
    try:
        response = chat.invoke(
            [
                SystemMessage(content=_SKILL_BUILDER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Skill builder LLM call failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Skill builder LLM call failed: {exc}") from exc

    text = _coerce_content(getattr(response, "content", response)).strip()
    if not text:
        raise HTTPException(status_code=502, detail="Skill builder LLM returned an empty response.")
    parsed = _parse_skill_builder_json(text)
    return {
        "object": "skill.build_draft",
        "draft": _skill_builder_draft_from_output(request, parsed),
    }


@app.get("/v1/skills/{skill_id}")
def get_skill(
    skill_id: str,
    runtime: Runtime = Depends(get_runtime_or_503),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    store = _require_skill_store(runtime)
    accessible_skill_family_ids = _skill_accessible_family_ids(ctx)
    record = store.get_skill_pack(
        skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Skill not found.")
    chunks = store.get_skill_chunks(skill_id, tenant_id=ctx.tenant_id)
    visible_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    payload = _serialize_skill_pack(
        record,
        visible_records=visible_records,
        skill_health_by_family=_skill_health_map(store, tenant_id=ctx.tenant_id),
    )
    payload["chunks"] = chunks
    payload["chunk_count"] = len(chunks)
    return payload


@app.post("/v1/skills")
def create_skill(
    request: SkillPackUpsertRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> Dict[str, Any]:
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    if x_admin_token is None:
        if not _skill_management_allowed(ctx):
            raise HTTPException(status_code=403, detail="Skill mutations require an admin token or manage permission.")
    else:
        require_admin_token(runtime.settings, x_admin_token=x_admin_token)
    store = _require_skill_store(runtime)
    accessible_skill_family_ids = None if x_admin_token is not None else _skill_accessible_family_ids(ctx)
    visible_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    provisional_id = str(request.skill_id or f"skill-{uuid.uuid4().hex[:10]}")
    pack = _materialize_skill_pack(
        request,
        tenant_id=ctx.tenant_id,
        default_owner_user_id=ctx.user_id,
        source_path=f"api://skills/{provisional_id}.md",
        force_skill_id=provisional_id,
    )
    record = _skill_pack_to_record(pack, tenant_id=ctx.tenant_id)
    if _skill_kind(record) in EXECUTABLE_SKILL_KINDS and not _executable_skills_enabled(runtime.settings):
        raise HTTPException(status_code=403, detail="Executable skills are disabled.")
    dependency_validation = _simulate_skill_validation(record, visible_records=visible_records)
    if bool(record.enabled) and str(record.status or "").strip().lower() == "active" and not bool(
        dependency_validation.get("is_valid", False)
    ):
        _raise_skill_dependency_conflict(
            action="create",
            message="Cannot create an active skill while dependency requirements are invalid.",
            validation=dependency_validation,
        )
    store.upsert_skill_pack(record, pack.chunks)
    created = store.get_skill_pack(
        pack.skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    refreshed_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    return {
        "object": "skill",
        "created": True,
        "chunk_count": len(pack.chunks),
        "data": _serialize_skill_pack(
            created or record,
            visible_records=refreshed_records,
            skill_health_by_family=_skill_health_map(store, tenant_id=ctx.tenant_id),
        ),
    }


@app.put("/v1/skills/{skill_id}")
def update_skill(
    skill_id: str,
    request: SkillPackUpsertRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> Dict[str, Any]:
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    if x_admin_token is None:
        if not _skill_management_allowed(ctx, version_parent=skill_id):
            raise HTTPException(status_code=403, detail="Skill mutations require an admin token or manage permission.")
    else:
        require_admin_token(runtime.settings, x_admin_token=x_admin_token)
    store = _require_skill_store(runtime)
    accessible_skill_family_ids = None if x_admin_token is not None else _skill_accessible_family_ids(ctx)
    existing = store.get_skill_pack(
        skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    if existing is None:
        raise HTTPException(status_code=404, detail="Skill not found.")
    visible_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    pack = _materialize_skill_pack(
        request,
        tenant_id=ctx.tenant_id,
        default_owner_user_id=ctx.user_id,
        source_path=f"api://skills/{skill_id}.md",
        existing=existing,
        new_version=True,
    )
    record = _skill_pack_to_record(pack, tenant_id=ctx.tenant_id)
    if _skill_kind(record) in EXECUTABLE_SKILL_KINDS and not _executable_skills_enabled(runtime.settings):
        raise HTTPException(status_code=403, detail="Executable skills are disabled.")
    dependency_validation = _simulate_skill_validation(record, visible_records=visible_records)
    if bool(record.enabled) and str(record.status or "").strip().lower() == "active" and not bool(
        dependency_validation.get("is_valid", False)
    ):
        _raise_skill_dependency_conflict(
            action="update",
            message="Cannot update an active skill version while dependency requirements are invalid.",
            validation=dependency_validation,
        )
    store.upsert_skill_pack(record, pack.chunks)
    created = store.get_skill_pack(
        pack.skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    refreshed_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    return {
        "object": "skill",
        "updated": True,
        "previous_skill_id": skill_id,
        "chunk_count": len(pack.chunks),
        "data": _serialize_skill_pack(
            created or record,
            visible_records=refreshed_records,
            skill_health_by_family=_skill_health_map(store, tenant_id=ctx.tenant_id),
        ),
    }


@app.post("/v1/skills/{skill_id}/activate")
def activate_skill(
    skill_id: str,
    request: SkillStatusRequest | None = None,
    runtime: Runtime = Depends(get_runtime_or_503),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> Dict[str, Any]:
    del request
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    if x_admin_token is None:
        if not _skill_management_allowed(ctx, version_parent=skill_id):
            raise HTTPException(status_code=403, detail="Skill mutations require an admin token or manage permission.")
    else:
        require_admin_token(runtime.settings, x_admin_token=x_admin_token)
    store = _require_skill_store(runtime)
    accessible_skill_family_ids = None if x_admin_token is not None else _skill_accessible_family_ids(ctx)
    record = store.get_skill_pack(
        skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Skill not found.")
    visible_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    validation = build_transition_validation(
        visible_records,
        overrides={skill_id: {"status": "active", "enabled": True}},
        primary_skill_id=skill_id,
        action="activate",
    ).to_dict()
    if not bool(validation.get("is_valid", False)):
        _raise_skill_dependency_conflict(
            action="activate",
            message="Cannot activate this skill because the active dependency graph would become invalid.",
            validation=validation,
        )
    store.set_skill_status(skill_id, tenant_id=ctx.tenant_id, status="active", enabled=True)
    updated = store.get_skill_pack(
        skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    refreshed_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    return {
        "object": "skill",
        "action": "activate",
        "data": _serialize_skill_pack(
            updated,
            visible_records=refreshed_records,
            skill_health_by_family=_skill_health_map(store, tenant_id=ctx.tenant_id),
        ),
    }


@app.post("/v1/skills/{skill_id}/deactivate")
def deactivate_skill(
    skill_id: str,
    request: SkillStatusRequest | None = None,
    runtime: Runtime = Depends(get_runtime_or_503),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> Dict[str, Any]:
    del request
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    if x_admin_token is None:
        if not _skill_management_allowed(ctx, version_parent=skill_id):
            raise HTTPException(status_code=403, detail="Skill mutations require an admin token or manage permission.")
    else:
        require_admin_token(runtime.settings, x_admin_token=x_admin_token)
    store = _require_skill_store(runtime)
    accessible_skill_family_ids = None if x_admin_token is not None else _skill_accessible_family_ids(ctx)
    record = store.get_skill_pack(
        skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Skill not found.")
    visible_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    validation = build_transition_validation(
        visible_records,
        overrides={skill_id: {"status": "archived", "enabled": False}},
        primary_skill_id=skill_id,
        action="deactivate",
    ).to_dict()
    if not bool(validation.get("is_valid", True)):
        _raise_skill_dependency_conflict(
            action="deactivate",
            message="Cannot deactivate this skill because active dependents would break.",
            validation=validation,
        )
    store.set_skill_status(skill_id, tenant_id=ctx.tenant_id, status="archived", enabled=False)
    updated = store.get_skill_pack(
        skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    refreshed_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    return {
        "object": "skill",
        "action": "deactivate",
        "data": _serialize_skill_pack(
            updated,
            visible_records=refreshed_records,
            skill_health_by_family=_skill_health_map(store, tenant_id=ctx.tenant_id),
        ),
    }


@app.post("/v1/skills/{skill_id}/rollback")
def rollback_skill(
    skill_id: str,
    request: SkillRollbackRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> Dict[str, Any]:
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    if x_admin_token is None:
        if not _skill_management_allowed(ctx, version_parent=skill_id):
            raise HTTPException(status_code=403, detail="Skill mutations require an admin token or manage permission.")
    else:
        require_admin_token(runtime.settings, x_admin_token=x_admin_token)
    store = _require_skill_store(runtime)
    accessible_skill_family_ids = None if x_admin_token is not None else _skill_accessible_family_ids(ctx)
    current = store.get_skill_pack(
        skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    target = store.get_skill_pack(
        request.target_skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    if current is None or target is None:
        raise HTTPException(status_code=404, detail="Skill version not found.")
    if (current.version_parent or current.skill_id) != (target.version_parent or target.skill_id):
        raise HTTPException(status_code=400, detail="Rollback target must belong to the same skill family.")
    visible_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    validation = build_transition_validation(
        visible_records,
        overrides={
            skill_id: {"status": "archived", "enabled": False},
            target.skill_id: {"status": "active", "enabled": True},
        },
        primary_skill_id=target.skill_id,
        action="rollback",
    ).to_dict()
    if not bool(validation.get("is_valid", False)):
        _raise_skill_dependency_conflict(
            action="rollback",
            message="Cannot roll back to this skill version because the active dependency graph would become invalid.",
            validation=validation,
        )
    store.set_skill_status(skill_id, tenant_id=ctx.tenant_id, status="archived", enabled=False)
    store.set_skill_status(target.skill_id, tenant_id=ctx.tenant_id, status="active", enabled=True)
    updated = store.get_skill_pack(
        target.skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    refreshed_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    return {
        "object": "skill",
        "action": "rollback",
        "rolled_back_from": skill_id,
        "data": _serialize_skill_pack(
            updated,
            visible_records=refreshed_records,
            skill_health_by_family=_skill_health_map(store, tenant_id=ctx.tenant_id),
        ),
    }


@app.post("/v1/skills/{skill_id}/preview-execution")
def preview_skill_execution(
    skill_id: str,
    request: SkillExecutionPreviewRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
) -> Dict[str, Any]:
    if not _executable_skills_enabled(runtime.settings):
        raise HTTPException(status_code=403, detail="Executable skills are disabled.")
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    if x_admin_token is not None:
        require_admin_token(runtime.settings, x_admin_token=x_admin_token)
    store = _require_skill_store(runtime)
    accessible_skill_family_ids = None if x_admin_token is not None else _skill_accessible_family_ids(ctx)
    record = store.get_skill_pack(
        skill_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Skill not found.")
    if x_admin_token is None and not _skill_management_allowed(
        ctx,
        version_parent=str(record.version_parent or record.skill_id),
    ):
        raise HTTPException(status_code=403, detail="Skill execution preview requires manage permission.")
    if _skill_kind(record) not in EXECUTABLE_SKILL_KINDS:
        raise HTTPException(status_code=400, detail="Skill is not executable.")
    preview = build_skill_execution_preview(
        record,
        input_text=request.input,
        arguments=dict(request.arguments or {}),
    )
    return {
        "object": "skill.execution_preview",
        "data": preview.to_dict(),
    }


@app.post("/v1/skills/preview")
def preview_skill_search(
    request: SkillPreviewRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    ctx = get_request_context(
        runtime,
        conversation_id=None,
        request_id="",
        tenant_id=x_tenant_id,
        user_id=x_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    store = _require_skill_store(runtime)
    accessible_skill_family_ids = _skill_accessible_family_ids(ctx)
    visible_records = _list_visible_skill_records(
        store,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    record_by_id = {item.skill_id: item for item in visible_records}
    skill_health_by_family = _skill_health_map(store, tenant_id=ctx.tenant_id)
    matches = store.vector_search(
        request.query,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        top_k=max(1, min(int(request.top_k), 8)),
        agent_scope=request.agent_scope,
        tool_tags=list(request.tool_tags),
        task_tags=list(request.task_tags),
        enabled_only=True,
        accessible_skill_family_ids=accessible_skill_family_ids,
    )
    return {
        "object": "skill.preview",
        "query": request.query,
        "matches": [
            {
                "skill_id": match.skill_id,
                "name": match.name,
                "agent_scope": match.agent_scope,
                "chunk_index": match.chunk_index,
                "score": match.score,
                "content": match.content,
                "retrieval_profile": match.retrieval_profile,
                "controller_hints": dict(match.controller_hints),
                "coverage_goal": match.coverage_goal,
                "result_mode": match.result_mode,
                "visibility": match.visibility,
                "status": match.status,
                "version_parent": match.version_parent,
                "kind": getattr(match, "kind", "retrievable"),
                "execution_config": (
                    SkillExecutionConfig.from_raw(dict(getattr(match, "execution_config", {}) or {})).to_dict()
                    if str(getattr(match, "kind", "retrievable") or "retrievable").strip().lower() in EXECUTABLE_SKILL_KINDS
                    else {}
                ),
                "dependency_validation": (
                    _dependency_validation_for_record(record_by_id[match.skill_id], visible_records=visible_records)
                    if match.skill_id in record_by_id
                    else {}
                ),
                "skill_health": dict(
                    skill_health_by_family.get(str(match.version_parent or match.skill_id)) or {}
                ),
            }
            for match in matches
        ],
    }


@app.get("/v1/mcp/connections")
def list_mcp_connections(
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_mcp_tool_plane_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    rows = _get_mcp_store(runtime).list_connections(
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        include_disabled=True,
    )
    return {"object": "mcp.connections", "data": [_serialize_mcp_connection(item) for item in rows]}


@app.post("/v1/mcp/connections")
def create_mcp_connection(
    request: McpConnectionCreateRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_mcp_tool_plane_enabled(runtime)
    _require_mcp_self_service_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    try:
        record = _get_mcp_service(runtime).create_connection(
            tenant_id=ctx.tenant_id,
            owner_user_id=ctx.user_id,
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
    _emit_mcp_api_event(runtime, "mcp_connection_created", ctx, {"connection_id": record.connection_id})
    return {"object": "mcp.connection", "data": _serialize_mcp_connection(record)}


@app.patch("/v1/mcp/connections/{connection_id}")
def update_mcp_connection(
    connection_id: str,
    request: McpConnectionUpdateRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_mcp_tool_plane_enabled(runtime)
    _require_mcp_self_service_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    try:
        record = _get_mcp_service(runtime).update_connection(
            connection_id,
            tenant_id=ctx.tenant_id,
            owner_user_id=ctx.user_id,
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
    _emit_mcp_api_event(runtime, "mcp_connection_updated", ctx, {"connection_id": connection_id})
    return {"object": "mcp.connection", "data": _serialize_mcp_connection(record)}


@app.delete("/v1/mcp/connections/{connection_id}")
def delete_mcp_connection(
    connection_id: str,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_mcp_tool_plane_enabled(runtime)
    _require_mcp_self_service_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    deleted = _get_mcp_store(runtime).delete_connection(connection_id, tenant_id=ctx.tenant_id, owner_user_id=ctx.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="MCP connection not found.")
    _emit_mcp_api_event(runtime, "mcp_connection_deleted", ctx, {"connection_id": connection_id})
    return {"object": "mcp.connection.deleted", "connection_id": connection_id, "deleted": True}


@app.post("/v1/mcp/connections/{connection_id}/test")
def test_mcp_connection(
    connection_id: str,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_mcp_tool_plane_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    store = _get_mcp_store(runtime)
    connection = store.get_connection(connection_id, tenant_id=ctx.tenant_id, owner_user_id=ctx.user_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="MCP connection not found.")
    try:
        health = _get_mcp_service(runtime).test_connection(connection)
    except McpClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    _emit_mcp_api_event(runtime, "mcp_connection_tested", ctx, {"connection_id": connection_id, "health": health})
    return {"object": "mcp.connection_test", "connection_id": connection_id, "health": health}


@app.post("/v1/mcp/connections/{connection_id}/refresh-tools")
def refresh_mcp_tools(
    connection_id: str,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_mcp_tool_plane_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    try:
        tools = _get_mcp_service(runtime).refresh_tools(
            connection_id,
            tenant_id=ctx.tenant_id,
            owner_user_id=ctx.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except McpClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    _emit_mcp_api_event(runtime, "mcp_tools_refreshed", ctx, {"connection_id": connection_id, "tool_count": len(tools)})
    return {"object": "mcp.tools", "connection_id": connection_id, "data": [_serialize_mcp_tool(item) for item in tools]}


@app.get("/v1/mcp/connections/{connection_id}/tools")
def list_mcp_tools(
    connection_id: str,
    include_disabled: bool = True,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_mcp_tool_plane_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    connection = _get_mcp_store(runtime).get_connection(connection_id, tenant_id=ctx.tenant_id, owner_user_id=ctx.user_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="MCP connection not found.")
    rows = _get_mcp_store(runtime).list_tool_catalog(
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
        connection_id=connection_id,
        include_disabled=include_disabled,
    )
    return {"object": "mcp.tools", "connection_id": connection_id, "data": [_serialize_mcp_tool(item) for item in rows]}


@app.patch("/v1/mcp/connections/{connection_id}/tools/{tool_id}")
def update_mcp_tool(
    connection_id: str,
    tool_id: str,
    request: McpToolCatalogUpdateRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_mcp_tool_plane_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    connection = _get_mcp_store(runtime).get_connection(connection_id, tenant_id=ctx.tenant_id, owner_user_id=ctx.user_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="MCP connection not found.")
    existing_tool_ids = {
        item.tool_id
        for item in _get_mcp_store(runtime).list_tool_catalog(
            tenant_id=ctx.tenant_id,
            owner_user_id=ctx.user_id,
            connection_id=connection_id,
            include_disabled=True,
        )
    }
    if tool_id not in existing_tool_ids:
        raise HTTPException(status_code=404, detail="MCP tool not found.")
    record = _get_mcp_store(runtime).update_tool_catalog(
        tool_id,
        tenant_id=ctx.tenant_id,
        owner_user_id=ctx.user_id,
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
    _emit_mcp_api_event(runtime, "mcp_tool_catalog_updated", ctx, {"connection_id": connection_id, "tool_id": tool_id})
    return {"object": "mcp.tool", "data": _serialize_mcp_tool(record)}


@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionsRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_collection_id: Optional[str] = Header(None, alias="X-Collection-ID"),
):
    _require_gateway_bearer_auth(runtime.settings, authorization)
    if request.model != runtime.settings.gateway_model_id:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

    request_user_email = _request_user_email(
        x_user_email,
        x_openwebui_user_email,
        request.userEmail,
        request.metadata.get("user_email"),
    )
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=request_user_email,
    )
    logger.info(
        "chat_completions request tenant=%s user=%s conversation=%s request_id=%s model=%s stream=%s",
        ctx.tenant_id,
        ctx.user_id,
        ctx.conversation_id,
        ctx.request_id or "-",
        request.model,
        request.stream,
    )

    history, user_text = _to_langchain_history(request.messages)
    session = ChatSession.from_context(ctx, messages=history)
    openwebui_client = any(
        bool(str(value or "").strip())
        for value in (x_openwebui_chat_id, x_openwebui_message_id, x_openwebui_user_id)
    )
    request_metadata = dict(request.metadata or {})
    if request_user_email:
        request_metadata["user_email"] = request_user_email
    if request.max_tokens is not None:
        request_metadata["chat_max_output_tokens"] = int(request.max_tokens)
    if openwebui_client:
        request_metadata.setdefault("openwebui_client", True)
        helper_task_type = infer_openwebui_helper_task_type(user_text)
        if helper_task_type:
            request_metadata.setdefault("openwebui_helper_task_type", helper_task_type)
    scope_metadata = merge_scope_metadata(
        runtime.settings,
        {
            **request_metadata,
            "collection_id": (
                request_metadata.get("collection_id")
                or x_collection_id
                or getattr(runtime.settings, "default_collection_id", "default")
            ),
        },
    )
    session.metadata.update(scope_metadata)
    _apply_request_access_snapshot(
        runtime,
        session,
        user_email=request_user_email,
        request_metadata=request_metadata,
        display_name=ctx.user_id,
    )

    force_agent = bool(request.metadata.get("force_agent", False))
    requested_agent = _requested_agent_override(request, runtime)

    prompt_text = "\n".join(_coerce_content(m.content) for m in request.messages)
    prompt_tokens = _estimate_tokens(prompt_text)

    if request.stream:
        return StreamingResponse(
            _stream_with_progress(
                request.model,
                session,
                user_text,
                runtime.bot,
                force_agent=force_agent,
                requested_agent=requested_agent,
                prompt_tokens=prompt_tokens,
                runtime=runtime,
                request_metadata=request_metadata,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        answer = runtime.bot.process_turn(
            session,
            user_text=user_text,
            upload_paths=[],
            force_agent=force_agent,
            requested_agent=requested_agent,
            request_metadata=request_metadata,
        )
    except TypeError as exc:
        if "request_metadata" not in str(exc):
            raise
        answer = runtime.bot.process_turn(
            session,
            user_text=user_text,
            upload_paths=[],
            force_agent=force_agent,
            requested_agent=requested_agent,
        )
    artifacts = latest_assistant_artifacts(getattr(session, "messages", []) or [])
    if artifacts:
        artifacts = _present_download_artifacts(runtime, artifacts)
    metadata = latest_assistant_metadata(
        getattr(session, "messages", []) or [],
        keys=[
            "job_id",
            "long_output",
            "turn_outcome",
            "clarification",
            "rag_retrieval_summary",
            "retrieval_mode",
            "tool_calls_used",
        ],
    )
    payload = _build_openai_completion_payload(
        request.model,
        answer,
        prompt_tokens,
        artifacts=artifacts,
        metadata=metadata,
    )
    return JSONResponse(payload)


@app.post("/v1/sessions/{session_id}/compact")
def compact_session_context(
    session_id: str,
    request: SessionCompactRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    if session_id != ctx.session_id:
        if x_admin_token is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        require_admin_token(runtime.settings, x_admin_token=x_admin_token)
    try:
        payload = runtime.bot.kernel.compact_session_context(
            session_id,
            preview=bool(request.preview),
            reason=str(request.reason or "manual"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return payload


@app.post("/v1/connector/chat")
async def connector_chat(
    request: Request,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_collection_id: Optional[str] = Header(None, alias="X-Collection-ID"),
):
    auth_result = require_connector_bearer_auth(
        runtime.settings,
        authorization,
        origin=request.headers.get("origin"),
        client_host=getattr(request.client, "host", ""),
    )
    content_type = str(request.headers.get("content-type") or "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        payload_text = str(form.get("payload") or "").strip()
        if not payload_text:
            raise HTTPException(status_code=400, detail="multipart connector requests must include a payload field.")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="connector payload must be valid JSON.") from exc
        multipart_files = [value for key, value in form.multi_items() if key == "files" and hasattr(value, "filename")]
    else:
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover - FastAPI may raise different JSON decode errors
            raise HTTPException(status_code=400, detail="connector body must be valid JSON.") from exc
        multipart_files = []

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="connector payload must be a JSON object.")

    messages = _extract_connector_payload_messages(payload)
    assistant_message_id = f"msg_{uuid.uuid4().hex}"
    text_block_id = f"text_{uuid.uuid4().hex}"
    conversation_id = _first_non_empty(
        x_conversation_id,
        str(payload.get("conversationId") or ""),
        str(payload.get("id") or ""),
    ) or runtime.settings.default_conversation_id
    request_id = _first_non_empty(
        x_request_id,
        str(payload.get("requestId") or ""),
        _connector_last_message_id(messages),
        uuid.uuid4().hex,
    ) or uuid.uuid4().hex
    tenant_id = _first_non_empty(x_tenant_id, str(payload.get("tenantId") or "")) or runtime.settings.default_tenant_id
    user_id = _first_non_empty(
        x_user_id,
        str(payload.get("userId") or ""),
        str(payload.get("user") or ""),
    ) or runtime.settings.default_user_id
    user_email = _request_user_email(
        x_user_email,
        str(payload.get("userEmail") or ""),
        str((payload.get("metadata") or {}).get("user_email") or "") if isinstance(payload.get("metadata"), dict) else "",
        str((payload.get("requestMetadata") or {}).get("user_email") or "") if isinstance(payload.get("requestMetadata"), dict) else "",
    )

    metadata_sources: List[Dict[str, Any]] = []
    for key in ("body", "metadata", "requestMetadata"):
        value = payload.get(key)
        if isinstance(value, dict):
            metadata_sources.append(value)
    merged_metadata: Dict[str, Any] = {}
    for source in metadata_sources:
        merged_metadata.update(source)

    upload_collection_id = str(
        payload.get("uploadCollectionId")
        or merged_metadata.get("upload_collection_id")
        or merged_metadata.get("collection_id")
        or x_collection_id
        or runtime.settings.default_collection_id
    ).strip() or runtime.settings.default_collection_id
    kb_collection_id = str(
        payload.get("kbCollectionId")
        or merged_metadata.get("kb_collection_id")
        or runtime.settings.default_collection_id
    ).strip() or runtime.settings.default_collection_id
    model_id = str(payload.get("model") or runtime.settings.gateway_model_id).strip() or runtime.settings.gateway_model_id
    if model_id != runtime.settings.gateway_model_id:
        raise HTTPException(status_code=400, detail=f"Unsupported connector model: {model_id}")

    upstream_messages = _connector_messages_to_openai_messages(messages)
    if not upstream_messages or str(upstream_messages[-1].get("role") or "").strip().lower() != "user":
        raise HTTPException(
            status_code=400,
            detail="connector messages must end with a user message containing text or file attachments.",
        )
    last_user_query = _connector_message_text(upstream_messages[-1])

    raw_source_ids = payload.get("sourceIds")
    resolved_raw_source_ids = [str(item).strip() for item in list(raw_source_ids or []) if str(item).strip()]
    multipart_uploads: List[Dict[str, Any]] = []
    for index, upload in enumerate(multipart_files):
        filename = str(getattr(upload, "filename", "") or f"upload_{index + 1}")
        media_type = str(getattr(upload, "content_type", "") or "application/octet-stream")
        content = await upload.read()
        source_id = (
            resolved_raw_source_ids[index]
            if index < len(resolved_raw_source_ids)
            else f"connector:{conversation_id}:multipart:{index}:{filename}"
        )
        multipart_uploads.append(
            {
                "filename": filename,
                "media_type": media_type,
                "content": content,
                "source_id": source_id,
            }
        )
    message_uploads = await _connector_message_uploads(messages)
    upload_items = multipart_uploads + message_uploads
    request_metadata = _connector_payload_metadata(
        payload,
        auth_result=auth_result,
        upload_collection_id=upload_collection_id,
        kb_collection_id=kb_collection_id,
        user_email=user_email,
    )
    helper_task_type = str(request_metadata.get("openwebui_helper_task_type") or "").strip()

    async def event_stream():
        tracker = None if helper_task_type else TurnStatusTracker(turn_started_at=time.monotonic())
        status_id = f"status:{assistant_message_id}"

        yield _connector_stream_sse({"type": "start", "messageId": assistant_message_id})
        yield _connector_stream_sse(
            {
                "type": "data-metadata",
                "data": {
                    "createdAt": int(time.time() * 1000),
                    "model": model_id,
                    "conversationId": conversation_id,
                },
            }
        )
        if tracker is not None:
            for snapshot in tracker.start_snapshots(time.monotonic()):
                yield _connector_status_part(status_id, snapshot)

        internal_headers = _build_internal_scope_headers(
            runtime.settings,
            conversation_id=conversation_id,
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
            user_email=user_email,
            collection_id=upload_collection_id,
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://connector.internal",
            timeout=None,
        ) as client:
            try:
                if upload_items:
                    if tracker is not None:
                        for snapshot in tracker.transition_phase(
                            PHASE_UPLOADING,
                            now=time.monotonic(),
                            source_event_type="upload_started",
                            label="Uploading inputs",
                            detail="Uploading files to the session workspace",
                        ):
                            yield _connector_status_part(status_id, snapshot)
                    internal_authorization = _internal_gateway_auth_headers(runtime.settings).get("Authorization")
                    upload_result = await upload_files(
                        files=[
                            UploadFile(
                                file=BytesIO(bytes(item["content"])),
                                filename=str(item["filename"]),
                            )
                            for item in upload_items
                        ],
                        source_type="upload",
                        collection_id=upload_collection_id,
                        source_ids=[str(item.get("source_id") or "") for item in upload_items],
                        runtime=runtime,
                        authorization=internal_authorization,
                        x_conversation_id=conversation_id,
                        x_request_id=request_id,
                        x_tenant_id=tenant_id,
                        x_user_id=user_id,
                        x_user_email=user_email,
                        x_collection_id=upload_collection_id,
                        x_upload_source_ids=None,
                    )
                else:
                    upload_result = {"filenames": []}

                if tracker is not None:
                    waiting_detail = "Waiting for the gateway to stream runtime progress"
                    if list(upload_result.get("filenames") or []):
                        waiting_detail = f"{len(list(upload_result.get('filenames') or []))} uploaded file(s) ready"
                    initial_phase, initial_label = _initial_connector_status(last_user_query)
                    for snapshot in tracker.transition_phase(
                        initial_phase,
                        now=time.monotonic(),
                        source_event_type="chat_request_started",
                        label=initial_label,
                        detail=waiting_detail,
                    ):
                        yield _connector_status_part(status_id, snapshot)

                chat_request = {
                    "model": model_id,
                    "messages": upstream_messages,
                    "stream": True,
                    "metadata": request_metadata,
                }
            except Exception as exc:
                if tracker is not None:
                    failure = tracker.failure_snapshot(time.monotonic())
                    if failure is not None:
                        yield _connector_status_part(status_id, failure)
                yield _connector_stream_sse({"type": "error", "errorText": str(exc)})
                yield _connector_stream_sse({"type": "finish"})
                yield "data: [DONE]\n\n"
                return

            try:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers={**internal_headers, "Content-Type": "application/json"},
                    json=chat_request,
                ) as chat_response:
                    if chat_response.status_code >= 400:
                        detail = await chat_response.aread()
                        if tracker is not None:
                            failure = tracker.failure_snapshot(time.monotonic())
                            if failure is not None:
                                yield _connector_status_part(status_id, failure)
                        yield _connector_stream_sse(
                            {
                                "type": "error",
                                "errorText": detail.decode("utf-8", errors="replace") or "Connector chat bridge failed.",
                            }
                        )
                        yield _connector_stream_sse({"type": "finish"})
                        yield "data: [DONE]\n\n"
                        return

                    text_started = False
                    text_finished = False
                    latest_metadata_payload: Dict[str, Any] = {}
                    upstream_events: asyncio.Queue[tuple[str | None, str]] = asyncio.Queue()
                    upstream_error: BaseException | None = None

                    async def pump_upstream_events() -> None:
                        nonlocal upstream_error
                        try:
                            async for event_name, data in _aiter_sse_events(chat_response):
                                await upstream_events.put((event_name, data))
                        except Exception as exc:
                            upstream_error = exc
                        finally:
                            await upstream_events.put((None, ""))

                    upstream_task = asyncio.create_task(pump_upstream_events())
                    try:
                        while True:
                            try:
                                if tracker is None:
                                    event_name, data = await upstream_events.get()
                                else:
                                    timeout = tracker.seconds_until_next_heartbeat(
                                        time.monotonic(),
                                        interval_seconds=STATUS_HEARTBEAT_SECONDS,
                                    )
                                    if timeout is None:
                                        event_name, data = await upstream_events.get()
                                    else:
                                        event_name, data = await asyncio.wait_for(upstream_events.get(), timeout=timeout)
                            except asyncio.TimeoutError:
                                if tracker is not None:
                                    snapshot = tracker.heartbeat_snapshot(time.monotonic())
                                    if snapshot is not None:
                                        yield _connector_status_part(status_id, snapshot)
                                continue

                            if event_name is None:
                                break
                            if event_name == "status":
                                try:
                                    payload_data = json.loads(data)
                                except json.JSONDecodeError:
                                    payload_data = {"description": data}
                                if isinstance(payload_data, dict):
                                    yield _connector_status_part(
                                        str(payload_data.get("status_id") or status_id),
                                        payload_data,
                                    )
                                continue
                            if event_name == "progress":
                                try:
                                    payload_data = json.loads(data)
                                except json.JSONDecodeError:
                                    payload_data = {"label": data}
                                if tracker is not None:
                                    for snapshot in tracker.progress_snapshots(payload_data, time.monotonic()):
                                        yield _connector_status_part(status_id, snapshot)
                                continue
                            if event_name == "artifacts":
                                try:
                                    artifact_payload = json.loads(data)
                                except json.JSONDecodeError:
                                    artifact_payload = []
                                for part in _artifact_parts_from_gateway_payload(artifact_payload):
                                    yield _connector_stream_sse(part)
                                continue
                            if event_name == "metadata":
                                try:
                                    metadata_payload = json.loads(data)
                                except json.JSONDecodeError:
                                    metadata_payload = {"raw": data}
                                latest_metadata_payload = dict(metadata_payload) if isinstance(metadata_payload, dict) else {}
                                yield _connector_stream_sse({"type": "data-metadata", "data": metadata_payload})
                                continue
                            if data == "[DONE]":
                                break
                            try:
                                chunk_payload = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            deltas, saw_finish = _text_deltas_from_chunk(chunk_payload)
                            if deltas and tracker is not None and tracker.active_phase != PHASE_SYNTHESIZING:
                                for snapshot in tracker.transition_phase(
                                    PHASE_SYNTHESIZING,
                                    now=time.monotonic(),
                                    source_event_type="content_delta",
                                    label="Synthesizing answer",
                                    detail="Grounding final response",
                                ):
                                    yield _connector_status_part(status_id, snapshot)
                            if deltas and not text_started:
                                yield _connector_stream_sse({"type": "text-start", "id": text_block_id})
                                text_started = True
                            for delta in deltas:
                                yield _connector_stream_sse({"type": "text-delta", "id": text_block_id, "delta": delta})
                            if saw_finish and text_started and not text_finished:
                                yield _connector_stream_sse({"type": "text-end", "id": text_block_id})
                                text_finished = True
                        if upstream_error is not None:
                            raise upstream_error
                        if text_started and not text_finished:
                            yield _connector_stream_sse({"type": "text-end", "id": text_block_id})
                        if tracker is not None:
                            for snapshot in tracker.completion_snapshots(
                                time.monotonic(),
                                metadata=latest_metadata_payload,
                            ):
                                yield _connector_status_part(status_id, snapshot)
                        yield _connector_stream_sse({"type": "finish"})
                        yield "data: [DONE]\n\n"
                    finally:
                        upstream_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await upstream_task
            except Exception as exc:
                if tracker is not None:
                    failure = tracker.failure_snapshot(time.monotonic())
                    if failure is not None:
                        yield _connector_status_part(status_id, failure)
                yield _connector_stream_sse({"type": "error", "errorText": str(exc)})
                yield _connector_stream_sse({"type": "finish"})
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )


@app.get("/v1/capabilities/catalog")
def get_capabilities_catalog(
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    session = SimpleNamespace(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        metadata={"access_summary": dict(ctx.access_summary or {})},
        access_summary=dict(ctx.access_summary or {}),
    )
    effective = resolve_effective_capabilities(
        settings=runtime.settings,
        stores=runtime.bot.ctx.stores,
        session=session,
        registry=runtime.bot.kernel.registry,
        access_summary=dict(ctx.access_summary or {}),
    )
    return build_capability_catalog(
        settings=runtime.settings,
        stores=runtime.bot.ctx.stores,
        session=session,
        registry=runtime.bot.kernel.registry,
        tool_definitions=getattr(runtime.bot.kernel, "tool_definitions", {}),
        effective=effective,
    )


@app.get("/v1/users/me/capabilities")
def get_my_capabilities(
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    session = SimpleNamespace(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        metadata={"access_summary": dict(ctx.access_summary or {})},
        access_summary=dict(ctx.access_summary or {}),
    )
    effective = resolve_effective_capabilities(
        settings=runtime.settings,
        stores=runtime.bot.ctx.stores,
        session=session,
        registry=runtime.bot.kernel.registry,
        access_summary=dict(ctx.access_summary or {}),
    )
    return {
        "object": "user.capabilities",
        "tenant_id": ctx.tenant_id,
        "user_id": ctx.user_id,
        "profile": CapabilityProfile.from_dict(effective.to_dict()).to_dict(),
        "effective_capabilities": effective.to_dict(),
    }


@app.put("/v1/users/me/capabilities")
def update_my_capabilities(
    request: CapabilityProfileRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    raw = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    profile = CapabilityProfile.from_dict(raw)
    saved = save_capability_profile(
        settings=runtime.settings,
        stores=runtime.bot.ctx.stores,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        profile=profile,
    )
    session = SimpleNamespace(
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        metadata={"capability_profile": saved.to_dict(), "access_summary": dict(ctx.access_summary or {})},
        access_summary=dict(ctx.access_summary or {}),
    )
    effective = resolve_effective_capabilities(
        settings=runtime.settings,
        stores=runtime.bot.ctx.stores,
        session=session,
        registry=runtime.bot.kernel.registry,
        profile=saved,
        access_summary=dict(ctx.access_summary or {}),
    )
    return {
        "object": "user.capabilities",
        "tenant_id": ctx.tenant_id,
        "user_id": ctx.user_id,
        "profile": saved.to_dict(),
        "effective_capabilities": effective.to_dict(),
    }


def _task_payload_from_job(runtime: Runtime, job: Any) -> Dict[str, Any]:
    artifacts = [
        normalize_artifact(item)
        for item in list((getattr(job, "metadata", {}) or {}).get("artifacts") or [])
        if isinstance(item, dict)
    ]
    if artifacts:
        artifacts = _present_download_artifacts(runtime, artifacts)
    return {
        "object": "task",
        "task_id": getattr(job, "job_id", ""),
        "job_id": getattr(job, "job_id", ""),
        "worker_agent": getattr(job, "agent_name", ""),
        "status": getattr(job, "status", ""),
        "dependencies": list((getattr(job, "metadata", {}) or {}).get("depends_on") or []),
        "output_artifact_path": getattr(job, "output_path", "") or getattr(job, "result_path", ""),
        "progress_summary": getattr(job, "result_summary", "") or getattr(job, "description", ""),
        "recent_tool_activity": list((getattr(job, "metadata", {}) or {}).get("recent_tool_activity") or []),
        "token_counts": {
            "estimated": int(getattr(job, "estimated_token_cost", 0) or 0),
            "actual": int(getattr(job, "actual_token_cost", 0) or 0),
        },
        "warnings": list((getattr(job, "metadata", {}) or {}).get("warnings") or []),
        "errors": [str(getattr(job, "last_error", "") or "")] if str(getattr(job, "last_error", "") or "").strip() else [],
        "artifacts": artifacts,
        "metadata": dict(getattr(job, "metadata", {}) or {}),
    }


@app.get("/v1/tasks")
def list_tasks(
    status: str = "",
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    jobs = runtime.bot.kernel.job_manager.list_jobs(session_id=ctx.session_id)
    if status:
        jobs = [job for job in jobs if str(getattr(job, "status", "") or "") == status]
    return {
        "object": "task.list",
        "tasks": [_task_payload_from_job(runtime, job) for job in jobs],
    }


@app.get("/v1/tasks/{task_id}")
def get_task(
    task_id: str,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    job = runtime.bot.kernel.job_manager.get_job(task_id)
    if job is None or str(getattr(job, "session_id", "") or "") != ctx.session_id:
        raise HTTPException(status_code=404, detail="Task not found.")
    return _task_payload_from_job(runtime, job)


@app.post("/v1/tasks/{task_id}/stop")
def stop_task(
    task_id: str,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    job = runtime.bot.kernel.job_manager.get_job(task_id)
    if job is None or str(getattr(job, "session_id", "") or "") != ctx.session_id:
        raise HTTPException(status_code=404, detail="Task not found.")
    stopped = runtime.bot.kernel.job_manager.stop_job(task_id)
    if stopped is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return _task_payload_from_job(runtime, stopped)


@app.get("/v1/jobs/{job_id}")
def get_job_status(
    job_id: str,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    job = runtime.bot.kernel.job_manager.get_job(job_id)
    if job is None or str(job.session_id or "") != ctx.session_id:
        raise HTTPException(status_code=404, detail="Job not found.")
    artifacts = [
        normalize_artifact(item)
        for item in list((job.metadata or {}).get("artifacts") or [])
        if isinstance(item, dict)
    ]
    if artifacts:
        artifacts = _present_download_artifacts(runtime, artifacts)
    job_manager = runtime.bot.kernel.job_manager
    mailbox_summary = job_manager.mailbox_summary(job.job_id) if hasattr(job_manager, "mailbox_summary") else {}
    team_mailbox_summary = (
        job_manager.team_mailbox_summary(job.session_id)
        if bool(getattr(runtime.settings, "team_mailbox_enabled", False)) and hasattr(job_manager, "team_mailbox_summary")
        else {}
    )
    return {
        "object": "job",
        "job_id": getattr(job, "job_id", ""),
        "session_id": getattr(job, "session_id", ""),
        "agent_name": getattr(job, "agent_name", ""),
        "status": getattr(job, "status", ""),
        "tenant_id": getattr(job, "tenant_id", ""),
        "user_id": getattr(job, "user_id", ""),
        "priority": getattr(job, "priority", ""),
        "queue_class": getattr(job, "queue_class", ""),
        "scheduler_state": getattr(job, "scheduler_state", ""),
        "description": getattr(job, "description", ""),
        "result_summary": getattr(job, "result_summary", ""),
        "output_path": getattr(job, "output_path", ""),
        "result_path": getattr(job, "result_path", ""),
        "last_error": getattr(job, "last_error", ""),
        "enqueued_at": getattr(job, "enqueued_at", ""),
        "started_at": getattr(job, "started_at", ""),
        "estimated_token_cost": int(getattr(job, "estimated_token_cost", 0) or 0),
        "actual_token_cost": int(getattr(job, "actual_token_cost", 0) or 0),
        "budget_block_reason": getattr(job, "budget_block_reason", ""),
        "updated_at": getattr(job, "updated_at", ""),
        "artifacts": artifacts,
        "mailbox": mailbox_summary,
        "team_mailbox": team_mailbox_summary,
        "metadata": dict(getattr(job, "metadata", {}) or {}),
    }


@app.get("/v1/jobs/{job_id}/mailbox")
def get_job_mailbox(
    job_id: str,
    status_filter: str = "",
    request_type: str = "",
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    job = runtime.bot.kernel.job_manager.get_job(job_id)
    if job is None or str(job.session_id or "") != ctx.session_id:
        raise HTTPException(status_code=404, detail="Job not found.")
    messages = runtime.bot.kernel.job_manager.list_mailbox_messages(
        job_id,
        status_filter=status_filter,
        request_type=request_type,
    )
    return {
        "object": "worker_mailbox",
        "job_id": job_id,
        "mailbox": runtime.bot.kernel.job_manager.mailbox_summary(job_id),
        "data": [_serialize_mailbox_message(item) for item in messages],
    }


@app.post("/v1/jobs/{job_id}/mailbox/{message_id}/respond")
def respond_job_mailbox(
    job_id: str,
    message_id: str,
    request: WorkerMailboxRespondRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    job = runtime.bot.kernel.job_manager.get_job(job_id)
    if job is None or str(job.session_id or "") != ctx.session_id:
        raise HTTPException(status_code=404, detail="Job not found.")
    messages = runtime.bot.kernel.job_manager.list_mailbox_messages(job_id)
    target = next((item for item in messages if item.message_id == message_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Mailbox request not found.")
    allow_approval = False
    if target.message_type == "approval_request":
        if x_admin_token is not None:
            require_admin_token(runtime.settings, x_admin_token=x_admin_token)
            allow_approval = True
        elif _worker_request_approval_allowed(ctx, job_id=job_id):
            allow_approval = True
        else:
            raise HTTPException(status_code=403, detail="Approval requests require an admin token or worker_request approve permission.")
    try:
        result = runtime.bot.kernel.job_manager.respond_to_request(
            job_id,
            message_id,
            response=request.response,
            responder=ctx.user_id or "operator",
            decision=request.decision,
            allow_approval=allow_approval,
            metadata={"source": "api"},
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    resolved_request, response_message = result
    if request.resume:
        runtime.bot.kernel.job_manager.continue_job(job_id, runtime.bot.kernel._job_runner)
    runtime.bot.kernel._sync_pending_worker_request_for_session(job.session_id)
    refreshed = runtime.bot.kernel.job_manager.get_job(job_id) or job
    return {
        "object": "worker_mailbox_response",
        "job_id": job_id,
        "status": getattr(refreshed, "status", ""),
        "request": _serialize_mailbox_message(resolved_request),
        "response": _serialize_mailbox_message(response_message),
    }


@app.get("/v1/sessions/{session_id}/team-mailbox/channels")
def list_team_mailbox_channels(
    session_id: str,
    status_filter: str = "active",
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_team_mailbox_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    if session_id != ctx.session_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    job_manager = runtime.bot.kernel.job_manager
    channels = job_manager.list_team_channels(session_id, status_filter=status_filter)
    return {
        "object": "team_mailbox_channels",
        "session_id": session_id,
        "summary": job_manager.team_mailbox_summary(session_id),
        "data": [_serialize_team_channel(item) for item in channels],
    }


@app.post("/v1/sessions/{session_id}/team-mailbox/channels")
def create_team_mailbox_channel(
    session_id: str,
    request: TeamMailboxChannelRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_team_mailbox_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    if session_id != ctx.session_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    job_manager = runtime.bot.kernel.job_manager
    member_job_ids = _validate_team_mailbox_job_ids(job_manager, session_id, request.member_job_ids)
    try:
        channel = job_manager.create_team_channel(
            session_id=session_id,
            name=request.name,
            purpose=request.purpose,
            member_agents=request.member_agents,
            member_job_ids=member_job_ids,
            metadata={**dict(request.metadata or {}), "created_by": ctx.user_id or "operator", "source": "api"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"object": "team_mailbox_channel", "channel": _serialize_team_channel(channel)}


@app.get("/v1/sessions/{session_id}/team-mailbox/messages")
def list_team_mailbox_messages(
    session_id: str,
    channel_id: str = "",
    message_type: str = "",
    status_filter: str = "open",
    limit: int = 20,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_team_mailbox_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    if session_id != ctx.session_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    job_manager = runtime.bot.kernel.job_manager
    rows = job_manager.list_team_messages(
        session_id,
        channel_id=channel_id,
        message_type=message_type,
        status_filter=status_filter,
        limit=limit,
    )
    return {
        "object": "team_mailbox_messages",
        "session_id": session_id,
        "summary": job_manager.team_mailbox_summary(session_id, channel_id=channel_id),
        "data": [_serialize_team_message(item) for item in rows],
    }


@app.post("/v1/sessions/{session_id}/team-mailbox/channels/{channel_id}/messages")
def post_team_mailbox_message(
    session_id: str,
    channel_id: str,
    request: TeamMailboxMessageRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_team_mailbox_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
    )
    if session_id != ctx.session_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    job_manager = runtime.bot.kernel.job_manager
    channel = next(
        (item for item in job_manager.list_team_channels(session_id, status_filter="") if item.channel_id == channel_id),
        None,
    )
    target_job_ids = _validate_team_mailbox_job_ids(
        job_manager,
        session_id,
        request.target_job_ids,
        channel=channel,
    )
    try:
        message = job_manager.post_team_message(
            session_id=session_id,
            channel_id=channel_id,
            content=request.content,
            source_agent="operator",
            source_job_id="",
            target_agents=request.target_agents,
            target_job_ids=target_job_ids,
            message_type=request.message_type,
            subject=request.subject,
            payload=request.payload,
            metadata={"source": "api", "user_id": ctx.user_id},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"object": "team_mailbox_message", "message": _serialize_team_message(message)}


@app.post("/v1/sessions/{session_id}/team-mailbox/channels/{channel_id}/messages/{message_id}/respond")
def respond_team_mailbox_message(
    session_id: str,
    channel_id: str,
    message_id: str,
    request: TeamMailboxRespondRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    _require_team_mailbox_enabled(runtime)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    if session_id != ctx.session_id:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = runtime.bot.kernel.job_manager.list_team_messages(
        session_id,
        channel_id=channel_id,
        status_filter="",
        limit=500,
    )
    target = next((item for item in messages if item.message_id == message_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Team mailbox request not found.")
    allow_approval = False
    if target.message_type == "approval_request":
        if x_admin_token is not None:
            require_admin_token(runtime.settings, x_admin_token=x_admin_token)
            allow_approval = True
        elif _worker_request_approval_allowed(ctx, job_id=channel_id):
            allow_approval = True
        else:
            raise HTTPException(status_code=403, detail="Approval requests require an admin token or worker_request approve permission.")
    try:
        result = runtime.bot.kernel.job_manager.respond_team_message(
            session_id,
            channel_id,
            message_id,
            response=request.response,
            responder_agent="operator",
            responder_job_id="",
            decision=request.decision,
            allow_approval=allow_approval,
            resolve=request.resolve,
            metadata={"source": "api", "user_id": ctx.user_id},
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Team mailbox request not found.")
    resolved_request, response_message = result
    return {
        "object": "team_mailbox_response",
        "session_id": session_id,
        "channel_id": channel_id,
        "request": _serialize_team_message(resolved_request),
        "response": _serialize_team_message(response_message),
    }


@app.get("/v1/files/{download_id}")
def download_file(
    download_id: str,
    conversation_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    expires: Optional[int] = None,
    sig: Optional[str] = None,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
):
    resolved_conversation_id = _first_non_empty(conversation_id, x_conversation_id, x_openwebui_chat_id)
    resolved_tenant_id = _first_non_empty(tenant_id, x_tenant_id) or runtime.settings.default_tenant_id
    resolved_user_id = _first_non_empty(user_id, x_user_id, x_openwebui_user_id) or runtime.settings.default_user_id
    signed_access_granted = verify_download_token(
        download_id=download_id,
        tenant_id=resolved_tenant_id,
        user_id=resolved_user_id,
        conversation_id=str(resolved_conversation_id or runtime.settings.default_conversation_id),
        expires=expires or 0,
        sig=sig,
        secret=runtime.settings.download_url_secret,
    )
    if not signed_access_granted:
        _require_gateway_bearer_auth(runtime.settings, authorization)

    ctx = get_request_context(
        runtime,
        conversation_id=resolved_conversation_id,
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=resolved_tenant_id,
        user_id=resolved_user_id,
    )
    state = runtime.bot.kernel.transcript_store.load_session_state(ctx.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found for requested file.")
    downloads = {
        str(key): normalize_artifact(value)
        for key, value in dict(state.metadata.get("downloads") or {}).items()
        if isinstance(value, dict)
    }
    artifact = downloads.get(download_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Download file not found.")
    filename = Path(str(artifact.get("filename") or "")).name
    if not filename:
        raise HTTPException(status_code=404, detail="Download file metadata is invalid.")
    file_path = Path(state.workspace_root or runtime.settings.workspace_dir / filesystem_key(ctx.session_id)) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Download file is no longer available.")
    return FileResponse(
        path=file_path,
        media_type=str(artifact.get("content_type") or "application/octet-stream"),
        filename=filename,
    )


@app.get("/v1/documents/{doc_id}/source")
def document_source_file(
    doc_id: str,
    conversation_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    expires: Optional[int] = None,
    sig: Optional[str] = None,
    disposition: str = "attachment",
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
):
    resolved_conversation_id = _first_non_empty(conversation_id, x_conversation_id, x_openwebui_chat_id)
    resolved_tenant_id = _first_non_empty(tenant_id, x_tenant_id) or runtime.settings.default_tenant_id
    resolved_user_id = _first_non_empty(user_id, x_user_id, x_openwebui_user_id) or runtime.settings.default_user_id
    signed_access_granted = verify_download_token(
        download_id=doc_id,
        tenant_id=resolved_tenant_id,
        user_id=resolved_user_id,
        conversation_id=str(resolved_conversation_id or runtime.settings.default_conversation_id),
        expires=expires or 0,
        sig=sig,
        secret=runtime.settings.download_url_secret,
    )
    if not signed_access_granted:
        _require_gateway_bearer_auth(runtime.settings, authorization)

    ctx = get_request_context(
        runtime,
        conversation_id=resolved_conversation_id,
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=resolved_tenant_id,
        user_id=resolved_user_id,
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    state = _load_or_create_session_state(runtime, ctx)
    _apply_request_access_snapshot(runtime, state, user_email=ctx.user_email, display_name=ctx.user_id)
    doc_store = getattr(getattr(getattr(runtime.bot, "ctx", None), "stores", None), "doc_store", None)
    if doc_store is None or not hasattr(doc_store, "get_document"):
        raise HTTPException(status_code=503, detail="Document store is unavailable.")
    try:
        record = doc_store.get_document(doc_id, ctx.tenant_id)
    except TypeError:
        record = doc_store.get_document(doc_id=doc_id, tenant_id=ctx.tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    _require_collection_use_access(
        runtime,
        state,
        collection_id=str(getattr(record, "collection_id", "") or ""),
    )
    content_disposition_type = "inline" if str(disposition or "").strip().lower() == "inline" else "attachment"
    source_metadata = dict(getattr(record, "source_metadata", {}) or {})
    blob_ref = blob_ref_from_record(record)
    filename = (
        Path(str(source_metadata.get("original_filename") or "")).name
        or Path(str(getattr(record, "source_display_path", "") or "")).name
        or Path(str(getattr(record, "title", "") or "")).name
    )
    if blob_ref is not None:
        blob_store = build_blob_store(runtime.settings)
        if blob_ref.backend != "local":
            if not blob_store.exists(blob_ref):
                raise HTTPException(status_code=404, detail="Document source file is no longer available.")
            download_filename = (filename or doc_id).replace('"', "")
            return StreamingResponse(
                blob_store.iter_bytes(blob_ref),
                media_type=(
                    str(getattr(record, "source_content_type", "") or "")
                    or blob_ref.content_type
                    or _document_source_media_type(record, Path(filename or getattr(record, "title", "source")))
                ),
                headers={"Content-Disposition": f'{content_disposition_type}; filename="{download_filename}"'},
            )
        try:
            local_source = blob_store.materialize_to_path(blob_ref)
            source_path = _safe_document_source_path(runtime.settings, str(local_source))
        except Exception:
            source_path = _safe_document_source_path(runtime.settings, str(getattr(record, "source_path", "") or ""))
    else:
        source_path = _safe_document_source_path(runtime.settings, str(getattr(record, "source_path", "") or ""))
    if not filename:
        filename = source_path.name
    return FileResponse(
        path=source_path,
        media_type=_document_source_media_type(record, source_path),
        filename=filename,
        content_disposition_type=content_disposition_type,
    )


@app.get("/v1/graphs")
def list_graphs(
    collection_id: str = "",
    limit: int = 20,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    state = _load_or_create_session_state(runtime, ctx)
    _apply_request_access_snapshot(runtime, state, user_email=ctx.user_email, display_name=ctx.user_id)
    if str(collection_id or "").strip():
        _require_collection_use_access(runtime, state, collection_id=collection_id)
    service = GraphService(runtime.settings, runtime.bot.ctx.stores, session=state)
    return {"object": "graph.index.list", "graphs": service.list_indexes(collection_id=collection_id, limit=limit)}


@app.get("/v1/graphs/{graph_id}")
def inspect_graph(
    graph_id: str,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    state = _load_or_create_session_state(runtime, ctx)
    _apply_request_access_snapshot(runtime, state, user_email=ctx.user_email, display_name=ctx.user_id)
    _require_graph_use_access(runtime, state, graph_id=graph_id)
    service = GraphService(runtime.settings, runtime.bot.ctx.stores, session=state)
    payload = service.inspect_index(graph_id)
    if payload.get("error"):
        raise HTTPException(status_code=404, detail=str(payload["error"]))
    _persist_session_state(runtime, state)
    return {"object": "graph.index", **payload}


@app.post("/v1/graphs/index")
def index_graph(
    request: GraphIndexRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    del request, x_conversation_id, x_openwebui_chat_id, x_request_id, x_openwebui_message_id, x_tenant_id, x_user_id, x_openwebui_user_id
    raise HTTPException(status_code=403, detail="Graph creation and refresh are admin-managed in the control panel.")


@app.post("/v1/graphs/import")
def import_graph(
    request: GraphImportRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    del request, x_conversation_id, x_openwebui_chat_id, x_request_id, x_openwebui_message_id, x_tenant_id, x_user_id, x_openwebui_user_id
    raise HTTPException(status_code=403, detail="Graph creation and refresh are admin-managed in the control panel.")


@app.post("/v1/graphs/query")
def query_graph(
    request: GraphQueryRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    state = _load_or_create_session_state(runtime, ctx)
    _apply_request_access_snapshot(runtime, state, user_email=ctx.user_email, display_name=ctx.user_id)
    if str(request.graph_id or "").strip():
        _require_graph_use_access(runtime, state, graph_id=str(request.graph_id or "").strip())
    elif str(request.collection_id or "").strip():
        _require_collection_use_access(runtime, state, collection_id=str(request.collection_id or "").strip())
    service = GraphService(runtime.settings, runtime.bot.ctx.stores, session=state)
    if str(request.graph_id or "").strip():
        payload = {
            "object": "graph.query.result",
            **service.query_index(
                str(request.graph_id or "").strip(),
                query=request.query,
                methods=list(request.methods or []),
                limit=int(request.limit),
            ),
        }
        _persist_session_state(runtime, state)
        return payload
    payload = {
        "object": "graph.query.result",
        **service.query_across_graphs(
            request.query,
            collection_id=str(request.collection_id or ""),
            graph_ids=[],
            methods=list(request.methods or []),
            limit=int(request.limit),
            top_k_graphs=int(request.top_k_graphs),
        ),
    }
    _persist_session_state(runtime, state)
    return payload


@app.post("/v1/ingest/documents")
def ingest_documents(
    request: IngestDocumentsRequest,
    runtime: Runtime = Depends(get_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_collection_id: Optional[str] = Header(None, alias="X-Collection-ID"),
) -> Dict[str, Any]:
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    logger.info(
        "ingest_documents request tenant=%s user=%s conversation=%s request_id=%s source_type=%s files=%d",
        ctx.tenant_id,
        ctx.user_id,
        ctx.conversation_id,
        ctx.request_id or "-",
        request.source_type,
        len(request.paths),
    )

    valid_paths, missing = _expand_request_paths(request.paths)
    effective_collection_id = str(
        request.collection_id
        or x_collection_id
        or getattr(runtime.settings, "default_collection_id", "default")
    )
    state = _load_or_create_session_state(runtime, ctx)
    state.metadata = merge_scope_metadata(
        runtime.settings,
        {
            **dict(state.metadata or {}),
            "collection_id": effective_collection_id,
            "upload_collection_id": effective_collection_id,
            "user_email": ctx.user_email,
        },
    )
    _apply_request_access_snapshot(
        runtime,
        state,
        user_email=ctx.user_email,
        request_metadata={"collection_id": effective_collection_id, "upload_collection_id": effective_collection_id},
        display_name=ctx.user_id,
    )
    _require_collection_use_access(runtime, state, collection_id=effective_collection_id)

    if request.index_preview:
        preview = _preview_index_metadata_for_paths(
            runtime.settings,
            valid_paths,
            metadata_profile=request.metadata_profile,
            metadata_enrichment=request.metadata_enrichment,
            providers=getattr(getattr(runtime.bot, "ctx", None), "providers", None),
            source_metadata_by_path=dict(request.source_metadata or {}),
            source_display_paths=dict(request.source_display_paths or {}),
        )
        preview.update(
            {
                "object": "ingest.preview",
                "tenant_id": ctx.tenant_id,
                "collection_id": effective_collection_id,
                "missing_paths": missing,
            }
        )
        if missing and not valid_paths:
            preview["status"] = "failed"
        elif missing:
            preview["status"] = "partial"
        return preview

    doc_ids = ingest_paths(
        runtime.settings,
        runtime.bot.ctx.stores,
        valid_paths,
        source_type=request.source_type,
        tenant_id=ctx.tenant_id,
        collection_id=effective_collection_id,
        source_display_paths=dict(request.source_display_paths or {}),
        source_identities=dict(request.source_identities or {}),
        source_metadata_by_path=dict(request.source_metadata or {}),
        metadata_profile=request.metadata_profile,
        metadata_enrichment=request.metadata_enrichment,
        providers=getattr(getattr(runtime.bot, "ctx", None), "providers", None),
    )

    # Copy ingested files into the active session workspace keyed by session_id.
    # We proactively create the workspace so uploads are available to the data
    # analyst at /workspace/<filename> even before the first chat turn runs.
    ws_conversation_id = (
        request.conversation_id
        or _first_non_empty(x_conversation_id, x_openwebui_chat_id)
        or runtime.settings.default_conversation_id
    )
    ws_session_id = f"{ctx.tenant_id}:{ctx.user_id}:{ws_conversation_id}"
    workspace = SessionWorkspace.for_session(ws_session_id, runtime.settings.workspace_dir)
    workspace.open()
    ws_root = workspace.root
    workspace_copies: List[str] = []
    for p in valid_paths:
        try:
            shutil.copy2(p, ws_root / p.name)
            workspace_copies.append(p.name)
            logger.debug("ingest_documents: copied %s into workspace %s", p.name, ws_root)
        except Exception as cp_exc:
            logger.warning(
                "ingest_documents: could not copy %s to workspace %s: %s",
                p.name, ws_root, cp_exc,
            )

    result: Dict[str, Any] = {
        "object": "ingest.result",
        "tenant_id": ctx.tenant_id,
        "collection_id": effective_collection_id,
        "ingested_count": len(doc_ids),
        "doc_ids": doc_ids,
        "missing_paths": missing,
        "metadata_summary": _index_metadata_summary_for_doc_ids(
            runtime.bot.ctx.stores,
            ctx.tenant_id,
            doc_ids,
            metadata_profile=request.metadata_profile,
        ),
    }
    if workspace_copies:
        result["workspace_copies"] = workspace_copies
    return result


@app.post("/v1/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    source_type: str = "upload",
    collection_id: str = "",
    metadata_profile: str = "auto",
    metadata_enrichment: str = "deterministic",
    index_preview: bool = False,
    source_ids: Optional[List[str]] = Form(None),
    runtime: Runtime = Depends(get_upload_runtime_or_503),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_openwebui_chat_id: Optional[str] = Header(None, alias="X-OpenWebUI-Chat-Id"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    x_openwebui_message_id: Optional[str] = Header(None, alias="X-OpenWebUI-Message-Id"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_openwebui_user_id: Optional[str] = Header(None, alias="X-OpenWebUI-User-Id"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
    x_openwebui_user_email: Optional[str] = Header(None, alias="X-OpenWebUI-User-Email"),
    x_collection_id: Optional[str] = Header(None, alias="X-Collection-ID"),
    x_upload_source_ids: Optional[str] = Header(None, alias="X-Upload-Source-Ids"),
) -> Dict[str, Any]:
    """Accept multipart file uploads from Open WebUI or another client.

    Saves files to the uploads directory, ingests them into the KB,
    and optionally copies them into the session workspace.
    """
    _require_gateway_bearer_auth(runtime.settings, authorization)
    ctx = get_request_context(
        runtime,
        conversation_id=_first_non_empty(x_conversation_id, x_openwebui_chat_id),
        request_id=_first_non_empty(x_request_id, x_openwebui_message_id),
        tenant_id=x_tenant_id,
        user_id=_first_non_empty(x_user_id, x_openwebui_user_id),
        user_email=_request_user_email(x_user_email, x_openwebui_user_email),
    )
    logger.info(
        "upload_files request tenant=%s conversation=%s files=%d",
        ctx.tenant_id,
        ctx.conversation_id,
        len(files),
    )

    blob_store = build_blob_store(runtime.settings)
    staging_root = (
        blob_store.cache_dir
        / "incoming"
        / (_safe_blob_key_part(ctx.request_id) or uuid.uuid4().hex)
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    effective_collection_id = str(
        collection_id
        or x_collection_id
        or getattr(runtime.settings, "default_collection_id", "default")
    )
    resolved_source_ids = _parse_source_upload_ids(source_ids, x_upload_source_ids)
    state = _load_or_create_session_state(runtime, ctx)
    state.metadata = merge_scope_metadata(
        runtime.settings,
        {
            **dict(state.metadata or {}),
            "collection_id": effective_collection_id,
            "upload_collection_id": effective_collection_id,
            "user_email": ctx.user_email,
        },
    )
    _apply_request_access_snapshot(
        runtime,
        state,
        user_email=ctx.user_email,
        request_metadata={"collection_id": effective_collection_id, "upload_collection_id": effective_collection_id},
        display_name=ctx.user_id,
    )
    _require_collection_use_access(runtime, state, collection_id=effective_collection_id)
    metadata = dict(state.metadata or {})
    seen_source_ids = [str(item) for item in list(metadata.get("source_upload_ids") or []) if str(item)]
    seen_source_id_set = set(seen_source_ids)

    saved_paths: List[Path] = []
    source_display_paths_by_path: Dict[str, str] = {}
    source_identities_by_path: Dict[str, str] = {}
    source_metadata_by_path: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    skipped_source_ids: List[str] = []
    skipped_filenames: List[str] = []
    remembered_source_ids: List[str] = []
    for index, file in enumerate(files):
        source_id = resolved_source_ids[index] if index < len(resolved_source_ids) else ""
        if source_id and source_id in seen_source_id_set:
            skipped_source_ids.append(source_id)
            skipped_filenames.append(file.filename or source_id)
            continue
        try:
            original_filename = Path(str(file.filename or f"upload_{uuid.uuid4().hex}")).name or f"upload_{uuid.uuid4().hex}"
            dest = staging_root / f"{index:04d}_{uuid.uuid4().hex}_{original_filename}"
            await _stream_upload_to_path(file, dest)
            content_type = str(file.content_type or mimetypes.guess_type(original_filename)[0] or "")
            blob_ref = None
            ingest_path = dest
            if not index_preview:
                blob_ref = blob_store.put_file(
                    dest,
                    key=_upload_object_key(
                        ctx=ctx,
                        collection_id=effective_collection_id,
                        filename=original_filename,
                        index=index,
                    ),
                    content_type=content_type,
                )
                if blob_ref.backend == "local":
                    ingest_path = blob_store.materialize_to_path(blob_ref)
            saved_paths.append(ingest_path)
            path_key = str(ingest_path.resolve())
            source_display_paths_by_path[path_key] = original_filename
            if source_id:
                source_identities_by_path[path_key] = f"upload:{source_id}"
            source_metadata = {
                "source_origin_id": source_id,
                "client_source_id": source_id,
                "original_filename": original_filename,
                "mime_type": content_type,
                "upload_conversation_id": ctx.conversation_id,
                "upload_request_id": ctx.request_id,
            }
            if blob_ref is not None:
                source_metadata.update(
                    {
                        "blob_ref": blob_ref.to_dict(),
                        "source_uri": blob_ref.uri,
                        "storage_backend": blob_ref.backend,
                        "object_bucket": blob_ref.bucket,
                        "object_key": blob_ref.key,
                    }
                )
            source_metadata_by_path[path_key] = source_metadata
            if source_id:
                remembered_source_ids.append(source_id)
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    if index_preview:
        preview_filenames = [
            source_display_paths_by_path.get(str(p.resolve())) or p.name
            for p in saved_paths
        ]
        preview = _preview_index_metadata_for_paths(
            runtime.settings,
            saved_paths,
            metadata_profile=metadata_profile,
            metadata_enrichment=metadata_enrichment,
            providers=getattr(getattr(runtime.bot, "ctx", None), "providers", None),
            source_metadata_by_path=source_metadata_by_path,
            source_display_paths=source_display_paths_by_path,
        )
        preview.update(
            {
                "object": "upload.preview",
                "tenant_id": ctx.tenant_id,
                "collection_id": effective_collection_id,
                "filenames": preview_filenames,
                "skipped_source_ids": list(skipped_source_ids),
                "skipped_filenames": list(skipped_filenames),
            }
        )
        if errors:
            preview["status"] = "partial" if saved_paths else "failed"
            preview["errors"] = [*list(preview.get("errors") or []), *errors]
        return preview

    doc_ids: List[str] = []
    if saved_paths:
        doc_ids = ingest_paths(
            runtime.settings,
            runtime.bot.ctx.stores,
            saved_paths,
            source_type=source_type,
            tenant_id=ctx.tenant_id,
            collection_id=effective_collection_id,
            source_display_paths=source_display_paths_by_path,
            source_identities=source_identities_by_path,
            source_metadata_by_path=source_metadata_by_path,
            metadata_profile=metadata_profile,
            metadata_enrichment=metadata_enrichment,
            providers=getattr(getattr(runtime.bot, "ctx", None), "providers", None),
        )

    ws_session_id = f"{ctx.tenant_id}:{ctx.user_id}:{ctx.conversation_id}"
    workspace = SessionWorkspace.for_session(ws_session_id, runtime.settings.workspace_dir)
    workspace.open()
    ws_root = workspace.root
    workspace_copies: List[str] = []
    for p in saved_paths:
        try:
            workspace_name = Path(source_display_paths_by_path.get(str(p.resolve())) or p.name).name
            shutil.copy2(p, ws_root / workspace_name)
            workspace_copies.append(workspace_name)
        except Exception as exc:
            logger.warning("upload_files: could not copy %s to workspace %s: %s", p.name, ws_root, exc)

    for doc_id in doc_ids:
        if doc_id not in state.uploaded_doc_ids:
            state.uploaded_doc_ids.append(doc_id)
    if remembered_source_ids:
        merged_source_ids = list(seen_source_ids)
        for source_id in remembered_source_ids:
            if source_id not in seen_source_id_set:
                merged_source_ids.append(source_id)
                seen_source_id_set.add(source_id)
        metadata["source_upload_ids"] = merged_source_ids
    active_uploaded_doc_ids = [
        str(doc_id)
        for doc_id in list(getattr(state, "uploaded_doc_ids", []) or [])
        if str(doc_id)
    ]
    upload_manifest: Dict[str, Any] = {
        "object": "upload.manifest",
        "tenant_id": ctx.tenant_id,
        "conversation_id": ctx.conversation_id,
        "collection_id": effective_collection_id,
        "active_uploaded_doc_ids": active_uploaded_doc_ids,
        "doc_ids": list(doc_ids),
        "filenames": [source_display_paths_by_path.get(str(p.resolve())) or p.name for p in saved_paths],
        "skipped_source_ids": list(skipped_source_ids),
        "skipped_filenames": list(skipped_filenames),
        "errors": list(errors),
        "source_type": source_type,
        "metadata_profile": metadata_profile,
        "metadata_enrichment": metadata_enrichment,
        "repository_source": "agent_document_repository",
    }
    runtime_diagnostics = dict(getattr(runtime, "diagnostics", {}) or {})
    if runtime_diagnostics:
        upload_manifest["runtime_diagnostics"] = runtime_diagnostics
        upload_manifest["warnings"] = [
            "Uploaded into the agent document repository while the chat runtime was degraded."
        ]
    if workspace_copies:
        upload_manifest["workspace_copies"] = workspace_copies
    metadata["collection_id"] = effective_collection_id
    metadata["upload_collection_id"] = effective_collection_id
    metadata["uploaded_doc_ids"] = active_uploaded_doc_ids
    metadata["last_upload_manifest"] = upload_manifest
    state.metadata = metadata
    state.workspace_root = str(ws_root)
    _persist_session_state(runtime, state)

    result: Dict[str, Any] = {
        "object": "upload.result",
        "tenant_id": ctx.tenant_id,
        "collection_id": effective_collection_id,
        "ingested_count": len(doc_ids),
        "doc_ids": doc_ids,
        "uploaded_doc_ids": active_uploaded_doc_ids,
        "active_uploaded_doc_ids": active_uploaded_doc_ids,
        "filenames": [source_display_paths_by_path.get(str(p.resolve())) or p.name for p in saved_paths],
        "errors": errors,
        "document_source_policy": "agent_repository_only",
        "upload_manifest": upload_manifest,
        "metadata_summary": _index_metadata_summary_for_doc_ids(
            runtime.bot.ctx.stores,
            ctx.tenant_id,
            doc_ids,
            metadata_profile=metadata_profile,
        ),
    }
    if runtime_diagnostics:
        result["runtime_diagnostics"] = runtime_diagnostics
    if skipped_source_ids:
        result["skipped_source_ids"] = skipped_source_ids
    if skipped_filenames:
        result["skipped_filenames"] = skipped_filenames
    if workspace_copies:
        result["workspace_copies"] = workspace_copies
    return result
