from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from agentic_chatbot_next.runtime.clarification import normalize_clarification_sensitivity
from agentic_chatbot_next.runtime.deep_rag import normalize_deep_rag_mode

_PROCESS_STARTED_AT = datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Settings:
    # --- Backend selection ---
    database_backend: str       # postgres
    vector_store_backend: str   # pgvector
    object_store_backend: str   # local | s3 | azure_blob (future)
    skills_backend: str         # local | s3 | azure_blob (future)
    prompts_backend: str        # local | s3 | azure_blob (future)

    # --- Providers ---
    llm_provider: str  # ollama | azure | nvidia
    embeddings_provider: str  # ollama | azure
    judge_provider: str  # ollama | azure | nvidia (defaults to llm_provider)

    # --- Ollama ---
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embed_model: str
    ollama_judge_model: str
    ollama_temperature: float
    ollama_num_predict: int | None
    demo_ollama_num_predict: int | None
    chat_max_output_tokens: int | None
    demo_chat_max_output_tokens: int | None
    judge_max_output_tokens: int | None
    agent_chat_model_overrides: dict[str, str]
    agent_judge_model_overrides: dict[str, str]
    agent_chat_max_output_tokens: dict[str, int]

    # --- Azure OpenAI (optional) ---
    azure_openai_api_key: str | None
    azure_openai_endpoint: str | None
    azure_openai_api_version: str | None
    azure_openai_chat_deployment: str | None
    azure_openai_judge_deployment: str | None
    azure_openai_embed_deployment: str | None
    azure_temperature: float
    judge_temperature: float
    nvidia_openai_endpoint: str | None
    nvidia_api_token: str | None
    nvidia_chat_model: str | None
    nvidia_judge_model: str | None
    nvidia_temperature: float
    http2_enabled: bool
    ssl_verify: bool
    ssl_cert_file: Path | None
    tiktoken_enabled: bool
    tiktoken_cache_dir: Path | None

    # --- Runtime limits ---
    max_agent_steps: int
    max_tool_calls: int
    max_parallel_tool_calls: int
    deferred_tool_discovery_enabled: bool
    deferred_tool_discovery_top_k: int
    deferred_tool_discovery_require_search: bool
    mcp_tool_plane_enabled: bool
    mcp_user_self_service_enabled: bool
    mcp_require_https: bool
    mcp_allow_private_network: bool
    mcp_connection_timeout_seconds: int
    mcp_tool_call_timeout_seconds: int
    mcp_catalog_refresh_seconds: int
    mcp_secret_encryption_key: str | None
    worker_job_wait_timeout_seconds: int
    llm_http_timeout_seconds: int
    llm_http_connect_timeout_seconds: int

    # --- RAG defaults ---
    rag_top_k_vector: int
    rag_top_k_keyword: int
    rag_max_retries: int
    rag_min_evidence_chunks: int
    max_rag_agent_steps: int  # step budget for the RAG loop agent
    max_parallel_collection_probes: int
    max_collection_discovery_collections: int

    # --- Text splitting ---
    chunk_size: int
    chunk_overlap: int

    # --- PostgreSQL ---
    pg_dsn: str          # e.g. "postgresql://user:pass@localhost:5432/ragdb"
    embedding_dim: int   # must match embedding model output (768 nomic, 1536 ada-002)

    # --- Storage paths (files only, no more vector/bm25 index dirs) ---
    project_root: Path
    data_dir: Path
    kb_dir: Path
    kb_extra_dirs: tuple[Path, ...]
    uploads_dir: Path
    kb_source_uri: str
    uploads_source_uri: str
    default_collection_id: str
    skill_packs_dir: Path
    runtime_dir: Path
    agents_dir: Path

    # --- Prompt/skills locations ---
    skills_dir: Path
    prompts_dir: Path
    shared_skills_path: Path
    general_agent_skills_path: Path
    rag_agent_skills_path: Path
    supervisor_agent_skills_path: Path
    utility_agent_skills_path: Path
    basic_chat_skills_path: Path
    planner_agent_skills_path: Path
    finalizer_agent_skills_path: Path
    judge_grading_prompt_path: Path
    judge_rewrite_prompt_path: Path
    grounded_answer_prompt_path: Path
    rag_synthesis_prompt_path: Path

    # --- Runtime compatibility (deprecated / ignored by live runtime) ---
    agent_runtime_mode: str          # env: AGENT_RUNTIME_MODE
    planner_max_tasks: int           # env: PLANNER_MAX_TASKS (default: 8)

    # --- Scratchpad ---
    clear_scratchpad_per_turn: bool  # wipe session.scratchpad after each turn
    memory_enabled: bool             # env: MEMORY_ENABLED (default: True)
    memory_manager_mode: str         # env: MEMORY_MANAGER_MODE (default: "shadow")
    memory_selector_model: str       # env: MEMORY_SELECTOR_MODEL (default: "")
    memory_writer_model: str         # env: MEMORY_WRITER_MODEL (default: "")
    memory_candidate_top_k: int      # env: MEMORY_CANDIDATE_TOP_K (default: 16)
    memory_context_token_budget: int # env: MEMORY_CONTEXT_TOKEN_BUDGET (default: 1600)
    memory_shadow_mode: bool         # env: MEMORY_SHADOW_MODE (default: False)

    # --- OCR (PaddleOCR, optional) ---
    docling_enabled: bool    # env: DOCLING_ENABLED (default: False)
    ocr_enabled: bool        # env: USE_PADDLE_OCR (default: True)
    ocr_language: str        # env: OCR_LANGUAGE   (default: "en")
    ocr_use_gpu: bool        # env: OCR_USE_GPU    (default: False)
    ocr_min_page_chars: int  # env: OCR_MIN_PAGE_CHARS (default: 50)
                             # PDF pages with fewer extracted chars trigger OCR fallback

    # --- Langfuse (optional) ---
    langfuse_host: str | None
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_debug: bool

    # --- Context defaults (CLI/demo compatibility) ---
    default_tenant_id: str
    default_user_id: str
    default_conversation_id: str

    # --- OpenAI-compatible gateway ---
    gateway_model_id: str
    gateway_shared_bearer_token: str | None
    download_url_secret: str | None
    download_url_ttl_seconds: int
    connector_secret_api_key: str | None
    connector_publishable_api_key: str | None
    connector_allowed_origins: tuple[str, ...]
    connector_publishable_rate_limit_per_minute: int
    authz_enabled: bool

    # --- LLM Router ---
    llm_router_enabled: bool            # env: LLM_ROUTER_ENABLED (default: True)
    llm_router_mode: str               # env: LLM_ROUTER_MODE (default: "hybrid")
    llm_router_confidence_threshold: float  # env: LLM_ROUTER_CONFIDENCE_THRESHOLD (default: 0.70)
    router_patterns_path: Path          # env: ROUTER_PATTERNS_PATH (default: data/router/intent_patterns.json)
    router_feedback_enabled: bool       # env: ROUTER_FEEDBACK_ENABLED (default: True)
    router_feedback_rephrase_window_seconds: int  # env: ROUTER_FEEDBACK_REPHRASE_WINDOW_SECONDS (default: 600)
    router_feedback_neutral_sample_rate: float  # env: ROUTER_FEEDBACK_NEUTRAL_SAMPLE_RATE (default: 0.10)
    router_feedback_tenant_daily_review_cap: int  # env: ROUTER_FEEDBACK_TENANT_DAILY_REVIEW_CAP (default: 25)
    router_retrain_governance: str      # env: ROUTER_RETRAIN_GOVERNANCE (default: "human_reviewed")

    # --- Provider circuit breakers ---
    llm_circuit_breaker_enabled: bool
    llm_circuit_breaker_window_size: int
    llm_circuit_breaker_min_samples: int
    llm_circuit_breaker_error_rate_threshold: float
    llm_circuit_breaker_consecutive_failures: int
    llm_circuit_breaker_open_seconds: int

    # --- Web search fallback (opt-in) ---
    tavily_api_key: str | None          # env: TAVILY_API_KEY
    web_search_enabled: bool            # env: WEB_SEARCH_ENABLED (default: False)

    # --- GraphRAG / Neo4j ---
    graph_search_enabled: bool          # env: GRAPH_SEARCH_ENABLED (default: False)
    graph_ingest_enabled: bool          # env: GRAPH_INGEST_ENABLED (default: False)
    graph_backend: str                  # env: GRAPH_BACKEND (default: "microsoft_graphrag")
    graph_import_enabled: bool          # env: GRAPH_IMPORT_ENABLED (default: True)
    graph_source_planning_enabled: bool # env: GRAPH_SOURCE_PLANNING_ENABLED (default: True)
    retrieval_decomposition_enabled: bool  # env: RETRIEVAL_DECOMPOSITION_ENABLED (default: False)
    entity_linking_enabled: bool       # env: ENTITY_LINKING_ENABLED (default: False)
    section_first_retrieval_enabled: bool  # env: SECTION_FIRST_RETRIEVAL_ENABLED (default: False)
    retrieval_quality_verifier_enabled: bool  # env: RETRIEVAL_QUALITY_VERIFIER_ENABLED (default: False)
    graphrag_projects_dir: Path         # env: GRAPHRAG_PROJECTS_DIR (default: data/graphrag/projects)
    graphrag_use_container: bool        # env: GRAPHRAG_USE_CONTAINER (default: False)
    graphrag_container_image: str       # env: GRAPHRAG_CONTAINER_IMAGE (default: "graphrag:latest")
    graphrag_cli_command: str           # env: GRAPHRAG_CLI_COMMAND (default: "graphrag")
    graphrag_llm_provider: str          # env: GRAPHRAG_LLM_PROVIDER (default: "openai")
    graphrag_base_url: str              # env: GRAPHRAG_BASE_URL
    graphrag_api_key: str | None        # env: GRAPHRAG_API_KEY
    graphrag_chat_model: str            # env: GRAPHRAG_CHAT_MODEL
    graphrag_index_chat_model: str      # env: GRAPHRAG_INDEX_CHAT_MODEL
    graphrag_community_report_mode: str  # env: GRAPHRAG_COMMUNITY_REPORT_MODE
    graphrag_community_report_chat_model: str  # env: GRAPHRAG_COMMUNITY_REPORT_CHAT_MODEL
    graphrag_embed_model: str           # env: GRAPHRAG_EMBED_MODEL
    graphrag_concurrency: int           # env: GRAPHRAG_CONCURRENCY (default: 4)
    graphrag_request_timeout_seconds: int  # env: GRAPHRAG_REQUEST_TIMEOUT_SECONDS (default: 60)
    graphrag_index_request_timeout_seconds: int  # env: GRAPHRAG_INDEX_REQUEST_TIMEOUT_SECONDS (default: GRAPHRAG_REQUEST_TIMEOUT_SECONDS)
    graphrag_community_report_request_timeout_seconds: int  # env: GRAPHRAG_COMMUNITY_REPORT_REQUEST_TIMEOUT_SECONDS
    graphrag_community_report_max_input_length: int  # env: GRAPHRAG_COMMUNITY_REPORT_MAX_INPUT_LENGTH
    graphrag_community_report_max_length: int  # env: GRAPHRAG_COMMUNITY_REPORT_MAX_LENGTH
    graphrag_job_timeout_seconds: int   # env: GRAPHRAG_JOB_TIMEOUT_SECONDS (default: 600; 0 disables the hard job timeout)
    graphrag_timeout_seconds: int       # env: GRAPHRAG_TIMEOUT_SECONDS (legacy compatibility alias for request timeout)
    graphrag_stale_run_after_seconds: int  # env: GRAPHRAG_STALE_RUN_AFTER_SECONDS (default: 900)
    graphrag_default_query_method: str  # env: GRAPHRAG_DEFAULT_QUERY_METHOD (default: "local")
    graphrag_artifact_cache_ttl_seconds: int  # env: GRAPHRAG_ARTIFACT_CACHE_TTL_SECONDS (default: 300)
    graph_query_cache_ttl_seconds: int  # env: GRAPH_QUERY_CACHE_TTL_SECONDS (default: 900)
    graph_sql_enabled: bool             # env: GRAPH_SQL_ENABLED (default: True)
    graph_sql_allowed_views: tuple[str, ...]  # env: GRAPH_SQL_ALLOWED_VIEWS
    neo4j_uri: str | None               # env: NEO4J_URI
    neo4j_username: str | None          # env: NEO4J_USERNAME
    neo4j_password: str | None          # env: NEO4J_PASSWORD
    neo4j_database: str | None          # env: NEO4J_DATABASE
    neo4j_timeout_seconds: int          # env: NEO4J_TIMEOUT_SECONDS (default: 15)

    # --- Data Analyst / Sandbox ---
    sandbox_docker_image: str           # env: SANDBOX_DOCKER_IMAGE (default: "agentic-chatbot-sandbox:py312")
    sandbox_timeout_seconds: int        # env: SANDBOX_TIMEOUT_SECONDS (default: 180)
    sandbox_memory_limit: str           # env: SANDBOX_MEMORY_LIMIT (default: "512m")
    data_analyst_max_steps: int         # env: DATA_ANALYST_MAX_STEPS (default: 10)
    data_analyst_skills_path: Path      # constructed: skills_dir / "data_analyst_agent.md"
    data_analyst_nlp_chat_model: str    # env: DATA_ANALYST_NLP_CHAT_MODEL (default: "")
    data_analyst_nlp_batch_size: int    # env: DATA_ANALYST_NLP_BATCH_SIZE (default: 5)
    data_analyst_nlp_temperature: float # env: DATA_ANALYST_NLP_TEMPERATURE (default: 0.0)

    # --- Session Workspace ---
    workspace_dir: Path                 # env: WORKSPACE_DIR (default: data/workspaces)
    workspace_session_ttl_hours: int    # env: WORKSPACE_SESSION_TTL_HOURS (default: 24; 0=keep forever)

    # --- Skills retrieval / DB-first KB ---
    seed_demo_kb_on_startup: bool       # env: SEED_DEMO_KB_ON_STARTUP (default: True)
    skill_search_top_k: int             # env: SKILL_SEARCH_TOP_K (default: 4)
    skill_context_max_chars: int        # env: SKILL_CONTEXT_MAX_CHARS (default: 4000)
    executable_skills_enabled: bool     # env: EXECUTABLE_SKILLS_ENABLED (default: False)
    skill_packs_hot_reload_enabled: bool  # env: SKILL_PACKS_HOT_RELOAD_ENABLED (default: False)
    skill_packs_hot_reload_interval_seconds: int  # env: SKILL_PACKS_HOT_RELOAD_INTERVAL_SECONDS
    team_mailbox_enabled: bool          # env: TEAM_MAILBOX_ENABLED (default: False)
    team_mailbox_max_channels_per_session: int  # env: TEAM_MAILBOX_MAX_CHANNELS_PER_SESSION
    team_mailbox_max_open_messages_per_channel: int  # env: TEAM_MAILBOX_MAX_OPEN_MESSAGES_PER_CHANNEL
    team_mailbox_claim_limit: int       # env: TEAM_MAILBOX_CLAIM_LIMIT
    runtime_job_retention_hours: int    # env: RUNTIME_JOB_RETENTION_HOURS (default: 168)
    max_worker_concurrency: int         # env: MAX_WORKER_CONCURRENCY (default: 4)
    worker_scheduler_enabled: bool      # env: WORKER_SCHEDULER_ENABLED (default: True)
    worker_scheduler_urgent_reserved_slots: int  # env: WORKER_SCHEDULER_URGENT_RESERVED_SLOTS (default: 1)
    worker_scheduler_tenant_budget_tokens_per_minute: int  # env: WORKER_SCHEDULER_TENANT_BUDGET_TOKENS_PER_MINUTE (default: 24000)
    worker_scheduler_tenant_budget_burst_tokens: int  # env: WORKER_SCHEDULER_TENANT_BUDGET_BURST_TOKENS (default: 48000)
    enable_coordinator_mode: bool       # env: ENABLE_COORDINATOR_MODE (default: False)
    max_revision_rounds: int            # env: MAX_REVISION_ROUNDS (default: 4)
    clarification_sensitivity: int      # env: CLARIFICATION_SENSITIVITY (default: 50)
    deep_rag_default_mode: str          # env: DEEP_RAG_DEFAULT_MODE (default: "auto")
    deep_rag_max_parallel_lanes: int    # env: DEEP_RAG_MAX_PARALLEL_LANES (default: 3)
    deep_rag_full_read_chunk_threshold: int  # env: DEEP_RAG_FULL_READ_CHUNK_THRESHOLD (default: 24)
    deep_rag_sync_reflection_rounds: int  # env: DEEP_RAG_SYNC_REFLECTION_ROUNDS (default: 1)
    deep_rag_background_threshold: int  # env: DEEP_RAG_BACKGROUND_THRESHOLD (default: 4)
    runtime_events_enabled: bool        # env: RUNTIME_EVENTS_ENABLED (default: True)
    session_hydrate_window_messages: int  # env: SESSION_HYDRATE_WINDOW_MESSAGES (default: 40)
    session_transcript_page_size: int     # env: SESSION_TRANSCRIPT_PAGE_SIZE (default: 100)
    context_budget_enabled: bool       # env: CONTEXT_BUDGET_ENABLED (default: False)
    context_window_tokens: int         # env: CONTEXT_WINDOW_TOKENS (default: 32768)
    context_target_ratio: float        # env: CONTEXT_TARGET_RATIO (default: 0.72)
    context_autocompact_threshold: float  # env: CONTEXT_AUTOCOMPACT_THRESHOLD (default: 0.85)
    context_tool_result_max_tokens: int   # env: CONTEXT_TOOL_RESULT_MAX_TOKENS (default: 2000)
    context_tool_results_total_tokens: int  # env: CONTEXT_TOOL_RESULTS_TOTAL_TOKENS (default: 8000)
    context_microcompact_target_tokens: int  # env: CONTEXT_MICROCOMPACT_TARGET_TOKENS (default: 2400)
    context_compact_recent_messages: int  # env: CONTEXT_COMPACT_RECENT_MESSAGES (default: 12)
    context_restore_recent_files: int  # env: CONTEXT_RESTORE_RECENT_FILES (default: 10)
    context_restore_recent_skills: int  # env: CONTEXT_RESTORE_RECENT_SKILLS (default: 6)
    agent_definitions_json: str         # env: AGENT_DEFINITIONS_JSON (deprecated / ignored by live runtime)

    # --- Control panel ---
    control_panel_enabled: bool
    control_panel_admin_token: str | None
    control_panel_overlay_dir: Path
    control_panel_runtime_env_path: Path
    control_panel_prompt_overlays_dir: Path
    control_panel_agent_overlays_dir: Path
    control_panel_audit_log_path: Path
    control_panel_static_dir: Path


def _getenv(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _as_int(name: str, default: int) -> int:
    v = _getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _as_optional_int(name: str) -> int | None:
    v = _getenv(name)
    if v is None:
        return None
    try:
        parsed = int(v)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _as_float(name: str, default: float) -> float:
    v = _getenv(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _as_bool(name: str, default: bool) -> bool:
    v = _getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")


def _as_router_mode(name: str, default: str) -> str:
    value = str(_getenv(name, default) or default).strip().lower()
    return value if value in {"hybrid", "llm_only"} else default


def _as_memory_manager_mode(name: str, default: str) -> str:
    value = str(_getenv(name, default) or default).strip().lower()
    return value if value in {"shadow", "selector", "live"} else default


def _resolve_path(raw: str, *, base: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _as_path_tuple(name: str, *, base: Path) -> tuple[Path, ...]:
    raw = _getenv(name, "")
    if raw is None:
        return ()

    items: list[Path] = []
    seen: set[Path] = set()
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        resolved = _resolve_path(value, base=base)
        if resolved in seen:
            continue
        seen.add(resolved)
        items.append(resolved)
    return tuple(items)


def _as_str_tuple(name: str) -> tuple[str, ...]:
    raw = _getenv(name, "")
    if raw is None:
        return ()
    items: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        value = str(part or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        items.append(value)
    return tuple(items)


def _normalise_agent_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def _as_agent_model_override_map(*, kind: str) -> dict[str, str]:
    prefix = "AGENT_"
    suffix = f"_{kind}"
    overrides: dict[str, str] = {}

    for env_name, raw_value in os.environ.items():
        if not env_name.startswith(prefix) or not env_name.endswith(suffix):
            continue
        clean_value = str(raw_value or "").strip()
        if not clean_value:
            continue
        agent_token = env_name[len(prefix) : -len(suffix)]
        agent_name = _normalise_agent_name(agent_token)
        if not agent_name:
            continue
        overrides[agent_name] = clean_value

    return overrides


def _as_agent_int_override_map(*, kind: str) -> dict[str, int]:
    prefix = "AGENT_"
    suffix = f"_{kind}"
    overrides: dict[str, int] = {}

    for env_name, raw_value in os.environ.items():
        if not env_name.startswith(prefix) or not env_name.endswith(suffix):
            continue
        try:
            clean_value = int(str(raw_value or "").strip())
        except ValueError:
            continue
        if clean_value <= 0:
            continue
        agent_token = env_name[len(prefix) : -len(suffix)]
        agent_name = _normalise_agent_name(agent_token)
        if not agent_name:
            continue
        overrides[agent_name] = clean_value

    return overrides


def _apply_env_overrides(env_overrides: dict[str, str | None] | None) -> None:
    for name, raw_value in dict(env_overrides or {}).items():
        key = str(name or "").strip()
        if not key:
            continue
        if raw_value is None:
            os.environ.pop(key, None)
            continue
        os.environ[key] = str(raw_value)


def runtime_settings_diagnostics(settings: Settings) -> dict[str, object]:
    fingerprint_payload = {
        "llm_provider": str(getattr(settings, "llm_provider", "") or ""),
        "judge_provider": str(getattr(settings, "judge_provider", "") or ""),
        "embeddings_provider": str(getattr(settings, "embeddings_provider", "") or ""),
        "ollama_chat_model": str(getattr(settings, "ollama_chat_model", "") or ""),
        "ollama_judge_model": str(getattr(settings, "ollama_judge_model", "") or ""),
        "ollama_embed_model": str(getattr(settings, "ollama_embed_model", "") or ""),
        "graphrag_chat_model": str(getattr(settings, "graphrag_chat_model", "") or ""),
        "graphrag_index_chat_model": str(getattr(settings, "graphrag_index_chat_model", "") or ""),
        "graphrag_community_report_mode": str(
            getattr(settings, "graphrag_community_report_mode", "") or ""
        ),
        "graphrag_community_report_chat_model": str(
            getattr(settings, "graphrag_community_report_chat_model", "") or ""
        ),
        "graphrag_embed_model": str(getattr(settings, "graphrag_embed_model", "") or ""),
        "graph_backend": str(getattr(settings, "graph_backend", "") or ""),
        "graph_search_enabled": bool(getattr(settings, "graph_search_enabled", False)),
        "graph_ingest_enabled": bool(getattr(settings, "graph_ingest_enabled", False)),
        "llm_router_mode": str(getattr(settings, "llm_router_mode", "") or ""),
        "llm_router_confidence_threshold": float(
            getattr(settings, "llm_router_confidence_threshold", 0.0) or 0.0
        ),
        "max_agent_steps": int(getattr(settings, "max_agent_steps", 0) or 0),
        "max_tool_calls": int(getattr(settings, "max_tool_calls", 0) or 0),
        "max_parallel_tool_calls": int(getattr(settings, "max_parallel_tool_calls", 0) or 0),
        "deferred_tool_discovery_enabled": bool(getattr(settings, "deferred_tool_discovery_enabled", False)),
        "deferred_tool_discovery_top_k": int(getattr(settings, "deferred_tool_discovery_top_k", 0) or 0),
        "deferred_tool_discovery_require_search": bool(getattr(settings, "deferred_tool_discovery_require_search", True)),
        "mcp_tool_plane_enabled": bool(getattr(settings, "mcp_tool_plane_enabled", False)),
        "mcp_user_self_service_enabled": bool(getattr(settings, "mcp_user_self_service_enabled", True)),
        "mcp_require_https": bool(getattr(settings, "mcp_require_https", True)),
        "mcp_allow_private_network": bool(getattr(settings, "mcp_allow_private_network", False)),
        "mcp_connection_timeout_seconds": int(getattr(settings, "mcp_connection_timeout_seconds", 0) or 0),
        "mcp_tool_call_timeout_seconds": int(getattr(settings, "mcp_tool_call_timeout_seconds", 0) or 0),
        "mcp_catalog_refresh_seconds": int(getattr(settings, "mcp_catalog_refresh_seconds", 0) or 0),
        "mcp_secret_configured": bool(getattr(settings, "mcp_secret_encryption_key", "") or ""),
        "max_rag_agent_steps": int(getattr(settings, "max_rag_agent_steps", 0) or 0),
        "max_parallel_collection_probes": int(getattr(settings, "max_parallel_collection_probes", 0) or 0),
        "max_collection_discovery_collections": int(
            getattr(settings, "max_collection_discovery_collections", 0) or 0
        ),
        "planner_max_tasks": int(getattr(settings, "planner_max_tasks", 0) or 0),
        "skill_context_max_chars": int(getattr(settings, "skill_context_max_chars", 0) or 0),
        "team_mailbox_enabled": bool(getattr(settings, "team_mailbox_enabled", False)),
        "team_mailbox_max_channels_per_session": int(getattr(settings, "team_mailbox_max_channels_per_session", 0) or 0),
        "team_mailbox_max_open_messages_per_channel": int(getattr(settings, "team_mailbox_max_open_messages_per_channel", 0) or 0),
        "team_mailbox_claim_limit": int(getattr(settings, "team_mailbox_claim_limit", 0) or 0),
        "context_budget_enabled": bool(getattr(settings, "context_budget_enabled", False)),
        "context_window_tokens": int(getattr(settings, "context_window_tokens", 0) or 0),
        "context_target_ratio": float(getattr(settings, "context_target_ratio", 0.0) or 0.0),
        "context_autocompact_threshold": float(
            getattr(settings, "context_autocompact_threshold", 0.0) or 0.0
        ),
        "docling_enabled": bool(getattr(settings, "docling_enabled", False)),
        "ocr_enabled": bool(getattr(settings, "ocr_enabled", False)),
        "retrieval_decomposition_enabled": bool(
            getattr(settings, "retrieval_decomposition_enabled", False)
        ),
        "entity_linking_enabled": bool(getattr(settings, "entity_linking_enabled", False)),
        "section_first_retrieval_enabled": bool(
            getattr(settings, "section_first_retrieval_enabled", False)
        ),
        "retrieval_quality_verifier_enabled": bool(
            getattr(settings, "retrieval_quality_verifier_enabled", False)
        ),
        "control_panel_runtime_env_path": str(
            getattr(settings, "control_panel_runtime_env_path", "") or ""
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "process_started_at": _PROCESS_STARTED_AT,
        "settings_fingerprint": fingerprint,
        "loaded_overlay_env_path": str(
            getattr(settings, "control_panel_runtime_env_path", "") or ""
        ),
        "providers": {
            "llm_provider": fingerprint_payload["llm_provider"],
            "judge_provider": fingerprint_payload["judge_provider"],
            "embeddings_provider": fingerprint_payload["embeddings_provider"],
        },
        "models": {
            "chat_model": fingerprint_payload["ollama_chat_model"],
            "judge_model": fingerprint_payload["ollama_judge_model"],
            "embedding_model": fingerprint_payload["ollama_embed_model"],
            "graphrag_chat_model": fingerprint_payload["graphrag_chat_model"],
            "graphrag_index_chat_model": fingerprint_payload["graphrag_index_chat_model"],
            "graphrag_community_report_mode": fingerprint_payload["graphrag_community_report_mode"],
            "graphrag_community_report_chat_model": fingerprint_payload["graphrag_community_report_chat_model"],
            "graphrag_embedding_model": fingerprint_payload["graphrag_embed_model"],
        },
    }


def load_settings(
    dotenv_path: str | None = None,
    *,
    env_overrides: dict[str, str | None] | None = None,
) -> Settings:
    """Load settings from environment (and optional .env)."""

    original_env = dict(os.environ)
    load_dotenv(dotenv_path=dotenv_path)

    # config.py -> agentic_chatbot -> src -> repo_root
    project_root = Path(__file__).resolve().parents[2]
    initial_data_dir = Path(_getenv("DATA_DIR", str(project_root / "data")))
    initial_overlay_dir = Path(
        _getenv(
            "CONTROL_PANEL_OVERLAY_DIR",
            str(initial_data_dir / "control_panel" / "overlays"),
        )
    )
    initial_runtime_env_path = Path(
        _getenv(
            "CONTROL_PANEL_RUNTIME_ENV_PATH",
            str(initial_overlay_dir / "runtime.env"),
        )
    )
    load_dotenv(dotenv_path=initial_runtime_env_path, override=True)
    for name, raw_value in original_env.items():
        os.environ[name] = raw_value
    _apply_env_overrides(env_overrides)

    data_dir = Path(_getenv("DATA_DIR", str(project_root / "data")))

    # Backends
    database_backend = str(_getenv("DATABASE_BACKEND", "postgres")).lower()
    vector_store_backend = str(_getenv("VECTOR_STORE_BACKEND", "pgvector")).lower()
    object_store_backend = str(_getenv("OBJECT_STORE_BACKEND", "local")).lower()
    skills_backend = str(_getenv("SKILLS_BACKEND", "local")).lower()
    prompts_backend = str(_getenv("PROMPTS_BACKEND", "local")).lower()

    llm_provider = str(_getenv("LLM_PROVIDER", "ollama")).lower()
    embeddings_provider = str(_getenv("EMBEDDINGS_PROVIDER", llm_provider)).lower()
    judge_provider = str(_getenv("JUDGE_PROVIDER", llm_provider)).lower()

    # Ollama
    ollama_base_url = str(_getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_chat_model = str(_getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b"))
    ollama_embed_model = str(_getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"))
    ollama_judge_model = str(_getenv("OLLAMA_JUDGE_MODEL", ollama_chat_model))
    ollama_temperature = _as_float("OLLAMA_TEMPERATURE", 0.2)
    ollama_num_predict = _as_optional_int("OLLAMA_NUM_PREDICT")
    demo_ollama_num_predict = _as_optional_int("DEMO_OLLAMA_NUM_PREDICT")
    chat_max_output_tokens = _as_optional_int("CHAT_MAX_OUTPUT_TOKENS")
    demo_chat_max_output_tokens = _as_optional_int("DEMO_CHAT_MAX_OUTPUT_TOKENS")
    judge_max_output_tokens = _as_optional_int("JUDGE_MAX_OUTPUT_TOKENS")
    agent_chat_model_overrides = _as_agent_model_override_map(kind="CHAT_MODEL")
    agent_judge_model_overrides = _as_agent_model_override_map(kind="JUDGE_MODEL")
    agent_chat_max_output_tokens = _as_agent_int_override_map(kind="MAX_OUTPUT_TOKENS")

    # Azure OpenAI (optional)
    azure_openai_api_key = _getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = _getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version = _getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    azure_openai_chat_deployment = _getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", _getenv("AZURE_OPENAI_DEPLOYMENT"))
    azure_openai_judge_deployment = _getenv("AZURE_OPENAI_JUDGE_DEPLOYMENT", azure_openai_chat_deployment)
    azure_openai_embed_deployment = _getenv(
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
        _getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
    )
    azure_temperature = _as_float("AZURE_TEMPERATURE", 0.2)
    nvidia_openai_endpoint = _getenv("NVIDIA_OPENAI_ENDPOINT")
    nvidia_api_token = _getenv("NVIDIA_API_TOKEN", _getenv("Token"))
    nvidia_chat_model = _getenv("NVIDIA_CHAT_MODEL")
    nvidia_judge_model = _getenv("NVIDIA_JUDGE_MODEL", nvidia_chat_model)
    nvidia_temperature = _as_float("NVIDIA_TEMPERATURE", 0.0)
    judge_temperature = _as_float("JUDGE_TEMPERATURE", 0.0)
    http2_enabled = _as_bool("HTTP2_ENABLED", True)
    ssl_verify = _as_bool("SSL_VERIFY", True)
    ssl_cert_raw = _getenv("SSL_CERT_FILE", _getenv("APP_SSL_CERT_FILE"))
    ssl_cert_file = _resolve_path(ssl_cert_raw, base=project_root) if ssl_cert_raw else None
    tiktoken_enabled = _as_bool("TIKTOKEN_ENABLED", True)
    tiktoken_cache_raw = _getenv("TIKTOKEN_CACHE_DIR")
    tiktoken_cache_dir = _resolve_path(tiktoken_cache_raw, base=project_root) if tiktoken_cache_raw else None

    if ssl_cert_file and ssl_verify:
        # Ensure non-httpx paths (e.g. tiktoken/urllib) trust corporate CA bundle.
        os.environ["SSL_CERT_FILE"] = str(ssl_cert_file)
        os.environ["REQUESTS_CA_BUNDLE"] = str(ssl_cert_file)
        os.environ["CURL_CA_BUNDLE"] = str(ssl_cert_file)

    if tiktoken_cache_dir:
        tiktoken_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

    # Runtime
    max_agent_steps = _as_int("MAX_AGENT_STEPS", 10)
    max_tool_calls = _as_int("MAX_TOOL_CALLS", 12)
    max_parallel_tool_calls = max(1, _as_int("MAX_PARALLEL_TOOL_CALLS", 4))
    deferred_tool_discovery_enabled = _as_bool("DEFERRED_TOOL_DISCOVERY_ENABLED", False)
    deferred_tool_discovery_top_k = max(1, _as_int("DEFERRED_TOOL_DISCOVERY_TOP_K", 8))
    deferred_tool_discovery_require_search = _as_bool("DEFERRED_TOOL_DISCOVERY_REQUIRE_SEARCH", True)
    mcp_tool_plane_enabled = _as_bool("MCP_TOOL_PLANE_ENABLED", False)
    mcp_user_self_service_enabled = _as_bool("MCP_USER_SELF_SERVICE_ENABLED", True)
    mcp_require_https = _as_bool("MCP_REQUIRE_HTTPS", True)
    mcp_allow_private_network = _as_bool("MCP_ALLOW_PRIVATE_NETWORK", False)
    mcp_connection_timeout_seconds = max(1, _as_int("MCP_CONNECTION_TIMEOUT_SECONDS", 15))
    mcp_tool_call_timeout_seconds = max(1, _as_int("MCP_TOOL_CALL_TIMEOUT_SECONDS", 60))
    mcp_catalog_refresh_seconds = max(1, _as_int("MCP_CATALOG_REFRESH_SECONDS", 3600))
    mcp_secret_encryption_key = _getenv("MCP_SECRET_ENCRYPTION_KEY")
    worker_job_wait_timeout_seconds = _as_int("WORKER_JOB_WAIT_TIMEOUT_SECONDS", 600)
    llm_http_timeout_seconds = _as_int("LLM_HTTP_TIMEOUT_SECONDS", 120)
    llm_http_connect_timeout_seconds = _as_int("LLM_HTTP_CONNECT_TIMEOUT_SECONDS", 20)

    # RAG
    rag_top_k_vector = _as_int("RAG_TOPK_VECTOR", 12)
    rag_top_k_keyword = _as_int("RAG_TOPK_BM25", 12)
    rag_max_retries = _as_int("RAG_MAX_RETRIES", 2)
    rag_min_evidence_chunks = _as_int("RAG_MIN_EVIDENCE_CHUNKS", 2)
    max_rag_agent_steps = _as_int("MAX_RAG_AGENT_STEPS", 8)
    max_parallel_collection_probes = max(1, _as_int("MAX_PARALLEL_COLLECTION_PROBES", 4))
    max_collection_discovery_collections = max(1, _as_int("MAX_COLLECTION_DISCOVERY_COLLECTIONS", 25))

    # Text splitting
    chunk_size = _as_int("CHUNK_SIZE", 900)
    chunk_overlap = _as_int("CHUNK_OVERLAP", 150)

    # PostgreSQL
    pg_dsn = str(_getenv("PG_DSN", "postgresql://localhost:5432/ragdb"))
    embedding_dim = _as_int("EMBEDDING_DIM", 768)


    # Paths
    kb_dir = Path(_getenv("KB_DIR", str(data_dir / "kb")))
    kb_extra_dirs = _as_path_tuple("KB_EXTRA_DIRS", base=project_root)
    uploads_dir = Path(_getenv("UPLOADS_DIR", str(data_dir / "uploads")))
    kb_source_uri = str(_getenv("KB_SOURCE_URI", f"file://{kb_dir}"))
    uploads_source_uri = str(_getenv("UPLOADS_SOURCE_URI", f"file://{uploads_dir}"))
    default_collection_id = str(_getenv("DEFAULT_COLLECTION_ID", "default"))
    skill_packs_dir = Path(_getenv("SKILL_PACKS_DIR", str(data_dir / "skill_packs")))
    runtime_dir = Path(_getenv("RUNTIME_DIR", str(data_dir / "runtime")))
    agents_dir = Path(_getenv("AGENTS_DIR", str(data_dir / "agents")))

    skills_dir = Path(_getenv("SKILLS_DIR", str(data_dir / "skills")))
    prompts_dir = Path(_getenv("PROMPTS_DIR", str(data_dir / "prompts")))

    shared_skills_path = Path(_getenv("SHARED_SKILLS_PATH", str(skills_dir / "skills.md")))
    general_agent_skills_path = Path(_getenv("GENERAL_AGENT_SKILLS_PATH", str(skills_dir / "general_agent.md")))
    rag_agent_skills_path = Path(_getenv("RAG_AGENT_SKILLS_PATH", str(skills_dir / "rag_agent.md")))
    supervisor_agent_skills_path = Path(_getenv("SUPERVISOR_AGENT_SKILLS_PATH", str(skills_dir / "supervisor_agent.md")))
    utility_agent_skills_path = Path(_getenv("UTILITY_AGENT_SKILLS_PATH", str(skills_dir / "utility_agent.md")))
    basic_chat_skills_path = Path(_getenv("BASIC_CHAT_SKILLS_PATH", str(skills_dir / "basic_chat.md")))
    planner_agent_skills_path = Path(_getenv("PLANNER_AGENT_SKILLS_PATH", str(skills_dir / "planner_agent.md")))
    finalizer_agent_skills_path = Path(_getenv("FINALIZER_AGENT_SKILLS_PATH", str(skills_dir / "finalizer_agent.md")))

    judge_grading_prompt_path = Path(_getenv("JUDGE_GRADING_PROMPT_PATH", str(prompts_dir / "judge_grading.txt")))
    judge_rewrite_prompt_path = Path(_getenv("JUDGE_REWRITE_PROMPT_PATH", str(prompts_dir / "judge_rewrite.txt")))
    grounded_answer_prompt_path = Path(_getenv("GROUNDED_ANSWER_PROMPT_PATH", str(prompts_dir / "grounded_answer.txt")))
    rag_synthesis_prompt_path = Path(_getenv("RAG_SYNTHESIS_PROMPT_PATH", str(prompts_dir / "rag_synthesis.txt")))

    # Runtime compatibility
    agent_runtime_mode = str(_getenv("AGENT_RUNTIME_MODE", "") or "").lower()
    planner_max_tasks = _as_int("PLANNER_MAX_TASKS", 8)

    # Scratchpad
    clear_scratchpad_per_turn = _as_bool("CLEAR_SCRATCHPAD_PER_TURN", True)
    memory_enabled = _as_bool("MEMORY_ENABLED", True)
    memory_manager_mode = _as_memory_manager_mode("MEMORY_MANAGER_MODE", "shadow")
    memory_selector_model = str(_getenv("MEMORY_SELECTOR_MODEL", "") or "").strip()
    memory_writer_model = str(_getenv("MEMORY_WRITER_MODEL", "") or "").strip()
    memory_candidate_top_k = max(4, _as_int("MEMORY_CANDIDATE_TOP_K", 16))
    memory_context_token_budget = max(400, _as_int("MEMORY_CONTEXT_TOKEN_BUDGET", 1600))
    memory_shadow_mode = _as_bool("MEMORY_SHADOW_MODE", False)

    # OCR
    docling_enabled    = _as_bool("DOCLING_ENABLED", False)
    ocr_enabled        = _as_bool("USE_PADDLE_OCR", True)
    ocr_language       = str(_getenv("OCR_LANGUAGE", "en"))
    ocr_use_gpu        = _as_bool("OCR_USE_GPU", False)
    ocr_min_page_chars = _as_int("OCR_MIN_PAGE_CHARS", 50)

    # Langfuse
    langfuse_host = _getenv("LANGFUSE_HOST", "http://localhost:3000")
    langfuse_public_key = _getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = _getenv("LANGFUSE_SECRET_KEY")
    langfuse_debug = _as_bool("LANGFUSE_DEBUG", False)

    # Context defaults
    default_tenant_id = str(_getenv("DEFAULT_TENANT_ID", "local-dev"))
    default_user_id = str(_getenv("DEFAULT_USER_ID", "local-cli"))
    default_conversation_id = str(_getenv("DEFAULT_CONVERSATION_ID", "local-session"))

    # Gateway model config
    gateway_model_id = str(_getenv("GATEWAY_MODEL_ID", "enterprise-agent"))
    gateway_shared_bearer_token = _getenv("GATEWAY_SHARED_BEARER_TOKEN", "")
    download_url_secret = _getenv("DOWNLOAD_URL_SECRET", "")
    download_url_ttl_seconds = _as_int("DOWNLOAD_URL_TTL_SECONDS", 900)
    connector_secret_api_key = _getenv(
        "CONNECTOR_SECRET_API_KEY",
        gateway_shared_bearer_token,
    )
    connector_publishable_api_key = _getenv("CONNECTOR_PUBLISHABLE_API_KEY", "")
    connector_allowed_origins = _as_str_tuple("CONNECTOR_ALLOWED_ORIGINS")
    connector_publishable_rate_limit_per_minute = _as_int(
        "CONNECTOR_PUBLISHABLE_RATE_LIMIT_PER_MINUTE",
        60,
    )
    authz_enabled = _as_bool("AUTHZ_ENABLED", False)

    # LLM Router
    llm_router_enabled = _as_bool("LLM_ROUTER_ENABLED", True)
    llm_router_mode = _as_router_mode("LLM_ROUTER_MODE", "hybrid")
    llm_router_confidence_threshold = _as_float("LLM_ROUTER_CONFIDENCE_THRESHOLD", 0.70)
    router_patterns_path = _resolve_path(
        _getenv("ROUTER_PATTERNS_PATH", str(data_dir / "router" / "intent_patterns.json")) or "",
        base=project_root,
    )
    router_feedback_enabled = _as_bool("ROUTER_FEEDBACK_ENABLED", True)
    router_feedback_rephrase_window_seconds = max(
        60,
        _as_int("ROUTER_FEEDBACK_REPHRASE_WINDOW_SECONDS", 600),
    )
    router_feedback_neutral_sample_rate = max(
        0.0,
        min(1.0, _as_float("ROUTER_FEEDBACK_NEUTRAL_SAMPLE_RATE", 0.10)),
    )
    router_feedback_tenant_daily_review_cap = max(
        1,
        _as_int("ROUTER_FEEDBACK_TENANT_DAILY_REVIEW_CAP", 25),
    )
    router_retrain_governance = str(
        _getenv("ROUTER_RETRAIN_GOVERNANCE", "human_reviewed") or "human_reviewed"
    ).strip().lower()

    # Circuit breaker
    llm_circuit_breaker_enabled = _as_bool("LLM_CIRCUIT_BREAKER_ENABLED", True)
    llm_circuit_breaker_window_size = _as_int("LLM_CIRCUIT_BREAKER_WINDOW_SIZE", 20)
    llm_circuit_breaker_min_samples = _as_int("LLM_CIRCUIT_BREAKER_MIN_SAMPLES", 6)
    llm_circuit_breaker_error_rate_threshold = _as_float("LLM_CIRCUIT_BREAKER_ERROR_RATE_THRESHOLD", 0.50)
    llm_circuit_breaker_consecutive_failures = _as_int("LLM_CIRCUIT_BREAKER_CONSECUTIVE_FAILURES", 3)
    llm_circuit_breaker_open_seconds = _as_int("LLM_CIRCUIT_BREAKER_OPEN_SECONDS", 30)

    # Web search fallback
    tavily_api_key = _getenv("TAVILY_API_KEY")
    web_search_enabled = _as_bool("WEB_SEARCH_ENABLED", False)

    # GraphRAG / Neo4j
    graph_search_enabled = _as_bool("GRAPH_SEARCH_ENABLED", False)
    graph_ingest_enabled = _as_bool("GRAPH_INGEST_ENABLED", graph_search_enabled)
    graph_backend = str(_getenv("GRAPH_BACKEND", "microsoft_graphrag") or "microsoft_graphrag").strip().lower()
    graph_import_enabled = _as_bool("GRAPH_IMPORT_ENABLED", True)
    graph_source_planning_enabled = _as_bool("GRAPH_SOURCE_PLANNING_ENABLED", True)
    retrieval_decomposition_enabled = _as_bool("RETRIEVAL_DECOMPOSITION_ENABLED", False)
    entity_linking_enabled = _as_bool("ENTITY_LINKING_ENABLED", False)
    section_first_retrieval_enabled = _as_bool("SECTION_FIRST_RETRIEVAL_ENABLED", False)
    retrieval_quality_verifier_enabled = _as_bool("RETRIEVAL_QUALITY_VERIFIER_ENABLED", False)
    graphrag_projects_dir = _resolve_path(
        _getenv("GRAPHRAG_PROJECTS_DIR", str(data_dir / "graphrag" / "projects")) or "",
        base=project_root,
    )
    graphrag_use_container = _as_bool("GRAPHRAG_USE_CONTAINER", False)
    graphrag_container_image = str(_getenv("GRAPHRAG_CONTAINER_IMAGE", "graphrag:latest") or "graphrag:latest")
    graphrag_cli_command = str(_getenv("GRAPHRAG_CLI_COMMAND", "graphrag") or "graphrag").strip()
    graphrag_llm_provider = str(_getenv("GRAPHRAG_LLM_PROVIDER", "openai") or "openai").strip().lower()
    graphrag_base_url = str(
        _getenv(
            "GRAPHRAG_BASE_URL",
            f"{ollama_base_url.rstrip('/')}/v1" if llm_provider == "ollama" and ollama_base_url else "",
        )
        or ""
    ).strip()
    graphrag_api_key = _getenv(
        "GRAPHRAG_API_KEY",
        "ollama" if llm_provider == "ollama" and ollama_base_url else "",
    )
    graphrag_chat_model = str(
        _getenv(
            "GRAPHRAG_CHAT_MODEL",
            ollama_chat_model if llm_provider == "ollama" else "",
        )
        or ""
    ).strip()
    graphrag_index_chat_model = str(
        _getenv("GRAPHRAG_INDEX_CHAT_MODEL", graphrag_chat_model)
        or graphrag_chat_model
        or ""
    ).strip()
    graphrag_community_report_mode = str(
        _getenv("GRAPHRAG_COMMUNITY_REPORT_MODE", "text") or "text"
    ).strip().lower()
    if graphrag_community_report_mode not in {"text", "graph"}:
        graphrag_community_report_mode = "text"
    graphrag_community_report_chat_model = str(
        _getenv("GRAPHRAG_COMMUNITY_REPORT_CHAT_MODEL", graphrag_index_chat_model)
        or graphrag_index_chat_model
        or graphrag_chat_model
        or ""
    ).strip()
    graphrag_embed_model = str(
        _getenv(
            "GRAPHRAG_EMBED_MODEL",
            ollama_embed_model if embeddings_provider == "ollama" else "",
        )
        or ""
    ).strip()
    graphrag_concurrency = _as_int("GRAPHRAG_CONCURRENCY", 4)
    graphrag_timeout_seconds = _as_int("GRAPHRAG_TIMEOUT_SECONDS", 60)
    graphrag_request_timeout_seconds = max(
        30,
        _as_int("GRAPHRAG_REQUEST_TIMEOUT_SECONDS", graphrag_timeout_seconds),
    )
    graphrag_index_request_timeout_seconds = max(
        30,
        _as_int("GRAPHRAG_INDEX_REQUEST_TIMEOUT_SECONDS", graphrag_request_timeout_seconds),
    )
    graphrag_community_report_request_timeout_seconds = max(
        30,
        _as_int(
            "GRAPHRAG_COMMUNITY_REPORT_REQUEST_TIMEOUT_SECONDS",
            graphrag_index_request_timeout_seconds,
        ),
    )
    graphrag_community_report_max_input_length = max(
        500,
        _as_int("GRAPHRAG_COMMUNITY_REPORT_MAX_INPUT_LENGTH", 4000),
    )
    graphrag_community_report_max_length = max(
        200,
        _as_int("GRAPHRAG_COMMUNITY_REPORT_MAX_LENGTH", 1200),
    )
    graphrag_job_timeout_raw = _getenv("GRAPHRAG_JOB_TIMEOUT_SECONDS")
    if graphrag_job_timeout_raw is None:
        graphrag_job_timeout_seconds = max(graphrag_index_request_timeout_seconds, 600)
    else:
        try:
            graphrag_job_timeout_seconds = max(0, int(graphrag_job_timeout_raw))
        except ValueError:
            graphrag_job_timeout_seconds = max(graphrag_index_request_timeout_seconds, 600)
    graphrag_timeout_seconds = graphrag_request_timeout_seconds
    graphrag_stale_run_after_seconds = _as_int("GRAPHRAG_STALE_RUN_AFTER_SECONDS", 900)
    graphrag_default_query_method = str(_getenv("GRAPHRAG_DEFAULT_QUERY_METHOD", "local") or "local").strip().lower()
    graphrag_artifact_cache_ttl_seconds = _as_int("GRAPHRAG_ARTIFACT_CACHE_TTL_SECONDS", 300)
    graph_query_cache_ttl_seconds = _as_int("GRAPH_QUERY_CACHE_TTL_SECONDS", 900)
    graph_sql_enabled = _as_bool("GRAPH_SQL_ENABLED", True)
    graph_sql_allowed_views = _as_str_tuple("GRAPH_SQL_ALLOWED_VIEWS")
    neo4j_uri = _getenv("NEO4J_URI")
    neo4j_username = _getenv("NEO4J_USERNAME")
    neo4j_password = _getenv("NEO4J_PASSWORD")
    neo4j_database = _getenv("NEO4J_DATABASE")
    neo4j_timeout_seconds = _as_int("NEO4J_TIMEOUT_SECONDS", 15)

    # Data Analyst / Sandbox
    sandbox_docker_image = str(_getenv("SANDBOX_DOCKER_IMAGE", "agentic-chatbot-sandbox:py312"))
    sandbox_timeout_seconds = _as_int("SANDBOX_TIMEOUT_SECONDS", 180)
    sandbox_memory_limit = str(_getenv("SANDBOX_MEMORY_LIMIT", "512m"))
    data_analyst_max_steps = _as_int("DATA_ANALYST_MAX_STEPS", 10)
    data_analyst_skills_path = Path(_getenv("DATA_ANALYST_SKILLS_PATH", str(skills_dir / "data_analyst_agent.md")))
    data_analyst_nlp_chat_model = str(_getenv("DATA_ANALYST_NLP_CHAT_MODEL", "") or "")
    data_analyst_nlp_batch_size = _as_int("DATA_ANALYST_NLP_BATCH_SIZE", 5)
    data_analyst_nlp_temperature = _as_float("DATA_ANALYST_NLP_TEMPERATURE", 0.0)

    # Session Workspace
    workspace_dir = Path(_getenv("WORKSPACE_DIR", str(data_dir / "workspaces")))
    workspace_session_ttl_hours = _as_int("WORKSPACE_SESSION_TTL_HOURS", 24)
    seed_demo_kb_on_startup = _as_bool("SEED_DEMO_KB_ON_STARTUP", True)
    skill_search_top_k = _as_int("SKILL_SEARCH_TOP_K", 4)
    skill_context_max_chars = _as_int("SKILL_CONTEXT_MAX_CHARS", 4000)
    executable_skills_enabled = _as_bool("EXECUTABLE_SKILLS_ENABLED", False)
    skill_packs_hot_reload_enabled = _as_bool("SKILL_PACKS_HOT_RELOAD_ENABLED", False)
    skill_packs_hot_reload_interval_seconds = max(1, _as_int("SKILL_PACKS_HOT_RELOAD_INTERVAL_SECONDS", 5))
    team_mailbox_enabled = _as_bool("TEAM_MAILBOX_ENABLED", False)
    team_mailbox_max_channels_per_session = max(1, _as_int("TEAM_MAILBOX_MAX_CHANNELS_PER_SESSION", 8))
    team_mailbox_max_open_messages_per_channel = max(1, _as_int("TEAM_MAILBOX_MAX_OPEN_MESSAGES_PER_CHANNEL", 50))
    team_mailbox_claim_limit = max(1, _as_int("TEAM_MAILBOX_CLAIM_LIMIT", 8))
    runtime_job_retention_hours = _as_int("RUNTIME_JOB_RETENTION_HOURS", 168)
    max_worker_concurrency = _as_int("MAX_WORKER_CONCURRENCY", 6)
    worker_scheduler_enabled = _as_bool("WORKER_SCHEDULER_ENABLED", True)
    worker_scheduler_urgent_reserved_slots = max(
        0,
        _as_int("WORKER_SCHEDULER_URGENT_RESERVED_SLOTS", 1),
    )
    worker_scheduler_tenant_budget_tokens_per_minute = max(
        0,
        _as_int("WORKER_SCHEDULER_TENANT_BUDGET_TOKENS_PER_MINUTE", 24000),
    )
    worker_scheduler_tenant_budget_burst_tokens = max(
        worker_scheduler_tenant_budget_tokens_per_minute,
        _as_int("WORKER_SCHEDULER_TENANT_BUDGET_BURST_TOKENS", 48000),
    )
    enable_coordinator_mode = _as_bool("ENABLE_COORDINATOR_MODE", False)
    max_revision_rounds = _as_int("MAX_REVISION_ROUNDS", 8)
    clarification_sensitivity = normalize_clarification_sensitivity(
        _getenv("CLARIFICATION_SENSITIVITY"),
    )
    deep_rag_default_mode = normalize_deep_rag_mode(
        _getenv("DEEP_RAG_DEFAULT_MODE"),
        default="auto",
    )
    deep_rag_max_parallel_lanes = max(1, _as_int("DEEP_RAG_MAX_PARALLEL_LANES", 3))
    deep_rag_full_read_chunk_threshold = max(6, _as_int("DEEP_RAG_FULL_READ_CHUNK_THRESHOLD", 24))
    deep_rag_sync_reflection_rounds = max(1, _as_int("DEEP_RAG_SYNC_REFLECTION_ROUNDS", 1))
    deep_rag_background_threshold = max(2, _as_int("DEEP_RAG_BACKGROUND_THRESHOLD", 4))
    runtime_events_enabled = _as_bool("RUNTIME_EVENTS_ENABLED", True)
    session_hydrate_window_messages = _as_int("SESSION_HYDRATE_WINDOW_MESSAGES", 80)
    session_transcript_page_size = _as_int("SESSION_TRANSCRIPT_PAGE_SIZE", 200)
    context_budget_enabled = _as_bool("CONTEXT_BUDGET_ENABLED", False)
    context_window_tokens = max(1024, _as_int("CONTEXT_WINDOW_TOKENS", 32768))
    context_target_ratio = min(0.95, max(0.25, _as_float("CONTEXT_TARGET_RATIO", 0.72)))
    context_autocompact_threshold = min(0.98, max(0.40, _as_float("CONTEXT_AUTOCOMPACT_THRESHOLD", 0.85)))
    context_tool_result_max_tokens = max(128, _as_int("CONTEXT_TOOL_RESULT_MAX_TOKENS", 2000))
    context_tool_results_total_tokens = max(512, _as_int("CONTEXT_TOOL_RESULTS_TOTAL_TOKENS", 8000))
    context_microcompact_target_tokens = max(256, _as_int("CONTEXT_MICROCOMPACT_TARGET_TOKENS", 2400))
    context_compact_recent_messages = max(2, _as_int("CONTEXT_COMPACT_RECENT_MESSAGES", 12))
    context_restore_recent_files = max(0, _as_int("CONTEXT_RESTORE_RECENT_FILES", 10))
    context_restore_recent_skills = max(0, _as_int("CONTEXT_RESTORE_RECENT_SKILLS", 6))
    agent_definitions_json = str(_getenv("AGENT_DEFINITIONS_JSON", ""))

    # Control panel
    control_panel_enabled = _as_bool("CONTROL_PANEL_ENABLED", True)
    control_panel_admin_token = _getenv("CONTROL_PANEL_ADMIN_TOKEN", "")
    control_panel_overlay_dir = Path(
        _getenv("CONTROL_PANEL_OVERLAY_DIR", str(data_dir / "control_panel" / "overlays"))
    )
    control_panel_runtime_env_path = Path(
        _getenv(
            "CONTROL_PANEL_RUNTIME_ENV_PATH",
            str(control_panel_overlay_dir / "runtime.env"),
        )
    )
    control_panel_prompt_overlays_dir = Path(
        _getenv(
            "CONTROL_PANEL_PROMPT_OVERLAYS_DIR",
            str(control_panel_overlay_dir / "prompts"),
        )
    )
    control_panel_agent_overlays_dir = Path(
        _getenv(
            "CONTROL_PANEL_AGENT_OVERLAYS_DIR",
            str(control_panel_overlay_dir / "agents"),
        )
    )
    control_panel_audit_log_path = Path(
        _getenv(
            "CONTROL_PANEL_AUDIT_LOG_PATH",
            str(data_dir / "control_panel" / "audit" / "events.jsonl"),
        )
    )
    control_panel_static_dir = _resolve_path(
        _getenv("CONTROL_PANEL_STATIC_DIR", str(project_root / "control_panel" / "dist")) or "",
        base=project_root,
    )

    # Ensure backend values are in allowed sets.
    if database_backend not in {"postgres"}:
        raise ValueError(f"Unsupported DATABASE_BACKEND={database_backend!r}. Supported: postgres")
    if vector_store_backend not in {"pgvector"}:
        raise ValueError(f"Unsupported VECTOR_STORE_BACKEND={vector_store_backend!r}. Supported: pgvector")
    if object_store_backend not in {"local", "s3", "azure_blob"}:
        raise ValueError(f"Unsupported OBJECT_STORE_BACKEND={object_store_backend!r}. Supported: local, s3, azure_blob")
    if skills_backend not in {"local", "s3", "azure_blob"}:
        raise ValueError(f"Unsupported SKILLS_BACKEND={skills_backend!r}. Supported: local, s3, azure_blob")
    if prompts_backend not in {"local", "s3", "azure_blob"}:
        raise ValueError(f"Unsupported PROMPTS_BACKEND={prompts_backend!r}. Supported: local, s3, azure_blob")
    if graph_backend not in {"microsoft_graphrag", "neo4j"}:
        raise ValueError(
            f"Unsupported GRAPH_BACKEND={graph_backend!r}. Supported: microsoft_graphrag, neo4j"
        )
    # Ensure base local directories exist.
    for p in [
        data_dir,
        kb_dir,
        uploads_dir,
        skills_dir,
        prompts_dir,
        workspace_dir,
        skill_packs_dir,
        runtime_dir,
        agents_dir,
        router_patterns_path.parent,
        control_panel_overlay_dir,
        control_panel_prompt_overlays_dir,
        control_panel_agent_overlays_dir,
        control_panel_audit_log_path.parent,
        graphrag_projects_dir,
        *kb_extra_dirs,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return Settings(
        database_backend=database_backend,
        vector_store_backend=vector_store_backend,
        object_store_backend=object_store_backend,
        skills_backend=skills_backend,
        prompts_backend=prompts_backend,
        llm_provider=llm_provider,
        embeddings_provider=embeddings_provider,
        judge_provider=judge_provider,
        ollama_base_url=ollama_base_url,
        ollama_chat_model=ollama_chat_model,
        ollama_embed_model=ollama_embed_model,
        ollama_judge_model=ollama_judge_model,
        ollama_temperature=ollama_temperature,
        ollama_num_predict=ollama_num_predict,
        demo_ollama_num_predict=demo_ollama_num_predict,
        chat_max_output_tokens=chat_max_output_tokens,
        demo_chat_max_output_tokens=demo_chat_max_output_tokens,
        judge_max_output_tokens=judge_max_output_tokens,
        agent_chat_model_overrides=agent_chat_model_overrides,
        agent_judge_model_overrides=agent_judge_model_overrides,
        agent_chat_max_output_tokens=agent_chat_max_output_tokens,
        azure_openai_api_key=azure_openai_api_key,
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_api_version=azure_openai_api_version,
        azure_openai_chat_deployment=azure_openai_chat_deployment,
        azure_openai_judge_deployment=azure_openai_judge_deployment,
        azure_openai_embed_deployment=azure_openai_embed_deployment,
        azure_temperature=azure_temperature,
        nvidia_openai_endpoint=nvidia_openai_endpoint,
        nvidia_api_token=nvidia_api_token,
        nvidia_chat_model=nvidia_chat_model,
        nvidia_judge_model=nvidia_judge_model,
        nvidia_temperature=nvidia_temperature,
        judge_temperature=judge_temperature,
        http2_enabled=http2_enabled,
        ssl_verify=ssl_verify,
        ssl_cert_file=ssl_cert_file,
        tiktoken_enabled=tiktoken_enabled,
        tiktoken_cache_dir=tiktoken_cache_dir,
        max_agent_steps=max_agent_steps,
        max_tool_calls=max_tool_calls,
        max_parallel_tool_calls=max_parallel_tool_calls,
        deferred_tool_discovery_enabled=deferred_tool_discovery_enabled,
        deferred_tool_discovery_top_k=deferred_tool_discovery_top_k,
        deferred_tool_discovery_require_search=deferred_tool_discovery_require_search,
        mcp_tool_plane_enabled=mcp_tool_plane_enabled,
        mcp_user_self_service_enabled=mcp_user_self_service_enabled,
        mcp_require_https=mcp_require_https,
        mcp_allow_private_network=mcp_allow_private_network,
        mcp_connection_timeout_seconds=mcp_connection_timeout_seconds,
        mcp_tool_call_timeout_seconds=mcp_tool_call_timeout_seconds,
        mcp_catalog_refresh_seconds=mcp_catalog_refresh_seconds,
        mcp_secret_encryption_key=mcp_secret_encryption_key,
        worker_job_wait_timeout_seconds=worker_job_wait_timeout_seconds,
        llm_http_timeout_seconds=llm_http_timeout_seconds,
        llm_http_connect_timeout_seconds=llm_http_connect_timeout_seconds,
        rag_top_k_vector=rag_top_k_vector,
        rag_top_k_keyword=rag_top_k_keyword,
        rag_max_retries=rag_max_retries,
        rag_min_evidence_chunks=rag_min_evidence_chunks,
        max_rag_agent_steps=max_rag_agent_steps,
        max_parallel_collection_probes=max_parallel_collection_probes,
        max_collection_discovery_collections=max_collection_discovery_collections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pg_dsn=pg_dsn,
        embedding_dim=embedding_dim,
        project_root=project_root,
        data_dir=data_dir,
        kb_dir=kb_dir,
        kb_extra_dirs=kb_extra_dirs,
        uploads_dir=uploads_dir,
        kb_source_uri=kb_source_uri,
        uploads_source_uri=uploads_source_uri,
        default_collection_id=default_collection_id,
        skill_packs_dir=skill_packs_dir,
        runtime_dir=runtime_dir,
        agents_dir=agents_dir,
        skills_dir=skills_dir,
        prompts_dir=prompts_dir,
        shared_skills_path=shared_skills_path,
        general_agent_skills_path=general_agent_skills_path,
        rag_agent_skills_path=rag_agent_skills_path,
        supervisor_agent_skills_path=supervisor_agent_skills_path,
        utility_agent_skills_path=utility_agent_skills_path,
        basic_chat_skills_path=basic_chat_skills_path,
        planner_agent_skills_path=planner_agent_skills_path,
        finalizer_agent_skills_path=finalizer_agent_skills_path,
        judge_grading_prompt_path=judge_grading_prompt_path,
        judge_rewrite_prompt_path=judge_rewrite_prompt_path,
        grounded_answer_prompt_path=grounded_answer_prompt_path,
        rag_synthesis_prompt_path=rag_synthesis_prompt_path,
        agent_runtime_mode=agent_runtime_mode,
        planner_max_tasks=planner_max_tasks,
        clear_scratchpad_per_turn=clear_scratchpad_per_turn,
        memory_enabled=memory_enabled,
        memory_manager_mode=memory_manager_mode,
        memory_selector_model=memory_selector_model,
        memory_writer_model=memory_writer_model,
        memory_candidate_top_k=memory_candidate_top_k,
        memory_context_token_budget=memory_context_token_budget,
        memory_shadow_mode=memory_shadow_mode,
        docling_enabled=docling_enabled,
        ocr_enabled=ocr_enabled,
        ocr_language=ocr_language,
        ocr_use_gpu=ocr_use_gpu,
        ocr_min_page_chars=ocr_min_page_chars,
        langfuse_host=langfuse_host,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_debug=langfuse_debug,
        default_tenant_id=default_tenant_id,
        default_user_id=default_user_id,
        default_conversation_id=default_conversation_id,
        gateway_model_id=gateway_model_id,
        gateway_shared_bearer_token=gateway_shared_bearer_token,
        download_url_secret=download_url_secret,
        download_url_ttl_seconds=download_url_ttl_seconds,
        connector_secret_api_key=connector_secret_api_key,
        connector_publishable_api_key=connector_publishable_api_key,
        connector_allowed_origins=connector_allowed_origins,
        connector_publishable_rate_limit_per_minute=connector_publishable_rate_limit_per_minute,
        authz_enabled=authz_enabled,
        llm_router_enabled=llm_router_enabled,
        llm_router_mode=llm_router_mode,
        llm_router_confidence_threshold=llm_router_confidence_threshold,
        router_patterns_path=router_patterns_path,
        router_feedback_enabled=router_feedback_enabled,
        router_feedback_rephrase_window_seconds=router_feedback_rephrase_window_seconds,
        router_feedback_neutral_sample_rate=router_feedback_neutral_sample_rate,
        router_feedback_tenant_daily_review_cap=router_feedback_tenant_daily_review_cap,
        router_retrain_governance=router_retrain_governance,
        llm_circuit_breaker_enabled=llm_circuit_breaker_enabled,
        llm_circuit_breaker_window_size=llm_circuit_breaker_window_size,
        llm_circuit_breaker_min_samples=llm_circuit_breaker_min_samples,
        llm_circuit_breaker_error_rate_threshold=llm_circuit_breaker_error_rate_threshold,
        llm_circuit_breaker_consecutive_failures=llm_circuit_breaker_consecutive_failures,
        llm_circuit_breaker_open_seconds=llm_circuit_breaker_open_seconds,
        tavily_api_key=tavily_api_key,
        web_search_enabled=web_search_enabled,
        graph_search_enabled=graph_search_enabled,
        graph_ingest_enabled=graph_ingest_enabled,
        graph_backend=graph_backend,
        graph_import_enabled=graph_import_enabled,
        graph_source_planning_enabled=graph_source_planning_enabled,
        retrieval_decomposition_enabled=retrieval_decomposition_enabled,
        entity_linking_enabled=entity_linking_enabled,
        section_first_retrieval_enabled=section_first_retrieval_enabled,
        retrieval_quality_verifier_enabled=retrieval_quality_verifier_enabled,
        graphrag_projects_dir=graphrag_projects_dir,
        graphrag_use_container=graphrag_use_container,
        graphrag_container_image=graphrag_container_image,
        graphrag_cli_command=graphrag_cli_command,
        graphrag_llm_provider=graphrag_llm_provider,
        graphrag_base_url=graphrag_base_url,
        graphrag_api_key=graphrag_api_key,
        graphrag_chat_model=graphrag_chat_model,
        graphrag_index_chat_model=graphrag_index_chat_model,
        graphrag_community_report_mode=graphrag_community_report_mode,
        graphrag_community_report_chat_model=graphrag_community_report_chat_model,
        graphrag_embed_model=graphrag_embed_model,
        graphrag_concurrency=graphrag_concurrency,
        graphrag_request_timeout_seconds=graphrag_request_timeout_seconds,
        graphrag_index_request_timeout_seconds=graphrag_index_request_timeout_seconds,
        graphrag_community_report_request_timeout_seconds=graphrag_community_report_request_timeout_seconds,
        graphrag_community_report_max_input_length=graphrag_community_report_max_input_length,
        graphrag_community_report_max_length=graphrag_community_report_max_length,
        graphrag_job_timeout_seconds=graphrag_job_timeout_seconds,
        graphrag_timeout_seconds=graphrag_timeout_seconds,
        graphrag_stale_run_after_seconds=graphrag_stale_run_after_seconds,
        graphrag_default_query_method=graphrag_default_query_method,
        graphrag_artifact_cache_ttl_seconds=graphrag_artifact_cache_ttl_seconds,
        graph_query_cache_ttl_seconds=graph_query_cache_ttl_seconds,
        graph_sql_enabled=graph_sql_enabled,
        graph_sql_allowed_views=graph_sql_allowed_views,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        neo4j_timeout_seconds=neo4j_timeout_seconds,
        sandbox_docker_image=sandbox_docker_image,
        sandbox_timeout_seconds=sandbox_timeout_seconds,
        sandbox_memory_limit=sandbox_memory_limit,
        data_analyst_max_steps=data_analyst_max_steps,
        data_analyst_skills_path=data_analyst_skills_path,
        data_analyst_nlp_chat_model=data_analyst_nlp_chat_model,
        data_analyst_nlp_batch_size=data_analyst_nlp_batch_size,
        data_analyst_nlp_temperature=data_analyst_nlp_temperature,
        workspace_dir=workspace_dir,
        workspace_session_ttl_hours=workspace_session_ttl_hours,
        seed_demo_kb_on_startup=seed_demo_kb_on_startup,
        skill_search_top_k=skill_search_top_k,
        skill_context_max_chars=skill_context_max_chars,
        executable_skills_enabled=executable_skills_enabled,
        skill_packs_hot_reload_enabled=skill_packs_hot_reload_enabled,
        skill_packs_hot_reload_interval_seconds=skill_packs_hot_reload_interval_seconds,
        team_mailbox_enabled=team_mailbox_enabled,
        team_mailbox_max_channels_per_session=team_mailbox_max_channels_per_session,
        team_mailbox_max_open_messages_per_channel=team_mailbox_max_open_messages_per_channel,
        team_mailbox_claim_limit=team_mailbox_claim_limit,
        runtime_job_retention_hours=runtime_job_retention_hours,
        max_worker_concurrency=max_worker_concurrency,
        worker_scheduler_enabled=worker_scheduler_enabled,
        worker_scheduler_urgent_reserved_slots=worker_scheduler_urgent_reserved_slots,
        worker_scheduler_tenant_budget_tokens_per_minute=worker_scheduler_tenant_budget_tokens_per_minute,
        worker_scheduler_tenant_budget_burst_tokens=worker_scheduler_tenant_budget_burst_tokens,
        enable_coordinator_mode=enable_coordinator_mode,
        max_revision_rounds=max_revision_rounds,
        clarification_sensitivity=clarification_sensitivity,
        deep_rag_default_mode=deep_rag_default_mode,
        deep_rag_max_parallel_lanes=deep_rag_max_parallel_lanes,
        deep_rag_full_read_chunk_threshold=deep_rag_full_read_chunk_threshold,
        deep_rag_sync_reflection_rounds=deep_rag_sync_reflection_rounds,
        deep_rag_background_threshold=deep_rag_background_threshold,
        runtime_events_enabled=runtime_events_enabled,
        session_hydrate_window_messages=session_hydrate_window_messages,
        session_transcript_page_size=session_transcript_page_size,
        context_budget_enabled=context_budget_enabled,
        context_window_tokens=context_window_tokens,
        context_target_ratio=context_target_ratio,
        context_autocompact_threshold=context_autocompact_threshold,
        context_tool_result_max_tokens=context_tool_result_max_tokens,
        context_tool_results_total_tokens=context_tool_results_total_tokens,
        context_microcompact_target_tokens=context_microcompact_target_tokens,
        context_compact_recent_messages=context_compact_recent_messages,
        context_restore_recent_files=context_restore_recent_files,
        context_restore_recent_skills=context_restore_recent_skills,
        agent_definitions_json=agent_definitions_json,
        control_panel_enabled=control_panel_enabled,
        control_panel_admin_token=control_panel_admin_token,
        control_panel_overlay_dir=control_panel_overlay_dir,
        control_panel_runtime_env_path=control_panel_runtime_env_path,
        control_panel_prompt_overlays_dir=control_panel_prompt_overlays_dir,
        control_panel_agent_overlays_dir=control_panel_agent_overlays_dir,
        control_panel_audit_log_path=control_panel_audit_log_path,
        control_panel_static_dir=control_panel_static_dir,
    )
