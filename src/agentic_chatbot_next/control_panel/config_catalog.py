from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence

from agentic_chatbot_next.providers.factory import normalise_agent_name


Getter = Callable[[Any], Any]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def mask_config_value(value: str, *, secret: bool) -> str:
    text = str(value or "")
    if not secret or not text:
        return text
    if len(text) <= 4:
        return "*" * len(text)
    return "*" * max(4, len(text) - 4) + text[-4:]


@dataclass(frozen=True)
class ConfigFieldSpec:
    env_name: str
    label: str
    group: str
    description: str
    getter: Getter
    kind: str = "string"
    choices: Sequence[str] = field(default_factory=tuple)
    secret: bool = False
    readonly: bool = False
    optional: bool = False
    reload_scope: str = "runtime_swap"
    ui_control: str = ""
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None

    def serialize_value(self, settings: Any) -> str:
        return _stringify(self.getter(settings))

    def normalize(self, raw: Any) -> str | None:
        if raw is None:
            return None
        if isinstance(raw, str) and not raw.strip():
            if self.optional:
                return None
            if self.kind == "string":
                return None
        if self.kind == "bool":
            text = str(raw).strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return "true"
            if text in {"0", "false", "no", "n", "off"}:
                return "false"
            raise ValueError("must be a boolean")
        if self.kind == "int":
            value = int(raw)
            if self.min_value is not None and value < int(self.min_value):
                raise ValueError(f"must be greater than or equal to {int(self.min_value)}")
            if self.max_value is not None and value > int(self.max_value):
                raise ValueError(f"must be less than or equal to {int(self.max_value)}")
            return str(value)
        if self.kind == "float":
            value = float(raw)
            if self.min_value is not None and value < float(self.min_value):
                raise ValueError(f"must be greater than or equal to {self.min_value}")
            if self.max_value is not None and value > float(self.max_value):
                raise ValueError(f"must be less than or equal to {self.max_value}")
            return str(value)
        text = str(raw)
        if self.kind == "enum":
            lowered = text.strip().lower()
            if lowered not in {choice.lower() for choice in self.choices}:
                raise ValueError(f"must be one of: {', '.join(self.choices)}")
            return lowered
        if not text.strip():
            return None
        return text

    def to_schema(self, settings: Any) -> Dict[str, Any]:
        current_value = self.serialize_value(settings)
        return {
            "env_name": self.env_name,
            "label": self.label,
            "group": self.group,
            "description": self.description,
            "kind": self.kind,
            "choices": list(self.choices),
            "secret": self.secret,
            "readonly": self.readonly,
            "optional": self.optional,
            "reload_scope": self.reload_scope,
            "ui_control": self.ui_control,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "value": mask_config_value(current_value, secret=self.secret),
            "is_configured": bool(current_value),
        }


class ConfigCatalog:
    def __init__(self, fields: Iterable[ConfigFieldSpec]) -> None:
        self.fields = list(fields)
        self.by_env_name = {field.env_name: field for field in self.fields}

    def schema(self, settings: Any) -> List[Dict[str, Any]]:
        return [field.to_schema(settings) for field in self.fields]

    def effective_values(self, settings: Any, *, masked: bool = True) -> Dict[str, str]:
        payload: Dict[str, str] = {}
        for field in self.fields:
            value = field.serialize_value(settings)
            payload[field.env_name] = mask_config_value(value, secret=masked and field.secret)
        return payload

    def validate_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, str | None] = {}
        errors: Dict[str, str] = {}
        for env_name, raw_value in dict(changes or {}).items():
            key = str(env_name or "").strip()
            field = self.by_env_name.get(key)
            if field is None:
                errors[key] = "unsupported config field"
                continue
            if field.readonly:
                errors[key] = "field is read-only in the control panel"
                continue
            try:
                normalized[key] = field.normalize(raw_value)
            except (TypeError, ValueError) as exc:
                errors[key] = str(exc)
        return {
            "valid": not errors,
            "errors": errors,
            "normalized_changes": normalized,
        }


def _agent_chat_getter(agent_name: str) -> Getter:
    normalized = normalise_agent_name(agent_name)
    return lambda settings: dict(getattr(settings, "agent_chat_model_overrides", {}) or {}).get(normalized, "")


def _agent_judge_getter(agent_name: str) -> Getter:
    normalized = normalise_agent_name(agent_name)
    return lambda settings: dict(getattr(settings, "agent_judge_model_overrides", {}) or {}).get(normalized, "")


def build_config_catalog(agent_names: Sequence[str] | None = None) -> ConfigCatalog:
    fields: List[ConfigFieldSpec] = [
        ConfigFieldSpec("DATABASE_BACKEND", "Database Backend", "Bootstrap", "Live runtime database backend.", getter=lambda s: getattr(s, "database_backend", ""), kind="enum", choices=("postgres",), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("VECTOR_STORE_BACKEND", "Vector Store Backend", "Bootstrap", "Live vector store backend.", getter=lambda s: getattr(s, "vector_store_backend", ""), kind="enum", choices=("pgvector",), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("PG_DSN", "Postgres DSN", "Bootstrap", "Primary PostgreSQL connection string.", getter=lambda s: getattr(s, "pg_dsn", ""), secret=True, readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("EMBEDDING_DIM", "Embedding Dimension", "Bootstrap", "Vector dimension for stored embeddings.", getter=lambda s: getattr(s, "embedding_dim", ""), kind="int", readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("DATA_DIR", "Data Directory", "Bootstrap", "Root data directory.", getter=lambda s: getattr(s, "data_dir", ""), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("KB_DIR", "KB Directory", "Bootstrap", "Knowledge-base source directory.", getter=lambda s: getattr(s, "kb_dir", ""), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("UPLOADS_DIR", "Uploads Directory", "Bootstrap", "Upload staging directory.", getter=lambda s: getattr(s, "uploads_dir", ""), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("SKILLS_DIR", "Skills Directory", "Bootstrap", "Base prompt directory.", getter=lambda s: getattr(s, "skills_dir", ""), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("PROMPTS_DIR", "Prompt Templates Directory", "Bootstrap", "RAG prompt template directory.", getter=lambda s: getattr(s, "prompts_dir", ""), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("SKILL_PACKS_DIR", "Skill Packs Directory", "Bootstrap", "Curated skill-pack source directory.", getter=lambda s: getattr(s, "skill_packs_dir", ""), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("RUNTIME_DIR", "Runtime Directory", "Bootstrap", "Session/job persistence directory.", getter=lambda s: getattr(s, "runtime_dir", ""), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("WORKSPACE_DIR", "Workspace Directory", "Bootstrap", "Per-session workspace directory.", getter=lambda s: getattr(s, "workspace_dir", ""), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("AGENTS_DIR", "Agents Directory", "Bootstrap", "Base markdown agent definition directory.", getter=lambda s: getattr(s, "agents_dir", ""), readonly=True, reload_scope="read_only"),
        ConfigFieldSpec("LLM_PROVIDER", "Chat Provider", "Providers", "Primary chat-model provider.", getter=lambda s: getattr(s, "llm_provider", ""), kind="enum", choices=("ollama", "azure", "nvidia")),
        ConfigFieldSpec("EMBEDDINGS_PROVIDER", "Embeddings Provider", "Providers", "Embedding model provider.", getter=lambda s: getattr(s, "embeddings_provider", ""), kind="enum", choices=("ollama", "azure")),
        ConfigFieldSpec("JUDGE_PROVIDER", "Judge Provider", "Providers", "Judge/routing model provider.", getter=lambda s: getattr(s, "judge_provider", ""), kind="enum", choices=("ollama", "azure", "nvidia")),
        ConfigFieldSpec("OLLAMA_BASE_URL", "Ollama Base URL", "Ollama", "Base URL for the Ollama API.", getter=lambda s: getattr(s, "ollama_base_url", "")),
        ConfigFieldSpec("OLLAMA_CHAT_MODEL", "Ollama Chat Model", "Ollama", "Default chat model for Ollama.", getter=lambda s: getattr(s, "ollama_chat_model", "")),
        ConfigFieldSpec("OLLAMA_EMBED_MODEL", "Ollama Embed Model", "Ollama", "Embedding model for Ollama.", getter=lambda s: getattr(s, "ollama_embed_model", "")),
        ConfigFieldSpec("OLLAMA_JUDGE_MODEL", "Ollama Judge Model", "Ollama", "Judge/routing model for Ollama.", getter=lambda s: getattr(s, "ollama_judge_model", "")),
        ConfigFieldSpec("OLLAMA_TEMPERATURE", "Ollama Temperature", "Ollama", "Default Ollama chat temperature.", getter=lambda s: getattr(s, "ollama_temperature", ""), kind="float"),
        ConfigFieldSpec("CHAT_MAX_OUTPUT_TOKENS", "Chat Max Output Tokens", "Ollama", "Optional global output cap for user-facing chat generation. Leave blank to let the provider decide.", getter=lambda s: getattr(s, "chat_max_output_tokens", ""), kind="int", optional=True),
        ConfigFieldSpec("DEMO_CHAT_MAX_OUTPUT_TOKENS", "Demo Chat Max Output Tokens", "Ollama", "Optional global output cap used in CLI and notebook demo flows. Leave blank to fall back to the normal chat policy.", getter=lambda s: getattr(s, "demo_chat_max_output_tokens", ""), kind="int", optional=True),
        ConfigFieldSpec("JUDGE_MAX_OUTPUT_TOKENS", "Judge Max Output Tokens", "Ollama", "Optional global output cap for routing and grading judge calls. Leave blank to let the provider decide.", getter=lambda s: getattr(s, "judge_max_output_tokens", ""), kind="int", optional=True),
        ConfigFieldSpec("OLLAMA_NUM_PREDICT", "Legacy Ollama Num Predict", "Ollama", "Legacy compatibility cap used only when the newer output-token fields are unset.", getter=lambda s: getattr(s, "ollama_num_predict", ""), kind="int", optional=True),
        ConfigFieldSpec("DEMO_OLLAMA_NUM_PREDICT", "Legacy Demo Num Predict", "Ollama", "Legacy compatibility cap for demo flows. Prefer DEMO_CHAT_MAX_OUTPUT_TOKENS for new setups.", getter=lambda s: getattr(s, "demo_ollama_num_predict", ""), kind="int", optional=True),
        ConfigFieldSpec("AZURE_OPENAI_API_KEY", "Azure API Key", "Azure", "Azure OpenAI API key.", getter=lambda s: getattr(s, "azure_openai_api_key", ""), secret=True),
        ConfigFieldSpec("AZURE_OPENAI_ENDPOINT", "Azure Endpoint", "Azure", "Azure OpenAI endpoint.", getter=lambda s: getattr(s, "azure_openai_endpoint", "")),
        ConfigFieldSpec("AZURE_OPENAI_API_VERSION", "Azure API Version", "Azure", "Azure OpenAI API version.", getter=lambda s: getattr(s, "azure_openai_api_version", "")),
        ConfigFieldSpec("AZURE_OPENAI_CHAT_DEPLOYMENT", "Azure Chat Deployment", "Azure", "Azure chat deployment name.", getter=lambda s: getattr(s, "azure_openai_chat_deployment", "")),
        ConfigFieldSpec("AZURE_OPENAI_JUDGE_DEPLOYMENT", "Azure Judge Deployment", "Azure", "Azure judge deployment name.", getter=lambda s: getattr(s, "azure_openai_judge_deployment", "")),
        ConfigFieldSpec("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "Azure Embeddings Deployment", "Azure", "Azure embeddings deployment name.", getter=lambda s: getattr(s, "azure_openai_embed_deployment", "")),
        ConfigFieldSpec("AZURE_TEMPERATURE", "Azure Temperature", "Azure", "Azure chat temperature.", getter=lambda s: getattr(s, "azure_temperature", ""), kind="float"),
        ConfigFieldSpec("JUDGE_TEMPERATURE", "Judge Temperature", "Azure", "Judge-model temperature.", getter=lambda s: getattr(s, "judge_temperature", ""), kind="float"),
        ConfigFieldSpec("NVIDIA_OPENAI_ENDPOINT", "NVIDIA Endpoint", "NVIDIA", "NVIDIA OpenAI-compatible endpoint.", getter=lambda s: getattr(s, "nvidia_openai_endpoint", "")),
        ConfigFieldSpec("NVIDIA_API_TOKEN", "NVIDIA API Token", "NVIDIA", "NVIDIA API token.", getter=lambda s: getattr(s, "nvidia_api_token", ""), secret=True),
        ConfigFieldSpec("NVIDIA_CHAT_MODEL", "NVIDIA Chat Model", "NVIDIA", "Default NVIDIA chat model.", getter=lambda s: getattr(s, "nvidia_chat_model", "")),
        ConfigFieldSpec("NVIDIA_JUDGE_MODEL", "NVIDIA Judge Model", "NVIDIA", "Default NVIDIA judge model.", getter=lambda s: getattr(s, "nvidia_judge_model", "")),
        ConfigFieldSpec("NVIDIA_TEMPERATURE", "NVIDIA Temperature", "NVIDIA", "NVIDIA chat temperature.", getter=lambda s: getattr(s, "nvidia_temperature", ""), kind="float"),
        ConfigFieldSpec("MAX_AGENT_STEPS", "Max Agent Steps", "Runtime", "Global default max steps for prompt-backed agents.", getter=lambda s: getattr(s, "max_agent_steps", ""), kind="int"),
        ConfigFieldSpec("MAX_TOOL_CALLS", "Max Tool Calls", "Runtime", "Global default max tool calls.", getter=lambda s: getattr(s, "max_tool_calls", ""), kind="int"),
        ConfigFieldSpec("MAX_PARALLEL_TOOL_CALLS", "Max Parallel Tool Calls", "Runtime", "Maximum number of conflict-free tool calls the general agent may run at once in one burst.", getter=lambda s: getattr(s, "max_parallel_tool_calls", 4), kind="int", min_value=1),
        ConfigFieldSpec("DEFERRED_TOOL_DISCOVERY_ENABLED", "Deferred Tool Discovery", "Runtime", "Hide deferred heavy tools from initial binding and expose them through search.", getter=lambda s: getattr(s, "deferred_tool_discovery_enabled", False), kind="bool"),
        ConfigFieldSpec("DEFERRED_TOOL_DISCOVERY_TOP_K", "Deferred Tool Top K", "Runtime", "Default maximum deferred tool candidates returned by discovery.", getter=lambda s: getattr(s, "deferred_tool_discovery_top_k", 8), kind="int", min_value=1),
        ConfigFieldSpec("DEFERRED_TOOL_DISCOVERY_REQUIRE_SEARCH", "Require Tool Search", "Runtime", "Require a deferred tool to be discovered before it can be invoked through the facade.", getter=lambda s: getattr(s, "deferred_tool_discovery_require_search", True), kind="bool"),
        ConfigFieldSpec("MCP_TOOL_PLANE_ENABLED", "MCP Tool Plane Enabled", "Runtime", "Enable user-owned Streamable HTTP MCP tools in the runtime registry.", getter=lambda s: getattr(s, "mcp_tool_plane_enabled", False), kind="bool"),
        ConfigFieldSpec("MCP_USER_SELF_SERVICE_ENABLED", "MCP User Self Service", "Runtime", "Allow authenticated users to create their own MCP connection profiles.", getter=lambda s: getattr(s, "mcp_user_self_service_enabled", True), kind="bool"),
        ConfigFieldSpec("MCP_REQUIRE_HTTPS", "MCP Require HTTPS", "Runtime", "Reject non-HTTPS MCP server URLs unless explicitly disabled for local development.", getter=lambda s: getattr(s, "mcp_require_https", True), kind="bool"),
        ConfigFieldSpec("MCP_ALLOW_PRIVATE_NETWORK", "MCP Private Network", "Runtime", "Allow MCP server URLs that point at localhost or private network ranges.", getter=lambda s: getattr(s, "mcp_allow_private_network", False), kind="bool"),
        ConfigFieldSpec("MCP_CONNECTION_TIMEOUT_SECONDS", "MCP Connect Timeout", "Runtime", "Timeout for MCP initialize and tools/list calls.", getter=lambda s: getattr(s, "mcp_connection_timeout_seconds", 15), kind="int", min_value=1),
        ConfigFieldSpec("MCP_TOOL_CALL_TIMEOUT_SECONDS", "MCP Tool Timeout", "Runtime", "Timeout for MCP tools/call invocations.", getter=lambda s: getattr(s, "mcp_tool_call_timeout_seconds", 60), kind="int", min_value=1),
        ConfigFieldSpec("MCP_CATALOG_REFRESH_SECONDS", "MCP Catalog Refresh", "Runtime", "Minimum intended interval between normal MCP catalog refreshes.", getter=lambda s: getattr(s, "mcp_catalog_refresh_seconds", 3600), kind="int", min_value=1),
        ConfigFieldSpec("MCP_SECRET_ENCRYPTION_KEY", "MCP Secret Key", "Runtime", "Key material used to encrypt MCP bearer tokens at rest.", getter=lambda s: getattr(s, "mcp_secret_encryption_key", ""), secret=True, optional=True),
        ConfigFieldSpec("MAX_RAG_AGENT_STEPS", "Max RAG Steps", "Runtime", "Direct RAG worker step budget.", getter=lambda s: getattr(s, "max_rag_agent_steps", ""), kind="int"),
        ConfigFieldSpec("WORKER_JOB_WAIT_TIMEOUT_SECONDS", "Worker Job Wait Timeout", "Runtime", "How long coordinator and RAG wait loops block for worker jobs before timing out.", getter=lambda s: getattr(s, "worker_job_wait_timeout_seconds", 600), kind="int", min_value=1),
        ConfigFieldSpec("LLM_HTTP_TIMEOUT_SECONDS", "LLM HTTP Timeout", "Runtime", "Overall HTTP timeout used for provider requests.", getter=lambda s: getattr(s, "llm_http_timeout_seconds", 120), kind="int", min_value=1),
        ConfigFieldSpec("LLM_HTTP_CONNECT_TIMEOUT_SECONDS", "LLM Connect Timeout", "Runtime", "HTTP connect timeout used for provider requests.", getter=lambda s: getattr(s, "llm_http_connect_timeout_seconds", 20), kind="int", min_value=1),
        ConfigFieldSpec("DATA_ANALYST_MAX_STEPS", "Data Analyst Max Steps", "Runtime", "Max steps for the data analyst agent.", getter=lambda s: getattr(s, "data_analyst_max_steps", ""), kind="int"),
        ConfigFieldSpec("MAX_WORKER_CONCURRENCY", "Max Worker Concurrency", "Runtime", "Maximum number of background workers the runtime will allow at once.", getter=lambda s: getattr(s, "max_worker_concurrency", ""), kind="int"),
        ConfigFieldSpec("WORKER_SCHEDULER_ENABLED", "Worker Scheduler Enabled", "Runtime", "Route worker jobs through the fairness scheduler instead of direct first-come execution.", getter=lambda s: getattr(s, "worker_scheduler_enabled", True), kind="bool"),
        ConfigFieldSpec("WORKER_SCHEDULER_URGENT_RESERVED_SLOTS", "Urgent Reserved Slots", "Runtime", "Number of worker slots held back for urgent backlog when urgent jobs are waiting.", getter=lambda s: getattr(s, "worker_scheduler_urgent_reserved_slots", 1), kind="int", min_value=0),
        ConfigFieldSpec("WORKER_SCHEDULER_TENANT_BUDGET_TOKENS_PER_MINUTE", "Tenant Tokens Per Minute", "Runtime", "Per-tenant refill rate for worker token budgets used by the scheduler.", getter=lambda s: getattr(s, "worker_scheduler_tenant_budget_tokens_per_minute", 24000), kind="int", min_value=0),
        ConfigFieldSpec("WORKER_SCHEDULER_TENANT_BUDGET_BURST_TOKENS", "Tenant Burst Budget", "Runtime", "Maximum burst tokens a tenant can accumulate before scheduler throttling resumes.", getter=lambda s: getattr(s, "worker_scheduler_tenant_budget_burst_tokens", 48000), kind="int", min_value=0),
        ConfigFieldSpec("MAX_REVISION_ROUNDS", "Max Revision Rounds", "Runtime", "Maximum coordinator revision loops before stopping.", getter=lambda s: getattr(s, "max_revision_rounds", ""), kind="int"),
        ConfigFieldSpec(
            "CLARIFICATION_SENSITIVITY",
            "Clarification Sensitivity",
            "Runtime",
            "How readily the runtime asks clarifying questions for soft ambiguity. Hard blockers still always ask.",
            getter=lambda s: getattr(s, "clarification_sensitivity", ""),
            kind="int",
            ui_control="slider",
            min_value=0,
            max_value=100,
            step=5,
        ),
        ConfigFieldSpec("RAG_MAX_RETRIES", "RAG Max Retries", "Runtime", "Adaptive retrieval retry count.", getter=lambda s: getattr(s, "rag_max_retries", ""), kind="int"),
        ConfigFieldSpec("RAG_MIN_EVIDENCE_CHUNKS", "Min Evidence Chunks", "Runtime", "Minimum evidence chunks to keep for synthesis.", getter=lambda s: getattr(s, "rag_min_evidence_chunks", ""), kind="int"),
        ConfigFieldSpec("RAG_TOPK_VECTOR", "Vector Top K", "Runtime", "Vector retrieval top-k.", getter=lambda s: getattr(s, "rag_top_k_vector", ""), kind="int"),
        ConfigFieldSpec("RAG_TOPK_BM25", "Keyword Top K", "Runtime", "Keyword retrieval top-k.", getter=lambda s: getattr(s, "rag_top_k_keyword", ""), kind="int"),
        ConfigFieldSpec("CHUNK_SIZE", "Chunk Size", "Runtime", "Chunk size for ingestion.", getter=lambda s: getattr(s, "chunk_size", ""), kind="int"),
        ConfigFieldSpec("CHUNK_OVERLAP", "Chunk Overlap", "Runtime", "Chunk overlap for ingestion.", getter=lambda s: getattr(s, "chunk_overlap", ""), kind="int"),
        ConfigFieldSpec("DOCLING_ENABLED", "Docling Enabled", "Runtime", "Opt in to Docling-based DOCX/XLSX parsing before the stable local fallback loaders.", getter=lambda s: getattr(s, "docling_enabled", False), kind="bool"),
        ConfigFieldSpec("USE_PADDLE_OCR", "Paddle OCR Enabled", "Runtime", "Enable OCR for scanned PDFs and image files. Keep disabled for faster local ingest unless you need it.", getter=lambda s: getattr(s, "ocr_enabled", False), kind="bool"),
        ConfigFieldSpec("DEFAULT_COLLECTION_ID", "Default Collection", "Runtime", "Default collection used when none is specified.", getter=lambda s: getattr(s, "default_collection_id", "")),
        ConfigFieldSpec("SESSION_HYDRATE_WINDOW_MESSAGES", "Session Hydrate Window", "Runtime", "Recent message window used when hydrating runtime state.", getter=lambda s: getattr(s, "session_hydrate_window_messages", ""), kind="int"),
        ConfigFieldSpec("SESSION_TRANSCRIPT_PAGE_SIZE", "Session Transcript Page Size", "Runtime", "Pagination size for transcript reads and hydration.", getter=lambda s: getattr(s, "session_transcript_page_size", ""), kind="int"),
        ConfigFieldSpec("CONTEXT_BUDGET_ENABLED", "Context Budget Enabled", "Runtime", "Enable context budgeting, autocompaction, microcompaction, and restore snapshots.", getter=lambda s: getattr(s, "context_budget_enabled", False), kind="bool"),
        ConfigFieldSpec("CONTEXT_WINDOW_TOKENS", "Context Window Tokens", "Runtime", "Estimated model context window used by the budget manager.", getter=lambda s: getattr(s, "context_window_tokens", 32768), kind="int", min_value=1024),
        ConfigFieldSpec("CONTEXT_TARGET_RATIO", "Context Target Ratio", "Runtime", "Fraction of the window used as the target prompt budget.", getter=lambda s: getattr(s, "context_target_ratio", 0.72), kind="float"),
        ConfigFieldSpec("CONTEXT_AUTOCOMPACT_THRESHOLD", "Autocompact Threshold", "Runtime", "Fraction of the window that triggers automatic conversation compaction.", getter=lambda s: getattr(s, "context_autocompact_threshold", 0.85), kind="float"),
        ConfigFieldSpec("CONTEXT_TOOL_RESULT_MAX_TOKENS", "Tool Result Max Tokens", "Runtime", "Per-tool-result model-visible budget before sidecar preservation and clipping.", getter=lambda s: getattr(s, "context_tool_result_max_tokens", 2000), kind="int", min_value=128),
        ConfigFieldSpec("CONTEXT_TOOL_RESULTS_TOTAL_TOKENS", "Tool Results Total Tokens", "Runtime", "Current-turn total tool-result budget before microcompaction.", getter=lambda s: getattr(s, "context_tool_results_total_tokens", 8000), kind="int", min_value=512),
        ConfigFieldSpec("CONTEXT_MICROCOMPACT_TARGET_TOKENS", "Microcompact Target Tokens", "Runtime", "Target token budget for current-turn tool chatter after microcompaction.", getter=lambda s: getattr(s, "context_microcompact_target_tokens", 2400), kind="int", min_value=256),
        ConfigFieldSpec("CONTEXT_COMPACT_RECENT_MESSAGES", "Compact Recent Messages", "Runtime", "Recent messages preserved outside compact summaries.", getter=lambda s: getattr(s, "context_compact_recent_messages", 12), kind="int", min_value=2),
        ConfigFieldSpec("CONTEXT_RESTORE_RECENT_FILES", "Restore Recent Files", "Runtime", "Recent file/doc handles restored after compaction.", getter=lambda s: getattr(s, "context_restore_recent_files", 10), kind="int", min_value=0),
        ConfigFieldSpec("CONTEXT_RESTORE_RECENT_SKILLS", "Restore Recent Skills", "Runtime", "Recent skill handles restored after compaction.", getter=lambda s: getattr(s, "context_restore_recent_skills", 6), kind="int", min_value=0),
        ConfigFieldSpec("SKILL_CONTEXT_MAX_CHARS", "Skill Context Max Chars", "Runtime", "Maximum skill context size injected into prompts.", getter=lambda s: getattr(s, "skill_context_max_chars", ""), kind="int"),
        ConfigFieldSpec("EXECUTABLE_SKILLS_ENABLED", "Executable Skills Enabled", "Runtime", "Enable explicit execute_skill tool binding for executable and hybrid skill packs.", getter=lambda s: getattr(s, "executable_skills_enabled", False), kind="bool"),
        ConfigFieldSpec("SKILL_PACKS_HOT_RELOAD_ENABLED", "Skill Packs Hot Reload", "Runtime", "Poll repo-authored skill packs for checksum changes after startup.", getter=lambda s: getattr(s, "skill_packs_hot_reload_enabled", False), kind="bool"),
        ConfigFieldSpec("SKILL_PACKS_HOT_RELOAD_INTERVAL_SECONDS", "Skill Reload Interval", "Runtime", "Polling interval for skill-pack hot reload when enabled.", getter=lambda s: getattr(s, "skill_packs_hot_reload_interval_seconds", 5), kind="int", min_value=1),
        ConfigFieldSpec("TEAM_MAILBOX_ENABLED", "Team Mailbox Enabled", "Runtime", "Enable async team mailbox channels for same-session agent coordination.", getter=lambda s: getattr(s, "team_mailbox_enabled", False), kind="bool"),
        ConfigFieldSpec("TEAM_MAILBOX_MAX_CHANNELS_PER_SESSION", "Team Mailbox Channels", "Runtime", "Maximum active team mailbox channels per session.", getter=lambda s: getattr(s, "team_mailbox_max_channels_per_session", 8), kind="int", min_value=1),
        ConfigFieldSpec("TEAM_MAILBOX_MAX_OPEN_MESSAGES_PER_CHANNEL", "Team Mailbox Open Messages", "Runtime", "Maximum open messages per team mailbox channel.", getter=lambda s: getattr(s, "team_mailbox_max_open_messages_per_channel", 50), kind="int", min_value=1),
        ConfigFieldSpec("TEAM_MAILBOX_CLAIM_LIMIT", "Team Mailbox Claim Limit", "Runtime", "Maximum team mailbox messages one claim call can return.", getter=lambda s: getattr(s, "team_mailbox_claim_limit", 8), kind="int", min_value=1),
        ConfigFieldSpec("GATEWAY_MODEL_ID", "Gateway Model ID", "Runtime", "Public model ID exposed by the API gateway.", getter=lambda s: getattr(s, "gateway_model_id", "")),
        ConfigFieldSpec("LLM_ROUTER_ENABLED", "Router Enabled", "Routing", "Enable hybrid router behavior.", getter=lambda s: getattr(s, "llm_router_enabled", ""), kind="bool"),
        ConfigFieldSpec("LLM_ROUTER_MODE", "Router Mode", "Routing", "Hybrid or LLM-only routing mode.", getter=lambda s: getattr(s, "llm_router_mode", ""), kind="enum", choices=("hybrid", "llm_only")),
        ConfigFieldSpec("LLM_ROUTER_CONFIDENCE_THRESHOLD", "Router Confidence Threshold", "Routing", "Confidence threshold before escalating to the judge model.", getter=lambda s: getattr(s, "llm_router_confidence_threshold", ""), kind="float"),
        ConfigFieldSpec("ROUTER_FEEDBACK_ENABLED", "Router Feedback Enabled", "Routing", "Persist router decisions and score them against later observable outcomes.", getter=lambda s: getattr(s, "router_feedback_enabled", True), kind="bool"),
        ConfigFieldSpec("ROUTER_FEEDBACK_REPHRASE_WINDOW_SECONDS", "Rephrase Window Seconds", "Routing", "Window for treating the next same-session user turn as a rephrase-like retry signal.", getter=lambda s: getattr(s, "router_feedback_rephrase_window_seconds", 600), kind="int", min_value=60),
        ConfigFieldSpec("ROUTER_FEEDBACK_NEUTRAL_SAMPLE_RATE", "Neutral Review Sample Rate", "Routing", "Share of neutral router outcomes to sample into the review pool.", getter=lambda s: getattr(s, "router_feedback_neutral_sample_rate", 0.10), kind="float", min_value=0.0, max_value=1.0, step=0.05),
        ConfigFieldSpec("ROUTER_FEEDBACK_TENANT_DAILY_REVIEW_CAP", "Tenant Daily Review Cap", "Routing", "Maximum router review samples collected per tenant each day.", getter=lambda s: getattr(s, "router_feedback_tenant_daily_review_cap", 25), kind="int", min_value=1),
        ConfigFieldSpec("ROUTER_RETRAIN_GOVERNANCE", "Router Retrain Governance", "Routing", "Controls whether retrain artifacts are advisory only or require explicit human review.", getter=lambda s: getattr(s, "router_retrain_governance", "human_reviewed"), kind="enum", choices=("human_reviewed", "manual")),
        ConfigFieldSpec("WEB_SEARCH_ENABLED", "Web Search Enabled", "Features", "Allow Tavily-backed web search fallback.", getter=lambda s: getattr(s, "web_search_enabled", ""), kind="bool"),
        ConfigFieldSpec("TAVILY_API_KEY", "Tavily API Key", "Features", "API key for Tavily web search.", getter=lambda s: getattr(s, "tavily_api_key", ""), secret=True),
        ConfigFieldSpec("GRAPH_SEARCH_ENABLED", "Graph Search Enabled", "Features", "Enable graph-augmented retrieval.", getter=lambda s: getattr(s, "graph_search_enabled", ""), kind="bool"),
        ConfigFieldSpec("GRAPH_INGEST_ENABLED", "Graph Ingest Enabled", "Features", "Enable graph ingestion during document sync.", getter=lambda s: getattr(s, "graph_ingest_enabled", ""), kind="bool"),
        ConfigFieldSpec("GRAPH_BACKEND", "Graph Backend", "Features", "Primary managed graph backend.", getter=lambda s: getattr(s, "graph_backend", ""), kind="enum", choices=("microsoft_graphrag", "neo4j")),
        ConfigFieldSpec("GRAPH_IMPORT_ENABLED", "Graph Import Enabled", "Features", "Allow registration of existing graph artifacts.", getter=lambda s: getattr(s, "graph_import_enabled", ""), kind="bool"),
        ConfigFieldSpec("GRAPH_SOURCE_PLANNING_ENABLED", "Graph Source Planning", "Features", "Let the retrieval controller shortlist graphs before querying them.", getter=lambda s: getattr(s, "graph_source_planning_enabled", ""), kind="bool"),
        ConfigFieldSpec("RETRIEVAL_DECOMPOSITION_ENABLED", "Retrieval Decomposition", "Features", "Split deep search into entity, relationship, confirmation, and synthesis phases.", getter=lambda s: getattr(s, "retrieval_decomposition_enabled", ""), kind="bool"),
        ConfigFieldSpec("ENTITY_LINKING_ENABLED", "Entity Linking Enabled", "Features", "Resolve canonical entity IDs across graph, vector, and SQL retrieval.", getter=lambda s: getattr(s, "entity_linking_enabled", ""), kind="bool"),
        ConfigFieldSpec("SECTION_FIRST_RETRIEVAL_ENABLED", "Section-first Retrieval", "Features", "Narrow to likely sections before chunk fanout during deep search.", getter=lambda s: getattr(s, "section_first_retrieval_enabled", ""), kind="bool"),
        ConfigFieldSpec("RETRIEVAL_QUALITY_VERIFIER_ENABLED", "Retrieval Quality Verifier", "Features", "Run retrieval-audit checks for deep RAG and coordinator verification.", getter=lambda s: getattr(s, "retrieval_quality_verifier_enabled", ""), kind="bool"),
        ConfigFieldSpec("GRAPH_SQL_ENABLED", "Structured Search Enabled", "Features", "Allow read-only SQL-backed metadata lookups during source planning.", getter=lambda s: getattr(s, "graph_sql_enabled", ""), kind="bool"),
        ConfigFieldSpec("GRAPH_SQL_ALLOWED_VIEWS", "Structured Search Views", "Features", "Comma-separated allowlist of read-only structured search views.", getter=lambda s: ",".join(getattr(s, "graph_sql_allowed_views", ()) or ())),
        ConfigFieldSpec("GRAPHRAG_PROJECTS_DIR", "GraphRAG Projects Dir", "Features", "Managed project root for GraphRAG indexes.", getter=lambda s: getattr(s, "graphrag_projects_dir", "")),
        ConfigFieldSpec("GRAPHRAG_USE_CONTAINER", "GraphRAG Use Container", "Features", "Run GraphRAG CLI inside a container instead of the local binary.", getter=lambda s: getattr(s, "graphrag_use_container", ""), kind="bool"),
        ConfigFieldSpec("GRAPHRAG_CONTAINER_IMAGE", "GraphRAG Container Image", "Features", "Container image used when GraphRAG runs in Docker.", getter=lambda s: getattr(s, "graphrag_container_image", "")),
        ConfigFieldSpec("GRAPHRAG_CLI_COMMAND", "GraphRAG CLI Command", "Features", "Local GraphRAG CLI command used when not running in Docker.", getter=lambda s: getattr(s, "graphrag_cli_command", "")),
        ConfigFieldSpec("GRAPHRAG_LLM_PROVIDER", "GraphRAG Model Provider", "Features", "LiteLLM/OpenAI-compatible provider name used for GraphRAG project builds.", getter=lambda s: getattr(s, "graphrag_llm_provider", "")),
        ConfigFieldSpec("GRAPHRAG_BASE_URL", "GraphRAG Base URL", "Features", "Optional OpenAI-compatible base URL used by the GraphRAG execution profile.", getter=lambda s: getattr(s, "graphrag_base_url", "")),
        ConfigFieldSpec("GRAPHRAG_API_KEY", "GraphRAG API Key", "Features", "API key used by the GraphRAG execution profile.", getter=lambda s: getattr(s, "graphrag_api_key", ""), secret=True, optional=True),
        ConfigFieldSpec("GRAPHRAG_CHAT_MODEL", "GraphRAG Chat Model", "Features", "Chat/completion model used for GraphRAG indexing and query workflows.", getter=lambda s: getattr(s, "graphrag_chat_model", "")),
        ConfigFieldSpec("GRAPHRAG_INDEX_CHAT_MODEL", "GraphRAG Index Chat Model", "Features", "Optional smaller completion model used only for GraphRAG indexing workflows such as extract_graph and community report generation.", getter=lambda s: getattr(s, "graphrag_index_chat_model", "")),
        ConfigFieldSpec("GRAPHRAG_COMMUNITY_REPORT_MODE", "Community Report Mode", "Features", "Choose the community-report workflow: 'text' keeps extraction standard but uses the lighter text summarization stage; 'graph' preserves the full graph-context report stage.", getter=lambda s: getattr(s, "graphrag_community_report_mode", "text"), kind="enum", choices=("text", "graph")),
        ConfigFieldSpec("GRAPHRAG_COMMUNITY_REPORT_CHAT_MODEL", "Community Report Chat Model", "Features", "Dedicated model used for GraphRAG community report generation.", getter=lambda s: getattr(s, "graphrag_community_report_chat_model", "")),
        ConfigFieldSpec("GRAPHRAG_EMBED_MODEL", "GraphRAG Embed Model", "Features", "Embedding model used for GraphRAG indexing and local/drift query workflows.", getter=lambda s: getattr(s, "graphrag_embed_model", "")),
        ConfigFieldSpec("GRAPHRAG_CONCURRENCY", "GraphRAG Concurrency", "Features", "Worker concurrency used by the GraphRAG execution profile.", getter=lambda s: getattr(s, "graphrag_concurrency", ""), kind="int"),
        ConfigFieldSpec("GRAPHRAG_REQUEST_TIMEOUT_SECONDS", "GraphRAG Request Timeout", "Features", "Per-request timeout written into GraphRAG model call settings.", getter=lambda s: getattr(s, "graphrag_request_timeout_seconds", getattr(s, "graphrag_timeout_seconds", "")), kind="int", min_value=1),
        ConfigFieldSpec("GRAPHRAG_INDEX_REQUEST_TIMEOUT_SECONDS", "GraphRAG Index Request Timeout", "Features", "Per-request timeout used specifically for GraphRAG indexing workflows when the indexing model is slower than query-time workflows.", getter=lambda s: getattr(s, "graphrag_index_request_timeout_seconds", getattr(s, "graphrag_request_timeout_seconds", getattr(s, "graphrag_timeout_seconds", ""))), kind="int", min_value=1),
        ConfigFieldSpec("GRAPHRAG_COMMUNITY_REPORT_REQUEST_TIMEOUT_SECONDS", "Community Report Request Timeout", "Features", "Per-request timeout used specifically for GraphRAG community report generation.", getter=lambda s: getattr(s, "graphrag_community_report_request_timeout_seconds", getattr(s, "graphrag_index_request_timeout_seconds", getattr(s, "graphrag_request_timeout_seconds", getattr(s, "graphrag_timeout_seconds", "")))), kind="int", min_value=1),
        ConfigFieldSpec("GRAPHRAG_COMMUNITY_REPORT_MAX_INPUT_LENGTH", "Community Report Max Input", "Features", "Token budget used to build each community report prompt context.", getter=lambda s: getattr(s, "graphrag_community_report_max_input_length", 4000), kind="int", min_value=1),
        ConfigFieldSpec("GRAPHRAG_COMMUNITY_REPORT_MAX_LENGTH", "Community Report Max Output", "Features", "Token budget requested for each generated community report.", getter=lambda s: getattr(s, "graphrag_community_report_max_length", 1200), kind="int", min_value=1),
        ConfigFieldSpec("GRAPHRAG_JOB_TIMEOUT_SECONDS", "GraphRAG Job Timeout", "Features", "Overall timeout budget for a managed GraphRAG build job. Set to 0 to disable the hard job timeout.", getter=lambda s: getattr(s, "graphrag_job_timeout_seconds", 0), kind="int", min_value=0),
        ConfigFieldSpec("GRAPHRAG_STALE_RUN_AFTER_SECONDS", "GraphRAG Stale Run Window", "Features", "How long a managed GraphRAG run may go without log activity before it is marked stalled.", getter=lambda s: getattr(s, "graphrag_stale_run_after_seconds", ""), kind="int"),
        ConfigFieldSpec("GRAPHRAG_DEFAULT_QUERY_METHOD", "GraphRAG Default Query", "Features", "Default query method used for managed graph search.", getter=lambda s: getattr(s, "graphrag_default_query_method", ""), kind="enum", choices=("local", "global", "drift")),
        ConfigFieldSpec("GRAPHRAG_ARTIFACT_CACHE_TTL_SECONDS", "GraphRAG Artifact Cache TTL", "Features", "TTL for cached GraphRAG artifact bundles used by the query adapter.", getter=lambda s: getattr(s, "graphrag_artifact_cache_ttl_seconds", ""), kind="int"),
        ConfigFieldSpec("GRAPH_QUERY_CACHE_TTL_SECONDS", "Graph Query Cache TTL", "Features", "TTL for cached graph query results.", getter=lambda s: getattr(s, "graph_query_cache_ttl_seconds", ""), kind="int"),
        ConfigFieldSpec("NEO4J_URI", "Neo4j URI", "Features", "Neo4j connection URI.", getter=lambda s: getattr(s, "neo4j_uri", "")),
        ConfigFieldSpec("NEO4J_USERNAME", "Neo4j Username", "Features", "Neo4j username.", getter=lambda s: getattr(s, "neo4j_username", "")),
        ConfigFieldSpec("NEO4J_PASSWORD", "Neo4j Password", "Features", "Neo4j password.", getter=lambda s: getattr(s, "neo4j_password", ""), secret=True),
        ConfigFieldSpec("NEO4J_DATABASE", "Neo4j Database", "Features", "Neo4j database name.", getter=lambda s: getattr(s, "neo4j_database", "")),
        ConfigFieldSpec("NEO4J_TIMEOUT_SECONDS", "Neo4j Timeout", "Features", "Neo4j query timeout in seconds.", getter=lambda s: getattr(s, "neo4j_timeout_seconds", ""), kind="int"),
        ConfigFieldSpec("SANDBOX_DOCKER_IMAGE", "Sandbox Docker Image", "Sandbox", "Docker image used for analyst code execution.", getter=lambda s: getattr(s, "sandbox_docker_image", "")),
        ConfigFieldSpec("SANDBOX_TIMEOUT_SECONDS", "Sandbox Timeout", "Sandbox", "Execution timeout for analyst sandbox tasks.", getter=lambda s: getattr(s, "sandbox_timeout_seconds", ""), kind="int"),
        ConfigFieldSpec("SANDBOX_MEMORY_LIMIT", "Sandbox Memory Limit", "Sandbox", "Memory limit for analyst sandbox containers.", getter=lambda s: getattr(s, "sandbox_memory_limit", "")),
        ConfigFieldSpec("DATA_ANALYST_NLP_CHAT_MODEL", "Analyst NLP Model", "Sandbox", "Optional chat-model override for NLP column tasks.", getter=lambda s: getattr(s, "data_analyst_nlp_chat_model", "")),
        ConfigFieldSpec("DATA_ANALYST_NLP_BATCH_SIZE", "Analyst NLP Batch Size", "Sandbox", "Batch size for LLM NLP column tasks.", getter=lambda s: getattr(s, "data_analyst_nlp_batch_size", ""), kind="int"),
        ConfigFieldSpec("DATA_ANALYST_NLP_TEMPERATURE", "Analyst NLP Temperature", "Sandbox", "Temperature for NLP column tasks.", getter=lambda s: getattr(s, "data_analyst_nlp_temperature", ""), kind="float"),
        ConfigFieldSpec("LANGFUSE_HOST", "Langfuse Host", "Observability", "Langfuse host URL.", getter=lambda s: getattr(s, "langfuse_host", "")),
        ConfigFieldSpec("LANGFUSE_PUBLIC_KEY", "Langfuse Public Key", "Observability", "Langfuse public key.", getter=lambda s: getattr(s, "langfuse_public_key", "")),
        ConfigFieldSpec("LANGFUSE_SECRET_KEY", "Langfuse Secret Key", "Observability", "Langfuse secret key.", getter=lambda s: getattr(s, "langfuse_secret_key", ""), secret=True),
        ConfigFieldSpec("LANGFUSE_DEBUG", "Langfuse Debug", "Observability", "Enable verbose Langfuse debugging.", getter=lambda s: getattr(s, "langfuse_debug", ""), kind="bool"),
    ]

    for agent_name in sorted({str(item).strip() for item in (agent_names or []) if str(item).strip()}):
        fields.append(
            ConfigFieldSpec(
                env_name=f"AGENT_{normalise_agent_name(agent_name).upper()}_CHAT_MODEL",
                label=f"{agent_name} Chat Model",
                group="Agent Models",
                description=f"Per-agent chat-model override for {agent_name}.",
                getter=_agent_chat_getter(agent_name),
            )
        )
        fields.append(
            ConfigFieldSpec(
                env_name=f"AGENT_{normalise_agent_name(agent_name).upper()}_JUDGE_MODEL",
                label=f"{agent_name} Judge Model",
                group="Agent Models",
                description=f"Per-agent judge-model override for {agent_name}.",
                getter=_agent_judge_getter(agent_name),
            )
        )
        fields.append(
            ConfigFieldSpec(
                env_name=f"AGENT_{normalise_agent_name(agent_name).upper()}_MAX_OUTPUT_TOKENS",
                label=f"{agent_name} Max Output Tokens",
                group="Agent Models",
                description=f"Optional per-agent chat output cap for {agent_name}. Leave blank to inherit the global chat output policy.",
                getter=lambda settings, agent_name=agent_name: dict(
                    getattr(settings, "agent_chat_max_output_tokens", {}) or {}
                ).get(normalise_agent_name(agent_name), ""),
                kind="int",
                optional=True,
            )
        )
    return ConfigCatalog(fields)
