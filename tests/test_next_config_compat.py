from __future__ import annotations

from agentic_chatbot_next.config import load_settings, runtime_settings_diagnostics
from agentic_chatbot_next.control_panel.config_catalog import build_config_catalog


FRONTEND_EVENT_ENV_NAMES = {
    "FRONTEND_EVENTS_ENABLED",
    "FRONTEND_EVENTS_SHOW_STATUS",
    "FRONTEND_EVENTS_SHOW_AGENTS",
    "FRONTEND_EVENTS_SHOW_TOOLS",
    "FRONTEND_EVENTS_SHOW_PARALLEL_GROUPS",
    "FRONTEND_EVENTS_SHOW_GUIDANCE",
    "FRONTEND_EVENTS_SHOW_SKILLS",
    "FRONTEND_EVENTS_SHOW_CONTEXT",
    "FRONTEND_EVENTS_SHOW_MEMORY_CONTEXT",
    "FRONTEND_EVENTS_DETAIL_LEVEL",
    "FRONTEND_EVENTS_PREVIEW_CHARS",
}


def test_deprecated_runtime_compat_env_vars_no_longer_block_settings_load(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_RUNTIME_MODE", "unexpected_legacy_value")
    monkeypatch.setenv("AGENT_DEFINITIONS_JSON", "{\"general\": {\"name\": \"ignored\"}}")

    settings = load_settings(
        env_overrides={
            "WORKER_JOB_WAIT_TIMEOUT_SECONDS": None,
            "LLM_HTTP_TIMEOUT_SECONDS": None,
            "LLM_HTTP_CONNECT_TIMEOUT_SECONDS": None,
            "SANDBOX_TIMEOUT_SECONDS": None,
        }
    )

    assert settings.agent_runtime_mode == "unexpected_legacy_value"
    assert settings.agent_definitions_json == "{\"general\": {\"name\": \"ignored\"}}"


def test_ocr_enabled_prefers_new_env_over_legacy_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OCR_ENABLED", "true")
    monkeypatch.setenv("USE_PADDLE_OCR", "false")

    settings = load_settings()

    assert settings.ocr_enabled is True


def test_deprecated_use_paddle_ocr_still_controls_ocr(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("OCR_ENABLED", raising=False)
    monkeypatch.setenv("USE_PADDLE_OCR", "false")

    settings = load_settings()

    assert settings.ocr_enabled is False


def test_agent_model_override_envs_are_parsed_and_normalized(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_GENERAL_CHAT_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("AGENT_DATA_ANALYST_CHAT_MODEL", "gpt-oss:20b")
    monkeypatch.setenv("AGENT_MEMORY_MAINTAINER_JUDGE_MODEL", "gpt-oss:20b")

    settings = load_settings(
        env_overrides={
            "WORKER_JOB_WAIT_TIMEOUT_SECONDS": None,
            "LLM_HTTP_TIMEOUT_SECONDS": None,
            "LLM_HTTP_CONNECT_TIMEOUT_SECONDS": None,
            "SANDBOX_TIMEOUT_SECONDS": None,
        }
    )

    assert settings.agent_chat_model_overrides["general"] == "gpt-oss:20b"
    assert settings.agent_chat_model_overrides["data_analyst"] == "gpt-oss:20b"
    assert settings.agent_judge_model_overrides["memory_maintainer"] == "gpt-oss:20b"


def test_output_token_envs_are_optional_and_agent_scoped(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHAT_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("DEMO_CHAT_MAX_OUTPUT_TOKENS", "6144")
    monkeypatch.setenv("JUDGE_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("AGENT_GENERAL_MAX_OUTPUT_TOKENS", "8192")

    settings = load_settings()

    assert settings.chat_max_output_tokens == 4096
    assert settings.demo_chat_max_output_tokens == 6144
    assert settings.judge_max_output_tokens == 1024
    assert settings.agent_chat_max_output_tokens["general"] == 8192


def test_blank_legacy_output_cap_envs_do_not_force_default_limits(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "")
    monkeypatch.setenv("DEMO_OLLAMA_NUM_PREDICT", "")
    monkeypatch.setenv("CHAT_MAX_OUTPUT_TOKENS", "")
    monkeypatch.setenv("DEMO_CHAT_MAX_OUTPUT_TOKENS", "")
    monkeypatch.setenv("JUDGE_MAX_OUTPUT_TOKENS", "")

    settings = load_settings()

    assert settings.ollama_num_predict is None
    assert settings.demo_ollama_num_predict is None
    assert settings.chat_max_output_tokens is None
    assert settings.demo_chat_max_output_tokens is None
    assert settings.judge_max_output_tokens is None


def test_session_history_window_settings_support_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.session_hydrate_window_messages == 80
    assert defaults.session_transcript_page_size == 200

    monkeypatch.setenv("SESSION_HYDRATE_WINDOW_MESSAGES", "12")
    monkeypatch.setenv("SESSION_TRANSCRIPT_PAGE_SIZE", "55")
    overrides = load_settings()

    assert overrides.session_hydrate_window_messages == 12
    assert overrides.session_transcript_page_size == 55


def test_frontend_event_settings_support_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    cleared = {name: None for name in FRONTEND_EVENT_ENV_NAMES}

    defaults = load_settings(env_overrides=cleared)
    assert defaults.frontend_events_enabled is True
    assert defaults.frontend_events_show_status is True
    assert defaults.frontend_events_show_agents is True
    assert defaults.frontend_events_show_tools is True
    assert defaults.frontend_events_show_parallel_groups is True
    assert defaults.frontend_events_show_guidance is True
    assert defaults.frontend_events_show_skills is True
    assert defaults.frontend_events_show_context is True
    assert defaults.frontend_events_show_memory_context is False
    assert defaults.frontend_events_detail_level == "safe_preview"
    assert defaults.frontend_events_preview_chars == 480

    overrides = load_settings(
        env_overrides={
            **cleared,
            "FRONTEND_EVENTS_SHOW_TOOLS": "false",
            "FRONTEND_EVENTS_SHOW_MEMORY_CONTEXT": "true",
            "FRONTEND_EVENTS_DETAIL_LEVEL": "metadata",
            "FRONTEND_EVENTS_PREVIEW_CHARS": "72",
        }
    )

    assert overrides.frontend_events_show_tools is False
    assert overrides.frontend_events_show_memory_context is True
    assert overrides.frontend_events_detail_level == "metadata"
    assert overrides.frontend_events_preview_chars == 72


def test_frontend_event_control_panel_schema_and_validation():
    catalog = build_config_catalog()
    fields = {field.env_name: field for field in catalog.fields}

    assert set(FRONTEND_EVENT_ENV_NAMES).issubset(fields)
    assert fields["FRONTEND_EVENTS_DETAIL_LEVEL"].choices == ("metadata", "safe_preview")
    assert fields["FRONTEND_EVENTS_PREVIEW_CHARS"].min_value == 0

    valid = catalog.validate_changes(
        {
            "FRONTEND_EVENTS_SHOW_TOOLS": "off",
            "FRONTEND_EVENTS_DETAIL_LEVEL": "metadata",
            "FRONTEND_EVENTS_PREVIEW_CHARS": "12",
        }
    )
    invalid = catalog.validate_changes(
        {
            "FRONTEND_EVENTS_SHOW_TOOLS": "maybe",
            "FRONTEND_EVENTS_DETAIL_LEVEL": "raw",
            "FRONTEND_EVENTS_PREVIEW_CHARS": "-1",
        }
    )

    assert valid["valid"] is True
    assert valid["normalized_changes"]["FRONTEND_EVENTS_SHOW_TOOLS"] == "false"
    assert valid["normalized_changes"]["FRONTEND_EVENTS_DETAIL_LEVEL"] == "metadata"
    assert invalid["valid"] is False
    assert set(invalid["errors"]) == {
        "FRONTEND_EVENTS_SHOW_TOOLS",
        "FRONTEND_EVENTS_DETAIL_LEVEL",
        "FRONTEND_EVENTS_PREVIEW_CHARS",
    }


def test_runtime_limit_defaults_restore_interactive_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("WORKER_JOB_WAIT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_HTTP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LLM_HTTP_CONNECT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SANDBOX_TIMEOUT_SECONDS", raising=False)

    settings = load_settings(
        env_overrides={
            "WORKER_JOB_WAIT_TIMEOUT_SECONDS": None,
            "LLM_HTTP_TIMEOUT_SECONDS": None,
            "LLM_HTTP_CONNECT_TIMEOUT_SECONDS": None,
            "SANDBOX_TIMEOUT_SECONDS": None,
        }
    )

    assert settings.max_agent_steps == 10
    assert settings.max_tool_calls == 12
    assert settings.max_parallel_tool_calls == 4
    assert settings.max_rag_agent_steps == 8
    assert settings.worker_job_wait_timeout_seconds == 600
    assert settings.llm_http_timeout_seconds == 120
    assert settings.llm_http_connect_timeout_seconds == 20
    assert settings.data_analyst_max_steps == 10
    assert settings.sandbox_timeout_seconds == 180
    assert settings.planner_max_tasks == 8
    assert settings.max_worker_concurrency == 6
    assert settings.max_revision_rounds == 8
    assert settings.skill_context_max_chars == 4000
    assert settings.retrieval_decomposition_enabled is False


def test_rag_top_k_defaults_are_15_each(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("RAG_TOPK_VECTOR", raising=False)
    monkeypatch.delenv("RAG_TOPK_BM25", raising=False)
    monkeypatch.delenv("RAG_BUDGET_MS", raising=False)
    monkeypatch.delenv("RAG_JUDGE_GRADE_MAX_CHUNKS", raising=False)

    settings = load_settings()

    assert settings.rag_top_k_vector == 15
    assert settings.rag_top_k_keyword == 15
    assert settings.rag_budget_ms == 210000
    assert settings.rag_budget_synthesis_reserve_ms == 30000
    assert settings.rag_heuristic_grading_enabled is True
    assert settings.rag_judge_grade_max_chunks == 12
    assert settings.rag_extractive_fallback_enabled is True
    assert settings.entity_linking_enabled is False
    assert settings.section_first_retrieval_enabled is False
    assert settings.retrieval_quality_verifier_enabled is False


def test_rerank_settings_default_to_mixedbread_ollama_adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    monkeypatch.delenv("RERANK_PROVIDER", raising=False)
    monkeypatch.delenv("RERANK_MODEL", raising=False)
    monkeypatch.delenv("RERANK_TOP_N", raising=False)
    monkeypatch.delenv("RERANK_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("RERANK_FALLBACK_TO_HEURISTICS", raising=False)

    settings = load_settings(
        env_overrides={
            "RERANK_ENABLED": None,
            "RERANK_PROVIDER": None,
            "RERANK_MODEL": None,
            "RERANK_TOP_N": None,
            "RERANK_TIMEOUT_SECONDS": None,
            "RERANK_FALLBACK_TO_HEURISTICS": None,
        }
    )

    assert settings.rerank_enabled is True
    assert settings.rerank_provider == "ollama"
    assert settings.rerank_model == "rjmalagon/mxbai-rerank-large-v2:1.5b-fp16"
    assert settings.rerank_top_n == 12
    assert settings.rerank_timeout_seconds == 30
    assert settings.rerank_fallback_to_heuristics is True


def test_max_parallel_tool_calls_supports_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.max_parallel_tool_calls == 4

    monkeypatch.setenv("MAX_PARALLEL_TOOL_CALLS", "6")
    overrides = load_settings()
    assert overrides.max_parallel_tool_calls == 6


def test_runtime_settings_diagnostics_include_models_overlay_and_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()

    diagnostics = runtime_settings_diagnostics(settings)

    assert diagnostics["process_started_at"]
    assert diagnostics["settings_fingerprint"]
    assert diagnostics["loaded_overlay_env_path"].endswith("runtime.env")
    assert diagnostics["models"]["chat_model"] == settings.ollama_chat_model
    assert diagnostics["models"]["judge_model"] == settings.ollama_judge_model
    assert diagnostics["models"]["graphrag_chat_model"] == settings.graphrag_chat_model
    assert diagnostics["models"]["graphrag_index_chat_model"] == settings.graphrag_index_chat_model
    assert diagnostics["models"]["rerank_model"] == settings.rerank_model


def test_runtime_timeout_settings_support_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WORKER_JOB_WAIT_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv("LLM_HTTP_TIMEOUT_SECONDS", "150")
    monkeypatch.setenv("LLM_HTTP_CONNECT_TIMEOUT_SECONDS", "25")
    monkeypatch.setenv("SANDBOX_TIMEOUT_SECONDS", "240")

    settings = load_settings()

    assert settings.worker_job_wait_timeout_seconds == 900
    assert settings.llm_http_timeout_seconds == 150
    assert settings.llm_http_connect_timeout_seconds == 25
    assert settings.sandbox_timeout_seconds == 240


def test_clarification_sensitivity_supports_defaults_and_boundaries(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.clarification_sensitivity == 50

    monkeypatch.setenv("CLARIFICATION_SENSITIVITY", "0")
    low = load_settings()
    assert low.clarification_sensitivity == 0

    monkeypatch.setenv("CLARIFICATION_SENSITIVITY", "50")
    balanced = load_settings()
    assert balanced.clarification_sensitivity == 50

    monkeypatch.setenv("CLARIFICATION_SENSITIVITY", "100")
    high = load_settings()
    assert high.clarification_sensitivity == 100


def test_deep_rag_settings_support_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.deep_rag_default_mode == "auto"
    assert defaults.deep_rag_max_parallel_lanes == 3
    assert defaults.deep_rag_full_read_chunk_threshold == 24
    assert defaults.deep_rag_sync_reflection_rounds == 1

    monkeypatch.setenv("DEEP_RAG_DEFAULT_MODE", "force")
    monkeypatch.setenv("DEEP_RAG_MAX_PARALLEL_LANES", "5")
    monkeypatch.setenv("DEEP_RAG_FULL_READ_CHUNK_THRESHOLD", "36")
    monkeypatch.setenv("DEEP_RAG_SYNC_REFLECTION_ROUNDS", "2")
    overrides = load_settings()

    assert overrides.deep_rag_default_mode == "force"
    assert overrides.deep_rag_max_parallel_lanes == 5
    assert overrides.deep_rag_full_read_chunk_threshold == 36
    assert overrides.deep_rag_sync_reflection_rounds == 2


def test_llm_router_mode_supports_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.llm_router_mode == "hybrid"

    monkeypatch.setenv("LLM_ROUTER_MODE", "llm_only")
    overrides = load_settings()
    assert overrides.llm_router_mode == "llm_only"

    monkeypatch.setenv("LLM_ROUTER_MODE", "not-a-real-mode")
    invalid = load_settings()
    assert invalid.llm_router_mode == "hybrid"


def test_memory_enabled_supports_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.memory_enabled is True

    monkeypatch.setenv("MEMORY_ENABLED", "false")
    overrides = load_settings()
    assert overrides.memory_enabled is False


def test_deferred_tool_discovery_settings_support_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.deferred_tool_discovery_enabled is False
    assert defaults.deferred_tool_discovery_top_k == 8
    assert defaults.deferred_tool_discovery_require_search is True

    monkeypatch.setenv("DEFERRED_TOOL_DISCOVERY_ENABLED", "true")
    monkeypatch.setenv("DEFERRED_TOOL_DISCOVERY_TOP_K", "12")
    monkeypatch.setenv("DEFERRED_TOOL_DISCOVERY_REQUIRE_SEARCH", "false")
    overrides = load_settings()

    assert overrides.deferred_tool_discovery_enabled is True
    assert overrides.deferred_tool_discovery_top_k == 12
    assert overrides.deferred_tool_discovery_require_search is False


def test_memory_manager_settings_support_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.memory_manager_mode == "shadow"
    assert defaults.memory_selector_model == ""
    assert defaults.memory_writer_model == ""
    assert defaults.memory_candidate_top_k == 16
    assert defaults.memory_context_token_budget == 1600
    assert defaults.memory_shadow_mode is False

    monkeypatch.setenv("MEMORY_MANAGER_MODE", "live")
    monkeypatch.setenv("MEMORY_SELECTOR_MODEL", "selector-small")
    monkeypatch.setenv("MEMORY_WRITER_MODEL", "writer-small")
    monkeypatch.setenv("MEMORY_CANDIDATE_TOP_K", "24")
    monkeypatch.setenv("MEMORY_CONTEXT_TOKEN_BUDGET", "2200")
    monkeypatch.setenv("MEMORY_SHADOW_MODE", "true")
    overrides = load_settings()

    assert overrides.memory_manager_mode == "live"
    assert overrides.memory_selector_model == "selector-small"
    assert overrides.memory_writer_model == "writer-small"
    assert overrides.memory_candidate_top_k == 24
    assert overrides.memory_context_token_budget == 2200
    assert overrides.memory_shadow_mode is True

    monkeypatch.setenv("MEMORY_MANAGER_MODE", "not-a-real-mode")
    invalid = load_settings()
    assert invalid.memory_manager_mode == "shadow"


def test_team_mailbox_settings_support_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.team_mailbox_enabled is False
    assert defaults.team_mailbox_max_channels_per_session == 8
    assert defaults.team_mailbox_max_open_messages_per_channel == 50
    assert defaults.team_mailbox_claim_limit == 8

    monkeypatch.setenv("TEAM_MAILBOX_ENABLED", "true")
    monkeypatch.setenv("TEAM_MAILBOX_MAX_CHANNELS_PER_SESSION", "3")
    monkeypatch.setenv("TEAM_MAILBOX_MAX_OPEN_MESSAGES_PER_CHANNEL", "12")
    monkeypatch.setenv("TEAM_MAILBOX_CLAIM_LIMIT", "4")
    overrides = load_settings()

    assert overrides.team_mailbox_enabled is True
    assert overrides.team_mailbox_max_channels_per_session == 3
    assert overrides.team_mailbox_max_open_messages_per_channel == 12
    assert overrides.team_mailbox_claim_limit == 4


def test_router_feedback_and_worker_scheduler_settings_support_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    defaults = load_settings()
    assert defaults.router_feedback_enabled is True
    assert defaults.router_feedback_rephrase_window_seconds == 600
    assert defaults.router_feedback_neutral_sample_rate == 0.10
    assert defaults.router_feedback_tenant_daily_review_cap == 25
    assert defaults.router_retrain_governance == "human_reviewed"
    assert defaults.worker_scheduler_enabled is True
    assert defaults.worker_scheduler_urgent_reserved_slots == 1
    assert defaults.worker_scheduler_tenant_budget_tokens_per_minute == 24000
    assert defaults.worker_scheduler_tenant_budget_burst_tokens == 48000

    monkeypatch.setenv("ROUTER_FEEDBACK_ENABLED", "false")
    monkeypatch.setenv("ROUTER_FEEDBACK_REPHRASE_WINDOW_SECONDS", "900")
    monkeypatch.setenv("ROUTER_FEEDBACK_NEUTRAL_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("ROUTER_FEEDBACK_TENANT_DAILY_REVIEW_CAP", "12")
    monkeypatch.setenv("ROUTER_RETRAIN_GOVERNANCE", "manual")
    monkeypatch.setenv("WORKER_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("WORKER_SCHEDULER_URGENT_RESERVED_SLOTS", "2")
    monkeypatch.setenv("WORKER_SCHEDULER_TENANT_BUDGET_TOKENS_PER_MINUTE", "12000")
    monkeypatch.setenv("WORKER_SCHEDULER_TENANT_BUDGET_BURST_TOKENS", "18000")
    overrides = load_settings()

    assert overrides.router_feedback_enabled is False
    assert overrides.router_feedback_rephrase_window_seconds == 900
    assert overrides.router_feedback_neutral_sample_rate == 0.25
    assert overrides.router_feedback_tenant_daily_review_cap == 12
    assert overrides.router_retrain_governance == "manual"
    assert overrides.worker_scheduler_enabled is False
    assert overrides.worker_scheduler_urgent_reserved_slots == 2
    assert overrides.worker_scheduler_tenant_budget_tokens_per_minute == 12000
    assert overrides.worker_scheduler_tenant_budget_burst_tokens == 18000
