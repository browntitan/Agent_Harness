# Providers and Backend Config

## Provider roles

The runtime currently has three provider roles:

- chat LLM
- judge LLM
- embeddings

Supported today:

- chat LLM: `ollama`, `azure`, or `nvidia`
- judge LLM: `ollama`, `azure`, or `nvidia`
- embeddings: `ollama` or `azure`

Only the providers selected by `LLM_PROVIDER`, `JUDGE_PROVIDER`, and
`EMBEDDINGS_PROVIDER` are validated.

The OpenAI-compatible gateway model ID is separate from the underlying runtime model names.
Keep `GATEWAY_MODEL_ID=enterprise-agent` stable unless you intentionally want to change the
public API contract.

## What each provider role does

### Chat LLM

Used by:

- `run_basic_chat()`
- all `react`-mode runtime agents via `run_general_agent()`:
  - `general`
  - `utility`
  - `data_analyst`
- the bounded `DataAnalystNlpRunner` used by `run_nlp_column_task`
- `run_rag_contract()` for the `rag_worker` path and upload-summary kickoff
- planner worker model calls
- finalizer worker model calls
- verifier worker model calls

The dedicated `memory_maintainer` mode does not currently use chat or judge providers. It
runs local heuristic extraction.

When `MEMORY_ENABLED=false`, the runtime disables that memory-maintenance path entirely and also
removes the memory tool surface from prompt-backed agents.

### Judge LLM

Used by:

- LLM-router escalation inside `route_turn()` when deterministic routing confidence is low
- `run_rag_contract()` grading / grounded-answer support
- `rag_agent_tool` calls made by `general` or `verifier`

### Embeddings

Used by:

- KB ingest and retrieval
- skill-pack indexing and retrieval

## Optional graph backend

The runtime now supports a managed graph layer for GraphRAG. `GRAPH_BACKEND` defaults to
`microsoft_graphrag`; `neo4j` is retained as an optional compatibility backend.

This is not a fourth LLM provider role. It is a feature-flagged retrieval backend that
augments the normal PostgreSQL / pgvector stack when enabled.

Relevant settings:

- `GRAPH_SEARCH_ENABLED`
- `GRAPH_INGEST_ENABLED`
- `GRAPH_BACKEND`
- `GRAPHRAG_PROJECTS_DIR`
- `GRAPHRAG_USE_CONTAINER`
- `GRAPHRAG_CONTAINER_IMAGE`
- `GRAPHRAG_CLI_COMMAND`
- `GRAPHRAG_LLM_PROVIDER`
- `GRAPHRAG_BASE_URL`
- `GRAPHRAG_API_KEY`
- `GRAPHRAG_CHAT_MODEL`
- `GRAPHRAG_INDEX_CHAT_MODEL`
- `GRAPHRAG_COMMUNITY_REPORT_MODE`
- `GRAPHRAG_COMMUNITY_REPORT_CHAT_MODEL`
- `GRAPHRAG_EMBED_MODEL`
- `GRAPHRAG_CONCURRENCY`
- `GRAPHRAG_REQUEST_TIMEOUT_SECONDS`
- `GRAPHRAG_INDEX_REQUEST_TIMEOUT_SECONDS`
- `GRAPHRAG_COMMUNITY_REPORT_REQUEST_TIMEOUT_SECONDS`
- `GRAPHRAG_COMMUNITY_REPORT_MAX_INPUT_LENGTH`
- `GRAPHRAG_COMMUNITY_REPORT_MAX_LENGTH`
- `GRAPHRAG_JOB_TIMEOUT_SECONDS`
- `GRAPHRAG_STALE_RUN_AFTER_SECONDS`
- `GRAPHRAG_DEFAULT_QUERY_METHOD`
- `GRAPHRAG_ARTIFACT_CACHE_TTL_SECONDS`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- `NEO4J_TIMEOUT_SECONDS`

Operational notes:

- managed GraphRAG catalog records, graph sources, graph runs, query cache rows, and canonical
  entities are PostgreSQL-backed
- Neo4j is optional and lazily used when the Neo4j backend is selected/configured
- PostgreSQL / pgvector remains the default and the citation source of truth
- the graph layer is integrated through the existing runtime, graph tools, and `graph_manager`,
  not a top-level LangGraph rewrite of the whole application
- graph prompts, graph-bound skills, research tuning, build/refresh actions, and graph
  admin flows are operator-managed through the control panel

## Circuit-breaker layer

The live runtime now wraps chat and judge models with provider-level circuit breakers inside
`src/agentic_chatbot_next/runtime/kernel_providers.py`.

Important properties:

- breaker scope is per provider role plus resolved model identity
- chat and judge are wrapped; embeddings are not
- availability failures count toward the breaker:
  - timeouts
  - network / transport failures
  - provider 429s
  - provider 5xx responses
- local parsing, JSON extraction, or schema-validation errors do not count toward breaker state

Default policy:

- enabled
- rolling window size `20`
- minimum samples `6`
- error-rate threshold `0.50`
- immediate open after `3` consecutive availability failures
- open interval `30` seconds
- half-open allows one probe call

Graceful degradation:

- judge breaker open -> LLM router skips to deterministic routing
- requested AGENT chat breaker open -> one downgrade attempt to `basic`
- if `basic` is also unavailable -> persisted degraded-service assistant response

## Ollama example

```bash
LLM_PROVIDER=ollama
JUDGE_PROVIDER=ollama
EMBEDDINGS_PROVIDER=ollama

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2:1b
OLLAMA_JUDGE_MODEL=llama3.2:1b
OLLAMA_EMBED_MODEL=nomic-embed-text:latest
OLLAMA_TEMPERATURE=0.2
JUDGE_TEMPERATURE=0.0
EMBEDDING_DIM=768
KB_EXTRA_DIRS=./docs
```

## Ollama throughput tuning and benchmarking

If you care about single-request tokens per second on a local Ollama setup, the biggest levers are model size, whether the model fits fully on GPU, and how much prompt/context you ask the model to process before generation.

Practical ways to make it faster:

- Pick a smaller or more aggressively quantized model.
- Keep the request context shorter with smaller prompts and, when appropriate, a lower `num_ctx`.
- Keep the model warm with `keep_alive` so repeated runs do not pay a reload penalty.
- Prefer a setup where `ollama ps` shows `100% GPU`; mixed CPU/GPU execution is usually much slower.
- Keep `OLLAMA_NUM_PARALLEL` low for single-user latency benchmarks. Higher parallelism can improve aggregate throughput across concurrent requests, but it often reduces per-request speed and increases memory pressure.
- On long-context workloads, `OLLAMA_FLASH_ATTENTION=1` can help memory efficiency. That matters most when context growth would otherwise push part of the model or KV cache out of fast memory.
- If you enable flash attention, `OLLAMA_KV_CACHE_TYPE=q8_0` or `q4_0` can reduce KV-cache memory use further. The tradeoff is lower cache precision, which can slightly affect quality.
- Avoid benchmarking Ollama through Docker on macOS when you care about GPU speed. Use the native Ollama app/service path instead.

Tradeoffs to keep in mind:

- Smaller models and heavier quantization usually increase speed, but you give up reasoning quality, retrieval fidelity, and long-form consistency.
- Lower `num_ctx` and shorter prompts improve prefill speed and reduce memory pressure, but you lose available context and may truncate agent/RAG inputs.
- `keep_alive` mainly improves cold-start latency, not steady-state decode speed, and it keeps memory reserved longer.
- Higher concurrency settings can raise total server throughput under load, but they usually make an individual interactive request slower.
- Flash attention and KV-cache quantization are most useful when memory is the bottleneck. They are not guaranteed to move decode speed much on already-comfortable short-context runs.

The benchmark command below measures prompt throughput, decode throughput, and end-to-end throughput using Ollama's `/api/generate` timing fields:

```bash
python run.py benchmark-ollama-throughput
```

Useful variants:

```bash
# Compare multiple local models with the same prompt shape.
python run.py benchmark-ollama-throughput \
  --model llama3.2:1b \
  --model qwen3.5:9b \
  --model llama3.2:1b

# Change the prompt and output size.
python run.py benchmark-ollama-throughput \
  --model qwen3.5:9b \
  --context-words 4000 \
  --num-predict 512

# Force a smaller active context window to see whether memory pressure is the bottleneck.
python run.py benchmark-ollama-throughput \
  --model llama3.2:1b \
  --num-ctx 8192

# Save JSON output for later comparison.
python run.py benchmark-ollama-throughput \
  --model llama3.2:1b \
  --model qwen3.5:9b \
  --output tmp/ollama-throughput.json
```

Notes:

- If you omit `--model`, the command uses `OLLAMA_CHAT_MODEL`.
- On some Macs, `localhost:11434` can resolve to the wrong listener when Docker is also binding that port. The benchmark command automatically falls back to `127.0.0.1` unless you pass `--no-localhost-fallback`.
- Compare models with the same `--context-words`, `--num-predict`, and `--runs` values, otherwise the numbers are not apples-to-apples.
- Look primarily at `avg_gen_tps` for decode speed and `avg_prompt_tps` for long-prompt/prefill speed.

## Per-agent runtime overrides

The live runtime can override chat and judge model selection per agent role without changing
the public gateway model ID.

Environment pattern:

```bash
AGENT_<AGENT_NAME>_CHAT_MODEL=...
AGENT_<AGENT_NAME>_JUDGE_MODEL=...
```

Examples:

```bash
AGENT_GENERAL_CHAT_MODEL=llama3.2:1b
AGENT_DATA_ANALYST_CHAT_MODEL=llama3.2:1b
AGENT_MEMORY_MAINTAINER_JUDGE_MODEL=llama3.2:1b
```

Notes:

- agent names are normalized to lowercase with underscores
- if no per-agent override is set, the agent inherits the shared provider defaults
- overrides affect chat and judge only; embeddings remain shared/global
- overrides only matter for roles that actually invoke providers; today
  `memory_maintainer` overrides are effectively inert

## Output-length controls

The runtime now treats output caps as optional.

Recommended envs:

```bash
CHAT_MAX_OUTPUT_TOKENS=
DEMO_CHAT_MAX_OUTPUT_TOKENS=
JUDGE_MAX_OUTPUT_TOKENS=
AGENT_GENERAL_MAX_OUTPUT_TOKENS=
```

Current policy:

- when these settings are blank, the runtime does not force a provider-side output cap
- `CHAT_MAX_OUTPUT_TOKENS` applies to normal user-facing chat generation
- `DEMO_CHAT_MAX_OUTPUT_TOKENS` is the demo/notebook/CLI override
- `JUDGE_MAX_OUTPUT_TOKENS` applies to routing, grading, and judge-model calls
- `AGENT_<AGENT_NAME>_MAX_OUTPUT_TOKENS` is an optional per-agent chat override
- per-agent output caps affect chat generation only; they do not change judge or embedding calls

Legacy compatibility:

- `OLLAMA_NUM_PREDICT` and `DEMO_OLLAMA_NUM_PREDICT` still work as fallback aliases
- the newer `*_MAX_OUTPUT_TOKENS` settings are the canonical operator surface going forward

## Data analyst NLP override

`run_nlp_column_task` uses the same provider family as the rest of the live
runtime.

Default behavior:

- it reuses the current `data_analyst` chat bundle
- batching, validation, and repair are handled in the tool layer

Optional override behavior:

- if `DATA_ANALYST_NLP_CHAT_MODEL` is set, the runtime rebuilds a chat bundle in
  the same provider family with that chat-model override
- embeddings stay shared with the base provider bundle

Relevant settings:

- `DATA_ANALYST_NLP_CHAT_MODEL`: optional small-model override for bounded NLP
- `DATA_ANALYST_NLP_BATCH_SIZE`: row batch size per LLM call, default `5`
- `DATA_ANALYST_NLP_TEMPERATURE`: override temperature, default `0.0`

Implementation nuance:

- when `DATA_ANALYST_NLP_CHAT_MODEL` is unset, the tool reuses the base provider
  bundle and does not build a separate analyst-NLP provider stack

## Analyst sandbox image

The data-analyst sandbox is now an offline Docker image contract rather than a runtime
package-install step.

Relevant settings and commands:

- `SANDBOX_DOCKER_IMAGE`
- `SANDBOX_TIMEOUT_SECONDS`
- `SANDBOX_MEMORY_LIMIT`
- `python run.py build-sandbox-image`
- `python run.py doctor --strict`

Current local/dev default:

- `SANDBOX_DOCKER_IMAGE=agentic-chatbot-sandbox:py312`

Operational notes:

- the sandbox keeps `--network none`
- analyst packages must already exist in the image
- `doctor --strict` and notebook preflight both verify that the configured image exists locally
  and can import `pandas`, `numpy`, `openpyxl`, `xlrd`, `matplotlib`, and `pillow`
- if the probe fails, rebuild the image or point `SANDBOX_DOCKER_IMAGE` at a compatible local
  image

## Azure OpenAI example

```bash
LLM_PROVIDER=azure
JUDGE_PROVIDER=azure
EMBEDDINGS_PROVIDER=azure

AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.us/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_JUDGE_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002
EMBEDDING_DIM=1536
```

## Mixed-provider setups

Mixed setups are allowed. Example:

```bash
LLM_PROVIDER=azure
JUDGE_PROVIDER=azure
EMBEDDINGS_PROVIDER=ollama
```

## Operational checks

Useful commands:

```bash
python run.py doctor
python run.py migrate
python run.py index-skills
python run.py sync-kb
```

## Related runtime settings

The provider layer now feeds the live next runtime directly. Important nearby settings are:

- `LLM_ROUTER_ENABLED`
- `LLM_ROUTER_MODE`
- `LLM_ROUTER_CONFIDENCE_THRESHOLD`
- `MEMORY_ENABLED`
- `MEMORY_MANAGER_MODE`
- `MEMORY_SELECTOR_MODEL`
- `MEMORY_WRITER_MODEL`
- `MEMORY_CANDIDATE_TOP_K`
- `MEMORY_CONTEXT_TOKEN_BUDGET`
- `ROUTER_PATTERNS_PATH`
- `ENABLE_COORDINATOR_MODE`
- `RUNTIME_EVENTS_ENABLED`
- `MAX_PARALLEL_TOOL_CALLS`
- `DEFERRED_TOOL_DISCOVERY_ENABLED`
- `DEFERRED_TOOL_DISCOVERY_TOP_K`
- `DEFERRED_TOOL_DISCOVERY_REQUIRE_SEARCH`
- `MCP_TOOL_PLANE_ENABLED`
- `MCP_USER_SELF_SERVICE_ENABLED`
- `MCP_REQUIRE_HTTPS`
- `MCP_ALLOW_PRIVATE_NETWORK`
- `MCP_CONNECTION_TIMEOUT_SECONDS`
- `MCP_TOOL_CALL_TIMEOUT_SECONDS`
- `MCP_CATALOG_REFRESH_SECONDS`
- `AUTHZ_ENABLED`
- `TEAM_MAILBOX_ENABLED`
- `TEAM_MAILBOX_MAX_CHANNELS_PER_SESSION`
- `TEAM_MAILBOX_MAX_OPEN_MESSAGES_PER_CHANNEL`
- `TEAM_MAILBOX_CLAIM_LIMIT`
- `WORKER_SCHEDULER_ENABLED`
- `WORKER_SCHEDULER_URGENT_RESERVED_SLOTS`
- `WORKER_SCHEDULER_TENANT_BUDGET_TOKENS_PER_MINUTE`
- `WORKER_SCHEDULER_TENANT_BUDGET_BURST_TOKENS`
- `CONTEXT_BUDGET_ENABLED`
- `LLM_CIRCUIT_BREAKER_ENABLED`
- `LLM_CIRCUIT_BREAKER_WINDOW_SIZE`
- `LLM_CIRCUIT_BREAKER_MIN_SAMPLES`
- `LLM_CIRCUIT_BREAKER_ERROR_RATE_THRESHOLD`
- `LLM_CIRCUIT_BREAKER_CONSECUTIVE_FAILURES`
- `LLM_CIRCUIT_BREAKER_OPEN_SECONDS`

`LLM_ROUTER_MODE` supports:

- `hybrid`: deterministic/config-driven routing first, then LLM escalation below the confidence threshold
- `llm_only`: LLM router is primary for normal text turns, with deterministic fast paths still preserved for attachments and `force_agent`, plus deterministic fallback on LLM failure
- `MAX_REVISION_ROUNDS`
- `DATA_ANALYST_NLP_CHAT_MODEL`
- `DATA_ANALYST_NLP_BATCH_SIZE`
- `DATA_ANALYST_NLP_TEMPERATURE`

Most of those settings do not change provider construction, but they do affect
how the runtime uses the configured models. The main exception is
`DATA_ANALYST_NLP_CHAT_MODEL`, which can trigger a dedicated bounded-NLP chat
override bundle for `run_nlp_column_task`.

`LLM_ROUTER_ENABLED=false` keeps the runtime on deterministic/config-driven routing only.
When `LLM_ROUTER_ENABLED=true`, `LLM_ROUTER_MODE` selects between the `hybrid` and `llm_only`
behaviors described above.

Graph settings are adjacent operator knobs rather than model-provider settings, but they are
documented here because they affect how the retrieval subsystem uses the configured runtime.
