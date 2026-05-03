# Next Runtime Status

This file is a cutover snapshot for the next-runtime transition. Keep the facts aligned with the
live repo, but read it as historical/status context rather than the primary current-state
architecture guide.

`src/agentic_chatbot_next/` is no longer just a foundation package. It is the live runtime
used by the CLI and FastAPI gateway.

This is also an intentional breaking change for in-process callers: the removed
`agentic_chatbot.runtime.*` package and `agentic_chatbot.agents.orchestrator.ChatbotApp`
compatibility layer are no longer supported import paths.

## What changed

The original purpose of `agentic_chatbot_next` was to stage a cleaner runtime with:

- filesystem-safe runtime paths
- markdown-frontmatter agent definitions
- typed runtime contracts
- managed memory with file projections
- a cleaner session-kernel boundary

That cutover is now complete enough that the live entrypoints use the next runtime.

The runtime has also been hardened beyond the original foundation pass with:

- a thinner `RuntimeKernel` facade over extracted helper modules
- fail-fast agent markdown schema validation
- config-driven router intent patterns
- validated `metadata.requested_agent` overrides for API/demo/operator control
- provider-level circuit breakers and graceful degradation paths
- bounded coordinator revision rounds
- `research_coordinator` as the routed manager for deep corpus campaigns
- `rag_researcher` as a manual/delegated ReAct RAG researcher
- deferred `rag_workbench` tools for query planning, structure navigation, evidence grading,
  pruning, validation, and controller-hint construction
- optional reranking through the local Ollama reranker defaults
- frontend event policy and context-budget controls
- runtime skill CRUD and scoped skill versioning through the gateway
- capability profiles, RBAC clipping, deferred tool discovery, and MCP tool catalogs
- coordinator-owned typed handoffs for worker campaigns
- long-form writing through a workspace-backed multi-call composer
- managed GraphRAG catalogs and graph-augmented retrieval alongside PostgreSQL / pgvector
- an explicit offline analyst sandbox image contract checked by `doctor --strict`
- `MEMORY_ENABLED` as a runtime-wide feature flag

## Current role in the repository

Live runtime surface:

- `src/agentic_chatbot_next/app/service.py`
- `src/agentic_chatbot_next/runtime/*`
- `src/agentic_chatbot_next/router/*`
- `src/agentic_chatbot_next/tools/*`
- `src/agentic_chatbot_next/memory/*`
- `data/agents/*.md`

## Live source of truth

### Agent definitions

The live runtime resolves agents from `data/agents/*.md`. Markdown frontmatter is now the
authoritative agent-definition format.

Historical JSON agent artifacts are not part of the current runtime contract.

### Runtime paths

Runtime artifacts are keyed through `filesystem_key(...)` in
`src/agentic_chatbot_next/runtime/context.py`.

That applies to:

- session directories
- job directories
- workspace directories
- memory projection directories

### Memory

Live memory is managed in PostgreSQL when `MEMORY_ENABLED=true`. The current store uses
`memory_records`, `memory_observations`, and `memory_episodes`, with the older key/value
`memory` table retained for import and compatibility flows.

`data/memory/...` is now an inspection projection and fallback path. The projector writes:

- `index.json`
- `MEMORY.md`
- `topics/*.md`
- `groups/*.md`

When `MEMORY_ENABLED=false`, the runtime skips managed memory-store initialization, hides the
memory tool surface, disables post-turn memory management and projections, and refuses
`memory_maintainer` worker launches.

### Analyst sandbox

The live data-analyst sandbox is no longer a generic `python:3.12-slim` container plus runtime
package installation. The supported contract is a prebuilt offline image:

- default image tag: `agentic-chatbot-sandbox:py312`
- build command: `python run.py build-sandbox-image`
- readiness gate: `python run.py doctor --strict`

The demo notebook preflight uses the same readiness contract and can build the image during local
bootstrap when Docker is available.

## Cutover status

The hard cut is complete: the live runtime now owns its config, provider factories, Postgres
primitives, sandbox exceptions, and low-level ingest helpers under `src/agentic_chatbot_next/`.

That live surface now includes:

- `/v1/skills` runtime CRUD for DB-backed skill versions
- `/v1/mcp` self-service Streamable HTTP MCP connection and cached tool catalog management
- `/v1/capabilities/catalog` and `/v1/users/me/capabilities` for effective user capability
  profiles
- summarized streaming progress milestones for the UI timeline
- safe frontend audit/context events such as `agent_context_loaded`, gated by
  `FRONTEND_EVENTS_*`
- coordinator-owned typed handoff artifacts for planned worker campaigns
- optional session/job-scoped team mailbox coordination for typed peer status, handoff, and
  question flows
- `graph_manager` as a routable-or-worker graph specialist
- `research_coordinator` as the router-preferred manager for corpus-scale research campaigns
- `rag_researcher` as a non-routable but manually selectable/delegable RAG research specialist
- managed GraphRAG backed by PostgreSQL graph stores, with Neo4j compatibility guarded by
  backend settings

The import-boundary test now enforces that runtime code, tests, examples, and notebook helpers
do not import `agentic_chatbot.*`.

## Acceptance verification

The authoritative live acceptance gate is the optional scenario harness plus the executed notebook
smoke in `tests/test_next_acceptance_harness.py`.

Verified local operator flow:

```bash
python -m pip install -r new_demo_notebook/requirements.txt
ollama list
docker info
python run.py build-sandbox-image
python run.py doctor --strict
python run.py migrate
python run.py sync-kb --collection-id default
python run.py index-skills
RUN_NEXT_RUNTIME_ACCEPTANCE=1 pytest -m acceptance tests/test_next_acceptance_harness.py
RUN_NEXT_RUNTIME_NOTEBOOK_ACCEPTANCE=1 pytest -m acceptance tests/test_next_acceptance_harness.py -k notebook
```

The long-timeout harness-only env vars for this flow are:

- `NEXT_RUNTIME_GATEWAY_TIMEOUT_SECONDS`
- `NEXT_RUNTIME_JOB_WAIT_SECONDS`
- `NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS`
- `NEXT_RUNTIME_NOTEBOOK_EXECUTION_TIMEOUT_SECONDS`
- `SANDBOX_TIMEOUT_SECONDS`

`NEXT_RUNTIME_SERVER_READY_TIMEOUT_SECONDS` and
`NEXT_RUNTIME_NOTEBOOK_EXECUTION_TIMEOUT_SECONDS` are operator env knobs for the notebook and
acceptance helpers only. They are not part of `config.Settings`, the public CLI contract, or
the public API surface.

`metadata.requested_agent` is part of the public chat metadata contract, not a notebook-only
hack. The notebook delegated-grounded-RAG scenario uses it to demonstrate
`general -> rag_agent_tool`
execution even though the normal runtime often routes bounded grounded lookups directly to
`rag_worker`.

`new_demo_notebook/` remains a supported harness for demos and acceptance coverage, but it is
support infrastructure around the live next runtime rather than part of the runtime package
boundary itself.

The curated notebook itself is now scenario-first after setup/startup rather than a shared
appendix-style run. Each main scenario owns its own ingest or upload prep, prints inline
progress/tool/artifact logs as it runs, and then renders structured summaries for traces,
artifacts, metadata, and jobs. That main path now also includes a synchronous long-form writing
showcase sized below the background thresholds so operators can inspect `metadata.long_output`
plus workspace draft/manifest previews in one notebook run.

Acceptance evidence is written to:

- `new_demo_notebook/.artifacts/server.log`
- `new_demo_notebook/.artifacts/executed/agentic_system_showcase.executed.ipynb`
- `data/runtime/sessions/<filesystem_key(session_id)>/`
- `data/runtime/jobs/<filesystem_key(job_id)>/`
- `data/workspaces/<filesystem_key(session_id)>/`
- `data/memory/tenants/<tenant>/users/<user>/...` memory projections when enabled

For long-form writing, the expected split is:

- `data/workspaces/<filesystem_key(session_id)>/` for the generated draft, manifest, and any
  optional per-section files
- `data/runtime/jobs/<filesystem_key(job_id)>/` for background-job state, transcript, and event
  records
- session metadata for the assistant-facing download artifact registration

## Nearby operator settings

Important graph-related operator knobs now include:

- `GRAPH_SEARCH_ENABLED`
- `GRAPH_INGEST_ENABLED`
- `GRAPH_BACKEND`
- `GRAPHRAG_PROJECTS_DIR`
- `GRAPHRAG_CLI_COMMAND`
- `GRAPHRAG_CHAT_MODEL`
- `GRAPHRAG_EMBED_MODEL`
- `GRAPHRAG_DEFAULT_QUERY_METHOD`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- `NEO4J_TIMEOUT_SECONDS`

Those settings keep graph retrieval optional. The live runtime defaults to managed Microsoft
GraphRAG and still works without Neo4j.

Nearby non-graph defaults worth keeping aligned with code:

- `MAX_REVISION_ROUNDS=8` is the configured default; coordinator execution may apply lower
  effective caps by workflow type
- `RERANK_ENABLED=true`, `RERANK_PROVIDER=ollama`, and
  `RERANK_MODEL=rjmalagon/mxbai-rerank-large-v2:1.5b-fp16` are the local reranker defaults
- `CONTEXT_BUDGET_ENABLED=false` keeps context budgeting opt-in, but the manager is wired into
  the kernel/query loop
- `FRONTEND_EVENTS_DETAIL_LEVEL=safe_preview` is the default UI transparency posture
