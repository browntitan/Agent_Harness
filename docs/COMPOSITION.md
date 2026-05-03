# Composition

The live system is composed around `RuntimeService`.

## Composition order

1. transport layer creates request/session scope
2. `RuntimeService` prepares uploads, workspace, and route metadata
3. `RuntimeKernel` turns the live session into persisted `SessionState`
4. `AgentRegistry` selects the active `AgentDefinition`
5. `QueryLoop` dispatches the selected agent mode and injects shared context
6. context budgeting optionally compacts history/tool results before model execution
7. capabilities, authorization, tools, worker jobs, notifications, MCP catalogs, and memory
   operate inside that runtime context

## Runtime layers

### Service layer

Owns:

- route selection
- degraded routing / fallback decisions
- validation of optional `metadata.requested_agent` overrides for API/demo/operator turns
- eager workspace open plus upload/workspace copy behavior
- upload summary kickoff
- initial agent choice, including requested-agent override application after routing
- defaulting broad corpus research to `research_coordinator` when router/deep-RAG policy
  identifies repository-scale grounded work
- effective capability profile resolution for user-visible agents, tools, collections, skills,
  MCP tools, and fast-path policy
- FastAPI-facing request scoping for runtime-effective skill CRUD under `/v1/skills`
  using `X-Tenant-ID` and `X-User-ID`
- FastAPI-facing MCP, capability, task/job, team mailbox, graph, and admin-control-panel APIs

### Kernel layer

Owns:

- persistence
- events
- provider resolution and breaker-aware wrapping
- jobs
- worker scheduling and token-budget admission when `WORKER_SCHEDULER_ENABLED=true`
- coordinator orchestration, including document-research campaigns
- `research_coordinator` execution, which reuses coordinator mode with a research-specific
  worker allow-list
- coordinator-owned typed handoff artifact validation, persistence, and worker injection
- notification drain
- post-turn memory maintenance only when `MEMORY_ENABLED=true`

Implementation note:

- `RuntimeKernel` is the facade
- `kernel_events.py`, `kernel_providers.py`, and `kernel_coordinator.py` hold the
  extracted concern-specific logic behind that facade

### Loop layer

Owns:

- prompt construction for prompt-backed modes
- execution dispatch by mode
- manual/delegated `rag_researcher` ReAct execution for exploratory RAG research
- handoff to `general_agent.py` for tool-using `react` execution
- direct `run_rag_contract(...)` dispatch for `rag_worker`
- skill-to-hint resolution for direct RAG execution
- direct heuristic extraction for `memory_maintainer` when `MEMORY_ENABLED=true`
- managed-memory context injection only when memory is enabled, with file projections/fallbacks
- skill-context injection
- runtime-owned graph-augmented retrieval decisions when GraphRAG is enabled
- data-analyst workspace handoff into the prebuilt offline sandbox image configured by
  `SANDBOX_DOCKER_IMAGE`
- context-budget ledgers for prompt sections, autocompaction, and tool-result sidecars

## Persistence split

- PostgreSQL: documents, chunks, skill embeddings
- PostgreSQL: runtime-authored skill versions, scope metadata, and skill bodies
- PostgreSQL: managed memory records, observations, episodes, legacy key/value imports, and
  requirement statements
- PostgreSQL: access-control rows, capability profiles, MCP connections/tool catalogs, graph
  indexes/sources/runs/query-cache rows, and canonical entities
- retrieval reranker state is request metadata only; reranker model residency is managed by the
  configured provider such as local Ollama
- managed GraphRAG project artifacts with optional Neo4j compatibility backend
- `data/runtime`: session/job state, transcripts, events, notifications
- `data/runtime`: context compaction records and large tool-result sidecars when context
  budgeting is enabled
- `data/workspaces`: sandbox-visible files
- `data/memory`: memory projections and fallback files when `MEMORY_ENABLED=true`

The analyst sandbox keeps the same workspace bind-mount contract under the new image model. What
changed is package provisioning: `doctor --strict`, notebook preflight, and
`python run.py build-sandbox-image` now treat the sandbox image itself as the analyst dependency
contract.

## Typed handoffs and peer follow-ups

Coordinator-owned worker handoffs now flow through typed session/job artifacts instead of
depending on unstructured cross-worker routing.

Current live artifact types:

- `analysis_summary`
- `entity_candidates`
- `keyword_windows`
- `doc_focus`
- `evidence_request`
- `evidence_response`

Those artifacts are validated by the coordinator layer, surfaced in runtime progress, and
then injected into downstream worker requests in a bounded form.

For ad hoc same-session follow-ups, prompt-backed agents may also use `invoke_agent` to queue
one bounded peer request through the same durable job/mailbox system. Typed handoffs remain
the preferred path when the workflow is a planned multi-worker campaign.

When `TEAM_MAILBOX_ENABLED=true`, planned campaigns can also create a session/job-scoped team
channel. The channel is an async coordination surface for typed status updates, handoffs,
questions, and operator-owned approval requests. It reuses the runtime transcript/job JSONL
store and does not change worker permissions, sandbox scope, or allowed-tool policy.

## Research Roles And Frontend Events

The runtime now separates three RAG-oriented patterns:

- `rag_worker`: direct stable-contract retrieval and synthesis
- `rag_researcher`: ReAct source-selection specialist using `rag_workbench` before final RAG
  synthesis
- `research_coordinator`: coordinator-mode manager for broad corpus campaigns

Streaming clients see only policy-filtered frontend events. `FRONTEND_EVENTS_*` settings decide
whether status, agents, tools, parallel groups, prompt/skill/context metadata, and safe previews
are forwarded; durable runtime events remain the source of truth.
