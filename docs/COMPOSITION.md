# Composition

The live system is composed around `RuntimeService`.

## Composition order

1. transport layer creates request/session scope
2. `RuntimeService` prepares uploads, workspace, and route metadata
3. `RuntimeKernel` turns the live session into persisted `SessionState`
4. `AgentRegistry` selects the active `AgentDefinition`
5. `QueryLoop` dispatches the selected agent mode and injects shared context
6. tools, worker jobs, notifications, and memory operate inside that runtime context

## Runtime layers

### Service layer

Owns:

- route selection
- degraded routing / fallback decisions
- validation of optional `metadata.requested_agent` overrides for API/demo/operator turns
- eager workspace open plus upload/workspace copy behavior
- upload summary kickoff
- initial agent choice, including requested-agent override application after routing
- FastAPI-facing request scoping for runtime-effective skill CRUD under `/v1/skills`
  using `X-Tenant-ID` and `X-User-ID`

### Kernel layer

Owns:

- persistence
- events
- provider resolution and breaker-aware wrapping
- jobs
- coordinator orchestration, including document-research campaigns
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
- handoff to `general_agent.py` for tool-using `react` execution
- direct `run_rag_contract(...)` dispatch for `rag_worker`
- skill-to-hint resolution for direct RAG execution
- direct heuristic extraction for `memory_maintainer` when `MEMORY_ENABLED=true`
- file-memory context injection only when memory is enabled
- skill-context injection
- runtime-owned graph-augmented retrieval decisions when GraphRAG is enabled
- data-analyst workspace handoff into the prebuilt offline sandbox image configured by
  `SANDBOX_DOCKER_IMAGE`

## Persistence split

- PostgreSQL: documents, chunks, skill embeddings
- PostgreSQL: runtime-authored skill versions, scope metadata, and skill bodies
- optional Neo4j: graph-backed entity and relationship retrieval for GraphRAG
- `data/runtime`: session/job state, transcripts, events, notifications
- `data/workspaces`: sandbox-visible files
- `data/memory`: file-backed durable memory when `MEMORY_ENABLED=true`

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
