# Observability with Langfuse and Local Runtime Events

This project has two observability layers:

1. LangChain callbacks, usually backed by Langfuse
2. durable local runtime artifacts written by the live `agentic_chatbot_next` runtime

For streaming chat turns, there is also a third, non-durable user-facing layer:

3. an ephemeral `progress` SSE stream derived from runtime events and callbacks

Langfuse is optional. The authoritative local ground truth for session and worker behavior is
the file-backed persistence written from `src/agentic_chatbot_next/runtime/*`.

## Langfuse layer

The repo-level observability surface re-exports the live next-runtime callbacks from
`src/agentic_chatbot_next/observability/__init__.py`.

The current callback implementation lives in
`src/agentic_chatbot_next/observability/callbacks.py`.

When Langfuse keys are set, the live runtime attaches callbacks for:

- chat turns
- upload ingest
- model calls inside general-agent execution
- model calls inside RAG execution
- planner, finalizer, and verifier direct invocations
- worker execution paths launched through the live next runtime

Provider breaker lifecycle is tracked through local runtime events rather than Langfuse-only
callbacks.

If callback setup fails, the runtime returns an empty callback list and continues.

## Local runtime observability

Runtime artifacts are keyed with `filesystem_key(...)` from
`src/agentic_chatbot_next/runtime/context.py`.

### Session files

```text
data/runtime/sessions/<filesystem_key(session_id)>/
  state.json
  transcript.jsonl
  events.jsonl
  notifications.jsonl
```

### Job files

```text
data/runtime/jobs/<filesystem_key(job_id)>/
  state.json
  transcript.jsonl
  events.jsonl
  mailbox.jsonl
  artifacts/
    output.md
    result.json
```

Background long-form writing uses the same durable job directory for job state, transcripts,
and progress events. The generated report itself does not live under
`data/runtime/jobs/.../artifacts/`; the long-form composer writes the user-facing draft,
manifest, and optional per-section files into the session workspace and then registers the
workspace draft as a download artifact.

Related local artifacts:

- `data/workspaces/<filesystem_key(session_id)>/`
- `data/memory/tenants/<tenant>/users/<user>/...` memory projections when enabled
- `new_demo_notebook/.artifacts/server.log`
- `new_demo_notebook/.artifacts/executed/agentic_system_showcase.executed.ipynb`

Artifact naming note:

- "runtime artifacts" in this document means durable local files under
  `data/runtime/...` plus related notebook and workspace evidence
- user-facing download artifacts are separate session metadata entries created by
  `return_file`; they point at workspace files and surface through chat
  `artifacts` plus `GET /v1/files/{download_id}`
- long-form writing reuses that same assistant-facing download contract, but the draft,
  `long_output_*_manifest.json`, and optional section files are produced directly by the
  runtime composer rather than by a tool-returned file handle
- those download artifacts are not the same thing as job
  `artifacts/output.md` or `artifacts/result.json`

## What gets recorded

### Session transcript

The session transcript records:

- accepted user messages
- assistant outputs
- task notifications

Task notifications may appear in more than one durable location:

- `notifications.jsonl`
- `state.json` under `pending_notifications`
- `transcript.jsonl` rows with `kind="notification"`

Operational nuance: `notifications.jsonl` can be empty after notification drain. For durable
acceptance triage, also inspect `state.json` and `transcript.jsonl`.

### Runtime events

The runtime emits structured `RuntimeEvent` records for:

- router decisions: `router_decision`
- turn lifecycle: `turn_accepted`, `turn_completed`, `turn_failed`
- BASIC-turn lifecycle: `basic_turn_started`, `basic_turn_completed`, `basic_turn_failed`
- agent lifecycle: `agent_run_started`, `agent_run_completed`
- AGENT-turn lifecycle: `agent_turn_started`, `agent_turn_completed`, `agent_turn_failed`
- coordinator phases:
  `coordinator_planning_started`, `coordinator_planning_completed`,
  `coordinator_batch_started`, `coordinator_finalizer_completed`,
  `coordinator_verifier_completed`, `coordinator_revision_round_started`,
  `coordinator_revision_limit_reached`
- worker lifecycle: `worker_agent_started`, `worker_agent_completed`
- job lifecycle: `job_created`, `job_started`, `job_completed`, `job_failed`, `job_stopped`
- mailbox lifecycle: `mailbox_enqueued`
- notification lifecycle: `notification_appended`
- memory extraction lifecycle:
  `memory_extraction_started`, `memory_extraction_completed`,
  `memory_extraction_failed`, `memory_extraction_skipped`
- managed memory manager lifecycle:
  `memory_manager_completed`, `memory_manager_failed`
- callback-driven model lifecycle: `model_start`, `model_end`, `model_error`
- callback-driven tool lifecycle: `tool_start`, `tool_end`, `tool_error`
- deferred tool lifecycle:
  `deferred_tool_catalog_built`, `deferred_tool_discovery_searched`,
  `deferred_tool_invoked`, `deferred_tool_denied`
- team mailbox lifecycle:
  `team_mailbox_channel_created`, `team_mailbox_message_posted`,
  `team_mailbox_message_claimed`, `team_mailbox_message_resolved`,
  `team_mailbox_digest_created`
- provider resilience and degraded-service lifecycle:
  `llm_circuit_breaker_opened`, `llm_circuit_breaker_half_opened`,
  `llm_circuit_breaker_closed`, `router_degraded_to_deterministic`,
  `agent_downgraded_to_basic`, `degraded_response_returned`

Worker execution failures currently show up through `job_failed`; the runtime does not emit a
separate `worker_agent_failed` event today.

Each event row carries the persisted `RuntimeEvent` envelope:

- `created_at`
- `session_id`
- `job_id`
- `agent_name`
- `tool_name`
- `payload`

When available, the payload also includes runtime-specific fields such as:

- `conversation_id`
- `route`
- `router_method`
- `suggested_agent`
- `requested_agent_override`
- `requested_agent_override_applied`
- long-form job ids, result summaries, and workspace output paths when a turn is routed through
  the long-form writing path
- coordinator worker and verifier metadata
- RAG worker task ids and doc scopes for worker lifecycle events
- capability-profile source and hidden/unavailable values when gateway responses include
  effective capability metadata
- scheduler state, queue class, token budget, and budget block reason on task/job responses

## Streaming progress layer

The streaming API path in `src/agentic_chatbot_next/api/main.py` now creates a
`LiveProgressSink`.

That sink is fed from:

- router and kernel lifecycle events
- worker/job lifecycle events
- LangChain callback events from `ProgressCallback`
- adaptive RAG controller milestones and document-focus updates

Unlike `events.jsonl`, this stream is not durable and is meant for the live inline status UI.

Current live progress event families include:

- `route_decision`
- `agent_selected`
- `agent_start`
- `decision_point`
- `phase_start`, `phase_update`, `phase_end`
- `task_plan`
- `worker_start`, `worker_end`
- `doc_focus`
- `tool_intent`
- `evidence_status`
- `handoff_prepared`, `handoff_consumed`
- `summary`
- `tool_call`, `tool_result`, `tool_error`

Those events are intentionally summarized. They expose routing, current phase, active worker
tasks, and current document/file focus without exposing raw chain-of-thought.

The scenario-first demo notebook consumes that same live stream per scenario cell. It prints the
progress/tool/artifact timeline inline while the cell runs, then renders structured summaries
afterward instead of asking users to inspect raw SSE payloads manually. The notebook helper now
also captures `event: metadata`, which lets the synchronous long-form writing showcase display
`metadata.long_output`, output/manifest filenames, and workspace previews without background-job
polling.

Common payload fields in that stream include:

- `type`
- `label`
- `detail`
- `agent`
- `job_id`
- `task_id`
- `status`
- `docs`
- `counts`
- `timestamp`
- `why`
- `waiting_on`

When `MEMORY_ENABLED=false`, the runtime does not emit managed memory or
`memory_extraction_*` event families in either the durable event log or the streaming progress
layer because the memory-maintenance paths are disabled entirely.

## Acceptance triage locations

When live acceptance fails, inspect artifacts in this order:

1. `new_demo_notebook/.artifacts/server.log`
2. `data/runtime/sessions/<filesystem_key(session_id)>/events.jsonl`
3. `data/runtime/sessions/<filesystem_key(session_id)>/state.json`
4. `data/runtime/sessions/<filesystem_key(session_id)>/transcript.jsonl`
5. `data/runtime/jobs/<filesystem_key(job_id)>/state.json`
6. `data/runtime/jobs/<filesystem_key(job_id)>/events.jsonl`
7. `data/runtime/jobs/<filesystem_key(job_id)>/artifacts/output.md`
8. `data/runtime/jobs/<filesystem_key(job_id)>/artifacts/result.json`
9. `data/workspaces/<filesystem_key(session_id)>/`
10. `data/memory/tenants/<tenant>/users/<user>/...` memory projections when enabled

For long-form writing failures or partial completions, inspect the workspace directory for the
generated draft plus `long_output_*_manifest.json` and any `long_output_*_section_*.md` files,
then correlate them with the background job `state.json` and `events.jsonl` when the request ran
asynchronously.

These files are the durable acceptance artifacts for server readiness, worker orchestration,
RAG grounding, data-analyst execution, coordinator job flow, and memory/notification
verification.

For MCP/capability issues, inspect the gateway response, control-panel operation logs, and
PostgreSQL-backed MCP/capability records first; local runtime files only show the tool
binding/invocation consequences once a chat turn or job runs.

## Why both layers exist

Langfuse is useful for centralized trace visualization.

Local runtime files are useful for:

- resume/debug behavior
- worker-job inspection
- auditability when external tracing is unavailable
- development setups where Langfuse is not configured

The live SSE progress layer is useful for:

- UI task summaries while a turn is still running
- showing routed agent, active workers, and current doc focus
- incremental UX feedback without waiting for the final assistant text

## Relevant settings

- `LANGFUSE_HOST`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_DEBUG`
- `RUNTIME_EVENTS_ENABLED`
- `RUNTIME_DIR`
- `MEMORY_MANAGER_MODE`
- `DEFERRED_TOOL_DISCOVERY_ENABLED`
- `MCP_TOOL_PLANE_ENABLED`
- `TEAM_MAILBOX_ENABLED`
- `WORKER_SCHEDULER_ENABLED`
- `CONTEXT_BUDGET_ENABLED`
- `AUTHZ_ENABLED`

## Operational takeaway

If Langfuse is enabled, use it for trace exploration.

If you need the durable local source of truth, inspect the artifacts written by
`src/agentic_chatbot_next/runtime/*`.
