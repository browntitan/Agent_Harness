# OpenAI-Compatible Gateway

Use the repo-root [`README.md`](../README.md) for canonical startup, restart, database, KB,
and Open WebUI operations guidance. This document is the gateway/API contract deep reference.

The live FastAPI gateway is `src/agentic_chatbot_next/api/main.py`.

It exposes the next runtime through OpenAI-style endpoints without changing the
internal runtime contracts.

## Supported endpoints

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/admin/runtime/diagnostics`
- `GET/POST/PATCH/PUT/DELETE /v1/admin/...` control-panel routes for overview, operations,
  access, capabilities, architecture, config, agents, prompts, graphs, uploads, and collections
- `GET /v1/models`
- `GET /v1/agents`
- `POST /v1/chat/completions`
- `POST /v1/connector/chat`
- `POST /v1/sessions/{session_id}/compact`
- `GET /v1/capabilities/catalog`
- `GET /v1/users/me/capabilities`
- `PUT /v1/users/me/capabilities`
- `GET /v1/tasks`
- `GET /v1/tasks/{task_id}`
- `POST /v1/tasks/{task_id}/stop`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/mailbox`
- `POST /v1/jobs/{job_id}/mailbox/{message_id}/respond`
- `GET /v1/sessions/{session_id}/team-mailbox/channels`
- `POST /v1/sessions/{session_id}/team-mailbox/channels`
- `GET /v1/sessions/{session_id}/team-mailbox/messages`
- `POST /v1/sessions/{session_id}/team-mailbox/channels/{channel_id}/messages`
- `POST /v1/sessions/{session_id}/team-mailbox/channels/{channel_id}/messages/{message_id}/respond`
- `GET /v1/files/{download_id}`
- `GET /v1/documents/{doc_id}/source`
- `GET /v1/graphs`
- `GET /v1/graphs/{graph_id}`
- `POST /v1/graphs/index`
- `POST /v1/graphs/import`
- `POST /v1/graphs/query`
- `POST /v1/ingest/documents`
- `POST /v1/upload`
- `GET /v1/skills`
- `GET /v1/skills/{skill_id}`
- `POST /v1/skills`
- `PUT /v1/skills/{skill_id}`
- `POST /v1/skills/{skill_id}/activate`
- `POST /v1/skills/{skill_id}/deactivate`
- `POST /v1/skills/{skill_id}/rollback`
- `POST /v1/skills/{skill_id}/preview-execution`
- `POST /v1/skills/preview`
- `GET /v1/mcp/connections`
- `POST /v1/mcp/connections`
- `PATCH /v1/mcp/connections/{connection_id}`
- `DELETE /v1/mcp/connections/{connection_id}`
- `POST /v1/mcp/connections/{connection_id}/test`
- `POST /v1/mcp/connections/{connection_id}/refresh-tools`
- `GET /v1/mcp/connections/{connection_id}/tools`
- `PATCH /v1/mcp/connections/{connection_id}/tools/{tool_id}`

## Live runtime binding

The gateway resolves its live runtime through `get_runtime_manager()` and serves the current
`RuntimeService` snapshot from `src/agentic_chatbot_next/app/service.py`.

The public model contract remains `GATEWAY_MODEL_ID=enterprise-agent`.
`POST /v1/chat/completions` rejects any other `model` value even if the runtime
internally uses different per-agent chat or judge model overrides.

`GET /health/ready` is KB-aware. It returns `200` only when providers are
healthy and the configured KB/docs corpus is indexed for the active collection.
When coverage is missing, it returns `503` with `reason`, `collection_id`,
`missing_sources`, and `suggested_fix`.

## Request scoping

The gateway now accepts explicit runtime scope headers in addition to the OpenAI-style
payload:

- `X-Conversation-ID`
- `X-Request-ID`
- `X-Tenant-ID`
- `X-User-ID`
- `X-User-Email`

For Open WebUI compatibility, the gateway also accepts these forwarded header aliases:

- `X-OpenWebUI-Chat-Id` as an alias for `X-Conversation-ID`
- `X-OpenWebUI-Message-Id` as an alias for `X-Request-ID`
- `X-OpenWebUI-User-Id` as an alias for `X-User-ID`
- `X-OpenWebUI-User-Email` as an alias for `X-User-Email`

The skill control plane uses those scope headers to resolve runtime-authored skill
visibility, precedence, and ownership without requiring a process restart.

Capability profiles, MCP catalogs, graph access, task/job mailboxes, team mailboxes, document
source downloads, and control-panel admin APIs also use these headers so user/tenant policy is
resolved consistently across chat-adjacent operations.

For retrieval-aware callers such as the OpenWebUI pipe, `POST /v1/chat/completions`
also supports these optional metadata keys. The default OpenWebUI deployment runs in
thin mode: OpenWebUI may transport uploaded bytes, but the backend document repository is
the only trusted retrieval and citation source.

- `metadata.collection_id`
- `metadata.upload_collection_id`
- `metadata.kb_collection_id`
- `metadata.available_kb_collection_ids`
- `metadata.kb_collection_confirmed`
- `metadata.user_email`
- `metadata.openwebui_thin_mode`
- `metadata.document_source_policy` with `agent_repository_only`
- `metadata.uploaded_doc_ids` using internal repository document ids
- top-level `userEmail`

## Per-user authorization

When `AUTHZ_ENABLED=true`, the runtime resolves a fresh RBAC snapshot from the trusted
request email on every chat turn, upload/ingest call, graph request, and worker-job start.

Current behavior:

- authorization is deny-by-default for protected resources
- protected resource types are KB collections, named graphs, tool grants, and skill families
- the Open WebUI pipe is expected to forward a trustworthy user email
- the chat-scoped upload collection remains implicitly usable by that chat/session
- graph access requires both a graph grant and a grant to the graph's backing collection
- tool access is the intersection of agent-allowed tools and user-allowed tools
- skill access is bound to the skill family (`version_parent`), not a single version id
- capability profiles can further enable or disable tools, tool groups, agents, collections,
  skills, MCP tools, plugins, permission mode, and fast-path policy

Skill mutation hardening:

- `POST /v1/skills`
- `PUT /v1/skills/{skill_id}`
- `POST /v1/skills/{skill_id}/activate`
- `POST /v1/skills/{skill_id}/deactivate`
- `POST /v1/skills/{skill_id}/rollback`

These mutation routes now require either:

- `X-Admin-Token` matching `CONTROL_PANEL_ADMIN_TOKEN`
- or an RBAC `skill_family:manage` grant when authz is enabled

`POST /v1/skills/{skill_id}/preview-execution` follows the same admin/manage rule because
it exposes executable instructions, although it does not mutate state.

The control-panel admin surface now also exposes:

- `GET /v1/admin/access/principals`
- `GET /v1/admin/access/roles`
- `GET /v1/admin/access/bindings`
- `GET /v1/admin/access/permissions`
- `GET /v1/admin/access/effective-access`

Backward compatibility rules:

- if only `collection_id` is provided, the gateway treats it as the active upload
  and KB collection for legacy single-collection callers
- if `upload_collection_id` and `kb_collection_id` are provided, the runtime keeps
  uploads chat-scoped while using the shared KB collection for coverage checks
- `kb_collection_id` selects the active KB collection for the chat; it does not
  define the full set of KB collections visible to the runtime
- callers can set `kb_collection_confirmed=false` when `kb_collection_id` is only
  a bootstrap default and the runtime should still ask the user to choose among
  multiple visible KB collections

## Gateway auth

When `GATEWAY_SHARED_BEARER_TOKEN` is set, these endpoints require
`Authorization: Bearer <token>`:

- `GET /v1/admin/runtime/diagnostics`
- `GET /v1/models`
- `GET /v1/agents`
- `POST /v1/chat/completions`
- `POST /v1/sessions/{session_id}/compact`
- `POST /v1/upload`
- `POST /v1/ingest/documents`
- `GET /v1/files/{download_id}`
- `GET /v1/documents/{doc_id}/source`
- `/v1/graphs...`
- `/v1/mcp...`
- `/v1/capabilities/catalog`
- `/v1/users/me/capabilities`
- `/v1/tasks...`
- `/v1/jobs...`
- `/v1/sessions/{session_id}/team-mailbox...`

Browser-safe download handoff uses signed URLs generated when
`DOWNLOAD_URL_SECRET` is set. Signed links remain valid for
`DOWNLOAD_URL_TTL_SECONDS`.

`POST /v1/connector/chat` uses a separate connector auth layer so browser-direct
or integration clients do not need the internal gateway bearer token:

- `CONNECTOR_SECRET_API_KEY` is the server-side integration key
- `CONNECTOR_PUBLISHABLE_API_KEY` is optional for trusted browser-direct apps
- publishable keys are restricted to `CONNECTOR_ALLOWED_ORIGINS`
- publishable keys are rate-limited by `CONNECTOR_PUBLISHABLE_RATE_LIMIT_PER_MINUTE`

If `CONNECTOR_SECRET_API_KEY` is unset, it falls back to
`GATEWAY_SHARED_BEARER_TOKEN` for local/dev compatibility.

## Requested-agent override

`POST /v1/chat/completions` also accepts an optional `metadata.requested_agent` field.

Use it when you explicitly want to start the AGENT path in a particular routable role for
operator control, notebook walkthroughs, or acceptance coverage.

Current behavior:

- the router still runs first and records the normal `BASIC` vs `AGENT` decision
- `force_agent=true` still only forces the AGENT route; it does not choose the initial agent
- when `metadata.requested_agent` is present and valid, the runtime uses it instead of the
  normal router-policy-selected starting agent
- the trace metadata keeps both `requested_agent_override` and
  `requested_agent_override_applied`

Validation rules:

- values are validated against the current routable non-`basic` registry surface
- current common values are `general`, `rag_worker`, `data_analyst`, `coordinator`, and
  `graph_manager`
- `memory_maintainer` is filtered out when `MEMORY_ENABLED=false`
- invalid values return `400` with the allowed value list

## Long-form writing

`POST /v1/chat/completions` also accepts an optional `metadata.long_output` object.

Use it when the caller wants the runtime to generate a long document across multiple model
calls, persist the draft into the session workspace, and return the finished draft as a normal
download artifact instead of trying to fit everything into one assistant response.

Supported fields:

- `enabled`
- `target_words`
- `target_sections`
- `delivery_mode`
- `background_ok`
- `output_format`
- `async_requested`

Current behavior:

- this is an opt-in runtime orchestration feature, not a separate agent mode
- the runtime still routes the turn first, then applies long-form writing on top of the selected
  route/agent
- the selected agent still contributes prompt/style guidance
- the full draft is written into the session workspace as Markdown by default
- the assistant history only keeps a short summary plus artifact metadata

Background behavior:

- the runtime stays synchronous for moderate requests
- it may queue a background job when:
  - `async_requested=true`
  - `target_words` is above roughly `3000`
  - `target_sections` is greater than `5`
  - the selected agent allows background jobs

Generated workspace files may include:

- the final draft, usually `long_output_<hash>_<slug>.md`
- `long_output_<hash>_manifest.json`
- optional per-section files such as `long_output_<hash>_section_01.md`

## Chat completions flow

`POST /v1/chat/completions`:

1. validates the requested gateway model id
2. builds a local request context using `X-Conversation-ID`
3. converts prior OpenAI-format messages into LangChain history
4. validates optional `metadata.requested_agent`
5. resolves request scope, authorization, and effective capability profile inputs
6. creates a `ChatSession`
7. calls `RuntimeService.process_turn(...)`
8. wraps the returned assistant text back into OpenAI-compatible JSON or SSE
   chunks

Request length control:

- `max_tokens` is honored for the user-facing chat generation step of that turn
- it overrides the active global or per-agent chat output-token settings
- it does not expand judge-model or embedding calls

When the final assistant message contains returned workspace files, the gateway
also exposes normalized download artifacts.

Non-stream behavior:

- the normal OpenAI-style response still contains `choices`
- returned files appear in a top-level `artifacts` array
- long-form responses may also include top-level `metadata.job_id`
- long-form responses may also include top-level `metadata.long_output`

Stream behavior with `stream=true`:

- named `progress` events may arrive before text chunks and continue while the turn runs
- standard OpenAI-style `chat.completion.chunk` payloads stream the assistant
  text
- if files were returned, the gateway emits `event: artifacts` before `[DONE]`
- long-form turns may also emit `event: metadata` carrying `job_id` and `long_output`
  details

Some clients may ignore `event: metadata`, but the API emits it for clients
that want to observe background job ids or long-form result metadata during streaming.

The live streaming path now creates a `LiveProgressSink` and feeds it from:

- router decisions
- kernel/runtime lifecycle events
- LangChain tool callbacks
- adaptive RAG controller phase updates

The gateway keeps the SSE event name as `progress` and expands the JSON payload shape.
Common fields now include:

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

Current progress event families include:

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

Router-derived progress and persisted route metadata may also include
`requested_agent_override` and `requested_agent_override_applied`.

In Open WebUI or another compatible client, that stream can power an inline reasoning/task
summary panel rather than a separate sidecar debugger view.

## Legacy Connector Endpoint

`POST /v1/connector/chat` remains available for browser-direct and integration clients.
Open WebUI is the supported v3 chat UI, so this endpoint is compatibility surface rather
than the primary user experience.

It keeps the existing gateway/runtime behavior but hides the glue work that a
consumer app would otherwise need to write:

1. accepts JSON chat messages
2. accepts file attachments in either JSON form or raw multipart
3. uploads those files through the existing `/v1/upload` path
4. forwards the chat turn to `/v1/chat/completions`
5. translates the gateway SSE into a structured UI message stream

### Request shapes

JSON requests are the best default for browser-direct clients:

```json
{
  "id": "chat_123",
  "messages": [
    {
      "id": "msg_user_1",
      "role": "user",
      "parts": [
        { "type": "text", "text": "Summarize the attached CSV." },
        {
          "type": "file",
          "id": "file_1",
          "filename": "sales.csv",
          "mediaType": "text/csv",
          "url": "data:text/csv;base64,..."
        }
      ]
    }
  ],
  "metadata": {
    "upload_collection_id": "owui-chat-123",
    "kb_collection_id": "default"
  }
}
```

For raw multipart clients, send:

- `payload`: JSON body as a string
- `files[]`: binary file parts

### Response stream

The connector returns an SSE response with:

- `x-vercel-ai-ui-message-stream: v1`
- `start`, `text-start`, `text-delta`, `text-end`, and `finish` parts
- transient `data-status` parts for live runtime progress
- `data-artifact` parts plus standard `file` parts for downloadable outputs
- `data-metadata` parts for message-level runtime details such as job ids

### Compatibility split

Use the gateway like this in downstream apps:

- Open WebUI: installed pipe against `/v1/chat/completions`
- Browser-direct compatibility: `POST /v1/connector/chat`
- OpenAI-compatible server clients: base URL `/v1`

That keeps normal OpenAI-compatible access for server-side model calls while v3 keeps
Open WebUI as the supported chat interface.

Artifact objects use the normalized shape from `runtime/artifacts.py`:

```text
download_id
artifact_ref
filename
label
download_url
content_type
size_bytes
session_id
conversation_id
```

Long-form chat responses reuse the same artifact contract. The only difference is that the
artifact usually points at a workspace-backed report draft rather than a file published by a
tool like `return_file`.

The top-level `metadata` object on chat responses may now also include:

- `job_id`
- `long_output`

`metadata.long_output` currently carries high-level runtime information such as:

- `title`
- `output_filename`
- `manifest_filename`
- `section_count`
- `background`

## Capability and MCP APIs

`GET /v1/capabilities/catalog` returns the server-visible catalog of agents, tools, tool
groups, skill-pack selectors, MCP tools, collections, permission modes, and fast-path policies.

`GET /v1/users/me/capabilities` returns the effective profile for the current tenant/user.
`PUT /v1/users/me/capabilities` updates that profile when allowed. Profile fields include
enabled/disabled tools, enabled tool groups, skill-pack ids, MCP tool ids, enabled agents,
enabled collections, plugin preferences, `permission_mode`, and `fast_path_policy`.

The MCP API manages user-owned Streamable HTTP MCP connections and cached tool catalogs:

- connection list/create/update/delete
- connection test
- catalog refresh
- per-connection tool list
- per-tool enable/visibility metadata updates

The runtime does not bind remote MCP calls directly from the connection profile. It refreshes
catalog rows into Postgres, exposes those rows as dynamic `mcp__...` tool definitions, and
invokes the remote `tools/call` endpoint only when an allowed tool is actually called.

## Runtime skill API

The gateway also exposes a runtime skill control plane for retrievable, executable, and
hybrid skill packs.

Current supported operations:

- list skills: `GET /v1/skills`
- inspect one skill: `GET /v1/skills/{skill_id}`
- create a new skill version: `POST /v1/skills`
- update an existing skill record: `PUT /v1/skills/{skill_id}`
- activate a skill: `POST /v1/skills/{skill_id}/activate`
- deactivate a skill: `POST /v1/skills/{skill_id}/deactivate`
- rollback to a previous version family: `POST /v1/skills/{skill_id}/rollback`
- preview executable prompt/config rendering: `POST /v1/skills/{skill_id}/preview-execution`
- preview retrieval resolution without mutating live state: `POST /v1/skills/preview`

Operational rules:

- skill updates are runtime-effective on the next retrieval/search; restart is not required
- API-created executable skill updates are runtime-effective on the next tool binding or
  execution; file-authored packs can opt into polling hot reload with
  `SKILL_PACKS_HOT_RELOAD_ENABLED=true`
- file-authored skill packs still exist, but runtime-authored skills are first-class
- executable skills require `EXECUTABLE_SKILLS_ENABLED=true`
- scope/precedence is high-level:
  - user-private override
  - tenant-shared
  - global default
- status/version model is high-level:
  - `draft`
  - `active`
  - `archived`
  - version families linked through `version_parent`

## Graph API

The gateway also exposes a managed graph catalog/query surface.

Current supported operations:

- list graphs: `GET /v1/graphs`
- inspect one graph: `GET /v1/graphs/{graph_id}`
- query one graph or a shortlist of relevant graphs: `POST /v1/graphs/query`
- attempt index/create: `POST /v1/graphs/index`
- attempt import/register: `POST /v1/graphs/import`

Current behavior:

- graph list, inspect, and query require the normal gateway bearer token
- graph calls use the same conversation/request/tenant/user headers as chat so graph access stays
  session-scoped, tenant-scoped, and user-filtered
- graph visibility is enforced consistently across list, inspect, and query:
  - `private` graphs are owner-only within the tenant
  - `tenant` graphs are visible to any user in the same tenant
  - `global` currently behaves the same as tenant-visible because managed graph storage is still tenant-scoped
- inspect and query calls persist `active_graph_ids` into the session state so graph-aware follow-up turns can stay grounded in the same graph set
- `POST /v1/graphs/index` and `POST /v1/graphs/import` are intentionally blocked with `403`
  because graph creation and refresh are admin-managed in the control panel today
- graph query returns the same high-level `graph.query.result` object whether it targets one
  explicit graph or a shortlist across a collection
- graph payloads now include readiness, visibility, query backend, supported query methods, and graph context summary fields so callers can explain why a graph was or was not used

`graph_manager` is the routable graph specialist for chat turns that need graph-backed
evidence, graph relationships, graph inventory, or source planning. Direct graph API calls are
still useful for UI panels and operators, while chat turns go through router/policy and the
agent tool plane.

## Task, Job, and Mailbox APIs

`GET /v1/tasks` and `GET /v1/tasks/{task_id}` expose high-level task/job snapshots. Stop
requests go through `POST /v1/tasks/{task_id}/stop`.

`GET /v1/jobs/{job_id}` returns the durable job state. Job payloads may include priority,
queue class, scheduler state, token budget counters, budget block reasons, result summaries,
artifacts, and long-form metadata.

Worker mailbox APIs let a parent/operator inspect pending worker questions and respond to
question requests. Approval requests require the operator/admin path; answering a mailbox item
does not broaden the worker's allowed tools.

Team mailbox endpoints are session-scoped and only useful when `TEAM_MAILBOX_ENABLED=true`.
They manage channels and typed messages for status updates, handoffs, questions, and responses.
They are coordination surfaces, not shared permission scopes.

## Download file flow

`GET /v1/files/{download_id}`:

1. resolves conversation scope from `conversation_id` or `X-Conversation-ID`
2. loads the persisted session state
3. reads the session `downloads` metadata
4. validates that the requested file still exists in the session workspace
5. serves the file with `FileResponse`

This endpoint only serves session-registered workspace files. It is the public
handoff path used by `return_file`.

When signed query params are present (`tenant_id`, `user_id`, `conversation_id`,
`expires`, `sig`), the endpoint allows browser downloads without a bearer token.

`GET /v1/documents/{doc_id}/source` is the protected source-file download path for indexed
documents. It uses the same request scope and authorization checks as retrieval/graph access
and is separate from session workspace artifact downloads.

## Job status flow

`GET /v1/jobs/{job_id}`:

1. requires the normal gateway bearer token
2. resolves the session scope from the same conversation/user headers as chat
3. loads the persisted job record from the runtime job manager
4. validates that the job belongs to the active session scope
5. returns high-level job state plus any normalized download artifacts

Current response fields include:

- `status`
- `result_summary`
- `output_path`
- `artifacts`
- `metadata`

For long-form writing jobs, `metadata.long_output_result` is the main high-level payload to
inspect after completion. It may include:

- `title`
- `output_filename`
- `manifest_filename`
- `section_count`

## Document ingest flow

`POST /v1/ingest/documents`:

1. resolves the request context
2. ingests files through `agentic_chatbot_next.rag.ingest_paths(...)`
3. opens the canonical session workspace using
   `SessionWorkspace.for_session(...)` with the derived runtime `session_id`
4. copies ingested files into that workspace
5. returns ingest metadata

This keeps upload scope aligned with later data-analyst turns.

## Multipart upload flow

`POST /v1/upload` supports OpenWebUI file-bridge idempotency through
`source_ids` form fields or `X-Upload-Source-Ids`. Previously-seen source ids are
stored in session metadata and skipped on later re-uploads for the same chat
scope. These external source ids are dedupe hints only; answer-time retrieval uses
internal uploaded document ids returned by the backend.

`POST /v1/upload`:

1. accepts multipart files from clients such as Open WebUI
2. saves them into `UPLOADS_DIR`
3. ingests them into the KB
4. opens the canonical session workspace with
   `SessionWorkspace.for_session(...)`
5. copies the uploaded files into that workspace
6. returns upload metadata

The response includes `active_uploaded_doc_ids` and `upload_manifest` so clients can
forward internal repository ids on the next chat turn. OpenWebUI source ids must not be
used as evidence identifiers.

This endpoint is still different from `/v1/ingest/documents`, but the
difference is transport shape, not workspace behavior:

- use `/v1/ingest/documents` for host-visible server paths
- use `/v1/upload` for browser-style multipart uploads
- both endpoints now seed the canonical session workspace for the active
  derived runtime `session_id`

## In-process usage

Prefer `RuntimeService`.

```python
from agentic_chatbot_next.config import load_settings
from agentic_chatbot_next.providers import build_providers
from agentic_chatbot_next.app.service import RuntimeService

settings = load_settings()
providers = build_providers(settings)
service = RuntimeService.create(settings, providers)
session = RuntimeService.create_local_session(settings, conversation_id="my-chat-001")

answer = service.process_turn(session, user_text="Summarize the auth policy.")
```
