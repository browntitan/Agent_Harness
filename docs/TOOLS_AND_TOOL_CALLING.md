# Tools and Tool Calling

The live runtime uses tool calling through `src/agentic_chatbot_next`.

## Runtime tool model

The primary runtime abstraction is `ToolDefinition` plus a bound `ToolContext` in
`src/agentic_chatbot_next/tools/`.

Each tool definition carries:

- `name`
- `group`
- `builder`
- `description`
- `args_schema`
- `read_only`
- `destructive`
- `background_safe`
- `concurrency_key`
- `requires_workspace`
- `serializer`
- `should_defer`
- `search_hint`
- `defer_reason`
- `defer_priority`
- `eager_for_agents`
- `metadata`

The registry binds those definitions to the current session/job context and produces
LangChain-compatible tools for agent execution.

This surface remains conservative even when external tools are enabled:

- tools are Python-defined in-repo
- optional user-owned MCP tools are loaded from cached DB catalogs
- skills are retrieved prompt context or explicit executable workflows
- agents are markdown-defined roles

When `MCP_TOOL_PLANE_ENABLED=true`, authenticated users can add Streamable HTTP MCP
connections through `/v1/mcp/connections` or the control panel. The runtime refreshes each
connection's tool catalog into Postgres and then turns those cached rows into dynamic
`ToolDefinition` entries named `mcp__{connection_slug}__{tool_slug}`. Normal prompt/tool
binding never calls the remote MCP server; only an actual tool invocation does.

## Top-level runtime tool groups

### Utility cluster

Exposed through next-runtime tool definitions such as:

- `calculator`
- `list_indexed_docs`
- `search_skills`

### Discovery cluster

- `discover_tools`
- `call_deferred_tool`

When `DEFERRED_TOOL_DISCOVERY_ENABLED=true`, heavy tools can be marked
`should_defer=true`. Deferred tools are omitted from the initial model-bound tool list
unless the active agent is listed in `eager_for_agents`. The runtime auto-adds the two
discovery facade tools when the active agent has at least one deferred target that still
passes `ToolPolicyService`.

The v1 flow is explicit:

- `discover_tools(query, group="", top_k=0)` searches deferred tool cards using
  `search_hint`, keywords, descriptions, schema fields, and group/name matches
- `call_deferred_tool(tool_name, arguments)` invokes a deferred target only after it was
  returned by `discover_tools` in the current turn
- direct tool policy is rechecked at invocation time, so discovery never grants extra tools,
  workspace access, background safety, read/write authority, or executable-skill permissions

Both facade tools use the `deferred_tool_discovery` concurrency key so a same-burst search
and call are serialized in model order.

### MCP / Plugin cluster

MCP tools are not a separate command plane. They enter the same registry, discovery, and
policy path as in-repo tools:

- connection profiles live in `mcp_connections`
- cached tool rows live in `mcp_tool_catalog`
- registry names use `mcp__{connection_slug}__{tool_slug}`
- `mcp__*` and `mcp__{connection_slug}__*` selectors can be used in agent `allowed_tools`
- MCP tools default to `should_defer=true`, `read_only=false`, `destructive=true`, and
  `background_safe=false`

V1 supports Streamable HTTP MCP tools only. It intentionally does not support stdio servers,
package installation, resources, prompts, sampling, elicitation, or server-initiated local
filesystem access.

MCP invocation policy is checked twice: discovery only shows tools the active agent may use,
and `call_deferred_tool` rechecks ownership, tenant visibility, agent allow-list, read-only
mode, background-job safety, authz grants, and executable-skill clipping immediately before
the remote `tools/call`. A remote MCP tool can never expand local workspace access, sandbox
permissions, worker scope, or the agent's allowed tool surface.

### Skills cluster

- `execute_skill`

`execute_skill` runs active `executable` or `hybrid` skill packs from the same `/v1/skills`
control plane used for retrievable packs. It is only bound when
`EXECUTABLE_SKILLS_ENABLED=true` and only for agents that include `execute_skill` in
`allowed_tools`.

Execution is explicit and auditable:

- `context: inline` returns rendered instructions to the current agent loop
- `context: fork` creates a synchronous worker job, clips tools to the skill allow-list, and
  returns the worker result
- recursive `execute_skill` inside executable skills is denied

### Memory cluster

- `memory_save`
- `memory_load`
- `memory_list`

When `MEMORY_ENABLED=true`, the memory tools prefer the managed PostgreSQL memory store
(`memory_records`, observations, and episodes). `data/memory/...` is now a projection for
human inspection plus a fallback path when the managed store is absent. Explicit tool saves
call the managed `save_explicit(...)` path and then project the active session view back to
`index.json`, `MEMORY.md`, `topics/*.md`, and `groups/*.md` when a projector is available.
When `MEMORY_ENABLED=false`, those memory tools are omitted from the bound tool surface.

### RAG gateway cluster

- `rag_agent_tool`
- `resolve_indexed_docs`
- `search_indexed_docs`
- `read_indexed_doc`
- `compare_indexed_docs`
- `document_extract`
- `document_compare`
- `document_consolidation_campaign`
- `template_transform`
- `evidence_binder`
- `extract_requirement_statements`
- `export_requirement_statements`

This is how prompt-backed agents reach grounded document reasoning without receiving the
entire internal RAG specialist controller surface directly. `rag_agent_tool` fronts the
next-runtime `run_rag_contract()` flow and returns the stable JSON RAG contract. Indexed-doc,
document, template, evidence, and requirements helpers expose exact corpus operations as
first-class tools while preserving the direct RAG controller as the synthesis path.

Caller-facing retrieval controls now include:

- `search_mode`
- `max_search_rounds`

### Graph gateway cluster

- `list_graph_indexes`
- `inspect_graph_index`
- `list_graph_documents`
- `search_graph_index`
- `explain_source_plan`
- `index_graph_corpus`
- `import_existing_graph`
- `refresh_graph_index`

The live public gateway only exposes graph listing, inspection, and query directly. The
mutation-style graph tools above remain bound for authorized runtime agents, while the
OpenAI-compatible `/v1/graphs/index` and `/v1/graphs/import` endpoints currently return `403`
because graph creation and refresh are admin-managed through the control panel.

With deferred discovery enabled, graph-heavy tools are hidden from general-purpose prompt
surfaces and found through `discover_tools`:

- `list_graph_documents`, `search_graph_index`, and `explain_source_plan` are deferred for
  general agents but remain eager for `graph_manager`
- `index_graph_corpus`, `import_existing_graph`, and `refresh_graph_index` are deferred
  everywhere unless a later agent config explicitly makes them eager
- `list_graph_indexes` and `inspect_graph_index` stay eager because they are cheap
  inventory/routing tools

### Analyst cluster

- `load_dataset`
- `inspect_columns`
- `execute_code`
- `run_nlp_column_task`
- `return_file`
- `scratchpad_write`
- `scratchpad_read`
- `scratchpad_list`
- `workspace_write`
- `workspace_read`
- `workspace_list`

`run_nlp_column_task` is the bounded provider-backed NLP path for row-level text
classification and summarization. `return_file` registers an existing workspace
file as a user-facing download artifact without moving it out of the session
workspace.

### Orchestration cluster

- `spawn_worker`
- `message_worker`
- `request_parent_question`
- `request_parent_approval`
- `list_worker_requests`
- `respond_worker_request`
- `create_team_channel`
- `post_team_message`
- `list_team_messages`
- `claim_team_messages`
- `respond_team_message`
- `invoke_agent`
- `list_jobs`
- `stop_job`

These are only exposed to agents that allow worker orchestration.

Worker mailbox requests are typed. A worker can pause itself with
`request_parent_question` when a missing answer blocks safe progress, or with
`request_parent_approval` when an operator decision is required. Parent/coordinator agents
may answer `question_request` items through `respond_worker_request`; approval requests must
be resolved through the gateway/control-panel API with an admin token or
`worker_request` approve permission. Approval responses do not grant extra tools or broader
scope; normal `ToolPolicyService` checks still apply after the worker resumes.

When `TEAM_MAILBOX_ENABLED=true`, allowed agents can also use a session-scoped team mailbox:

- `create_team_channel(name, purpose, member_agents=[], member_job_ids=[])`
- `post_team_message(channel_id, content, message_type="message", target_agents=[], target_job_ids=[], subject="", payload={})`
- `list_team_messages(channel_id="", message_type="", status_filter="open", limit=20)`
- `claim_team_messages(channel_id, limit=0, message_type="")`
- `respond_team_message(channel_id, message_id, response, decision="", resolve=true)`

Team channels are async coordination surfaces over the same job/transcript store. They support
typed messages such as `status_update`, `handoff`, `question_request`, and
`approval_request`, but they are not shared permission scopes. Agents may answer team
questions through `respond_team_message`; team approval decisions are rejected from agent tools
and must be resolved through the operator/API path with admin or approval authority. The team
mailbox never elevates tool access, sandbox permissions, skill scope, or worker scope.

## Tool surfaces by runtime agent

### `general`

- utility tools
- memory tools when `MEMORY_ENABLED=true`
- `rag_agent_tool`
- indexed-doc helpers
- graph gateway read/search tools
- `search_skills`
- orchestration tools

`general` may delegate to `coordinator`, `rag_worker`, `data_analyst`, `utility`,
`graph_manager`, or `memory_maintainer` because those worker roles are explicitly allowed in
the live registry.

`memory_maintainer` is only actually launchable when `MEMORY_ENABLED=true`.

For broad corpus-mining tasks, `general` should prefer `coordinator` rather than trying to
simulate a long-running research campaign through repeated direct tool calls.

### `coordinator`

- orchestration tools only

`coordinator` is not a normal ReAct worker. Its runtime mode is `coordinator`, and the
kernel handles planning, task batching, finalization, and optional verification around it.

For document-research campaigns, `planner` now emits one or more `rag_worker` tasks with
focused `doc_scope`, `skill_queries`, and structured RAG hint fields. `coordinator`
remains the owner of durable worker spawning and progress tracking.

### `utility`

- calculator
- document listing
- memory tools when `MEMORY_ENABLED=true`
- skill search
- bounded peer follow-up through `invoke_agent`
- team mailbox tools when `TEAM_MAILBOX_ENABLED=true`

### `data_analyst`

- dataset inspection
- bounded NLP column tasks
- Docker execution
- calculator
- scratchpad tools
- workspace tools
- file return
- skill search
- parent question/approval tools when running inside a worker job context
- bounded peer follow-up through `invoke_agent`
- team mailbox tools when `TEAM_MAILBOX_ENABLED=true`

The live `data_analyst` role is intentionally mixed-mode:

- `execute_code` is the open-ended pandas, statistics, and chart path
- `run_nlp_column_task` is the bounded LLM path for row-level text work
- `return_file` publishes derived workspace files through the API download flow

`execute_code` now assumes the configured offline analyst image already contains the analyst
stack. The runtime no longer installs packages inside the sandbox at turn time.

### `verifier`

- `rag_agent_tool`
- `list_indexed_docs`
- `resolve_indexed_docs`
- `read_indexed_doc`
- `compare_indexed_docs`
- `list_graph_indexes`
- `inspect_graph_index`
- `search_graph_index`
- `explain_source_plan`
- `search_skills`

`verifier` also has its own runtime mode (`verifier`) rather than sharing the generic
`react` path.

### `graph_manager`

- `list_graph_indexes`
- `inspect_graph_index`
- `search_graph_index`
- `explain_source_plan`
- `rag_agent_tool`
- parent question/approval tools when running inside a worker job context
- `invoke_agent`
- team mailbox tools when `TEAM_MAILBOX_ENABLED=true`

`graph_manager` is a `react` role with `metadata.role_kind=top_level_or_worker`. It can be
selected directly by router fast paths or requested-agent overrides for graph-backed evidence,
GraphRAG, graph inventory, relationship, and source-planning turns. It can also still be
launched by `general` or `coordinator` when a broader task needs managed-graph inspection or
graph-aware source planning.

### `rag_worker`

No top-level tool exposure. It delegates to the next-runtime RAG contract flow, which uses
a direct Python adaptive retrieval and synthesis pipeline.

That means grounded document work now appears in two live shapes:

- direct specialist starts, where the router or policy layer begins in `rag_worker`
- delegated tool-path starts, where `general` or `verifier` calls `rag_agent_tool`

The demo notebook keeps both shapes visible. It can pin `general` via
`metadata.requested_agent=general` when it wants to showcase tool traces rather than the direct
specialist path.

Internally, the live RAG path now uses backend-agnostic retrieval operations equivalent to:

- `search_corpus(...)`
- `grep_corpus(...)`
- `read_document(...)`
- `fetch_chunk_window(...)`
- `prune_chunks(...)`

Those operations are runtime-owned Python helpers inside the adaptive controller, not
top-level LangChain tools exposed to the agent registry.

The live `rag_worker` also consumes structured retrieval hints from planner/coordinator
payloads and indexed skill-pack metadata:

- `research_profile`
- `coverage_goal`
- `result_mode`
- `controller_hints`

It may also consume coordinator-owned typed handoff artifacts when the task was produced by
an upstream document-research campaign:

- `analysis_summary`
- `entity_candidates`
- `keyword_windows`
- `doc_focus`
- `evidence_request`
- `evidence_response`

That is now the primary way to steer corpus discovery, inventory output, process-flow
detection, and negative-evidence reporting for the direct contract path.

Design boundary:

- `rag_worker` does not receive `spawn_worker`
- `rag_worker` is not a durable worker manager
- prompt-driven agents may still call `search_skills`, but `rag_worker` does not depend on
  free-form skill search as its primary control surface
- coordinator-owned typed handoffs remain the preferred collaboration path for planned
  multi-worker campaigns
- prompt-backed agents may now use `invoke_agent` for bounded same-session peer follow-ups,
  and `rag_worker` may make one bounded async peer delegation before final synthesis without
  becoming a full ReAct tool user

### `memory_maintainer`

- registry-declared memory tools only

Current implementation note: the dedicated `memory_maintainer` mode bypasses ReAct/tool
calling today and runs direct heuristic extraction in
`QueryLoop._run_memory_maintainer(...)`. The runtime removes that path entirely when
`MEMORY_ENABLED=false`.

## Additional RAG helper modules

The repo also contains helper tool factories under
`src/agentic_chatbot_next/rag/specialist_tools.py` and
`src/agentic_chatbot_next/rag/extended_tools.py`.

Those modules expose operations such as:

- document resolution
- search across docs or collections
- clause and requirement extraction
- document diff / comparison
- chunk window fetches
- collection listing
- scratchpad helpers
- optional web-search helpers
- managed graph helpers:
  - `list_graph_indexes`
  - `inspect_graph_index`
  - `search_graph_index`
  - `explain_source_plan`
- compatibility graph traversal helpers:
  - `graph_search_local`
  - `graph_search_global`

Most of these remain repository-level helper utilities, but the live runtime now binds a
larger exact-doc/document-research subset as first-class tools:

- `resolve_indexed_docs`
- `search_indexed_docs`
- `read_indexed_doc`
- `compare_indexed_docs`
- `document_extract`
- `document_compare`
- `document_consolidation_campaign`
- `template_transform`
- `evidence_binder`
- `extract_requirement_statements`
- `export_requirement_statements`

The broader adaptive retrieval path still uses equivalent Python-level operations inside
`src/agentic_chatbot_next/rag/adaptive.py`, but the managed `microsoft_graphrag` catalog/query
surface is now the default graph path for end-user graph discovery and query workflows.

## Fallback behavior

If a model wrapper does not support tool calling:

- `run_general_agent()` falls back to a plan-execute loop
- the tool interfaces remain the same from the runtime perspective

This keeps agent behavior functional even when native tool binding is unavailable.

## Safety and metadata

The runtime uses tool metadata primarily for:

- shaping the visible tool surface
- distinguishing read-only vs world-changing operations
- grouping tools by capability
- documenting orchestration permissions through agent config

The current central policy layer is `ToolPolicyService`, which enforces:

- allowed-tool membership per agent
- workspace requirements
- background-job safety
- worker request tools only being callable from worker job contexts
- team mailbox tools only being callable when `TEAM_MAILBOX_ENABLED=true`
- read-only restrictions for non-effectful modes
- memory-only access for the memory maintainer
- executable-skill allow-list clipping while a forked skill is running
- capability-profile clipping of enabled/disabled tools, tool groups, MCP tools, agents,
  collections, and skill packs
- RBAC grants when `AUTHZ_ENABLED=true`, including tool, collection, graph, and skill-family
  permissions

Deferred discovery is layered on top of that same policy. A deferred target must already be
in the active agent's `allowed_tools`, must be visible under authz, and must pass all normal
mode/job/workspace/skill checks when discovered and again when invoked through
`call_deferred_tool`.

For `memory_maintainer`, that policy is mostly defensive today because the live
`memory_maintainer` mode does not execute a ReAct tool loop.

The repo now has a consent mailbox for worker approval requests. It is not an automatic
risky-tool interception layer; workers must explicitly ask, and approval never elevates
tool access beyond the agent's existing allow-list.

The team mailbox uses the same no-elevation rule. It can coordinate peers and campaign
workers, but it cannot make a caller a member of another channel, target workers outside the
caller's `allowed_worker_agents`, or approve risky actions from inside an agent loop.

## Observability tie-in

Tool execution is now observable through both:

- LangChain callbacks when external tracing is configured
- local `tool_start`, `tool_end`, and `tool_error` events in `data/runtime/*`

Deferred discovery also emits `deferred_tool_catalog_built`,
`deferred_tool_discovery_searched`, `deferred_tool_invoked`, and `deferred_tool_denied`
events when the feature is enabled.

Team mailbox operations emit `team_mailbox_channel_created`,
`team_mailbox_message_posted`, `team_mailbox_message_claimed`,
`team_mailbox_message_resolved`, and `team_mailbox_digest_created` when the feature is
enabled.

Direct `rag_worker` execution does not produce `rag_agent_tool` wrapper events because it bypasses
the top-level tool layer. Delegated grounded RAG through `general` or `verifier` still surfaces
the normal tool lifecycle.

For streaming API turns, those tool callbacks are also folded into the live `progress`
timeline shown in the UI status panel.

That progress stream now includes summarized intent and evidence events such as
`tool_intent`, `decision_point`, `evidence_status`, `handoff_prepared`, and
`handoff_consumed` in addition to raw tool lifecycle markers.
