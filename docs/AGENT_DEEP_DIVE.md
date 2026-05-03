# Agent Deep Dive

This document describes the live agent system in `agentic_chatbot_next`.

## Entry point

The live entry point is `RuntimeService.process_turn(...)`.

## Agent definitions

Agents are data, not hard-coded classes.

Each role is defined in `data/agents/*.md` and loaded into `AgentDefinition`.

Important fields:

- `mode`
- `prompt_file`
- `allowed_tools`
- `allowed_worker_agents`
- `memory_scopes`
- `max_steps`
- `max_tool_calls`
- `metadata`

Those markdown frontmatter blocks are now schema-validated at load time. Invalid agent files
raise path-qualified startup errors before the registry is used.

## Live execution modes

The next runtime currently uses these agent modes:

- `basic`
- `react`
- `rag`
- `planner`
- `finalizer`
- `verifier`
- `coordinator`
- `memory_maintainer`

Two current role distinctions matter:

- `research_coordinator` uses the existing `coordinator` mode but is the preferred manager for
  deep corpus research and long-running document campaigns
- `rag_researcher` uses `react` mode for exploratory RAG research and is manual/delegated
  rather than routable

## Execution ownership

The runtime splits execution responsibilities across three layers:

- `RuntimeService` handles eager workspace open, upload ingest/upload-summary kickoff,
  routing, requested-agent override validation, capability-profile scoping, and handoff into
  the kernel
- `RuntimeKernel` owns persisted session state, jobs, notifications, and worker orchestration
- `QueryLoop` dispatches execution by agent mode and injects prompt/skill/memory context

`RuntimeKernel` remains the stable public entrypoint, but internally it now delegates to:

- `kernel_events.py`
- `kernel_providers.py`
- `kernel_coordinator.py`

For `react` agents, `QueryLoop` delegates to `agentic_chatbot_next.general_agent.run_general_agent(...)`.
That helper uses LangGraph ReAct when tool binding is available and a plan-execute fallback
when it is not.

When a tool publishes a workspace file, the final assistant runtime message may
also carry `artifacts` metadata. The FastAPI gateway exposes that metadata
through chat responses and `/v1/files/{download_id}`.

Execution-mode nuance:

- `react`, `planner`, `finalizer`, and `verifier` are prompt-backed model executions
- `rag_worker` is a direct `run_rag_contract(...)` call in answer mode
- delegated deep-search evidence jobs can also run `rag_worker` in evidence-only mode
- `memory_maintainer` is a direct heuristic extractor and does not currently run an LLM or
  ReAct loop

Long-form writing nuance:

- `metadata.long_output` is an opt-in orchestration path above the normal mode system
- it reuses the chosen agent's prompt/style guidance
- it does not register a separate `long_output` agent mode
- synchronous runs write a workspace-backed draft inside the active turn
- background runs reuse the durable job system and return user-facing artifacts rather than
  coordinator handoff artifacts

## Role summary

### `basic`

- no tools
- direct chat execution

### `general`

- default AGENT entry
- utility tools
- memory tools when `MEMORY_ENABLED=true`
- RAG gateway
- graph gateway inventory tools
- `search_skills`
- orchestration tools for delegation and job control
- may delegate to `coordinator`, `rag_worker`, `data_analyst`, `utility`,
  `graph_manager`, or `memory_maintainer`; deep corpus research normally routes to
  `research_coordinator` before `general` needs to choose that path itself

### `coordinator`

- manager role for multi-step tasks
- planner/finalizer/verifier orchestration
- worker batching and notifications
- owner of durable document-research campaigns
- may launch delegated specialists including `graph_manager`
- bounded finalizer/verifier revision rounds controlled by `MAX_REVISION_ROUNDS`

### `research_coordinator`

- manager role for long-running deep research over indexed corpora
- uses the same kernel-owned coordinator mode as `coordinator`
- selected by router/deep-RAG policy for repository-scale, corpus-wide, or multi-hop document
  research
- can launch `planner`, `rag_worker`, `rag_researcher`, `general`, `graph_manager`,
  `finalizer`, and `verifier`
- has no analyst sandbox or terminal/code-execution tools by default

### `utility`

- calculator
- document listing
- managed-memory tools when `MEMORY_ENABLED=true`
- `search_skills`
- bounded same-session peer follow-up through `invoke_agent`
- team mailbox tools when `TEAM_MAILBOX_ENABLED=true`

### `data_analyst`

- dataset loading and multi-sheet profiling through `profile_dataset`
- column inspection
- bounded LLM-backed column NLP
- Docker sandbox execution through the prebuilt offline analyst image
- scratchpad and workspace tools
- explicit file publication through `return_file`
- `search_skills` plus bounded peer follow-up through `invoke_agent`
- worker parent-question/approval tools when running in a worker job
- team mailbox tools when `TEAM_MAILBOX_ENABLED=true`

### `rag_worker`

- specialist grounded retrieval path
- returns the preserved RAG contract in normal execution
- can also act as an internal evidence-only worker for adaptive deep retrieval
- owns the fast or deep retrieval decision through `run_rag_contract(...)`
- consumes structured RAG hints from planner/coordinator payloads and indexed skill packs
- does not own durable worker spawning
- may enqueue one bounded async peer follow-up to `data_analyst`, `utility`, or `general`
  when the direct RAG path decides a specialist continuation is more useful than answering now

### `rag_researcher`

- ReAct-style RAG research specialist for exploratory source selection before final synthesis
- not a normal router start; available through manual `metadata.requested_agent` override or
  coordinator delegation
- uses indexed-doc tools, deferred `rag_workbench` tools, graph source-planning/search tools,
  `search_skills`, and final `rag_agent_tool` synthesis
- plans query facets, inspects chunks/sections/structure, grades and prunes evidence, validates
  the evidence plan, then packages `controller_hints_json` for the final RAG contract call

### `graph_manager`

- graph retrieval and source-planning specialist with `metadata.role_kind=top_level_or_worker`
- inspects managed graph indexes and their source sets
- runs graph-backed evidence search across one graph or a shortlist of relevant graphs
- explains graph/vector/keyword/SQL source planning through `explain_source_plan`
- can start directly from graph fast-path routing or a valid requested-agent override
- may open one bounded same-session peer follow-up through `invoke_agent`

### `planner`

- JSON task-plan generator

### `finalizer`

- synthesis over task artifacts

### `verifier`

- output review / revision feedback
- can still read indexed docs directly, run `rag_agent_tool`, inspect graph indexes, and use
  `search_skills` during verification

### `memory_maintainer`

- explicit delegated helper for writing extracted memory entries into the managed memory path
- separate from the normal post-turn kernel memory-management path
- unavailable when `MEMORY_ENABLED=false`

## Context control

The next runtime uses several boundaries to keep context under control:

- scoped worker prompts
- bounded skill context
- bounded memory context
- capability-profile clipped tools, collections, agents, skills, and MCP tools
- session/job transcript separation
- worker mailboxes instead of raw conversation sharing
- typed handoff artifacts for structured worker-to-worker campaigns owned by `coordinator`
- bounded peer-agent messaging for ad hoc same-session follow-ups
- optional team mailbox channels for async typed campaign coordination when
  `TEAM_MAILBOX_ENABLED=true`

## Worker execution

Workers are durable jobs, not ad hoc threads of text.

Each worker has:

- a persisted job record
- a mailbox
- transcript rows
- event rows
- task notification output back into the parent session

For campaign-style work, workers can now also produce typed handoff artifacts such as:

- `analysis_summary`
- `entity_candidates`
- `keyword_windows`
- `doc_focus`
- `evidence_request`
- `evidence_response`

Those artifacts are still validated and routed by `coordinator` for planned campaigns.
Workers may now also open bounded same-session peer follow-ups through `invoke_agent`, but
typed handoffs remain the preferred path when the workflow is intentionally multi-stage.
With the team mailbox flag enabled, coordinator-created campaigns can additionally pass a
`team_channel_id` to workers so they can post or claim typed handoffs, status updates, and
questions without sharing raw transcripts or receiving broader authority.

There are now two live worker patterns:

- coordinator-owned workers for explicit multi-task orchestration
- runtime-owned internal RAG evidence workers launched through `KernelRagRuntimeBridge`

The second path is bounded and internal to deep retrieval. It reuses the same durable job
machinery, but the jobs return evidence payloads instead of user-facing synthesized text.

For large corpus-mining requests, the intended split is:

- `research_coordinator` owns the broad user-visible research campaign when router/deep-RAG
  policy identifies corpus-scale work
- `coordinator` remains the generic manager for delegated multi-step work
- `planner` decomposes the campaign into `rag_worker` tasks plus skill/hint seeds
- `rag_researcher` can handle exploratory source selection and evidence-plan preparation
- `rag_worker` performs bounded retrieval work
- `finalizer` and `verifier` handle merged synthesis and overclaim checks

## User-visible progress

The live UI does not expose raw chain-of-thought. Instead, the runtime emits
summarized progress milestones such as:

- `decision_point`
- `tool_intent`
- `evidence_status`
- `handoff_prepared`
- `handoff_consumed`
- `peer_dispatch`

Those events can also carry `why` and `waiting_on` so users can see what the agent is doing
without exposing internal scratch reasoning.
