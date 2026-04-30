# Agent Harness Audit

This file is a historical audit snapshot of the runtime-shape review that informed the current
docs. Keep the factual references current, but treat its recommendations and status labels as
change-history context rather than the main architecture reference.

This audit compares the live `agentic_chatbot_next` runtime against the external guidance
reviewed in the uploaded analysis pack:

- `ARCHITECTURE.md`
- `AGENTS.md`
- `CODE_FLOW.md`
- `BEST_PRACTICES.md`

The guidance is treated as advisory input for a general-purpose multi-task agent, not as a
strict blueprint for a coding-only agent.

## Adopt now

### 1. Remove the old runtime surface and stale compatibility layer

Why it matters:

- the repository had two competing runtime stories
- old import paths made the live architecture harder to understand
- docs stayed stale longer because the deprecated path was still present

Decision:

- adopt now

Status in this pass:

- remove the old `agentic_chatbot.runtime` package
- remove the old `ChatbotApp` compatibility surface
- migrate in-repo tests and imports onto next-runtime contracts

### 2. Make initial-agent selection more data-driven before expanding the agent set

Observed live behavior:

- router hinting now comes from config-backed deterministic patterns plus the LLM router
- `choose_agent_name(...)` in `router/policy.py` is now registry-aware through
  `build_router_targets(registry)`, and the live hint surface includes `coordinator`,
  `data_analyst`, `rag_worker`, and `graph_manager`

Why it matters:

- this is fine for the current `general` / `coordinator` / `data_analyst` / `rag_worker` shape
- it will become brittle once the system grows into a broader general-purpose agent with more
  top-level specialists

Decision:

- adopt next in implementation, but do not force the refactor in this pass

Recommended follow-up:

- keep expanding registry-aware/config-aware policy so new top-level specialists do not require
  router-code changes

### 3. Enrich tool policy before adding broader world-changing tools

Observed live behavior:

- the repo already has a centralized `ToolPolicyService`
- current policy covers allowed tools, read-only gating, workspace requirements,
  background-job safety, worker-context-only tools, team mailbox gating, executable-skill
  clipping, capability profiles, RBAC grants, deferred discovery, and MCP tool visibility

Why it matters:

- this is acceptable for the current bounded tool surface
- it is thinner than the richer permission and metadata model described in the external guidance

Decision:

- adopted after the original audit; continue tightening policy as new external-action tools
  are added

Recommended follow-up:

- add stronger approval and safety semantics before introducing more open-ended external-action tools

## Docs only

### 1. Query-loop ownership was overstated

Observed live behavior:

- `RuntimeService` owns routing and ingest
- `RuntimeKernel` owns persistence, jobs, notifications, and orchestration
- `QueryLoop` dispatches by agent mode and injects skill/memory context
- `general_agent.py` owns the actual `react` execution loop, using LangGraph ReAct or a
  plan-execute fallback

Decision:

- docs only

Action in this pass:

- update architecture and control-flow docs to describe `QueryLoop` as the dispatcher and
  `general_agent.py` as the live `react` executor

### 2. Router docs missed `rag_worker`

Observed live behavior:

- the live router can suggest `rag_worker`
- the router docs only listed `coordinator`, `data_analyst`, and empty string

Decision:

- docs only

Action in this pass:

- update router docs to include `rag_worker`

### 3. Gateway and runtime docs still described removed compatibility paths

Observed live behavior:

- several docs still referenced the removed compatibility wrapper and old runtime as active repo
  context

Decision:

- docs only

Action in this pass:

- update README and runtime docs to describe the repo as next-runtime-only

## Keep as-is

### 1. Keep the simpler next-runtime tool contract

Observed divergence:

- the external coding-agent guidance assumes a richer tool object with broader UI and policy
  metadata
- the live repo uses a simpler `ToolDefinition` plus centralized policy service

Why this is acceptable:

- the live tool plane is smaller and more internal
- the current abstraction already cleanly separates tool metadata, policy, and binding
- a heavier tool object is not required yet for this product

### 2. Keep tools and skills as separate abstractions

Observed live behavior:

- tools are bound runtime capabilities under `src/agentic_chatbot_next/tools/`
- skills are retrieved prompt context under `src/agentic_chatbot_next/skills/`

Why this is good:

- it matches the external guidance closely
- it keeps operating guidance separate from side-effecting actions

### 3. Keep managed memory, jobs, notifications, and verifier-style orchestration

Observed live behavior:

- early persistence is in place
- worker jobs, mailboxes, notifications, and event logs are durable
- memory is PostgreSQL-backed when `MEMORY_ENABLED=true`, with file projections/fallbacks
- coordinator mode already separates planning, execution, finalization, and verification

Why this is good:

- these are strong patterns for a general-purpose agent, not only for a coding agent
- they directly improve debuggability, resumability, and context control

### 4. Extension story is now live but guarded

Observed divergence:

- the external reference includes a larger MCP/plugin/deferred-tool extension plane
- the live next runtime now supports deferred tool discovery, executable/hybrid skills,
  capability profiles, RBAC clipping, and user-owned Streamable HTTP MCP tool catalogs

Why this is acceptable:

- the extension machinery is feature-gated and policy-clipped rather than always exposed
- cached MCP catalogs enter the same `ToolDefinition` and `ToolPolicyService` path as in-repo
  tools

Recommended trigger to revisit:

- revisit before adding stdio MCP servers, resource/prompt/sampling support, package
  installation, or broader third-party action tools

## Bottom line

The live next runtime already matches the external guidance in the areas that matter most for a
general-purpose agent:

- a persistent session kernel
- explicit routing
- durable worker jobs
- optional managed memory behind `MEMORY_ENABLED`, with file projections/fallbacks
- separate tools and skills
- verification-aware orchestration

The biggest issues were repository clarity and doc drift, not a fundamentally wrong live
architecture. The main architectural follow-ups are to keep initial-agent selection
registry-aware as the specialist set grows and to keep the permission model ahead of any
broader external-action tool surface.

Recent live runtime additions that the docs now treat as first-class behavior:

- `metadata.requested_agent` for explicit API/demo control of the initial AGENT role
- `MEMORY_ENABLED` as a real runtime-wide feature flag
- the offline analyst sandbox image contract built through `python run.py build-sandbox-image`
- `graph_manager` as a top-level-or-worker graph specialist
- capability profiles, RBAC, deferred tool discovery, executable skills, team mailboxes, worker
  scheduler metadata, and MCP tool catalogs as live guarded runtime surfaces
