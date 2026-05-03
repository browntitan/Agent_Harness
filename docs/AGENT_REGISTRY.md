# Agent Registry

The live agent registry is `src/agentic_chatbot_next/agents/registry.py`.

## Source of truth

Live agents are loaded from `data/agents/*.md`.

Each file contains markdown frontmatter parsed into `AgentDefinition`:

- `name`
- `mode`
- `description`
- `prompt_file`
- `skill_scope`
- `allowed_tools`
- `allowed_worker_agents`
- `preload_skill_packs`
- `memory_scopes`
- `max_steps`
- `max_tool_calls`
- `allow_background_jobs`
- `metadata`

Frontmatter is schema-validated before it becomes an `AgentDefinition`. Invalid markdown now
fails fast with path-qualified field errors instead of silently dropping bad keys.

The old JSON-based runtime agent-loading path is not the live source of truth anymore.

## Load order

`AgentRegistry.reload()`:

1. scans the repo `agents_dir`, usually `data/agents/*.md`
2. optionally scans the configured control-panel overlay dir after that
3. parses and validates frontmatter through `agentic_chatbot_next.agents.loader`
4. stores definitions by `name`, with later overlay files replacing same-name repo files

There is no live built-in-plus-JSON override chain in the next runtime registry.
The only active override layer is markdown-on-markdown through the optional overlay dir.

## Live agent set

Expected loaded live roles:

- `basic`
- `general`
- `coordinator`
- `research_coordinator`
- `utility`
- `data_analyst`
- `rag_worker`
- `rag_researcher`
- `graph_manager`
- `planner`
- `finalizer`
- `verifier`
- `memory_maintainer`

Routable starts come from `list_routable()` rather than the full loaded set. Current
top-level non-`basic` starts include `general`, `coordinator`, `data_analyst`, `rag_worker`,
`research_coordinator`, and `graph_manager`. `graph_manager` is intentionally both routable
and delegable: its metadata is `role_kind=top_level_or_worker` with
`entry_path=router_fast_path_or_delegated`.

`research_coordinator` is the manager selected for long-running or corpus-scale research
campaigns. It has `metadata.research_campaign_agent=true`, uses the normal coordinator mode,
and can launch `rag_researcher` alongside `rag_worker`, `graph_manager`, `general`,
`finalizer`, and `verifier`.

`rag_researcher` is loaded but not routable. It is a ReAct-style RAG research specialist with
`metadata.manual_override_allowed=true`; API/demo callers can select it through
`metadata.requested_agent`, and coordinators can delegate to it, but normal router policy does
not start there automatically.

The registry can still load `memory_maintainer`, but the runtime filters it out of the
requested-agent override surface and blocks launches when `MEMORY_ENABLED=false`.

## How the registry is used

`RuntimeKernel.process_agent_turn(...)`:

1. selects the requested initial agent
2. resolves it through `AgentRegistry`
3. records `active_agent`
4. builds tool policy and execution context from that definition

Coordinator workers are also resolved through the same registry.

`RuntimeService.list_requested_agent_overrides()` starts from routable non-`basic` roles and
then appends non-routable agents that explicitly set `manual_override_allowed`. That is why
`rag_researcher` is a valid manual override even though it is absent from `list_routable()`.
