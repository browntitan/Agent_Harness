# Patterns

## Session kernel first

**Why:** persistence, jobs, notifications, and routing are easier to reason about when one
runtime kernel owns them.

**How:** `RuntimeService` + `RuntimeKernel` + `QueryLoop`.

## Agent roles as data

**Why:** deployment-time role changes should not require code edits in the runtime loop.

**How:** `data/agents/*.md` -> `AgentDefinition` -> `AgentRegistry`.

## Tools separate from skills

**Why:** effectful actions and prompt guidance are different runtime concerns.

**How:** tools live under `src/agentic_chatbot_next/tools/`; skill loading and retrieval
live under `src/agentic_chatbot_next/skills/`.

## File-backed runtime traces, DB-backed runtime data

**Why:** traces, resumes, and worker inspection are simpler when runtime artifacts are directly
inspectable on disk, while searchable/shared state belongs in typed stores.

**How:** `data/runtime` and `data/workspaces` hold session/job/workspace artifacts.
PostgreSQL holds documents, chunks, skills, managed memory, requirements, access rows,
capability profiles, MCP catalogs, and graph metadata. `data/memory` is now an inspection
projection and fallback path when `MEMORY_ENABLED=true`.

## Capability-scoped tool plane

**Why:** agent allow-lists alone are not enough once runtime-authored skills, MCP tools,
tenant collections, and per-user access policies are live.

**How:** `ToolPolicyService`, effective capability profiles, and RBAC intersect agent
metadata with user/tenant grants before tools, skills, collections, agents, and MCP tools are
visible.

## Deferred heavy tools

**Why:** large graph, admin, requirements, and MCP surfaces should not crowd every prompt by
default.

**How:** `discover_tools` and `call_deferred_tool` expose policy-approved deferred tools only
after an explicit per-turn search, then recheck policy at invocation time.

## Prebuilt sandbox contract

**Why:** secure offline analyst execution is more reliable when package availability is fixed at
image-build time instead of depending on runtime installs.

**How:** `SANDBOX_DOCKER_IMAGE=agentic-chatbot-sandbox:py312`,
`python run.py build-sandbox-image`, and readiness checks in `doctor --strict` plus notebook
preflight.

## Tactical LangGraph

**Why:** ReAct is useful, but the whole runtime should not be trapped inside a graph.

**How:** only the react executor uses LangGraph; orchestration remains plain Python.
