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

## File-backed runtime state

**Why:** traces, resumes, and worker inspection are simpler when artifacts are directly
inspectable on disk.

**How:** `data/runtime`, `data/workspaces`, and `data/memory` when `MEMORY_ENABLED=true`.

## Prebuilt sandbox contract

**Why:** secure offline analyst execution is more reliable when package availability is fixed at
image-build time instead of depending on runtime installs.

**How:** `SANDBOX_DOCKER_IMAGE=agentic-chatbot-sandbox:py312`,
`python run.py build-sandbox-image`, and readiness checks in `doctor --strict` plus notebook
preflight.

## Tactical LangGraph

**Why:** ReAct is useful, but the whole runtime should not be trapped inside a graph.

**How:** only the react executor uses LangGraph; orchestration remains plain Python.
