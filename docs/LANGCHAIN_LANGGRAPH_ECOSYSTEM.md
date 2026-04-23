# LangChain / LangGraph Ecosystem

The live runtime is not “a LangGraph app” at the top level.

## Current split

### LangChain is used for

- chat-model abstractions
- tools
- message types
- callback plumbing

### LangGraph is used for

- the tactical ReAct executor inside `src/agentic_chatbot_next/general_agent.py`

## What is not a LangGraph flow

The following live orchestration is plain Python:

- routing
- session persistence
- coordinator planning/batching/finalization/verification
- worker jobs and mailbox continuation
- notification drain
- heuristic memory extraction

That logic lives in:

- `src/agentic_chatbot_next/app/service.py`
- `src/agentic_chatbot_next/runtime/kernel.py`
- `src/agentic_chatbot_next/runtime/query_loop.py`

## Direct model-call paths

These modes are currently direct model calls, not LangGraph flows:

- `basic`
- `planner`
- `finalizer`
- `verifier`

The `rag_worker` uses the next-owned `run_rag_contract()` adaptive retrieval and synthesis
path, including plain-Python deep-search fan-out, rather than a top-level LangGraph graph.

The `memory_maintainer` path is not a model flow at all in the live runtime; it is local
heuristic extraction over recent messages, and it disappears entirely when
`MEMORY_ENABLED=false`.

The `data_analyst` sandbox is also not a LangGraph flow. It is plain Python code executed inside
the offline Docker image configured by `SANDBOX_DOCKER_IMAGE`.
