# Graph Inventory Regression Audit

## Definite Causes

- `src/agentic_chatbot_next/router/semantic.py`
  Graph inventory was not represented as its own semantic scope, so downstream logic could only treat it as KB retrieval or `none`.
- `src/agentic_chatbot_next/router/llm_router.py`
  The LLM router schema and prompt did not support `requested_scope_kind="graph_indexes"`, which allowed graph inventory turns to collapse into knowledge-base retrieval.
- `src/agentic_chatbot_next/runtime/turn_contracts.py`
  Graph inventory was not treated as a lightweight inventory answer contract, so misclassified turns escalated into grounded synthesis.
- `src/agentic_chatbot_next/runtime/task_plan.py`
  Planner precedence allowed broad grounded-synthesis and document-research branches to run before a graph-inventory fast path, producing `multi_step_document_discovery`.
- `src/agentic_chatbot_next/runtime/deep_rag.py`
  Deep-RAG had no graph-inventory exemption and could still prefer `rag_worker`.

## Amplifiers

- `.env`
- `.env.example`
- `data/control_panel/overlays/runtime.env`
  Interactive defaults were widened beyond the old baseline, and retrieval-heavy features were always on.
- `src/agentic_chatbot_next/api/status_tracker.py`
- `src/agentic_chatbot_next/api/main.py`
  User-facing status defaulted to “Searching knowledge base,” which obscured the fact that graph inventory had been misrouted.
- `data/agents/rag_worker.md`
  The RAG worker envelope was widened, which made accidental entry into the RAG path more expensive.

## Non-Causal Or Secondary

- `data/skills/general_agent.md`
- `data/skills/rag_agent.md`
  Prompt wording was not the main latency driver, but the general-agent prompt still needed stronger graph-inventory guardrails.

## Recovery Summary

- Restore graph inventory as a first-class semantic scope and inventory contract.
- Force plain graph availability questions onto the direct `general -> list_graph_indexes` path.
- Keep deep-RAG, document search, and worker orchestration off that path.
- Restore the old interactive runtime envelope and expose active runtime diagnostics so stale processes are visible.
