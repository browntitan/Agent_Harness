---
name: rag_worker
mode: rag
description: Grounded document worker that returns the stable RAG contract.
prompt_file: rag_agent.md
skill_scope: rag
allowed_tools: []
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 8
max_tool_calls: 12
allow_background_jobs: false
metadata: {"role_kind": "top_level_or_worker", "entry_path": "router_fast_path_or_delegated", "expected_output": "rag_contract"}
---
RAG worker role definition for the next runtime.
