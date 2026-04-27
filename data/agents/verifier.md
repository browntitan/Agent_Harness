---
name: verifier
mode: verifier
description: Validation-focused worker for checking outputs and citations.
prompt_file: verifier_agent.md
skill_scope: verifier
allowed_tools: ["rag_agent_tool", "list_indexed_docs", "resolve_indexed_docs", "read_indexed_doc", "compare_indexed_docs", "document_extract", "document_compare", "document_consolidation_campaign", "list_graph_indexes", "inspect_graph_index", "search_graph_index", "explain_source_plan", "search_skills"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 16
max_tool_calls: 16
allow_background_jobs: false
metadata: {"role_kind": "worker", "entry_path": "coordinator_only", "expected_output": "verification_json"}
---
Verifier role definition for the next runtime.
