---
name: utility
mode: react
description: Calculation and document-listing specialist. Memory tools are optional and depend on MEMORY_ENABLED.
prompt_file: utility_agent.md
skill_scope: utility
allowed_tools: ["calculator", "list_indexed_docs", "memory_save", "memory_load", "memory_list", "search_skills", "request_parent_question", "request_parent_approval", "invoke_agent", "post_team_message", "list_team_messages", "claim_team_messages", "respond_team_message"]
allowed_worker_agents: ["rag_worker", "data_analyst", "general"]
preload_skill_packs: []
memory_scopes: ["conversation", "user"]
max_steps: 20
max_tool_calls: 24
allow_background_jobs: false
metadata: {"role_kind": "worker", "entry_path": "delegated", "expected_output": "user_text"}
---
Utility role definition for the next runtime.
