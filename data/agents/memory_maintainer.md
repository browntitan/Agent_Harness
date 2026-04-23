---
name: memory_maintainer
mode: memory_maintainer
description: Optional persistent-memory maintenance helper. Active only when MEMORY_ENABLED is true.
prompt_file: utility_agent.md
skill_scope: utility
allowed_tools: ["memory_save", "memory_load", "memory_list"]
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation", "user"]
max_steps: 8
max_tool_calls: 8
allow_background_jobs: true
metadata: {"role_kind": "maintenance", "entry_path": "delegated", "expected_output": "user_text"}
---
Memory maintainer role definition for the next runtime.
