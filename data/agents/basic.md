---
name: basic
mode: basic
description: Direct chat agent without tools.
prompt_file: basic_chat.md
skill_scope: basic
allowed_tools: []
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 2
max_tool_calls: 0
allow_background_jobs: false
metadata: {"role_kind": "top_level", "entry_path": "router_basic", "expected_output": "user_text"}
---
Basic role definition for the next runtime.
