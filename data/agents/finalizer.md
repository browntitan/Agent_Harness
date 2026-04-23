---
name: finalizer
mode: finalizer
description: Final synthesis agent over task artifacts.
prompt_file: finalizer_agent.md
skill_scope: finalizer
allowed_tools: []
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 12
max_tool_calls: 0
allow_background_jobs: false
metadata: {"role_kind": "worker", "entry_path": "coordinator_only", "expected_output": "user_text"}
---
Finalizer role definition for the next runtime.
