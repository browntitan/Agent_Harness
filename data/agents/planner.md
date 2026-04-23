---
name: planner
mode: planner
description: JSON planner for multi-step tasks.
prompt_file: planner_agent.md
skill_scope: planner
allowed_tools: []
allowed_worker_agents: []
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 16
max_tool_calls: 0
allow_background_jobs: false
metadata: {"role_kind": "worker", "entry_path": "coordinator_only", "expected_output": "task_plan_json"}
---
Planner role definition for the next runtime.
