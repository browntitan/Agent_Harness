---
name: data_analyst
mode: react
description: Tabular data analysis specialist using sandboxed Python tools.
prompt_file: data_analyst_agent.md
skill_scope: data_analyst
allowed_tools: ["load_dataset", "profile_dataset", "profile_workbook_status", "extract_workbook_status", "inspect_columns", "execute_code", "run_nlp_column_task", "return_file", "calculator", "scratchpad_write", "scratchpad_read", "scratchpad_list", "workspace_write", "workspace_read", "workspace_list", "search_skills", "request_parent_question", "request_parent_approval", "invoke_agent", "post_team_message", "list_team_messages", "claim_team_messages", "respond_team_message"]
allowed_worker_agents: ["rag_worker", "utility", "general"]
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 28
max_tool_calls: 36
allow_background_jobs: false
metadata: {"role_kind": "top_level_or_worker", "entry_path": "router_fast_path_or_delegated", "expected_output": "user_text", "execution_strategy": "plan_execute"}
---
Data analyst role definition for the next runtime.
