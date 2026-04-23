---
name: general
mode: react
description: Default session agent with direct RAG, inventory, and lightweight orchestration access. Memory tools are optional and depend on MEMORY_ENABLED.
prompt_file: general_agent.md
skill_scope: general
allowed_tools: ["calculator", "list_indexed_docs", "resolve_indexed_docs", "search_indexed_docs", "read_indexed_doc", "compare_indexed_docs", "extract_requirement_statements", "export_requirement_statements", "list_graph_indexes", "inspect_graph_index", "memory_save", "memory_load", "memory_list", "rag_agent_tool", "search_skills", "mcp__*", "spawn_worker", "message_worker", "request_parent_question", "request_parent_approval", "list_worker_requests", "respond_worker_request", "create_team_channel", "post_team_message", "list_team_messages", "claim_team_messages", "respond_team_message", "list_jobs", "stop_job"]
allowed_worker_agents: ["coordinator", "rag_worker", "data_analyst", "utility", "graph_manager", "memory_maintainer"]
preload_skill_packs: []
memory_scopes: ["conversation", "user"]
max_steps: 10
max_tool_calls: 12
allow_background_jobs: true
metadata: {"role_kind": "top_level", "entry_path": "default", "expected_output": "user_text", "delegates_complex_tasks_to": "coordinator"}
---
General role definition for the next runtime.
