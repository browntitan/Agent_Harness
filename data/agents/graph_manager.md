---
name: graph_manager
mode: react
description: Worker-only graph retrieval and planning specialist for managed GraphRAG inspection, discovery, and source selection.
prompt_file: graph_manager_agent.md
skill_scope: graph_manager
allowed_tools: ["list_graph_indexes", "inspect_graph_index", "search_graph_index", "explain_source_plan", "request_parent_question", "request_parent_approval", "invoke_agent", "post_team_message", "list_team_messages", "claim_team_messages", "respond_team_message"]
allowed_worker_agents: ["rag_worker", "general"]
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 24
max_tool_calls: 24
allow_background_jobs: false
metadata: {"role_kind": "top_level_or_worker", "entry_path": "router_fast_path_or_delegated", "expected_output": "user_text"}
---
Graph retrieval worker definition for the next runtime.
