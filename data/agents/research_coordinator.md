---
name: research_coordinator
mode: coordinator
description: Manager role for long-running deep research campaigns over indexed document corpora.
prompt_file: supervisor_agent.md
skill_scope: coordinator
allowed_tools: ["spawn_worker", "message_worker", "list_worker_requests", "respond_worker_request", "create_team_channel", "post_team_message", "list_team_messages", "claim_team_messages", "respond_team_message", "list_jobs", "stop_job"]
allowed_worker_agents: ["planner", "rag_worker", "rag_researcher", "general", "graph_manager", "finalizer", "verifier"]
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 12
max_tool_calls: 14
allow_background_jobs: true
metadata: {"role_kind": "manager", "entry_path": "router_fast_path_or_delegated", "expected_output": "cited_research_synthesis", "planner_agent": "planner", "finalizer_agent": "finalizer", "verifier_agent": "verifier", "verify_outputs": true, "research_campaign_agent": true}
---
Research coordinator role definition for deep corpus-scale RAG campaigns.
