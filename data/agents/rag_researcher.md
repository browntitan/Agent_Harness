---
name: rag_researcher
mode: react
description: ReAct-style RAG research specialist for exploratory grounded retrieval, source selection, query rewrites, and final citation-safe RAG synthesis.
prompt_file: rag_researcher_agent.md
skill_scope: rag_researcher
allowed_tools: ["list_indexed_docs", "search_indexed_docs", "resolve_indexed_docs", "read_indexed_doc", "compare_indexed_docs", "rag_agent_tool", "plan_rag_queries", "search_corpus_chunks", "grep_corpus_chunks", "fetch_chunk_window", "inspect_document_structure", "search_document_sections", "filter_indexed_docs", "grade_evidence_candidates", "prune_evidence_candidates", "validate_evidence_plan", "build_rag_controller_hints", "explain_source_plan", "search_graph_index", "search_skills", "invoke_agent", "post_team_message", "list_team_messages", "claim_team_messages", "respond_team_message"]
allowed_worker_agents: ["rag_worker", "graph_manager", "general"]
preload_skill_packs: []
memory_scopes: ["conversation"]
max_steps: 16
max_tool_calls: 20
allow_background_jobs: false
metadata: {"role_kind": "worker", "entry_path": "manual_or_delegated", "expected_output": "user_text", "manual_override_allowed": true}
---
RAG researcher role definition for exploratory ReAct-style grounded retrieval.
