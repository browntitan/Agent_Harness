# Agentic Chatbot Architecture Reference

## Technology Stack
- Core: LangChain, FastAPI, Pydantic, SQLAlchemy
- LLM Providers: Ollama (local), Azure OpenAI, Nvidia/OpenAI-compatible
- Storage: PostgreSQL + pgvector, file system / S3
- Tracing: Langfuse (optional), custom callbacks
- Graph DB: Neo4j (optional, for GraphRAG)

## Three-Layer Runtime Architecture
1. RuntimeService (app/service.py) — Orchestrator: resolve_turn_intent(), route_message(), choose_agent_name()
2. RuntimeKernel (runtime/kernel.py) — Engine: process_agent_turn(), run_basic_chat(), job management
3. QueryLoop (runtime/query_loop.py) — ReAct loop: run(), build_agent_tools(), PolicyAwareToolNode

## Complete Turn Flow
User Input → Session Hydration → resolve_turn_intent() → route_message() → choose_agent_name() → ContextBudgetManager → build_agent_tools() → query_loop.run() [ReAct loop] → post-processing/recovery → memory persistence → response return

## Router
- Modes: hybrid | llm_only | pattern_only
- Confidence threshold: 0.70 (llm_router_confidence_threshold)
- RouterDecision fields: route (BASIC|AGENT), confidence, suggested_agent
- Pattern matching (fast/deterministic) → LLM judge if confidence < threshold

## All 11 Agents
| Agent | Mode | Purpose |
|-------|------|---------|
| general | agentic | Default multi-tool general agent |
| basic | basic | No-tool fast chat path |
| coordinator | coordinator | Orchestrates worker agents via job system |
| data_analyst | agentic | Tabular data + Python code execution |
| rag_worker | rag | Grounded document retrieval + synthesis |
| graph_manager | agentic | GraphRAG / Neo4j queries |
| supervisor | agentic | Meta-agent for result validation |
| finalizer | agentic | Result formatting & post-processing |
| planner | agentic | Task decomposition for coordinator |
| verifier | agentic | Output verification |
| memory_maintainer | memory_maintainer | Manages persistent memory records |

## AgentDefinition Fields (contracts/agents.py)
name, mode, description, prompt_file, skill_scope, allowed_tools, allowed_worker_agents, memory_scopes, max_steps, max_tool_calls, allow_background_jobs, metadata

## Tool Categories
RAG Tools: search_knowledge_base, inspect_collection, search_graph_index, inspect_graph_index
Analysis Tools: execute_code, load_dataset, run_nlp_column_task, inspect_columns, return_file
Document Tools: list_documents, extract_requirement_statements, export_requirement_statements
Inventory Tools: list_kb_collections, list_graphs, list_uploaded_documents
Utility Tools: calculator, get_time, skill_search
MCP Tools: dynamic via discover_tools / call_deferred_tool

## RAG Pipeline (rag/engine.py)
1. Inventory Classification (is it a catalog query?)
2. Retrieval Scope Decision (uploads | KB | both | neither)
3. Collection Selection (auto or clarification)
4. Retrieval Controller — Hybrid search (vector + BM25 keyword)
5. Evidence Selection (top-K, relevance grading)
6. Answer Synthesis (LLM grounded)
7. RagContract output (answer, citations, confidence, followups)

ControllerHints: search_mode (vector|keyword|hybrid|graph|deep), coverage_goal, research_profile, result_mode

## Memory System (memory/)
Three tiers:
1. Session State — immutable message history, per-turn metadata
2. Long-Term Managed Memory — PostgreSQL, TTL, importance scoring
3. File-based fallback — FileMemoryStore

Memory types: decision, constraint, open_loop, task_state, profile_preference
Memory scopes: conversation, user, task, global
Manager flow: MemoryCandidateRetriever → MemorySelector → MemoryWriteManager → MemoryProjector

## Key Config (config.py — 210+ options)
max_agent_steps, max_tool_calls, max_parallel_tool_calls
rag_top_k_vector: 15, rag_top_k_keyword: 10
memory_enabled, memory_manager_mode (shadow|selector|live)
llm_provider (ollama|azure|nvidia), embeddings_provider
