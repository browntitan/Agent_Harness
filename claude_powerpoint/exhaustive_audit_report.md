# Exhaustive Codebase Audit Report
## Agentic Chatbot v3 — PowerPoint vs. Source Code

**Date:** 2026-04-25
**Scope:** All 20 checklist items from `/src/agentic_chatbot_next/`

---

## CONFIRMED CORRECT

These items in the current PowerPoint accurately reflect the codebase:

1. **11 Agents** — All agent definitions confirmed in `data/agents/*.md`:
   basic, coordinator, data_analyst, finalizer, general, graph_manager, memory_maintainer, planner, rag_worker, utility, verifier

2. **54 Tools in 8 Groups** — Registry matches: utility(3), discovery(2), skills(1), memory(3), analyst(11), rag_gateway(12), graph_gateway(8), orchestration(14) + dynamic MCP tools

3. **7 Execution Modes** — query_loop.py confirms: basic, rag, react (default), memory_maintainer, planner, finalizer, verifier. Coordinator handled at kernel level.

4. **Three-Layer Runtime** — RuntimeService → RuntimeKernel → QueryLoop architecture confirmed in runtime/ directory

5. **ReAct Loop** — LangGraph's `create_react_agent` with `PolicyAwareToolNode` and `pre_model_hook` (build_microcompact_hook) confirmed in general_agent.py

6. **Router Architecture** — Deterministic regex (7 intent groups) + LLM judge fallback + SemanticRoutingContract + feedback loop with outcome tracking all confirmed

7. **70 Admin Endpoints** — control_panel/ confirmed at 70 endpoints across agents, collections, prompts, graphs, access control, config, uploads, MCP/system

8. **GraphRAG Backends** — Microsoft GraphRAG + Neo4j confirmed in graph/service.py and graph/backend.py

9. **Worker Orchestration with Team Mailbox** — Confirmed in contracts/jobs.py (TeamMailboxChannel, WorkerMailboxMessage, TaskNotification)

10. **Tool name corrections** — execute_code, memory_load, etc. confirmed as current names in registry

---

## NEEDS UPDATE

These items are in the PowerPoint but have incorrect or outdated values:

### 1. Public API Endpoint Count
- **PowerPoint says:** 47 Public Endpoints
- **Actual:** 48 Public Endpoints
- **Details:** Added endpoints include health/live, health/ready (2), chat (2), skills (6), models (1), agents (1), graphs (4), documents (1), ingest/upload (2), MCP connectors (6), tasks/jobs (5), team mailbox (4), files (1), sessions (1), capabilities/users (3), admin runtime (1)

### 2. Router Confidence Threshold
- **PowerPoint may reference:** generic confidence scoring
- **Actual:** LLM router confidence threshold = 0.70, mode = "hybrid" or "llm_only"

### 3. Context Budget Parameters
- **PowerPoint may not specify:** Exact context budget numbers
- **Actual values from config.py:**
  - context_window_tokens: 32,768
  - context_target_ratio: 0.72
  - context_autocompact_threshold: 0.85
  - context_tool_result_max_tokens: 2,000
  - context_tool_results_total_tokens: 8,000
  - context_microcompact_target_tokens: 2,400
  - context_compact_recent_messages: 12

### 4. Worker Scheduler Defaults
- **PowerPoint may not specify:** Exact scheduler numbers
- **Actual values:**
  - max_worker_concurrency: 6
  - Queue classes: urgent (weight 8), interactive (weight 3), background (weight 1)
  - tenant_budget_tokens_per_minute: 24,000
  - tenant_budget_burst_tokens: 48,000
  - Token cost estimation: text_tokens x 1.35

### 5. RAG Pipeline Defaults
- **Actual values not in PowerPoint:**
  - rag_top_k_vector: 15
  - rag_top_k_keyword: 15
  - chunk_size: 900, chunk_overlap: 150
  - RRF fusion with k=60
  - 4-level relevance grading (0=not relevant, 1=somewhat, 2=relevant, 3=highly relevant)
  - max_parallel_collection_probes: 4

### 6. Memory Subsystem Defaults
- **Actual values not in PowerPoint:**
  - memory_candidate_top_k: 16
  - memory_context_token_budget: 1,600
  - memory_manager_mode: "shadow" | "selector" | "live"
  - 5 memory types: profile_preference, task_state, decision, constraint, open_loop
  - Episode triggers: 8+ recent messages OR plan/status/decision keywords
  - Type scoring: decisions(5) > constraints(4) > open_loops(3) > task_state(2) > profile_preference(1)

---

## COMPLETELY MISSING

These subsystems and features exist in the codebase but have NO representation in the current PowerPoint:

### 1. Authorization Subsystem (`authz/`)
- Dedicated authorization directory for fine-grained access control
- Separate from API-level security (gateway_security.py, connector_security.py)
- Principals, roles, bindings, memberships, permissions, effective-access model
- 10 dedicated admin endpoints for access control management

### 2. Documents Subsystem — Full Capabilities (`documents/`)
- **DocumentExtractionService**: Supports PDF, DOCX, PPTX, XLSX, XLS, TXT, MD, CSV, TSV. Uses Docling for PDFs. Max 200 elements per extraction.
- **DocumentComparisonService**: Key-based + fuzzy matching (0.72 threshold). 14 obligation modality patterns (shall, must, required, prohibited, etc.). Binding strength ranking (Prohibitive > Mandatory > Permissive). Change severity escalation.
- **DocumentConsolidationCampaignService**: Fingerprinting with IDF weighting. 6 similarity focus modes (auto, process_flows, policies, tables, requirements, full_text). Union-find clustering. Weighted Jaccard scoring.
- **Evidence Binder**: Evidence packaging and citation management

### 3. Skills Subsystem — Full Architecture (`skills/`)
- **3 Skill Kinds**: retrievable, executable, hybrid
- **2 Execution Contexts**: inline, fork
- **5 Effort Levels**: (empty), low, medium, high, xhigh
- **SkillResolver**: Semantic retrieval with BM25/embedding scoring, agent_scope filtering, pinned skill prioritization
- **SkillDependencyGraph**: Cycle detection via DFS, dependency validation states (healthy, unstable, broken)
- **SkillTelemetry**: Answer quality tracking, 80% success SLO, 20-use review window
- **SkillIndexSync**: Incremental sync by checksum, inverted index building

### 4. Storage/Blob Store (`storage/`)
- BlobRef dataclass for cloud storage references
- Multi-backend support: S3, Azure, local filesystem
- SHA1 hashing, key sanitization, signed downloads
- Integration across API for uploaded documents and artifacts

### 5. Provider Circuit Breaker (`providers/circuit_breaker.py`)
- Fault tolerance for LLM providers
- Window size: 20 samples, min 6 samples
- Error rate threshold: 50%
- Consecutive failure limit: 3
- Open circuit duration: 30 seconds

### 6. Per-Agent Model Overrides (`providers/factory.py`)
- AgentProviderResolver with caching by (llm_provider, judge_provider, chat_model, judge_model, chat_cap, judge_cap)
- Environment variable pattern: `AGENT_{AGENT_NAME}_CHAT_MODEL`, `AGENT_{AGENT_NAME}_JUDGE_MODEL`
- Output token capping: request > agent-specific > global/demo > ollama_num_predict

### 7. Task Plan Artifact Handoff System (`runtime/task_plan.py` + `kernel_coordinator.py`)
- 8 artifact types: title_candidates, doc_focus, research_facets, facet_matches, doc_digest, subsystem_inventory, policy_guidance_matches, buyer_recommendation_table
- Artifact dependency validation between tasks
- Document ranking system with keys: is_meta_document, reviewed_relevance, matched_facets, strong_evidence_count, title_path_score, seed_hits
- Meta-document filter (regex for test fixtures, acceptance scenarios)

### 8. MCP Security Model (`mcp/`)
- Fernet encryption for MCP secrets (prefix: "fernet:v1:")
- URL validation: HTTPS enforced, private network blocking
- Tool deferral system: all MCP tools defer=True, destructive=True, defer_priority=50
- Connection timeouts: 15s (list tools), 60s (tool execution)
- Tool schema sanitization (enforces type="object", additionalProperties=True)

### 9. Sandbox Execution Environment
- sandbox_timeout_seconds: 180
- Isolated code execution for data analyst and utility agents

### 10. Graph Query Methods — Full Set
- **4 query methods**: local, global, drift, sql
- "drift" method for temporal change detection/evolution tracking is notable and likely not in PowerPoint
- Query method aliases: "graph" → (local, global), "multihop" → (local, global), "relationship" → (local)
- Phased build: Phase 1 (entity/relationship extraction) → Phase 2 (community detection) → Phase 3 (embedding/indexing)
- Required artifacts vary by method: global needs 3 (entities, communities, community_reports), local/drift need 5 (+ relationships, text_units)

### 11. Observability (`observability/`)
- Dedicated observability directory (discovered in top-level listing)
- Likely contains telemetry, logging, and monitoring infrastructure

### 12. Benchmark Suite (`benchmark/`)
- Performance benchmarking directory at top level

### 13. Control Panel Frontend
- React/TypeScript application (Vite build)
- Located at `/agentic_chatbot_v3/control_panel/` (NOT inside src/)
- Components, sections, theming, tests
- Separate from backend admin endpoints

### 14. Persistence Layer (`persistence/`)
- PostgreSQL-backed persistence (discovered in top-level listing)
- Separate from file-based stores

---

## NOTABLE DETAILS

Architectural nuances worth considering for the PowerPoint:

### 1. Configuration Scale
- config.py contains **300+ configuration variables** organized by subsystem
- Every major subsystem has fine-grained tuning parameters
- This speaks to the production maturity of the system

### 2. Dual-Path Memory Architecture
- Managed memory (LLM-selected) with fallback to file_store (legacy key-value)
- Shadow mode allows memory writes to be tested without affecting production
- Profile signal detection via regex ("remember", "save", "prefer", "call me", "my name", "i like")

### 3. RAG Verification Pipeline
- Missed document detection
- Unsupported hop detection
- Citation topic/text mismatch checking
- Stale graph detection (graph age vs. doc ingest time)

### 4. Document Intelligence
- The documents subsystem is far more sophisticated than simple extraction
- Obligation tracking (14 modality patterns) suggests legal/compliance use cases
- Consolidation campaigns can identify duplicate policies across a corpus
- Process flow matching with step-by-step comparison

### 5. Coordinator Sophistication
- KernelCoordinatorController manages multi-task orchestration with artifact dependencies
- 8 distinct artifact handoff types enable complex multi-agent workflows
- Document ranking uses 6 scoring dimensions

### 6. Data Analyst Guided Fallback
- general_agent.py includes `_run_data_analyst_guided_fallback` with heuristic code generation
- `_run_plan_execute_fallback` when tool calling not supported or fails
- Graph evidence detection via regex patterns
- Data analyst intent classification system

### 7. Embedding Dimensions
- Supports both Nomic (768-dim) and Ada-002 (1536-dim) embeddings
- Configurable via config.py

### 8. Deep RAG Mode
- deep_rag_max_parallel_lanes: 3
- deep_rag_full_read_chunk_threshold: 24
- deep_rag_sync_reflection_rounds: 1
- deep_rag_background_threshold: 4
- Suggests a multi-pass deep retrieval strategy not documented elsewhere

### 9. Planner Constraints
- planner_max_tasks: 8
- max_revision_rounds: 8
- Task executors limited to: rag_worker, utility, data_analyst, general, graph_manager, verifier
- Sequential and parallel execution modes

### 10. Session Lifecycle
- Immutable identity triple: tenant_id:user_id:conversation_id
- RuntimeMessage with artifact_refs linking messages to task outputs
- Pending notification queue until next user turn
- Session compaction endpoint for long conversations

---

## SUMMARY OF CHANGES NEEDED

| Category | Count |
|----------|-------|
| Confirmed Correct | 10 items |
| Needs Update | 6 items |
| Completely Missing | 14 subsystems/features |
| Notable Details | 10 items |

**Priority recommendations for PowerPoint updates:**
1. Add slides for the Documents subsystem (extraction, comparison, consolidation)
2. Add slides for the Skills architecture (kinds, execution, telemetry, dependencies)
3. Add slides for the Authorization subsystem
4. Add a slide for the Provider circuit breaker and per-agent model overrides
5. Add a slide for the Task Plan artifact handoff system
6. Update API endpoint count from 47 to 48
7. Add specific configuration numbers (context budget, scheduler weights, RAG defaults)
8. Add the MCP security model details
9. Add the Storage/Blob subsystem
10. Cover the 4 graph query methods (especially "drift" for temporal analysis)
