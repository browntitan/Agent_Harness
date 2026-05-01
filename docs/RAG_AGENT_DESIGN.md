# RAG Agent Design

The live RAG path is now the next-runtime contract flow in `src/agentic_chatbot_next/rag/`.

## Live entrypoint

The runtime calls `run_rag_contract(...)`, which returns the preserved RAG contract:

- `answer`
- `citations`
- `used_citation_ids`
- `confidence`
- `retrieval_summary`
- `followups`
- `warnings`

Internal-only runtime wiring can also pass:

- `runtime_bridge`
- `progress_emitter`
- `allow_internal_fanout`

Internal worker payloads may also carry structured execution hints without changing the
public contract:

- `research_profile`
- `coverage_goal`
- `result_mode`
- `controller_hints`

Coordinator-owned worker handoffs can also seed the live RAG path without changing the
public contract:

- `analysis_summary`
- `entity_candidates`
- `keyword_windows`
- `doc_focus`
- `evidence_request`
- `evidence_response`

## Current invocation paths

The live next runtime uses the RAG contract flow in:

1. `rag_worker`
2. `rag_agent_tool` exposed to `general` and `verifier`
3. upload-summary kickoff from `RuntimeService.ingest_and_summarize_uploads(...)`

In normal routing, bounded grounded lookups often start directly in `rag_worker`. The delegated
tool path remains live for `general` and `verifier`, and notebook/API demos can explicitly pin
that path with `metadata.requested_agent=general` when they want to show tool traces.

`rag_worker` now has two live execution shapes:

- normal answer mode, which returns the stable RAG contract
- evidence-only worker mode, used internally by deep retrieval fan-out jobs

## Two grounded-answer shapes

The current runtime intentionally supports two user-visible grounded RAG shapes:

1. direct grounded RAG
   - route enters `rag_worker`
   - no top-level `rag_agent_tool` wrapper is involved
   - traces emphasize retrieval/controller milestones
2. delegated grounded RAG
   - route enters `general` or `verifier`
   - `rag_agent_tool` wraps `run_rag_contract(...)`
   - traces include tool lifecycle in addition to the grounded result

Both shapes end in the same stable contract and the same citation model.

## Current internal stages

The next-owned RAG flow is:

1. resolve execution hints from explicit task payloads, selected skill packs, and query
   heuristics
2. build a fast-path seed run from hybrid PostgreSQL retrieval
3. decide whether to stay fast or escalate to adaptive deep retrieval
4. in deep mode, iterate through rewritten, keyword, and semantic query variants
5. use backend-agnostic retrieval operations inside `CorpusRetrievalAdapter`:
   - `search_corpus(...)`
   - `grep_corpus(...)`
   - `read_document(...)`
   - `fetch_chunk_window(...)`
   - `prune_chunks(...)`
6. accumulate evidence in an internal ledger with round summaries and unresolved items
7. optionally request internal `rag_worker` evidence jobs through `RagRuntimeBridge`
8. merge worker evidence back into one ledger, re-grade, and choose bounded evidence docs
9. choose the answer style from the structured hints:
   - normal grounded synthesis
   - per-document inventory output
   - explicit negative-evidence reporting for exhaustive or corpus-wide searches
10. build grounded citations
11. synthesize one final answer from the merged evidence set
12. coerce the result into the stable contract

Implementation notes:

- `skill_context` and `task_context` remain boundary-compatible inputs
- live RAG still does not run a prompt-backed specialist agent loop, but it now consumes
  structured execution hints derived from skill packs and planner/coordinator payloads
- the live runtime now also exposes a narrow read-only indexed-document tool surface for
  exact file work:
  - `resolve_indexed_docs`
  - `search_indexed_docs`
  - `read_indexed_doc`
  - `compare_indexed_docs`
- adjacent document-research tools are also bound for prompt-backed agents:
  - `document_extract`
  - `document_compare`
  - `document_consolidation_campaign`
  - `template_transform`
  - `evidence_binder`
  - `extract_requirement_statements`
  - `export_requirement_statements`
- those tools share the same database-backed corpus and collection scoping as the adaptive
  controller; they are not a separate filesystem search path
- internal fan-out is bounded and non-recursive: evidence-only workers run with
  `allow_internal_fanout=false`

## Search modes

The live contract path supports:

- `fast`: one bounded hybrid retrieval and grading pass
- `auto`: fast by default, with automatic escalation when the query is complex or evidence
  is weak
- `deep`: multi-round retrieval with query refinement, chunk-window expansion,
  focused document reads, pruning, and optional internal worker fan-out

## What the live search actually hits

Answer-time retrieval is DB-first, not filesystem-first.

- the live search path queries indexed chunks stored in PostgreSQL / pgvector
- semantic search uses pgvector similarity over stored chunk embeddings
- keyword search uses PostgreSQL full-text search over stored chunk text
- the filesystem is primarily used earlier for ingest, uploads, workspace files, and source staging
- the final grounded answer is synthesized from selected chunk/document evidence returned by the stores

## Filtering and selection

The live path does more than fetch top-k chunks and stop.

- retrieval merges vector and keyword candidates
- title matches can be boosted when the query strongly suggests a specific document
- prompt catalogs, question-echo chunks, and irrelevant operational runbooks can be demoted during grading
- deep retrieval re-grades evidence, tracks coverage in an evidence ledger, and selects a bounded evidence set before final synthesis

## Performance controls

The controller is accuracy-first by default, but it now avoids avoidable repeated work:

- fast mode performs one fused hybrid retrieval pass, then optional graph/neighbor expansion
- vector and BM25 breadth stay configurable with `RAG_TOPK_VECTOR` and `RAG_TOPK_BM25`
- `RAG_BUDGET_MS` gives the RAG controller a soft per-request budget before the HTTP client timeout
- `RAG_BUDGET_SYNTHESIS_RESERVE_MS` reserves time for cited answer construction
- `RAG_HEURISTIC_GRADING_ENABLED` allows deterministic pre-grading to skip slow judge grading when evidence is already decisive
- `RAG_JUDGE_GRADE_MAX_CHUNKS` caps the uncertain candidate window sent to the judge model
- `RAG_EXTRACTIVE_FALLBACK_ENABLED` allows a concise cited extractive answer when the budget is exhausted before synthesis

For local Ollama-heavy runs, keep model-provider tuning outside the retrieval contract:
increase Ollama keep-alive, tune provider parallelism to available memory, and avoid loading
more large chat/judge models than the machine can keep resident. Langfuse callbacks are skipped
when the configured host cannot be resolved, unless `LANGFUSE_ALLOW_UNREACHABLE=true` is set.

## Skill-driven execution hints

RAG skill packs now support machine-readable metadata in addition to prose guidance:

- `retrieval_profile`
- `controller_hints`
- `coverage_goal`
- `result_mode`

Those hints are resolved through the live skill index and merged with explicit planner or
worker payload fields before retrieval starts.

This is how corpus-mining behavior is now shaped for the direct contract path without
turning `rag_worker` into a generic prompt-driven ReAct agent.

Coordinator handoffs are merged into the same hint-resolution flow. For example:

- `entity_candidates` and `keyword_windows` can bias query expansion
- `doc_focus` can narrow candidate reads
- `analysis_summary` and `evidence_request` can shape follow-up search intent
- `evidence_response` can be preserved as structured context for downstream synthesis

## Coordinator-owned research campaigns

The runtime now uses two different multi-step patterns for document work:

- direct `rag_worker` execution for small grounded lookups
- coordinator-owned document research campaigns for broad corpus discovery, inventories,
  exhaustive searches, and multi-document comparison

In the campaign path:

1. the router prefers `coordinator` for corpus-scale research asks
2. `planner` emits one or more `rag_worker` tasks with `doc_scope`, `skill_queries`, and
   structured RAG hint fields
3. `coordinator` launches durable workers, tracks progress, then hands results to
   `finalizer`
4. `verifier` checks citation sufficiency and overclaim risk for corpus-wide conclusions

`rag_worker` remains a retrieval specialist. It does not receive `spawn_worker` and is not
the owner of durable sub-agent orchestration.

The live direct path may now queue one bounded async peer follow-up through the runtime job
manager when a judge-model decision says a specialist continuation is better than answering
immediately. That keeps `rag_worker` non-ReAct while still allowing same-session escalation
to roles such as `data_analyst`, `utility`, or `general`.

## Optional GraphRAG augmentation

The live retrieval stack is still PostgreSQL + pgvector first, but it can now be augmented
by a managed GraphRAG catalog/query layer with optional Neo4j compatibility fallback.

Graph phases that are live in code today:

1. managed graph catalog records backed by PostgreSQL
2. GraphRAG project build flow that materializes source text into project `input/`, writes `settings.yaml`, and persists artifact metadata
3. graph-backed retrieval augmentation inside the adaptive controller
4. managed graph tools for list, inspect, query, and source-plan explanation
5. compatibility `graph_search_local` and `graph_search_global` wrappers that delegate to the managed graph path when it is available

Design boundary:

- graph traversal yields candidate document, clause, or chunk ids
- the existing text retrieval path still fetches the final quoted evidence
- citations remain grounded in text chunks, not graph facts alone

This means GraphRAG augments the retrieval controller, but does not replace the live
vector/keyword foundation or the stable RAG answer contract.

## Current RAG skill-pack families

The main RAG-oriented packs live under `data/skill_packs/rag/`:

- `citation_hygiene`
- `clause_extraction`
- `collection_scoping`
- `comparison_campaign`
- `corpus_discovery`
- `coverage_sufficiency_audit`
- `cross_document_inventory`
- `document_resolution`
- `empty_result_recovery`
- `graph_drift_followup`
- `graph_freshness_and_staleness_check`
- `graph_global_community_discovery`
- `graph_grounding_and_resolve_back`
- `graph_local_relationship_tracing`
- `graph_vs_vector_source_selection`
- `knowledge_base_search_guidance`
- `multi_document_comparison`
- `negative_evidence_reporting`
- `process_flow_identification`
- `retrieval_strategy`
- `windowed_keyword_followup`

## Retrieval summary

`retrieval_summary` now carries richer execution telemetry:

- `query_used`
- `steps`
- `tool_calls_used`
- `tool_call_log`
- `citations_found`
- `search_mode`
- `rounds`
- `strategies_used`
- `candidate_counts`
- `parallel_workers_used`

## Corpus model

The live corpus remains DB-first:

- `documents`
- `chunks`
- `collection_id` namespacing

When GraphRAG is enabled, the corpus is effectively dual-layer:

- PostgreSQL / pgvector remains the citation and chunk source of truth
- managed GraphRAG project artifacts and graph-index metadata are stored through the
  PostgreSQL-backed graph catalog
- optional Neo4j compatibility can store entity and relationship structure for multi-hop
  candidate discovery when `GRAPH_BACKEND=neo4j`

## Why the contract still matters

The stable contract is what lets the runtime reuse the same RAG flow across:

- direct upload summaries
- tool-wrapped usage from `general`
- verifier checks
- dedicated `rag_worker` execution
- internal deep-search evidence workers

The contract is stable even while the internal implementation continues to move behind the
next-runtime modules.
