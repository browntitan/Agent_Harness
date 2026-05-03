# RAG Tool Contract

`rag_agent_tool` remains backward-compatible.

It wraps `run_rag_contract()` and returns the same stable JSON contract used by the live
next runtime.

## Current positioning

In the live runtime:

- `general` uses `rag_agent_tool` for grounded document work from the top-level ReAct loop
- `verifier` can also use `rag_agent_tool`
- `rag_worker` bypasses the wrapper and calls `run_rag_contract()` directly

So the contract remains central even though not every RAG invocation goes through the tool.

Normal bounded grounded lookups may route straight to `rag_worker`. The delegated
`general -> rag_agent_tool` path is still supported and is sometimes pinned intentionally with
`metadata.requested_agent=general` in demos, tests, or operator troubleshooting so the tool
wrapper remains visible in traces.

## Input

Supported arguments:

- `query`
- `conversation_context`
- `preferred_doc_ids_csv`
- `must_include_uploads`
- `top_k_vector`
- `top_k_keyword`
- `max_retries`
- `search_mode`
- `max_search_rounds`
- `scratchpad_context_key`
- `research_profile`
- `coverage_goal`
- `result_mode`
- `controller_hints_json`
- `skill_context`

Coordinator-owned typed handoff artifacts are also internal-only. They can shape the live
`rag_worker` path, but direct tool callers do not send them as first-class tool arguments.

Important boundary:

- `rag_agent_tool` exposes caller-controlled retrieval knobs
- `controller_hints_json` is the public JSON-string form of controller hints; it must parse
  to a JSON object
- exact file targeting for named indexed docs is now also available through the read-only
  `resolve_indexed_docs`, `search_indexed_docs`, `read_indexed_doc`, and
  `compare_indexed_docs` tools
- adjacent document-research helpers such as `document_extract`, `document_compare`,
  `template_transform`, `evidence_binder`, `extract_requirement_statements`, and
  `export_requirement_statements` are separate tools; they do not change the
  `rag_agent_tool` JSON contract
- planner/coordinator payloads may add internal structured retrieval hints
- typed handoffs may add validated downstream context for worker-to-worker campaigns
- optional GraphRAG augmentation stays internal to the retrieval controller
- `skill_context` is optional synthesis guidance; it does not bypass citation grounding

That split is what lets the runtime support both direct grounded RAG and delegated
tool-wrapped grounded RAG without changing the contract shape.

Invalid controller-hint examples return a warning contract without running retrieval:

```json
{
  "answer": "",
  "citations": [],
  "used_citation_ids": [],
  "confidence": 0.0,
  "retrieval_summary": {
    "query_used": "find evidence",
    "search_mode": "none"
  },
  "followups": [],
  "warnings": ["INVALID_CONTROLLER_HINTS_JSON: expected a JSON object"]
}
```

## Output

```json
{
  "answer": "...",
  "citations": [
    {
      "citation_id": "...",
      "doc_id": "...",
      "title": "...",
      "source_type": "...",
      "location": "...",
      "snippet": "..."
    }
  ],
  "used_citation_ids": ["..."],
  "confidence": 0.84,
  "retrieval_summary": {
    "query_used": "...",
    "steps": 3,
    "tool_calls_used": 5,
    "tool_call_log": [],
    "citations_found": 4,
    "search_mode": "deep",
    "rounds": 2,
    "strategies_used": ["hybrid", "keyword", "grade", "worker"],
    "candidate_counts": {
      "unique_chunks": 12,
      "unique_docs": 4,
      "selected_docs": 3
    },
    "parallel_workers_used": true
  },
  "followups": [],
  "warnings": []
}
```

## Stability expectations

The next-runtime cutover did **not** change:

- key names
- citation object shape
- confidence field
- retrieval summary presence

The public shape is still stable even though the summary now exposes richer telemetry for
adaptive retrieval and internal worker fan-out.

That stability is what lets the tool remain a safe interface for callers outside the
specialist RAG worker path.

## Live implementation note

The live `rag_agent_tool` / `rag_worker` path is a direct Python pipeline over:

- adaptive candidate retrieval
- multi-round grading and evidence selection
- optional internal evidence-worker fan-out
- grounded answer synthesis
- stable-contract rendering

For direct callers, the tool still returns one final answer contract. The internal
fan-out path is runtime-owned and not exposed as a separate public tool surface.

For broad corpus-discovery campaigns, the public tool is still only one piece of the full
runtime story. The live system now prefers `research_coordinator`-owned multi-worker
planning above the tool layer, while preserving the same public contract for any direct RAG
tool call.

When GraphRAG is enabled, that remains true: graph traversal is an internal retrieval
augmentation step. End-user graph inventory and graph-backed evidence may also route directly
to `graph_manager`, but that routing choice does not change the public `rag_agent_tool`
contract.
