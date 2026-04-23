# Test Queries

Use these prompts to smoke-test the live agent in this worktree.

## Default Collection Agentic Test Matrix

This baseline suite assumes the `default` collection includes repo docs from `docs/*.md`
through `KB_EXTRA_DIRS=./docs`, especially:

- `ARCHITECTURE.md`
- `C4_ARCHITECTURE.md`
- `CONTROL_FLOW.md`
- `NEXT_RUNTIME_FOUNDATION.md`
- `OPENAI_GATEWAY.md`
- `PROVIDERS.md`
- `OBSERVABILITY_LANGFUSE.md`
- `RAG_AGENT_DESIGN.md`
- `RAG_TOOL_CONTRACT.md`
- `TOOLS_AND_TOOL_CALLING.md`
- `DATA_ANALYST_AGENT.md`
- `WORKSPACE.md`
- `ROUTER_RUBRIC.md`
- `AGENT_DEEP_DIVE.md`
- `SKILLS_PLAYBOOK.md`

Current worktree assumptions:

- `LLM_ROUTER_ENABLED=true`
- `LLM_ROUTER_MODE=hybrid`
- `MEMORY_ENABLED=true`
- `GRAPH_SEARCH_ENABLED=false`

If `ENABLE_COORDINATOR_MODE=true`, keep the same prompts but expect AGENT turns to begin in
`coordinator` even when the matrix below lists `general` or `rag_worker` as the likely start.

## Interface Notes

- Run the natural-routing prompts as normal chat turns.
- Run the tool-trace prompts with `metadata.requested_agent="general"` when you want to see
  top-level tool calls such as `list_indexed_docs`, `rag_agent_tool`, `resolve_indexed_docs`,
  `read_indexed_doc`, and `compare_indexed_docs`.
- For older API examples and trace docs that use assignment-style notation, this is the same
  override as `metadata.requested_agent=general`.
- Run the analyst prompts only after uploading:
  - `new_demo_notebook/demo_data/data_analyst/customer_reviews_100.csv`
  - `new_demo_notebook/demo_data/data_analyst/sales_performance.xlsx`

Minimal API shape for forced `general` starts:

```json
{
  "metadata": {
    "requested_agent": "general"
  }
}
```

## 1. Route Boundary And Inventory

- `Hello there`
  Expected start: `BASIC`
  Expected path: direct answer, no tools, no citations.

- `What does the router do when the judge model circuit breaker is open? Cite your sources.`
  Expected start: `AGENT -> rag_worker`
  Expected path: direct grounded RAG with citations from `ROUTER_RUBRIC.md` and `PROVIDERS.md`.

- `What knowledge bases do you have access to?`
  Expected start: `AGENT -> general`
  Expected path: metadata inventory through `list_indexed_docs(view="kb_collections")`.

- `What documents do we have access to?`
  Expected start: `AGENT -> general`
  Expected path: metadata inventory through `list_indexed_docs(view="session_access")`.

- `Can you list out all of the documents in the default collection?`
  Expected start: `AGENT -> general`
  Expected path: metadata inventory through `list_indexed_docs(source_type="kb", collection_id="default")`.

- `What's indexed?`
  Expected start: `AGENT -> general`
  Expected path: KB inventory fast-path through `list_indexed_docs(source_type="kb")`.

## 2. Direct Grounded Lookup

- `Summarize the main components of the runtime service described in ARCHITECTURE.md. Cite your sources.`
  Expected start: `AGENT -> rag_worker`
  Expected path: bounded grounded lookup with citations from `ARCHITECTURE.md`.

- `Explain how the service, kernel, and query loop fit together in the next runtime. Cite your sources.`
  Expected start: `AGENT -> rag_worker`
  Expected path: grounded synthesis across `ARCHITECTURE.md`, `CONTROL_FLOW.md`, and `NEXT_RUNTIME_FOUNDATION.md`.

- `What provider roles exist in this system, and what does each one do? Cite your sources.`
  Expected start: `AGENT -> rag_worker`
  Expected path: grounded retrieval from `PROVIDERS.md`.

- `How does the OpenAI-compatible gateway scope requests and support requested-agent overrides? Cite your sources.`
  Expected start: `AGENT -> rag_worker`
  Expected path: grounded retrieval from `OPENAI_GATEWAY.md` and `ROUTER_RUBRIC.md`.

- `Explain the RAG tool contract and its input/output shape. Cite your sources.`
  Expected start: `AGENT -> rag_worker`
  Expected path: grounded retrieval from `RAG_TOOL_CONTRACT.md` and `RAG_AGENT_DESIGN.md`.

- `What are the copy-on-write output rules for the data analyst agent? Cite your sources.`
  Expected start: `AGENT -> rag_worker`
  Expected path: grounded retrieval from `DATA_ANALYST_AGENT.md` and `WORKSPACE.md`.

## 3. Coordinator And Research Campaign

- `Investigate the major subsystems in this repo and give me a concise architectural walkthrough with citations.`
  Expected start: `AGENT -> coordinator`
  Expected path: planner/finalizer/verifier flow with grounded citations from `ARCHITECTURE.md`,
  `C4_ARCHITECTURE.md`, `COMPOSITION.md`, and `NEXT_RUNTIME_FOUNDATION.md`.

- `Identify all documents that discuss routing, agent selection, or requested-agent overrides.`
  Expected start: `AGENT -> coordinator`
  Expected path: corpus discovery campaign with likely `rag_worker` subtasks.

- `Compare the architecture, control-flow, and gateway docs and explain where routing decisions are made.`
  Expected start: `AGENT -> coordinator`
  Expected path: multi-document comparison campaign across `ARCHITECTURE.md`, `CONTROL_FLOW.md`,
  and `OPENAI_GATEWAY.md`.

- `Identify all documents that describe process flow, typed handoffs, or coordinator execution.`
  Expected start: `AGENT -> coordinator`
  Expected path: corpus inventory or campaign using `CONTROL_FLOW.md`, `COMPOSITION.md`,
  `RAG_AGENT_DESIGN.md`, and `AGENT_DEEP_DIVE.md`.

## 4. Forced Top-Level Tool Traces

Run this section with `metadata.requested_agent="general"` so the top-level agent stays on the
tool-wrapper path instead of starting directly in `rag_worker` or `coordinator`.

- `Summarize the Runtime service section of ARCHITECTURE.md with citations.`
  Expected tools: `resolve_indexed_docs` plus `read_indexed_doc`.
  Expected evidence: `ARCHITECTURE.md`.

- `Compare ARCHITECTURE.md and C4_ARCHITECTURE.md. Cite your sources.`
  Expected tools: `resolve_indexed_docs` plus `compare_indexed_docs`.
  Expected evidence: `ARCHITECTURE.md` and `C4_ARCHITECTURE.md`.

- `Explain how the platform documents tool-calling and tool grouping. Cite your sources.`
  Expected tools: `rag_agent_tool`.
  Expected evidence: `TOOLS_AND_TOOL_CALLING.md` and `ARCHITECTURE.md`.

- `What knowledge bases do you have access to?`
  Expected tools: `list_indexed_docs(view="kb_collections")`.

- `Can you list out all of the documents in the default collection?`
  Expected tools: `list_indexed_docs(source_type="kb", collection_id="default")`.

## 5. Upload And Analyst Tools

Upload both demo analyst files in the same conversation before issuing these prompts.

- `Provide sentiment analysis of all of the reviews in the reviews column.`
  Expected start: `AGENT -> data_analyst`
  Expected tools: `load_dataset`, `inspect_columns`, `run_nlp_column_task`.

- `Add sentiment_label and sentiment_score columns and return the file.`
  Expected start: `AGENT -> data_analyst`
  Expected tools: `run_nlp_column_task`, `return_file`.

- `Create a new tab that summarizes the correlation between marketing_spend_usd and revenue_usd.`
  Expected start: `AGENT -> data_analyst`
  Expected tools: `load_dataset`, `inspect_columns`, `execute_code`, `return_file`.

- `Generate a revenue-by-region chart and return the updated workbook.`
  Expected start: `AGENT -> data_analyst`
  Expected tools: `execute_code`, `return_file`.

Expected analyst-file behavior:

- returned files surface in the top-level `artifacts` array
- returned files are downloadable through `/v1/files/{download_id}`

Regression prompt:

- start a fresh chat, then ask `Add sentiment_label and sentiment_score columns and return the file.`
  Expected behavior: the runtime explains that no uploaded dataset is attached to this conversation yet.

## 6. Failure And Feature-Flag Checks

- `Summarize the contents of the non-existent customer_churn_strategy.pdf and cite it.`
  Expected path: AGENT failure handling with a clear missing-source message and no fabricated citations.

- `Use evidence from a file I have not uploaded yet and analyze it.`
  Expected path: upload-missing guardrail with guidance to upload or attach the source first.

- `Remember that when I ask for summaries, I prefer bullet points.`
  Then ask `What preferences do you remember for this conversation?`
  Expected path: memory round-trip through `memory_save` and `memory_load` or `memory_list`.
  This only applies because `MEMORY_ENABLED=true`.

- Optional only if you later enable graph search:
  `Identify all documents that describe process flow, typed handoffs, or coordinator execution, then summarize the relationship graph.`
  Expected path: same corpus as the coordinator campaign above, with graph augmentation available only when
  `GRAPH_SEARCH_ENABLED=true`.

## 7. Deep Research Diagnosis And Query Pack

Use this section when you want to test the agent as a corpus researcher instead of a bounded
single-question RAG responder.

### Why some deep-research prompts work better than others

You are not just using bad prompts. The current runtime shape makes some query styles much more
reliable than others.

- The router only sees a short recent summary, not the full transcript.
  Current implementation details:
  - `src/agentic_chatbot_next/app/service.py` uses `_summarise_history(messages, n=2)`.
  - `src/agentic_chatbot_next/router/llm_router.py` explicitly frames routing context as
    `Recent conversation history (last 2 turns)`.
- Grounded RAG also gets a short recent context window rather than the full conversation.
  Current implementation details:
  - `src/agentic_chatbot_next/runtime/query_loop.py` uses `_recent_conversation_context(session, limit=6)`.
  - that helper clips each message to `message.content[:300]`.
- The coordinator has a finalizer/verifier revision loop, but there is not yet a strong planner-side
  reflection loop that re-plans retrieval when corpus coverage is weak.
  Current implementation details:
  - bounded fallback planning lives in `src/agentic_chatbot_next/runtime/task_plan.py`
  - planner instructions are still comparatively compact in `data/skills/planner_agent.md`
- The planner, general, and finalizer skill text is still optimized for bounded, reliable work more
  than deep research over a document corpus.
  Current implementation details:
  - `data/skills/planner_agent.md`
  - `data/skills/general_agent.md`
  - `data/skills/finalizer_agent.md`

### What works best today

Prefer one-turn, self-contained prompts.

- do not rely on a prior candidate-doc turn unless you specifically want to test doc-focus carry-forward
- ask for both phases in one request:
  - identify relevant docs in `default`
  - inspect them directly
  - synthesize a structured answer
- ask for explicit coverage and output shape:
  - `search only the default collection`
  - `be exhaustive within the corpus`
  - `read the most relevant documents directly`
  - `organize by subsystem`
  - `call out missing or thin evidence`
- if you want true long-running deep-research behavior, prefer `metadata.long_output` rather than
  expecting the default synchronous chat answer to behave like a research report

### 7.1 Best Corpus Queries

- `Search only the knowledge base in the default collection. Identify the documents that describe the architecture, runtime layers, routing, coordinator execution, and persistence model of this repo. Then read the most relevant documents directly and produce a comprehensive subsystem map organized by subsystem, responsibilities, interfaces, and supporting documents. Be exhaustive within the corpus and call out thin evidence.`

- `Investigate the full lifecycle of an AGENT request in this system, from /v1/chat/completions through routing, kernel execution, query loop execution, coordinator orchestration, worker jobs, persistence, and observability. Search only the default collection, identify the relevant docs first, then synthesize a detailed end-to-end walkthrough with citations grouped by stage.`

- `Across the default collection, identify all documents that describe routing, agent selection, requested-agent overrides, and fallback behavior. Then produce a detailed explanation of how the router decides between BASIC, rag_worker, general, coordinator, and data_analyst, including failure and degraded-routing behavior.`

- `Search the default collection and investigate how tools, skills, and RAG cooperate in this runtime. Identify the relevant documents first, then synthesize a detailed explanation of tool groups, skill retrieval, RAG execution hints, and how those pieces influence agent behavior.`

- `Investigate the major cross-cutting subsystems of this repo across the default collection: provider selection, circuit breaker behavior, output-length controls, observability, workspace, and persistence. Read the relevant documents directly and produce a detailed architecture overview with one section per cross-cutting subsystem.`

- `Identify all documents in the default collection that describe coordinator execution, task planning, worker handoffs, verifier revisions, and final synthesis. Then explain the multi-stage research workflow of the agentic system, including where it currently reflects and where it does not.`

- `Search only the default collection and explain the relationship between ARCHITECTURE.md, C4_ARCHITECTURE.md, COMPOSITION.md, CONTROL_FLOW.md, and NEXT_RUNTIME_FOUNDATION.md. Produce a detailed synthesis of how each document contributes to understanding the system and where they overlap or differ.`

- `Investigate how the OpenAI-compatible gateway, router, and runtime service fit together in this repo. Search the default collection, identify all relevant documents, and produce a detailed explanation of request scoping, requested-agent overrides, long-form writing support, and runtime handoff into the kernel.`

- `Across the default collection, identify the documents that describe the data analyst agent, workspace model, copy-on-write outputs, and sandbox behavior. Then produce a detailed synthesis of how tabular-analysis work differs from normal rag_worker and coordinator execution.`

- `Investigate how LangChain, LangGraph, and the custom runtime are divided in this codebase. Search only the default collection, identify the relevant documents, and produce a detailed explanation of what is and is not a LangGraph flow in the live system.`

### 7.2 Best Long-Output Variants

Use the same prompts above, but add `metadata.long_output.enabled=true`.

Suggested long-output targets for this corpus:

- `target_words: 2500-4000`
- `target_sections: 5-7`
- `delivery_mode: hybrid`
- `background_ok: true`

Good long-output prompt shapes:

- `Produce a comprehensive architecture handbook for this repo from the default collection only. First identify relevant docs, then synthesize a sectioned report.`
- `Produce a deep research report on the runtime architecture, routing, RAG, and observability design of this repo using only the default collection.`

### 7.3 Query Styles To Avoid For Now

- `major subsystems` by itself with no required output structure
- multi-turn dependency like:
  - turn 1 `find candidate docs`
  - turn 2 `summarize those docs`
- prompts that assume the model will remember a lot of earlier instructions across many turns
- prompts that do not explicitly say:
  - corpus scope
  - deliverable structure
  - whether to identify docs first or synthesize directly

### 7.4 How To Judge Failures

When a deep-research run underperforms, sort the failure into one of these buckets:

- routing failure:
  it starts as a bounded `rag_worker` lookup instead of a corpus campaign
- planning failure:
  it identifies too few facets or launches too few worker searches
- synthesis failure:
  it finds the right docs but produces a shallow or incomplete answer
- context failure:
  a later turn ignores prior scope or earlier constraints

### 7.5 Practical Test Sequence

Run these three test types separately:

- discovery-only:
  `Identify all documents in the default collection that describe routing and coordinator execution. Return only document titles with short grounded justifications.`
- one-turn deep synthesis:
  use one of the queries in section `7.1` and expect a `coordinator`-style multi-step campaign
- long-form deep research:
  use one of the long-output prompt shapes in section `7.2` with `metadata.long_output`

## 8. Additional API And Long-Form Checks

These are still useful smoke tests even though they are not part of the default-collection chat matrix.

### Long-form writing

Synchronous hybrid request:

```bash
curl -X POST http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Conversation-ID: longform-sync-smoke" \
  -d '{
    "model": "enterprise-agent",
    "messages": [
      {"role": "user", "content": "Write a detailed implementation plan for the runtime architecture in this repo."}
    ],
    "metadata": {
      "long_output": {
        "enabled": true,
        "target_words": 2200,
        "target_sections": 4,
        "delivery_mode": "hybrid",
        "background_ok": true,
        "output_format": "markdown"
      }
    }
  }'
```

Background request:

```bash
curl -X POST http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Conversation-ID: longform-async-smoke" \
  -d '{
    "model": "enterprise-agent",
    "messages": [
      {"role": "user", "content": "Produce a comprehensive operator handbook for this system."}
    ],
    "metadata": {
      "long_output": {
        "enabled": true,
        "target_words": 5000,
        "target_sections": 7,
        "delivery_mode": "hybrid",
        "background_ok": true,
        "output_format": "markdown",
        "async_requested": true
      }
    }
  }'
```

Job polling:

```bash
curl http://127.0.0.1:18000/v1/jobs/<job_id>
```

`GET /v1/jobs/{job_id}` is the durable polling path for background long-form runs.

Artifact retrieval:

- download the final draft from the normal chat `artifacts` entry via `/v1/files/{download_id}`

Expected behavior:

- the chat response contains a short assistant summary instead of the full draft body
- the top-level `artifacts` array contains the generated Markdown or text document
- backgrounded runs include `metadata.job_id`
- response metadata includes `metadata.long_output`
- streamed progress shows phase milestones such as `phase_start`, `phase_update`, and `phase_end`
- clients receive progress updates during generation and the final artifact when the run completes

### Skill API smoke checks

Use the gateway and issue:

- `GET /v1/skills`
- `POST /v1/skills/preview`
- `POST /v1/skills`
- `POST /v1/skills/{skill_id}/activate`

Expected behavior:

- the gateway accepts `X-Tenant-ID` and `X-User-ID`
- skill CRUD becomes effective without restart
- preview lets you inspect the effect of a skill body or metadata change before activation
