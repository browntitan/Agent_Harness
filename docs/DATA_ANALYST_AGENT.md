# Data Analyst Agent

The data analyst agent is the live tabular-analysis specialist in
`agentic_chatbot_next`.

It is not a graph node or a separate runtime. In the live system it is a
markdown-defined `AgentDefinition(mode="react")` that stays on the guided
`plan_execute` path and works against a persistent session workspace.

## Overview

| Property | Value |
|---|---|
| Runtime agent name | `data_analyst` |
| Prompt file | `data/skills/data_analyst_agent.md` |
| Runtime mode | `react` |
| Execution strategy | `plan_execute` |
| Tools | `21` declared in `data/agents/data_analyst.md` |
| Sandbox | Docker container with bind-mounted workspace at `/workspace` |
| Primary file source | indexed KB documents plus persistent session workspace |
| NLP path | bounded provider-backed column task |
| File handoff | workspace artifact registration through `return_file` |

## Current tool set

The live data analyst runtime agent receives:

1. `load_dataset`
2. `profile_dataset`
3. `inspect_columns`
4. `execute_code`
5. `run_nlp_column_task`
6. `return_file`
7. `calculator`
8. `scratchpad_write`
9. `scratchpad_read`
10. `scratchpad_list`
11. `workspace_write`
12. `workspace_read`
13. `workspace_list`
14. `search_skills`
15. `request_parent_question`
16. `request_parent_approval`
17. `invoke_agent`
18. `post_team_message`
19. `list_team_messages`
20. `claim_team_messages`
21. `respond_team_message`

## Public interfaces

```text
load_dataset(doc_id="", sheet_name="")
profile_dataset(doc_id="", sheet_name="", sample_rows=5)
inspect_columns(doc_id="", columns="", sheet_name="")
execute_code(code, doc_ids="")
run_nlp_column_task(doc_id="", sheet_name="", column="", task="", classification_rules="", allowed_labels_csv="", batch_size=5, output_mode="", target_filename="", label_column="", score_column="")
return_file(filename="", label="")
scratchpad_write(key="", value="")
scratchpad_read(key="")
scratchpad_list()
workspace_write(filename="", content="")
workspace_read(filename="")
workspace_list()
request_parent_question(question="", context="")
request_parent_approval(action="", rationale="", payload={})
invoke_agent(agent_name="", task="")
post_team_message(channel_id="", content="", message_type="message", target_agents=[], target_job_ids=[], subject="", payload={})
list_team_messages(channel_id="", message_type="", status_filter="open", limit=20)
claim_team_messages(channel_id="", limit=0, message_type="")
respond_team_message(channel_id="", message_id="", response="", decision="", resolve=true)
```

- `load_dataset(doc_id="", sheet_name="")`
  loads CSV, `.xls`, or `.xlsx` files from the KB or the session workspace.
  For Excel inputs it returns the selected `sheet_name` and the workbook
  `sheet_names`.
- `profile_dataset(doc_id="", sheet_name="", sample_rows=5)`
  profiles CSV or Excel inputs, including all workbook sheets when no sheet is specified.
  It returns sheet names, shapes, columns, sample rows, source refs, warnings, and
  `operations: ["profile_dataset"]`.
- `inspect_columns(doc_id="", columns="", sheet_name="")`
  returns per-column statistics and `_meta` with `doc_id`, `sheet_name`, and
  `sheet_names`.
- `execute_code(code, doc_ids="")`
  is the open-ended numeric, statistical, and charting path. It runs Python in
  Docker against the bind-mounted `/workspace`.
- `run_nlp_column_task(...)`
  is the bounded LLM-backed NLP path for `sentiment`, `categorize`, `keywords`,
  and `summarize`. The tool owns the outer system prompt, batching, JSON
  validation, and repair logic. It is not an unrestricted prompt passthrough.
- `return_file(filename="", label="")`
  registers an existing workspace file for user download and returns the
  normalized artifact manifest. If `filename` is omitted, the tool uses the
  most recent analyst output when available.
- scratchpad tools hold short-lived planning state.
- workspace tools inspect or edit plain workspace files directly.
- `search_skills` retrieves skill-pack guidance, `calculator` is still available for quick
  arithmetic, and `invoke_agent` can open one bounded same-session follow-up to an allowed
  peer such as `utility`, `general`, or `rag_worker`.
- `request_parent_question` and `request_parent_approval` are only useful when the analyst is
  running as a worker job; normal top-level turns cannot use them to pause for parent input.
- Team mailbox tools are only callable when `TEAM_MAILBOX_ENABLED=true`; they coordinate typed
  status, handoff, question, and response messages without granting broader file, sandbox, or
  tool permissions.

## Invocation paths

The current runtime reaches this agent in two ways:

- directly, when the router suggests `data_analyst`
- indirectly, when `coordinator` delegates a tabular-analysis task
- indirectly, when the RAG bridge creates a tabular evidence task for spreadsheet or CSV
  evidence and invokes `data_analyst` as a bounded worker

## Operating workflow

The live prompt and tool surface steer the analyst into a plan-first workflow:

1. profile the dataset first with `profile_dataset`
2. decide whether the task is sandbox code, bounded NLP, or both
3. write derived outputs into the session workspace
4. verify the result
5. call `return_file` when the task creates a downloadable deliverable
6. summarize the result for the user

That behavior comes from the prompt plus `execution_strategy: plan_execute` in
the agent metadata, not from a custom graph wrapper.

## RAG tabular evidence handoff

The RAG controller can ask the runtime bridge for tabular evidence when retrieved evidence
points at a CSV or spreadsheet and the question asks for lookup, profiling, filtering,
aggregation, comparison, or similar row/column reasoning.

That handoff is still a worker job in the same session, not a new public endpoint. The
worker prompt requires `profile_dataset` first, then permits `inspect_columns` or
`execute_code` only if needed. The worker must return structured JSON with `summary`,
`findings`, `source_refs`, `operations`, `warnings`, and `confidence`; the RAG path converts
that result into citation-eligible tabular evidence for final synthesis.

## Workspace and file model

The normal execution path assumes a persistent workspace:

- `data/workspaces/<filesystem_key(session_id)>/`
- bind-mounted into Docker at `/workspace`

That workspace is opened in the live runtime by:

- every `RuntimeService.process_turn(...)` call when workspaces are enabled
- `POST /v1/ingest/documents`
- `POST /v1/upload`

Both API ingest paths now use the canonical
`SessionWorkspace.for_session(session_id, WORKSPACE_DIR)` workspace, where the
`session_id` comes from the active request context and is stored on
`SessionState`.

Scoped worker jobs inherit the same session workspace. There is no per-job
analyst workspace today.

### Copy-on-write output rules

- uploaded source files are preserved
- analyst mutations produce derived files such as
  `source__analyst_sentiment.csv` or `source__analyst_summary.xlsx`
- CSV outputs stay CSV for simple row and column edits
- workbook-style outputs are written as `.xlsx`
- Excel inputs preserve non-target sheets when a single sheet is updated
- `.xls` inputs remain readable, but writable outputs become `.xlsx`

`return_file` does not move files out of the workspace. It registers a
workspace file in session `downloads` metadata, adds a pending assistant
artifact, and lets the gateway expose the file through `/v1/files/{download_id}`
and chat `artifacts`.

## Typical flows

### Multi-sheet inspection

1. `load_dataset(..., sheet_name="...")`
2. `inspect_columns(..., sheet_name="...")`
3. `execute_code(...)` or `run_nlp_column_task(...)`

### Row-level NLP labeling

1. `load_dataset(...)`
2. `run_nlp_column_task(...)`
3. `return_file(...)` when the task transforms the dataset row by row

Sentiment defaults:

- labels: `positive`, `neutral`, `negative`
- appended columns: `sentiment_label`, `sentiment_score`
- row-level requests default to a chat summary plus a returned derived file
- distribution-only requests stay summary-only unless the user asks for a file

### Numeric analysis and charting

1. `load_dataset(...)`
2. `inspect_columns(...)`
3. `execute_code(...)`
4. optional `return_file(...)`

## Docker behavior

### Isolation properties

| Property | Value |
|---|---|
| Image | `SANDBOX_DOCKER_IMAGE` |
| Network | disabled |
| Timeout | `SANDBOX_TIMEOUT_SECONDS` |
| Memory cap | `SANDBOX_MEMORY_LIMIT` |
| Working directory | `/workspace` |
| Preinstalled packages | `pandas`, `openpyxl`, `xlrd`, `numpy`, `matplotlib`, `pillow` |

The container is ephemeral. Persistence comes from the workspace bind mount, not
from keeping the container alive.

The analyst sandbox is now an explicit offline image contract. The runtime does
not `pip install` packages at execution time. Instead, the configured
`SANDBOX_DOCKER_IMAGE` must already contain the analyst dependencies. The
default local/dev image is `agentic-chatbot-sandbox:py312`, and both
`python run.py doctor --strict` and the demo notebook preflight verify that the
image is present locally and can import the analyst stack with `--network none`.

If the image is missing or fails the readiness probe, rebuild it with:

```bash
python run.py build-sandbox-image
```

## Fallback and failure behavior

If no persistent workspace is available, analyst file tools return a clear
`No session workspace is available.` error instead of silently falling back to
a separate copy-into-container path.

If Docker is unavailable:

- the `data_analyst` runtime definition still exists
- `execute_code(...)` returns an error payload

If providers are unavailable, bounded NLP tasks fail, but the rest of the
analyst tool surface still exists.

## Why this agent still matters

The live next runtime uses the same kernel for all AGENT turns, but the data
analyst role still keeps tabular work isolated through:

- a narrower tool surface
- a persistent workspace
- a dataset-profile step that works across all workbook sheets
- explicit code-execution boundaries
- a bounded NLP helper for row-level text tasks
- explicit file publication through `return_file`
- optional skill lookup for analysis procedures
