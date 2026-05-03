# Router Rubric

The live router returns one of two routes:

- `BASIC`
- `AGENT`

## Live implementation

- deterministic rules: `src/agentic_chatbot_next/router/router.py`
- validated pattern config and normalization: `src/agentic_chatbot_next/router/patterns.py`
- default intent config: `data/router/intent_patterns.json`
- hybrid LLM escalation: `src/agentic_chatbot_next/router/llm_router.py`
- initial-agent selection: `src/agentic_chatbot_next/router/policy.py`

## Deterministic matcher behavior

The deterministic matcher now loads phrase/regex groups from config instead of relying on
hard-coded regex constants.

Before matching, the router normalizes text with:

- Unicode casefolding
- whitespace collapse
- accent-insensitive matching

That improves synonym and multilingual handling without forcing a code change for every new
phrase family.

Phrase matching is boundary-aware rather than raw substring matching. Short phrase entries such as
`plan` or `tab` match normalized whole words/phrases, while intentionally broad matching belongs
in the regex groups.

## Current agent hints

The router may suggest:

- `coordinator`
- `research_coordinator`
- `data_analyst`
- `graph_manager`
- `rag_worker`
- `""`

If no hint is returned, AGENT turns normally start in `general`.
If `ENABLE_COORDINATOR_MODE=true`, `policy.py` forces `coordinator` as the
initial AGENT role regardless of the router hint.

## Requested-agent override

The router hint is still advisory. API and notebook callers can pass
`metadata.requested_agent` to start the AGENT route in a specific validated role.

Important boundaries:

- the router still records its normal `BASIC` vs `AGENT` decision and reasons
- `force_agent=true` only forces AGENT; it does not choose the agent
- `requested_agent` is validated after routing against the routable non-`basic` registry roles
  plus explicit non-routable manual overrides
- common valid values are `general`, `coordinator`, `data_analyst`, `rag_worker`, and
  `graph_manager`
- `research_coordinator` is valid when the registry exposes it as the research campaign
  manager
- `rag_researcher` is also valid because its metadata sets `manual_override_allowed=true`;
  it is still not selected by normal router policy
- `memory_maintainer` is not a valid override target when `MEMORY_ENABLED=false`

## Hybrid fallback behavior

When `LLM_ROUTER_MODE=hybrid`, the router uses deterministic matching first. If
deterministic confidence is below `LLM_ROUTER_CONFIDENCE_THRESHOLD`, it can
escalate to the judge model.

When `LLM_ROUTER_MODE=llm_only`, the judge model becomes the primary router for
ordinary text turns. Hard deterministic fast paths are still preserved for
attachments and explicit `force_agent` requests.

If `LLM_ROUTER_ENABLED=false`, the judge-model path is skipped entirely and the runtime stays on
deterministic/config-driven routing.

If the judge model is unavailable or its circuit breaker is open:

- the router falls back to the deterministic decision
- the decision reason includes `llm_router_circuit_open` or `llm_router_failed`
- the runtime emits a degraded-routing event for observability

## Hard AGENT signals

- attachments or uploads
- search, retrieval, or citation requests
- document comparison
- high-stakes domains
- spreadsheet, CSV, workbook, or pandas-style requests
- clear multi-step workflows

## Typical `data_analyst` hints

The deterministic router is especially likely to suggest `data_analyst` when
the turn mentions spreadsheet-style analysis, especially alongside
attachments.

Common hints include:

- uploaded CSV, Excel, workbook, worksheet, or tab analysis
- requests to inspect or mutate a sheet, tab, or workbook
- add-column or add-sheet asks
- reviews-column or sentiment-analysis requests
- row classification language such as `classify rows`
- chart, scatter, correlation, group-by, or pivot-style analysis

## Typical `rag_worker` hints

The router is especially likely to suggest `rag_worker` for bounded grounded lookup tasks:

- cite or summarize a specific clause, section, or document
- answer a focused question from uploaded docs
- find the source for one claim
- perform a single grounded lookup that does not imply corpus-wide coverage

## Typical `graph_manager` hints

The router is especially likely to suggest `graph_manager` when the turn names graph-backed
evidence, GraphRAG, graph inventories, relationship/entity networks, dependencies,
source-planning, or named graph queries. These turns use `requested_scope_kind="graph_indexes"`
and can start directly in the graph specialist instead of first routing through RAG.

## Typical `research_coordinator` hints

The router is especially likely to suggest `research_coordinator` for long-running or
corpus-scale document research campaigns such as:

- `identify all documents/files that ...`
- corpus-wide inventories
- exhaustive or broad discovery requests
- compare many SOPs, policies, or operating procedures
- requests that imply multiple independent document-focused subtasks before synthesis

This keeps small lookups fast on `rag_worker` while routing corpus-scale work into
planner-led worker campaigns.

## Typical `coordinator` hints

`coordinator` remains the generic manager for explicit multi-step orchestration, delegated
manager work, and active-document follow-ups that do not need the dedicated deep research
campaign role.

## Typical BASIC signals

- greetings
- small talk
- lightweight general-knowledge questions
- conversational follow-ups that do not need tools
