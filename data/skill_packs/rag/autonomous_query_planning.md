---
name: Autonomous Query Planning
agent_scope: rag_researcher
tool_tags: plan_rag_queries, search_corpus_chunks, grep_corpus_chunks, rag_agent_tool
task_tags: query_planning, rewrite, retrieval, facets
version: 1
enabled: true
description: Break complex grounded questions into visible, bounded retrieval facets before evidence search.
keywords: query planning, rewrite, exact terms, semantic search, contradiction search
when_to_apply: Use when a question needs more than one retrieval lane or the first query may miss exact names, dates, versions, or conflicts.
avoid_when: Avoid for pure inventory or exact named-file reads that can be solved directly.
examples: multi-hop retrieval, entity query rewrite, authority check
---
# Autonomous Query Planning

## Workflow

Plan before searching when the request is exploratory, multi-document, authority-sensitive, or ambiguous.

Use these facets when useful:

- `semantic`: broad hybrid search for the whole question.
- `exact_terms`: keyword search for quoted text, identifiers, dates, part numbers, supplier names, clause numbers, or file names.
- `entity`: entity-focused search for names, programs, vendors, systems, or teams.
- `date_version`: current/latest/final/approved/draft/revision checks.
- `source_discovery`: document-finding query before scoped reads.
- `contradiction`: conflicting, superseded, obsolete, exception, or counter-evidence search.

Do not answer from the plan. Use the plan to choose retrieval tools, then send final factual synthesis through `rag_agent_tool`.
