---
name: Graph Freshness And Staleness Check
agent_scope: rag
tool_tags: inspect_graph_index, search_graph_index, read_indexed_doc
task_tags: graph, freshness, staleness
version: 2
enabled: true
description: Check graph freshness before trusting graph-backed evidence for time-sensitive questions.
keywords: graph freshness, stale graph, graph readiness
when_to_apply: Use when graph evidence may be outdated relative to document content.
avoid_when: Avoid strong graph-backed claims when freshness is uncertain.
examples: recent changes, stale graph concern
---
# Graph Freshness And Staleness Check

## Rule

Inspect freshness indicators and keep stale-graph risk explicit when it could materially affect the answer.
