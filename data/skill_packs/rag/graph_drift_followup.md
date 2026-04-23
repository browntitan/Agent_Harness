---
name: Graph Drift Followup
agent_scope: rag
tool_tags: search_graph_index, rag_agent_tool, read_indexed_doc
task_tags: graph, drift, followup
version: 2
enabled: true
description: Follow up on graph-detected drift or change signals with text-grounded evidence.
keywords: graph drift, change over time, follow-up
when_to_apply: Use when graph results suggest change or movement that needs text confirmation.
avoid_when: Avoid stopping at graph-level hints without grounding back into documents.
examples: changed process, evolving obligations
---
# Graph Drift Followup

## Rule

Treat graph drift as a lead, not a final answer. Resolve back into grounded text before making concrete claims.
