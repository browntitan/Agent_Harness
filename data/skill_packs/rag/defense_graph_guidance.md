---
name: Defense Graph Guidance
agent_scope: rag
tool_tags: list_graph_indexes, inspect_graph_index, search_graph_index, explain_source_plan
task_tags: graph, defense, source_planning
version: 2
enabled: true
description: Use graph tooling deliberately when the defense corpus benefits from relationship or community structure.
keywords: defense graph, relationships, graph guidance
when_to_apply: Use when entity relationships or graph-backed structure is likely to help more than plain text retrieval.
avoid_when: Avoid graph-first behavior for exact wording or clause-level reading tasks.
examples: relationship graph, defense topic community
---
# Defense Graph Guidance

## Rule

Inspect graph readiness first, then use graph search only when the question is about relationships, communities, or multi-hop structure rather than exact wording.
