---
name: Empty Result Recovery
agent_scope: rag
tool_tags: search_indexed_docs, rag_agent_tool, read_indexed_doc
task_tags: recovery, empty_results, retrieval
version: 2
enabled: true
description: Recover from weak or empty retrieval by changing scope, tool choice, or query shape rather than repeating the same failed search.
keywords: empty results, recovery, retrieval retry
when_to_apply: Use after empty or clearly irrelevant first-pass retrieval.
avoid_when: Avoid blind retries with the same query and same scope.
examples: no matches found, weak evidence pass
---
# Empty Result Recovery

## Workflow

Simplify the query, widen or narrow scope appropriately, or move to direct document reads when the first retrieval lane comes back thin.
