---
name: Collection Scoping
agent_scope: rag
tool_tags: list_indexed_docs, search_indexed_docs, rag_agent_tool
task_tags: collection, scope, kb
version: 2
enabled: true
description: Keep retrieval inside the right KB collection or upload scope before searching deeply.
keywords: collection scope, kb scope, uploads
when_to_apply: Use when collection choice or retrieval scope could materially change the answer.
avoid_when: Avoid silently widening across collections when the request implies a narrower scope.
examples: default collection, uploaded docs only, one KB collection
---
# Collection Scoping

## Rule

Confirm or infer the correct retrieval scope before deep retrieval. If multiple KB collections are visible and the choice matters, keep the ambiguity explicit.
