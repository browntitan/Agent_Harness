---
name: Graph Local Relationship Tracing
agent_scope: rag
tool_tags: search_graph_index, read_indexed_doc
task_tags: graph, relationships, tracing
version: 2
enabled: true
description: Use local graph search for entity-level relationship tracing when the question is about a specific node or edge.
keywords: local graph, relationship tracing, entity lookup
when_to_apply: Use for specific relationships, attributes, or local graph neighborhoods.
avoid_when: Avoid global graph search when the request is narrowly entity-centric.
examples: who depends on X, what relates to Y
---
# Graph Local Relationship Tracing

## Rule

Use local graph search for focused entity questions, then ground any document-specific claims back into text when needed.
