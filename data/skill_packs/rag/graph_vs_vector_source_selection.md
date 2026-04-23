---
name: Graph Vs Vector Source Selection
agent_scope: rag
tool_tags: explain_source_plan, search_graph_index, rag_agent_tool, read_indexed_doc
task_tags: graph, vector, routing, source_selection
version: 2
enabled: true
description: Choose the best retrieval lane among graph, staged RAG, and direct document reads.
keywords: source selection, graph vs vector, routing
when_to_apply: Use when the right retrieval lane is not obvious from the question alone.
avoid_when: Avoid graph-first behavior for exact wording, clause reads, or one-file questions.
examples: relationship question, file wording question
---
# Graph Vs Vector Source Selection

## Rule

Use graph for relationships and structure, staged RAG for broader grounded synthesis, and direct reads for exact sections or wording.
