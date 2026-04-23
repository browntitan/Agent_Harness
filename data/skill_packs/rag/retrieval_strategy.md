---
name: Retrieval Strategy
agent_scope: rag
tool_tags: rag_agent_tool, search_indexed_docs, read_indexed_doc
task_tags: retrieval, strategy, routing
version: 2
enabled: true
description: Choose the right retrieval posture for the shape of the question and escalate when the first pass is thin.
keywords: retrieval strategy, routing, staged retrieval
when_to_apply: Use on most grounded questions to pick the right first evidence path.
avoid_when: Avoid repeating the same weak retrieval lane when the first pass obviously missed.
examples: exact file question, broad concept question
---
# Retrieval Strategy

## Rule

Use direct reads for exact sections, candidate search for named-file discovery, and staged RAG for broader grounded synthesis across multiple documents.
