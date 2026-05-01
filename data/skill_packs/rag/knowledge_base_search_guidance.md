---
name: Knowledge Base Search Guidance
agent_scope: rag
tool_tags: list_indexed_docs, search_indexed_docs, resolve_indexed_docs, read_indexed_doc, rag_agent_tool
task_tags: kb, collection, retrieval, grounding, ambiguity
version: 1
enabled: true
description: Choose a fair, collection-aware search path without relying on prior knowledge of any corpus.
keywords: knowledge base search, scoped retrieval, collection choice, grounded answer, ambiguity
when_to_apply: Use when a user asks a KB-backed question, names a collection or file, or asks what indexed evidence is available.
avoid_when: Avoid assuming a collection or document contains an answer before retrieval confirms it.
examples: list available collections, search a named collection, resolve an indexed file, answer with citations
---
# Knowledge Base Search Guidance

## Workflow

Start from the user’s stated scope, then verify it through the available tools. Use `list_indexed_docs` for access or inventory questions, `search_indexed_docs` or `rag_agent_tool` for grounded lookup, and `resolve_indexed_docs` plus `read_indexed_doc` when the user names a file or document candidate.

If a collection, upload scope, graph, or document choice would materially change the answer, keep that ambiguity explicit instead of picking a hidden favorite. When evidence is thin or absent, report the searched scope and avoid implying the answer was known before retrieval.
