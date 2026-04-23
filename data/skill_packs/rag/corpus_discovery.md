---
name: Corpus Discovery
agent_scope: rag
tool_tags: search_indexed_docs, rag_agent_tool, read_indexed_doc
task_tags: corpus, discovery, inventory
version: 2
enabled: true
description: Broad-first document discovery for identifying which files in the corpus match a request.
retrieval_profile: corpus_discovery
coverage_goal: corpus_wide
result_mode: inventory
controller_hints: {"force_deep_search": true, "prefer_parallel_docs": true, "prefer_inventory_output": true}
keywords: corpus discovery, inventory, matching documents
when_to_apply: Use when the user asks which documents or files are relevant across a corpus.
avoid_when: Avoid implying exhaustive coverage if only a narrow evidence pass was possible.
examples: which docs mention auth, list relevant runbooks
---
# Corpus Discovery

## Workflow

Start broad, keep document-level provenance, and return a per-document inventory when the request is about finding matching files rather than summarizing one known file.
