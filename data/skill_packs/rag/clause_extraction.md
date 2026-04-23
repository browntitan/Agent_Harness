---
name: Clause Extraction
agent_scope: rag
tool_tags: resolve_indexed_docs, read_indexed_doc, rag_agent_tool
task_tags: clauses, legal, structured_docs
version: 2
enabled: true
description: Read the relevant section directly when the user asks for a specific clause or numbered provision.
keywords: clause extraction, numbered section, legal text
when_to_apply: Use for exact clause or section questions.
avoid_when: Avoid broad semantic retrieval when the user already identified the exact provision.
examples: clause 5, section 2.1, payment obligations clause
---
# Clause Extraction

## Workflow

Resolve the document if needed, then read the relevant section directly and answer from the retrieved text instead of relying on broad semantic search alone.
