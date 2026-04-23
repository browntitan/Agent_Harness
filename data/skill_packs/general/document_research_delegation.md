---
name: Document Research Delegation
agent_scope: general
tool_tags: rag_agent_tool, search_indexed_docs, spawn_worker
task_tags: documents, research, delegation
version: 2
enabled: true
description: Route broad document-research campaigns into coordinator-led workflows instead of flattening them into one direct response.
keywords: corpus discovery, deep research, coordinator, documents
when_to_apply: Use when the request spans many documents, requires staged evidence gathering, or asks for exhaustive coverage.
avoid_when: Avoid for single-file reads or small grounded follow-ups.
examples: identify relevant design docs, compare all policy documents
---
# Document Research Delegation

## Rule

Use direct grounded retrieval for bounded document questions. For corpus-wide discovery, staged comparison, or deep evidence campaigns, hand the task to `coordinator` with a self-contained brief and any known document scope.

## Guardrails

- Keep the brief explicit about desired output shape.
- Preserve document ids or candidate titles when they are already known.
