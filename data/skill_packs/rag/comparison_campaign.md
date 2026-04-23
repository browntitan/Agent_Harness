---
name: Comparison Campaign
agent_scope: rag
tool_tags: resolve_indexed_docs, compare_indexed_docs, read_indexed_doc
task_tags: comparison, diff, documents
version: 2
enabled: true
description: Compare multiple documents in a staged way that preserves evidence from each side.
retrieval_profile: comparison_campaign
coverage_goal: cross_document
result_mode: comparison
controller_hints: {"prefer_parallel_docs": true}
keywords: comparison campaign, document diff
when_to_apply: Use for document-vs-document or cross-document comparison requests.
avoid_when: Avoid flattening distinct document evidence into one generic answer too early.
examples: compare policy versions, contract differences
---
# Comparison Campaign

## Workflow

Resolve the target documents, compare them directly where possible, and keep side-specific evidence visible until final synthesis.
