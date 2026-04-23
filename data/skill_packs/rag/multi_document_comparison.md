---
name: Multi Document Comparison
agent_scope: rag
tool_tags: resolve_indexed_docs, compare_indexed_docs, read_indexed_doc
task_tags: comparison, multi_document, analysis
version: 2
enabled: true
description: Compare more than one document while preserving document-specific evidence and differences.
coverage_goal: cross_document
result_mode: comparison
keywords: multi-document comparison, diff
when_to_apply: Use when the user wants similarities, differences, or structured comparison across files.
avoid_when: Avoid turning comparison into a single-document summary.
examples: compare versions, compare procedures
---
# Multi Document Comparison

## Rule

Keep evidence separated by document until the important differences are clear, then synthesize the comparison without losing side-specific support.
