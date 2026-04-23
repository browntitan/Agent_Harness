---
name: Cross Document Inventory
agent_scope: rag
tool_tags: search_indexed_docs, rag_agent_tool, read_indexed_doc
task_tags: inventory, multi_document, synthesis
version: 2
enabled: true
description: Build a per-document inventory when several files contribute evidence to the same question.
coverage_goal: corpus_wide
result_mode: inventory
controller_hints: {"prefer_inventory_output": true}
keywords: cross-document inventory, per-document results
when_to_apply: Use when the user wants document-level matches or a breakdown by file.
avoid_when: Avoid flattening document-level results into a single paragraph.
examples: docs by subsystem, files mentioning incident response
---
# Cross Document Inventory

## Rule

Keep the output document-scoped so the user can see which files matched, why they matched, and where coverage is thin.
