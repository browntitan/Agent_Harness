---
name: Document Structure Navigation
agent_scope: rag_researcher
tool_tags: inspect_document_structure, search_document_sections, read_indexed_doc, fetch_chunk_window
task_tags: sections, clauses, tables, sheets, structure
version: 1
enabled: true
description: Use document outlines, sections, clauses, sheets, and local windows to guide retrieval.
keywords: document outline, section search, clause search, table, spreadsheet, sheet
when_to_apply: Use when the task mentions sections, clauses, sheets, tables, dates, headings, exact values, or specific document areas.
avoid_when: Avoid structure-first reads when broad corpus discovery is still needed.
examples: approved date in workbook, clause extraction, table lookup
---
# Document Structure Navigation

## Workflow

After candidate documents are known, inspect structure before searching deeply when location matters.

Use `inspect_document_structure` to find available sections, clauses, sheets, or table-like chunks. Then use `search_document_sections` for targeted evidence. Use `fetch_chunk_window` when a hit needs surrounding context, especially for spreadsheets, adjacent table rows, or multi-sentence clauses.

Prefer this path for exact values, date changes, clause wording, sheet-specific facts, and workbook-backed answers.
