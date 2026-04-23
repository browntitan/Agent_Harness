---
name: Corpus Coverage And Overclaim Check
agent_scope: verifier
tool_tags: rag_agent_tool, read_indexed_doc, search_indexed_docs
task_tags: verification, coverage, overclaim
version: 2
enabled: true
description: Catch claims of exhaustive coverage or absence when the evidence set does not support them.
keywords: overclaim, coverage, exhaustive
when_to_apply: Use when the answer implies all documents, no documents, or complete absence.
avoid_when: Avoid escalating minor phrasing differences that do not change the substance of the answer.
examples: all matching docs, no evidence found, exhaustive audit
---
# Corpus Coverage And Overclaim Check

## Rule

Require revision when the answer claims complete coverage, global absence, or exhaustive certainty without enough evidence breadth to support it.
