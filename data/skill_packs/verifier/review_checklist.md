---
name: Verifier Review Checklist
agent_scope: verifier
tool_tags: rag_agent_tool, search_skills, read_indexed_doc
task_tags: verification, critique, review
version: 2
enabled: true
description: Review answers for missing caveats, unsupported claims, and incorrect confidence.
keywords: review, verifier, critique
when_to_apply: Use for any answer that materially depends on evidence, orchestration, or multi-step work.
avoid_when: Avoid revision requests based only on wording preference.
examples: grounded answer review, multi-worker synthesis review
---
# Verifier Review Checklist

## Checklist

- Are material claims supported?
- Were conflicts, failed tasks, or thin evidence omitted?
- Does confidence match the evidence and workflow?
