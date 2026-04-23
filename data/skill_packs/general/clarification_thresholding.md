---
name: Clarification Thresholding
agent_scope: general
tool_tags: search_skills, list_indexed_docs, resolve_indexed_docs
task_tags: clarification, ambiguity, scope
version: 1
enabled: true
description: Ask only when ambiguity materially changes tools, evidence scope, or answer shape.
keywords: clarification, ambiguity, balanced autonomy
when_to_apply: Use when a request has multiple plausible interpretations.
avoid_when: Avoid when a reasonable assumption leads to the same execution path.
examples: ambiguous file name, unclear collection choice
---
# Clarification Thresholding

## Rule

Proceed when ambiguity is soft. Ask when scope, source selection, or deliverable format would materially change the work or make the answer misleading.
