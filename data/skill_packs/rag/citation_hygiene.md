---
name: Citation Hygiene
agent_scope: rag
tool_tags: rag_agent_tool, read_indexed_doc
task_tags: citations, grounding, evidence
version: 2
enabled: true
description: Keep grounded answers tied to the retrieved evidence that actually supports each claim.
keywords: citations, grounding, evidence
when_to_apply: Use whenever the answer depends on retrieved document content.
avoid_when: Avoid introducing unsupported claims or uncited synthesis.
examples: grounded summary, policy explanation
---
# Citation Hygiene

## Rule

Only present document facts that are supported by retrieved evidence, and keep the answer aligned with the cited chunks rather than memory.
