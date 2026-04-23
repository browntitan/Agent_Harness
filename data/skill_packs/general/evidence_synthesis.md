---
name: Evidence Synthesis
agent_scope: general
tool_tags: rag_agent_tool, read_indexed_doc, compare_indexed_docs
task_tags: synthesis, evidence, transparency
version: 1
enabled: true
description: Turn evidence-backed tool results into a direct answer without losing caveats or provenance.
keywords: synthesis, evidence, uncertainty, provenance
when_to_apply: Use after retrieval or comparison work has produced enough evidence to answer.
avoid_when: Avoid when the task still needs more evidence gathering.
examples: grounded summary, compared docs, partial answer
---
# Evidence Synthesis

## Rule

Lead with the answer, keep the evidence trail intact, and surface thin support or conflicts instead of smoothing them away.
