---
name: Finalizer Synthesis Rules
agent_scope: finalizer
tool_tags: rag_agent_tool
task_tags: synthesis, final_answer, transparency
version: 2
enabled: true
description: Compose the final answer from artifacts without dropping caveats, conflicts, or failed-task impacts.
keywords: synthesis, final answer, caveats
when_to_apply: Use for any final answer that merges multiple task outputs.
avoid_when: Avoid treating raw worker prose as authoritative when structured artifacts disagree.
examples: merged answer, revised answer after verification
---
# Finalizer Synthesis Rules

## Rule

Trust structured evidence over convenient prose, preserve material caveats, and keep conflicts visible when they change the meaning of the answer.
