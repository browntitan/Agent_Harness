---
name: Result Format Selection
agent_scope: general
tool_tags: search_skills, rag_agent_tool
task_tags: output, formatting, synthesis
version: 1
enabled: true
description: Pick the smallest answer shape that still matches the user's requested deliverable.
keywords: result format, concise, inventory, summary
when_to_apply: Use when multiple answer shapes would technically fit.
avoid_when: Avoid when the output shape is already explicitly constrained.
examples: summary vs inventory, concise vs detailed synthesis
---
# Result Format Selection

## Rule

Default to a direct answer plus supporting detail. Switch to inventories, tables, or structured breakdowns only when the request or evidence shape truly calls for them.
