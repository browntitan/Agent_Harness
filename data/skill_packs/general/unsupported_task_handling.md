---
name: Unsupported Task Handling
agent_scope: general
tool_tags: search_skills
task_tags: unsupported, boundaries, fallback
version: 1
enabled: true
description: Respond usefully when the runtime cannot fully perform the requested task.
keywords: unsupported, limitation, fallback
when_to_apply: Use when the user asks for capabilities outside the current runtime surface.
avoid_when: Avoid when the task is actually supported through an available tool path.
examples: unsupported integration, unavailable external system
---
# Unsupported Task Handling

## Rule

State the missing capability clearly, provide the best partial help the runtime can give, and suggest the closest supported next step instead of pretending the task was completed.
