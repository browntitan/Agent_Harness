---
name: General Task Intake
agent_scope: general
tool_tags: search_skills, list_indexed_docs, rag_agent_tool
task_tags: intake, triage, scope
version: 1
enabled: true
description: Quickly classify the task so the agent chooses the right direct path, grounded path, or delegated path.
keywords: intake, classify task, scope, triage
when_to_apply: Use at the start of broad or ambiguous requests.
avoid_when: Avoid when the task is already clearly in one narrow lane.
examples: mixed request, unclear scope, multi-part ask
---
# General Task Intake

## Workflow

Classify the request as direct utility work, grounded document work, data analysis, or orchestration-heavy work. Choose the narrowest path that can finish the task safely without over-planning.
