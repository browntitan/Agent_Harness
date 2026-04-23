---
name: General Failure Recovery
agent_scope: general
tool_tags: search_skills, rag_agent_tool, search_indexed_docs, invoke_agent
task_tags: failure, retry, recovery
version: 1
enabled: true
description: Recover from thin tool outputs, failed calls, or brittle first attempts without looping blindly.
keywords: failure recovery, retry, thin evidence
when_to_apply: Use after a tool failure or low-confidence first pass.
avoid_when: Avoid repetitive retries that do not change the execution strategy.
examples: empty search, ambiguous resolution, failed worker
---
# General Failure Recovery

## Workflow

Change the evidence path, narrow the scope, or escalate appropriately. Do not repeat the same weak call without changing query, scope, or tool choice.
