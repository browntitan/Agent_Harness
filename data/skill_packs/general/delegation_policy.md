---
name: General Delegation Policy
agent_scope: general
tool_tags: spawn_worker, invoke_agent, search_skills
task_tags: delegation, orchestration, triage
version: 2
enabled: true
description: Decide when the general agent should finish the task directly versus delegate it.
keywords: delegate, direct execution, worker, coordinator
when_to_apply: Use when the next step could be completed either directly or through a worker.
avoid_when: Avoid when the task is already clearly single-step and directly executable.
examples: simple lookup, multi-step comparison, background research
---
# General Delegation Policy

## Rule

Stay direct when the task can be completed reliably in the current turn with available tools. Delegate when planning, durable execution, specialist work, or parallel coverage will materially improve the result.

## Guardrails

- Prefer `coordinator` for multi-stage or verification-heavy work.
- Prefer `invoke_agent` only for bounded same-session follow-up, not full orchestration.
