---
name: Planner Task Decomposition
agent_scope: planner
tool_tags: spawn_worker, invoke_agent
task_tags: planning, decomposition, batching
version: 2
enabled: true
description: Decompose work into the fewest executable tasks that preserve correctness and handoff clarity.
keywords: task decomposition, batching, planning
when_to_apply: Use for any multi-step request that needs more than one worker or stage.
avoid_when: Avoid splitting simple tasks into artificial subproblems.
examples: compare docs then synthesize, inspect data then summarize
---
# Planner Task Decomposition

## Rule

Decompose only when sequencing, specialization, or verification demands it. Each task should be executable without rediscovering intent from the parent request.
