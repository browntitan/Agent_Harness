---
name: Long Running Work
agent_scope: general
tool_tags: spawn_worker, list_jobs, stop_job
task_tags: background, jobs, orchestration
version: 1
enabled: true
description: Decide when background execution is the right shape and keep the user informed about delegated work.
keywords: background jobs, long running work, worker
when_to_apply: Use when the task is likely to exceed a normal direct turn or needs durable progress tracking.
avoid_when: Avoid for fast direct tasks that can finish synchronously.
examples: large document campaign, multi-stage comparison
---
# Long Running Work

## Rule

Move work into a durable job when duration, parallelism, or resumability matters. Tell the user what was launched and what outcome to expect.
