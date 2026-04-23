---
name: Document Campaign Planning
agent_scope: planner
tool_tags: spawn_worker, rag_agent_tool
task_tags: planning, campaign, documents
version: 2
enabled: true
description: Plan long-running document research as staged worker campaigns with explicit evidence handoffs.
keywords: campaign planning, documents, staged retrieval
when_to_apply: Use for corpus-wide discovery, deep comparison, or exhaustive grounded research.
avoid_when: Avoid for bounded one-step document questions.
examples: identify all relevant architecture docs, exhaustive policy audit
---
# Document Campaign Planning

## Rule

Break deep document work into scoped discovery, evidence, and synthesis stages. Keep worker briefs explicit about document scope, result mode, and handoff artifacts.
