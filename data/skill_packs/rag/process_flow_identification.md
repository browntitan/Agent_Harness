---
name: Process Flow Identification
agent_scope: rag
tool_tags: rag_agent_tool, read_indexed_doc, search_indexed_docs
task_tags: workflow, process, flow
version: 2
enabled: true
description: Identify stepwise processes, handoffs, or approval flows across grounded sources.
retrieval_profile: process_flow_identification
coverage_goal: corpus_wide
controller_hints: {"process_flow_bias": true}
keywords: process flow, workflow, handoff
when_to_apply: Use for workflow, process, handoff, or stepwise execution questions.
avoid_when: Avoid answering from one isolated chunk when the process may span several sections or documents.
examples: approval flow, escalation process
---
# Process Flow Identification

## Workflow

Gather the relevant steps, preserve sequence and handoffs, and keep thin or missing stages visible instead of inferring them.
