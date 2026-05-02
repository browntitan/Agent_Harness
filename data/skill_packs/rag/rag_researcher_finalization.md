---
name: RAG Researcher Finalization
agent_scope: rag_researcher
tool_tags: build_rag_controller_hints, validate_evidence_plan, rag_agent_tool
task_tags: finalization, citations, controller_hints, synthesis
version: 1
enabled: true
description: Package exploratory findings into a safe final rag_agent_tool call.
keywords: final synthesis, controller hints, citations, RAG contract
when_to_apply: Use before giving a final factual answer after exploratory RAG research.
avoid_when: Avoid answering directly from raw chunks or graph leads.
examples: final cited answer, scoped synthesis, negative evidence answer
---
# RAG Researcher Finalization

## Rule

Exploratory tools produce leads. `rag_agent_tool` produces the final citation-safe answer.

Before the final call:

- validate that selected evidence is not empty or obviously off-topic
- build controller hints from selected doc ids and selected evidence candidates
- choose `search_mode="deep"` for multi-document, authority-sensitive, negative-evidence, or relationship-heavy work
- pass `coverage_goal` and `result_mode` when they materially describe the task

Preserve warnings, weak-evidence notes, and scope caveats from the final RAG contract.
