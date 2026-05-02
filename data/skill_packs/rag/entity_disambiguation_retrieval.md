---
name: Entity Disambiguation Retrieval
agent_scope: rag_researcher
tool_tags: plan_rag_queries, grep_corpus_chunks, search_corpus_chunks, grade_evidence_candidates, rag_agent_tool
task_tags: entities, disambiguation, suppliers, aliases
version: 1
enabled: true
description: Separate similar entity names, aliases, suppliers, programs, and relationship evidence before final synthesis.
keywords: entity disambiguation, similar names, suppliers, aliases, relationship
when_to_apply: Use when names are similar, entities may be confused, or a claim depends on who/which organization did what.
avoid_when: Avoid collapsing near-duplicate names into one entity without source confirmation.
examples: North Coast versus Northcoast, Halcyon Foundry versus Halcyon Microdevices
---
# Entity Disambiguation Retrieval

## Workflow

Search each entity name exactly, then search the surrounding relationship terms.

Keep separate evidence for:

- each entity name or alias
- the program/document each entity belongs to
- the action, issue, score, part, or relationship tied to that entity

If the question compares two similar names, preserve evidence from both sides and pass both source groups to `rag_agent_tool`.
