---
name: Evidence Grading And Pruning
agent_scope: rag_researcher
tool_tags: grade_evidence_candidates, prune_evidence_candidates, validate_evidence_plan, rag_agent_tool
task_tags: evidence, grading, pruning, sufficiency
version: 1
enabled: true
description: Grade and compact exploratory evidence before final citation-safe synthesis.
keywords: evidence grading, pruning, relevance, sufficiency, selected chunks
when_to_apply: Use after exploratory chunk, keyword, graph, or section searches return multiple candidate leads.
avoid_when: Avoid treating grades as final citation validation.
examples: noisy chunk results, multi-document evidence selection
---
# Evidence Grading And Pruning

## Rule

Grade candidates as planning evidence, not final proof.

Prefer a compact evidence set that includes:

- strongest direct support for the main claim
- at least one candidate from each necessary document or entity
- authority/version evidence when the question asks for current, final, latest, approved, or draft material
- conflict or obsolete/current contrast evidence when documents disagree

After grading, prune duplicates and over-representation from one document. Preserve meaningful conflicts and entity-disambiguation evidence even when it is not the highest-scoring semantic hit.

Final factual claims still need `rag_agent_tool`.
