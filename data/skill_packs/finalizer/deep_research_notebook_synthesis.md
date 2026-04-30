---
name: Deep Research Notebook Synthesis
agent_scope: finalizer
tool_tags: rag_agent_tool, compare_indexed_docs
task_tags: synthesis, inventory, campaign, deep_research
version: 1
enabled: true
description: Synthesize deep research notebooks while preserving triage decisions, citations, caveats, and negative evidence.
keywords: research notebook, triage notes, deep research synthesis, citations, negative evidence
when_to_apply: Use when coordinator artifacts include research_triage_note, research_notebook, facet matches, doc digests, or coverage ledgers.
avoid_when: Avoid for single-document answers where notebook-level aggregation would add noise.
examples: cited synthesis across all files, organize repository report, deep research final answer
---
# Deep Research Notebook Synthesis

## Rule

Read the `research_notebook` first, then use citation-bearing artifacts such as `doc_digest`, `facet_matches`, `subsystem_evidence`, and `research_coverage_ledger` to support the answer. Preserve caveats, unresolved questions, and negative evidence instead of smoothing them away.

If the notebook says evidence is thin or missing, say so directly and avoid overclaiming.
