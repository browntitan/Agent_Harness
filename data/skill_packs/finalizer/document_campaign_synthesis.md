---
name: Document Campaign Synthesis
agent_scope: finalizer
tool_tags: rag_agent_tool, compare_indexed_docs
task_tags: synthesis, inventory, campaign
version: 2
enabled: true
description: Preserve document-level provenance when synthesizing campaign-style research.
keywords: document inventory, campaign synthesis, provenance
when_to_apply: Use when multiple documents were reviewed separately and the user wants a merged result.
avoid_when: Avoid flattening per-document evidence into one generic paragraph when the document breakdown matters.
examples: relevant document list, subsystem summary by file
---
# Document Campaign Synthesis

## Rule

Preserve the per-document structure when it is part of the requested output or when collapsing it would hide relevance, coverage, or disagreement across files.
