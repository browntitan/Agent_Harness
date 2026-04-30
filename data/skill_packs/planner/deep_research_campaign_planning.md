---
name: Deep Research Campaign Planning
agent_scope: planner
tool_tags: rag_agent_tool, search_indexed_docs, read_indexed_doc, search_graph_index
task_tags: planning, campaign, documents, deep_research
version: 1
enabled: true
description: Plan deep research as staged corpus campaigns with shallow triage before expensive document review.
keywords: deep research, multi-hop, organize repository, corpus synthesis, triage notes, research notebook, defense repository
when_to_apply: Use for broad corpus research, repository organization, multi-hop evidence work, or synthesis across many files.
avoid_when: Avoid for focused factual questions that a single RAG lookup can answer.
examples: organize this repository of documents, synthesize across all files, defense program repository research
---
# Deep Research Campaign Planning

## Rule

For deep corpus-scale research, plan staged work in this order: clarify scope when needed, scan titles and paths, gather seed evidence, extract research facets, fan out facet searches, produce shallow document triage notes, deepen only relevant or partial documents, backfill thin evidence, and finalise with citations and explicit uncertainty.

Workers should write compact `research_triage_note` artifacts before full `doc_digest` artifacts. Triage should identify whether a document is worth deeper review, which topics it may cover, and what follow-up queries another worker should run.
