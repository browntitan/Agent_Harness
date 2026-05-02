---
name: Authority And Version Resolution
agent_scope: rag_researcher
tool_tags: plan_rag_queries, filter_indexed_docs, search_document_sections, validate_evidence_plan, rag_agent_tool
task_tags: authority, version, dates, current, drafts
version: 1
enabled: true
description: Resolve final/current/approved evidence against drafts, earlier plans, and superseded values.
keywords: current, latest, approved, final, draft, superseded, revision
when_to_apply: Use when documents may disagree by date, version, draft/final status, or authority.
avoid_when: Avoid silently preferring one value when the user needs reconciliation.
examples: latest approved date, draft versus final cost, authoritative source
---
# Authority And Version Resolution

## Rule

When the user asks for current, latest, approved, final, authoritative, changed, moved, or revised values, look for both:

- the earlier or draft value that explains what changed
- the final/current/approved source that establishes the answer

Do not cite only a draft unless the user asks for draft content. If two sources conflict, pass both documents into `rag_agent_tool` and ask it to synthesize the authority/version distinction.
