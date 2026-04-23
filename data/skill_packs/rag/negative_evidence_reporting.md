---
name: Negative Evidence Reporting
agent_scope: rag
tool_tags: rag_agent_tool, search_indexed_docs, read_indexed_doc
task_tags: negative_evidence, insufficiency, reporting
version: 2
enabled: true
description: Report absence and weak evidence transparently instead of implying certainty from silence.
coverage_goal: exhaustive
controller_hints: {"prefer_negative_evidence_reporting": true}
keywords: negative evidence, not found, uncertainty
when_to_apply: Use when the search did not find enough support for a strong yes or no answer.
avoid_when: Avoid saying something is absent when the search breadth was obviously insufficient.
examples: no mention found, uncertain absence
---
# Negative Evidence Reporting

## Rule

Say what was searched, what was not found, and why the remaining uncertainty still matters.
