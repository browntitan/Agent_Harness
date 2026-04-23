---
name: Coverage Sufficiency Audit
agent_scope: rag
tool_tags: rag_agent_tool, read_indexed_doc, search_indexed_docs
task_tags: coverage, sufficiency, validation
version: 2
enabled: true
description: Check whether the retrieved evidence is broad and strong enough before claiming the answer is complete.
retrieval_profile: corpus_discovery
coverage_goal: corpus_wide
controller_hints: {"enforce_sufficiency_check": true}
keywords: sufficiency, coverage audit, completeness
when_to_apply: Use before strong completeness claims or corpus-wide conclusions.
avoid_when: Avoid treating one strong chunk as proof of exhaustive coverage.
examples: all matching docs, no evidence found
---
# Coverage Sufficiency Audit

## Rule

Before claiming completeness, check whether enough distinct documents or sections were actually covered to support the requested conclusion.
