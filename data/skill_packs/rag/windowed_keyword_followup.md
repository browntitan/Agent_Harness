---
name: Windowed Keyword Followup
agent_scope: rag
tool_tags: read_indexed_doc, rag_agent_tool, search_indexed_docs
task_tags: keyword, followup, evidence_windows
version: 2
enabled: true
description: Use targeted follow-up reads when a first pass suggests the right area but needs more local evidence.
controller_hints: {"prefer_windowed_followup": true}
keywords: keyword follow-up, evidence windows, local read
when_to_apply: Use when the first pass identified a promising section but the answer needs nearby context.
avoid_when: Avoid staying at snippet level when local surrounding text matters.
examples: follow up on a heading, expand around a relevant chunk
---
# Windowed Keyword Followup

## Rule

After a promising first hit, read the relevant local area directly so the final answer reflects the surrounding context, not just one isolated match.
