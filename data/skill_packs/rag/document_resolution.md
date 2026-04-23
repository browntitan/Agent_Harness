---
name: Document Resolution
agent_scope: rag
tool_tags: search_indexed_docs, resolve_indexed_docs, read_indexed_doc
task_tags: resolution, documents, filenames
version: 2
enabled: true
description: Resolve user-named files to exact indexed documents before reading or comparing them.
keywords: document resolution, exact file, doc id
when_to_apply: Use when the user names a document, path, or filename candidate.
avoid_when: Avoid proceeding with low-confidence document matches as if they were exact.
examples: exact file read, ambiguous filename
---
# Document Resolution

## Rule

Prefer exact resolution before deep reading. If several candidates are plausible, keep the ambiguity visible instead of guessing the intended file.
