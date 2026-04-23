---
name: Utility Memory And Listing
agent_scope: utility
tool_tags: calculator, list_indexed_docs, memory_save, memory_load, memory_list
task_tags: utility, memory, listing
version: 2
enabled: true
description: Safe operating rules for calculations, document inventory, and durable memory lookups.
keywords: utility, memory, listing, calculator
when_to_apply: Use for quick operational tasks inside the utility agent lane.
avoid_when: Avoid turning inventory or memory lookups into broader grounded synthesis work.
examples: list docs, save preference, load memory key
---
# Utility Memory And Listing

## Rule

Use the narrowest utility tool that completes the request. Prefer exact lookup over inference, and keep the answer concise.
