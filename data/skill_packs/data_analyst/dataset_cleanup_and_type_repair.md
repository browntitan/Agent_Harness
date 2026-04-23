---
name: Dataset Cleanup And Type Repair
agent_scope: data_analyst
tool_tags: load_dataset, inspect_columns, execute_code, return_file
task_tags: cleanup, dtype, analyst
version: 2
enabled: true
description: Clean nulls, normalize types, and publish a derived output rather than mutating the source in place.
keywords: cleanup, type repair, analyst
when_to_apply: Use when data quality problems block downstream analysis.
avoid_when: Avoid silent coercions that materially change meaning without reporting them.
examples: parse dates, fix numeric types, fill nulls
---
# Dataset Cleanup And Type Repair

## Rule

Explain the repair strategy, run it on a derived output file, and report any rows or fields that still need manual review.
