---
name: Workbook Mutation Copy On Write
agent_scope: data_analyst
tool_tags: load_dataset, execute_code, return_file
task_tags: workbook, mutation, safety
version: 2
enabled: true
description: Apply spreadsheet mutations to a derived workbook instead of overwriting the uploaded source file.
keywords: copy on write, workbook safety, mutation
when_to_apply: Use when the task edits workbook contents, formulas, or labels.
avoid_when: Avoid in-place edits that destroy the original user upload.
examples: add columns, fix workbook values, relabel sheet data
---
# Workbook Mutation Copy On Write

## Rule

Create a derived output workbook, apply the requested change there, and return the derived file explicitly.
