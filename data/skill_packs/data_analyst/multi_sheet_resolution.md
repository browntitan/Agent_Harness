---
name: Multi Sheet Resolution
agent_scope: data_analyst
tool_tags: load_dataset, inspect_columns, execute_code
task_tags: workbook, excel, multi_sheet
version: 2
enabled: true
description: Resolve which workbook sheet matters and preserve sheet-specific assumptions when analyzing Excel files.
keywords: multi sheet, excel, workbook
when_to_apply: Use when the uploaded workbook has several sheets or the user did not specify one clearly.
avoid_when: Avoid assuming the first sheet is correct when another sheet is clearly named or required.
examples: workbook summary, sheet-specific analysis
---
# Multi Sheet Resolution

## Rule

Inspect the workbook structure, choose the relevant sheet explicitly, and keep that choice visible in the analysis or handoff.
