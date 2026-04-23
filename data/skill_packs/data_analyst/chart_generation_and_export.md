---
name: Chart Generation And Export
agent_scope: data_analyst
tool_tags: load_dataset, inspect_columns, execute_code, return_file
task_tags: charts, export, analyst
version: 2
enabled: true
description: Generate charts in the sandbox and publish them as user-facing artifacts when needed.
keywords: charts, export, visualization
when_to_apply: Use when the user explicitly wants a chart or visual deliverable.
avoid_when: Avoid creating charts when a text summary is the requested output.
examples: bar chart, trend chart, exported workbook
---
# Chart Generation And Export

## Workflow

Inspect the relevant columns first, generate the chart in sandbox code, and return the resulting file or artifact explicitly.
