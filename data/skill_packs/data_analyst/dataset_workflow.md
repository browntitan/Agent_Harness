---
name: Data Analyst Dataset Workflow
agent_scope: data_analyst
tool_tags: load_dataset, inspect_columns, execute_code, scratchpad_write
task_tags: workflow, analyst, pandas
version: 2
enabled: true
description: Standard plan-first workflow for tabular analysis in the sandboxed analyst environment.
keywords: workflow, inspect first, analyst
when_to_apply: Use for most non-trivial analyst requests.
avoid_when: Avoid jumping straight to code before understanding the dataset.
examples: load, inspect, plan, execute, verify
---
# Data Analyst Dataset Workflow

## Workflow

Load the dataset, inspect important columns, write the plan to the scratchpad, execute only the required code, and verify the result before answering.
