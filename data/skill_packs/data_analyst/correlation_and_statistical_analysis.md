---
name: Correlation And Statistical Analysis
agent_scope: data_analyst
tool_tags: load_dataset, inspect_columns, execute_code, scratchpad_write
task_tags: statistics, correlation, analyst
version: 2
enabled: true
description: Run correlation and other statistical checks after validating the data types and null handling strategy.
keywords: correlation, statistics, analyst
when_to_apply: Use when the user asks for relationships, significance, or comparative metrics across columns.
avoid_when: Avoid when the task is simple descriptive reporting.
examples: correlation matrix, variance analysis
---
# Correlation And Statistical Analysis

## Rule

Validate the columns first, then run the statistical procedure in sandbox code and report the assumptions or caveats that affect interpretation.
