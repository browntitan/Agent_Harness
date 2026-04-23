---
name: Groupby And Summary Reporting
agent_scope: data_analyst
tool_tags: load_dataset, inspect_columns, execute_code
task_tags: grouping, aggregation, summary
version: 2
enabled: true
description: Perform grouped aggregations and report the resulting summary clearly.
keywords: groupby, aggregation, summary
when_to_apply: Use when the request asks for totals, averages, rankings, or grouped comparisons.
avoid_when: Avoid implicit grouping on dirty columns you have not inspected.
examples: revenue by region, top categories, grouped averages
---
# Groupby And Summary Reporting

## Rule

Inspect grouping columns first, aggregate in sandbox code, and present the summary with clear units and any grouping caveats.
