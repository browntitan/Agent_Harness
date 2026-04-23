---
name: LLM NLP Row Labeling
agent_scope: data_analyst
tool_tags: load_dataset, run_nlp_column_task, return_file
task_tags: nlp, labeling, analyst
version: 2
enabled: true
description: Use the bounded NLP tool for row-level sentiment or category labeling over one text column.
keywords: row labeling, nlp, sentiment, categorize
when_to_apply: Use when the user wants consistent labels added to a dataset column.
avoid_when: Avoid using sandbox code for ad hoc prompt construction when the bounded NLP tool already fits.
examples: sentiment labels, support ticket categories
---
# LLM NLP Row Labeling

## Rule

Choose the bounded NLP tool, define the labels or rules clearly, and return the derived file when the task creates labeled output columns.
Use the canonical task value `sentiment` even when the user phrases the request as "sentiment analysis."
For row-level labeling requests, default to both an in-chat summary and a returned derived file with appended output columns.
Reserve summary-only delivery for requests that explicitly ask for overall findings, distribution, or in-chat analysis without a transformed dataset.
