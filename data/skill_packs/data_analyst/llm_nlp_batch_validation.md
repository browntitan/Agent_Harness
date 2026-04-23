---
name: LLM NLP Batch Validation
agent_scope: data_analyst
tool_tags: load_dataset, run_nlp_column_task, inspect_columns
task_tags: nlp, validation, analyst
version: 2
enabled: true
description: Validate bounded NLP outputs by checking labels, coverage, and obvious failure modes.
keywords: nlp validation, batch review, analyst
when_to_apply: Use after row-level NLP tasks where label quality or coverage matters.
avoid_when: Avoid treating the first NLP pass as automatically correct.
examples: sentiment QA, category QA
---
# LLM NLP Batch Validation

## Rule

Review allowed labels, empty-text handling, and the returned distribution so you can flag likely misclassification or low-coverage outputs.
