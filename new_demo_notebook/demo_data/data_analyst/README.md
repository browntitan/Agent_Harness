# Data Analyst Demo Data

This folder contains reusable CSV and Excel fixtures for the `data_analyst` agent.

## Files

- `customer_reviews_100.csv`: 100 review rows with duplicates, blank values, mixed sentiment, sarcasm, and ambiguous phrasing.
- `customer_reviews_multisheet.xlsx`: workbook version of the review set with `raw_reviews`, `metadata`, and `expected_sentiment_samples`.
- `sales_performance.xlsx`: multi-sheet sales workbook for correlation analysis, summary-tab generation, and chart export.
- `marketing_leads.csv`: lead-funnel dataset with missing values and cleanup opportunities.
- `support_tickets.xlsx`: mixed numeric and text support dataset for sentiment, categorization, SLA, and CSAT analysis.

## Suggested Prompts

- “Provide sentiment analysis of all reviews in the `reviews` column.”
- “Add `sentiment_label` and `sentiment_score` columns and return the file.”
- “Use this custom rubric: classify each review as promoter, neutral, or detractor.”
- “Create a new tab summarizing the correlation between `marketing_spend_usd` and `revenue_usd`.”
- “Generate a revenue-by-region chart and return the updated workbook.”
- “Clean missing values in the spend columns, summarize what changed, and return the file.”
- “Classify each support ticket message as positive, neutral, or negative and write the results back into a new column.”
- “Create an analyst summary workbook with a `SourceData` tab, a grouped summary tab, and one chart.”

## Regeneration

If you need to rebuild these fixtures, run:

```bash
python new_demo_notebook/demo_data/data_analyst/generate_data.py
```
