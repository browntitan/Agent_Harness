# Data Analyst Challenge Assets

Use these uploadable files with the `data_analyst` agent and the prompt suite in
`docs/AGENT_PATH_TEST_SUITE.md`.

## Files

- `regional_sales_messy.csv`: Q1 sales rows with currency strings, percent strings,
  mixed date formats, one duplicate invoice, one open pipeline row, one text-suffixed unit
  value, and one missing marketing-spend value.
- `customer_feedback_edge_cases.csv`: short customer comments for bounded NLP tests. It
  includes sarcasm, mixed sentiment, a blank comment, support praise, pricing complaints,
  reliability issues, and billing disputes.
- `ops_revenue_challenge.xlsx`: multi-sheet workbook with `orders`, `returns`,
  `region_targets`, `customer_accounts`, and `data_dictionary` sheets. It is designed for
  joins, return adjustments, target variance, margin checks, and segment-level discount
  leakage.
- `expected_metrics.json`: numeric answer key for workbook and messy-sales prompts.

## Ground Truth Checks

- Total net revenue after returns and retained restock fees: `160953.50`.
- Region target misses after returns: `SW`, `MW`, and `SE`.
- Region target variances:
  - `NE`: `+1762.00`
  - `SW`: `-2817.50`
  - `MW`: `-1769.50`
  - `SE`: `-221.50`
- Highest discount leakage by segment: `Mid-Market`, with `5145.00` in discount leakage
  and an `8.79%` discount rate.
- The messy CSV should detect `INV-010` as a duplicate invoice and exclude the `Open`
  `TBD` pipeline row from won-revenue summaries.

## Regenerate

From the repo root:

```bash
CODEX_NODE_MODULES=/path/to/bundled/node_modules node scripts/build_agent_path_test_assets.mjs
```
