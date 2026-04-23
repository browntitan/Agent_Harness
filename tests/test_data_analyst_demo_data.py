from __future__ import annotations

import csv
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_root() -> Path:
    return _repo_root() / "new_demo_notebook" / "demo_data" / "data_analyst"


def test_data_analyst_demo_data_pack_exists() -> None:
    root = _data_root()
    expected_files = {
        "README.md",
        "generate_data.py",
        "customer_reviews_100.csv",
        "customer_reviews_multisheet.xlsx",
        "sales_performance.xlsx",
        "marketing_leads.csv",
        "support_tickets.xlsx",
    }

    assert root.exists()
    assert expected_files.issubset({path.name for path in root.iterdir()})


def test_customer_reviews_csv_has_100_rows_with_blanks_and_duplicates() -> None:
    rows = list(
        csv.DictReader((_data_root() / "customer_reviews_100.csv").open(encoding="utf-8"))
    )

    review_values = [row["reviews"] for row in rows]
    non_blank_reviews = [value for value in review_values if value]

    assert len(rows) == 100
    assert "reviews" in rows[0]
    assert any(not value for value in review_values)
    assert len(non_blank_reviews) > len(set(non_blank_reviews))


def test_marketing_leads_csv_contains_missing_values_for_cleanup_workflows() -> None:
    rows = list(csv.DictReader((_data_root() / "marketing_leads.csv").open(encoding="utf-8")))

    assert len(rows) == 48
    assert any(not row["spend_usd"] for row in rows)
    assert any(not row["conversions"] for row in rows)


def test_customer_reviews_workbook_has_expected_sheets() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.load_workbook(_data_root() / "customer_reviews_multisheet.xlsx", read_only=True)

    assert workbook.sheetnames == ["raw_reviews", "metadata", "expected_sentiment_samples"]


def test_sales_and_support_workbooks_have_expected_structure() -> None:
    openpyxl = pytest.importorskip("openpyxl")

    sales = openpyxl.load_workbook(_data_root() / "sales_performance.xlsx", read_only=True)
    support = openpyxl.load_workbook(_data_root() / "support_tickets.xlsx", read_only=True)

    assert sales.sheetnames == ["sales_data", "targets", "rep_directory"]
    assert support.sheetnames == ["tickets", "team_roster", "qa_labels"]


def test_data_analyst_demo_readme_lists_sentiment_and_correlation_prompts() -> None:
    readme = (_data_root() / "README.md").read_text(encoding="utf-8")
    notebook_readme = (_repo_root() / "new_demo_notebook" / "README.md").read_text(encoding="utf-8")

    assert "sentiment analysis" in readme
    assert "correlation" in readme
    assert "new_demo_notebook/demo_data/data_analyst/README.md" in notebook_readme
