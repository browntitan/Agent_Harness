from __future__ import annotations

from agentic_chatbot_next.utils.json_utils import extract_json


def test_extract_json_handles_fenced_json_with_leading_text() -> None:
    text = """
    Here is the plan:

    ```json
    {"plan": [{"tool": "load_dataset", "args": {"doc_id": "regional_spend.csv"}}], "notes": "demo"}
    ```
    """

    payload = extract_json(text)

    assert payload == {
        "plan": [{"tool": "load_dataset", "args": {"doc_id": "regional_spend.csv"}}],
        "notes": "demo",
    }


def test_extract_json_handles_json_followed_by_trailing_explanation() -> None:
    text = (
        '{"plan": [{"tool": "workspace_list", "args": {}, "purpose": "inspect files"}], "notes": "demo"}'
        "\nNext I will execute the plan."
    )

    payload = extract_json(text)

    assert payload == {
        "plan": [{"tool": "workspace_list", "args": {}, "purpose": "inspect files"}],
        "notes": "demo",
    }


def test_extract_json_strips_reasoning_tags_before_parsing() -> None:
    text = """
    <think>
    Need a strict JSON answer.
    </think>

    {"answer": "ok", "warnings": []}
    """

    payload = extract_json(text)

    assert payload == {
        "answer": "ok",
        "warnings": [],
    }
