from __future__ import annotations

from agentic_chatbot_next.rag.discovery_precision import discovery_topic_label


def test_discovery_topic_label_skips_conversational_filler_for_inventory_like_queries() -> None:
    assert discovery_topic_label("can you list out all of the documents in the default collection") == "default"
