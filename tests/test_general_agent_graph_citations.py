from __future__ import annotations

import json

from agentic_chatbot_next.general_agent import _append_missing_graph_citations


def test_graph_manager_appends_clickable_citation_block_when_model_omits_it() -> None:
    tool_results = [
        {
            "tool": "search_graph_index",
            "output": json.dumps(
                {
                    "results": [
                        {
                            "doc_id": "DOC-1",
                            "title": "Asterion Planning Draft",
                            "summary": "North Coast delivery affects Asterion schedule.",
                            "citation_ids": ["DOC-1#graph"],
                        }
                    ],
                    "citations": [
                        {
                            "citation_id": "DOC-1#graph",
                            "doc_id": "DOC-1",
                            "title": "Asterion Planning Draft",
                            "url": "/v1/documents/DOC-1/source?conversation_id=conv",
                            "source_path": "/kb/asterion.md",
                        }
                    ],
                }
            ),
        }
    ]

    rendered = _append_missing_graph_citations(
        "North Coast is a critical Asterion supplier.",
        tool_results,
    )

    assert "Citations:" in rendered
    assert "- [DOC-1#graph] [Asterion Planning Draft](/v1/documents/DOC-1/source?conversation_id=conv)" in rendered
