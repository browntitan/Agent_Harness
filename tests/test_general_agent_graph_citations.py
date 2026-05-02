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
        "North Coast is a critical Asterion supplier (DOC-1#graph).",
        tool_results,
    )

    assert "Citations:" in rendered
    assert "DOC-1#graph" not in rendered
    assert "([Asterion Planning Draft](/v1/documents/DOC-1/source?conversation_id=conv))" in rendered
    assert "- [Asterion Planning Draft](/v1/documents/DOC-1/source?conversation_id=conv)" in rendered


def test_graph_manager_humanizes_collection_upload_citation_ids() -> None:
    citation_id = "COLLECTION_UPLOAD_935ee5d6d9#chunk0005"
    tool_results = [
        {
            "tool": "search_graph_index",
            "output": json.dumps(
                {
                    "results": [
                        {
                            "doc_id": "DOC-ASTERION",
                            "title": "asterion_issue_digest_draft.txt",
                            "summary": "Cost commentary identifies the approved change and residual risk.",
                            "citation_ids": [citation_id],
                        }
                    ],
                    "citations": [
                        {
                            "citation_id": citation_id,
                            "doc_id": "DOC-ASTERION",
                            "title": "asterion_issue_digest_draft.txt",
                            "url": "http://localhost:18000/v1/documents/DOC-ASTERION/source?disposition=inline",
                            "source_path": "/uploads/asterion_issue_digest_draft.txt",
                            "collection_id": "defense-rag-test",
                        }
                    ],
                }
            ),
        }
    ]

    rendered = _append_missing_graph_citations(
        f"The approved change drove the cost delta ({citation_id}).",
        tool_results,
    )

    assert citation_id not in rendered
    assert "([asterion_issue_digest_draft.txt](http://localhost:18000/v1/documents/DOC-ASTERION/source?disposition=inline))" in rendered
    assert "- [asterion_issue_digest_draft.txt](http://localhost:18000/v1/documents/DOC-ASTERION/source?disposition=inline)" in rendered
