from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document


def test_structured_evidence_arbitration_prefers_labeled_tabular_value() -> None:
    from agentic_chatbot_next.rag.engine import _apply_structured_evidence_arbitration

    structured_doc = Document(
        page_content=(
            "Tabular evidence for supplier workbook. Question: Which supplier caused the issue? "
            "Findings: Part: actuator housing | Supplier: Halcyon Foundry | "
            "Comment: Void indications and late castings. Source refs: sheet=Risks; row=7; cells=A7:D7"
        ),
        metadata={
            "doc_id": "workbook-doc",
            "chunk_id": "workbook-doc#tabular_01",
            "title": "supplier_status.xlsx",
            "source_type": "tabular_analysis",
            "chunk_type": "tabular_analysis",
            "is_synthetic_evidence": True,
            "sheet_name": "Risks",
            "cell_range": "A7:D7",
            "row_start": 7,
            "row_end": 7,
            "tabular_confidence": 0.92,
        },
    )
    distractor_doc = Document(
        page_content="A neighboring supplier note mentions Pelagic Switchgear in an unrelated status row.",
        metadata={"doc_id": "other-doc", "chunk_id": "other-doc#1", "title": "other.md"},
    )
    retrieval_run = SimpleNamespace(retrieval_verification={})

    corrected = _apply_structured_evidence_arbitration(
        {
            "answer": "The supplier was Pelagic Switchgear (other-doc#1).",
            "used_citation_ids": ["other-doc#1"],
            "warnings": [],
            "confidence_hint": 0.7,
        },
        query="Which supplier caused the issue?",
        selected_docs=[structured_doc, distractor_doc],
        retrieval_run=retrieval_run,
    )

    assert "Halcyon Foundry" in corrected["answer"]
    assert "Pelagic Switchgear" not in corrected["answer"]
    assert corrected["used_citation_ids"] == ["workbook-doc#tabular_01"]
    verification = retrieval_run.retrieval_verification["evidence_verification"]
    assert verification["action"] == "replaced_with_structured_extractive_answer"
    assert verification["conflicts"][0]["expected_value"] == "Halcyon Foundry"


def test_structured_evidence_arbitration_prefers_requested_value_over_context_note() -> None:
    from agentic_chatbot_next.rag.engine import _apply_structured_evidence_arbitration, _binding_evidence_candidates

    distractor = Document(
        page_content="Region context: West revenue commentary. Note: Use actual table rows for exact value. Owner: Analytics.",
        metadata={
            "chunk_id": "summary#1",
            "doc_id": "workbook-doc",
            "title": "sales.xlsx",
            "sheet_name": "Summary",
            "cell_range": "A1:C1",
            "is_synthetic_evidence": True,
            "tabular_confidence": 1.0,
        },
    )
    correct = Document(
        page_content="Region: West | revenue_usd: 613980 | marketing_spend_usd: 21000",
        metadata={
            "chunk_id": "data#5",
            "doc_id": "workbook-doc",
            "title": "sales.xlsx",
            "sheet_name": "Data",
            "cell_range": "A5:C5",
            "is_synthetic_evidence": True,
            "tabular_confidence": 1.0,
        },
    )
    query = "What is the revenue for the West region?"

    candidates = _binding_evidence_candidates(query, [distractor, correct])
    assert candidates[0]["citation_id"] == "data#5"

    retrieval_run = SimpleNamespace(retrieval_verification={})
    corrected = _apply_structured_evidence_arbitration(
        {
            "answer": "The revenue for West is unavailable in the provided evidence.",
            "used_citation_ids": [],
            "warnings": [],
            "confidence_hint": 0.4,
        },
        query=query,
        selected_docs=[distractor, correct],
        retrieval_run=retrieval_run,
    )

    assert "613980" in corrected["answer"]
    assert "West revenue commentary" not in corrected["answer"]
    assert corrected["used_citation_ids"] == ["data#5"]
    assert retrieval_run.retrieval_verification["evidence_verification"]["action"] == "replaced_with_structured_extractive_answer"


def test_structured_evidence_arbitration_regenerates_before_extractive_fallback() -> None:
    from agentic_chatbot_next.rag.engine import _apply_structured_evidence_arbitration

    structured_doc = Document(
        page_content="Supplier: Halcyon Foundry | Issue: void indications | Status: Open",
        metadata={
            "doc_id": "workbook-doc",
            "chunk_id": "workbook-doc#row_7",
            "title": "supplier_status.xlsx",
            "source_type": "tabular_analysis",
            "is_synthetic_evidence": True,
            "sheet_name": "Risks",
            "cell_range": "A7:C7",
            "tabular_confidence": 1.0,
        },
    )
    retrieval_run = SimpleNamespace(retrieval_verification={})

    def regenerate(candidate, missing_values):
        assert candidate["citation_id"] == "workbook-doc#row_7"
        assert missing_values[0]["value"] == "Halcyon Foundry"
        return {
            "answer": (
                "The supplier was Halcyon Foundry for the void indications issue, "
                "grounded in the structured row. (workbook-doc#row_7)"
            ),
            "used_citation_ids": ["workbook-doc#row_7"],
            "warnings": [],
            "confidence_hint": 0.8,
        }

    corrected = _apply_structured_evidence_arbitration(
        {
            "answer": "The supplier was another vendor.",
            "used_citation_ids": [],
            "warnings": [],
            "confidence_hint": 0.4,
        },
        query="Which supplier caused the issue?",
        selected_docs=[structured_doc],
        retrieval_run=retrieval_run,
        regenerate_answer=regenerate,
    )

    assert "Halcyon Foundry" in corrected["answer"]
    assert "STRUCTURED_EVIDENCE_VERIFIER_REGENERATED" in corrected["warnings"]
    assert retrieval_run.retrieval_verification["evidence_verification"]["action"] == "regenerated_with_verifier_feedback"
