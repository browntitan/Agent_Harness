from __future__ import annotations

from agentic_chatbot_next.runtime.task_plan import TaskSpec, build_fallback_plan


def test_clause_redline_policy_workflow_fallback_plan_has_extract_fanout_verify_synthesis() -> None:
    query = (
        "Look through the document I uploaded and extract all clauses and associated redlines, "
        "then loop through each clause/redline and search against the internal policy guidance collection. "
        "Return a recommended action for our buyer to write back to the supplier."
    )

    plan = build_fallback_plan(
        query,
        session_metadata={
            "uploaded_doc_ids": ["UPLOAD_123"],
            "requested_kb_collection_id": "internal policy guidance",
        },
    )

    assert [task["title"] for task in plan] == [
        "Extract clauses and redlines",
        "Search internal policy guidance per clause",
        "Verify clause coverage",
        "Draft buyer recommendation table",
    ]
    assert plan[1]["loop_over_artifact"] == "clause_redline_inventory.clauses"
    assert plan[1]["capability_requirements"]["collections"] == ["internal policy guidance"]
    assert plan[2]["executor"] == "verifier"
    assert plan[3]["expected_artifacts"] == ["buyer_recommendation_table"]


def test_task_spec_round_trips_capability_metadata() -> None:
    task = TaskSpec.from_dict(
        {
            "id": "task_1",
            "title": "Search policy",
            "executor": "rag_worker",
            "capability_requirements": {"collections": ["policy"]},
            "evidence_scope": {"source": "knowledge_base"},
            "loop_over_artifact": "clauses",
            "parallelization_key": "clause_id",
            "acceptance_criteria": ["Every clause has evidence or no-evidence."],
            "permission_requirements": ["read_only"],
            "expected_artifacts": ["policy_guidance_matches"],
        }
    )

    payload = task.to_dict()

    assert payload["capability_requirements"] == {"collections": ["policy"]}
    assert payload["evidence_scope"] == {"source": "knowledge_base"}
    assert payload["loop_over_artifact"] == "clauses"
    assert payload["parallelization_key"] == "clause_id"
    assert payload["acceptance_criteria"] == ["Every clause has evidence or no-evidence."]
    assert payload["permission_requirements"] == ["read_only"]
    assert payload["expected_artifacts"] == ["policy_guidance_matches"]
