from __future__ import annotations

import json
from types import SimpleNamespace

from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.runtime.artifacts import list_handoff_artifacts, register_handoff_artifact
from agentic_chatbot_next.runtime.kernel_coordinator import KernelCoordinatorController
from agentic_chatbot_next.runtime.task_plan import TaskExecutionState, TaskResult


class _FakeKernel:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, dict]] = []
        self.settings = SimpleNamespace()

    def _emit(self, event_type: str, session_id: str, *, agent_name: str, payload: dict) -> None:
        self.events.append((event_type, session_id, agent_name, payload))


def _controller() -> KernelCoordinatorController:
    return KernelCoordinatorController(_FakeKernel())


def _session() -> SessionState:
    return SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")


def _triage_payload(doc_id: str, *, relevance: str, action: str = "deep_review") -> dict:
    return {
        "document": {"doc_id": doc_id, "title": f"{doc_id}.md", "source_path": f"docs/{doc_id}.md"},
        "summary": f"Shallow summary for {doc_id}.",
        "relevance": relevance,
        "relevance_rationale": f"{doc_id} is {relevance}.",
        "key_topics": ["architecture"],
        "potential_queries": [f"{doc_id} architecture"],
        "recommended_next_action": action,
        "used_citation_ids": [f"{doc_id}:1"],
    }


def test_research_triage_note_handoff_validates_required_fields() -> None:
    controller = _controller()
    session = _session()
    task = {
        "id": "task_triage_doc_1",
        "doc_scope": ["doc_1"],
        "produces_artifacts": ["research_triage_note"],
        "handoff_schema": "research_inventory",
    }
    result = TaskResult(
        task_id="task_triage_doc_1",
        title="Triage doc_1",
        executor="general",
        status="completed",
        output=json.dumps(_triage_payload("doc_1", relevance="relevant")),
        artifact_ref="task:task_triage_doc_1",
    )

    artifacts = controller._prepare_handoff_artifacts(session_state=session, task=task, result=result)

    assert len(artifacts) == 1
    assert artifacts[0]["artifact_type"] == "research_triage_note"
    assert "finalizer" in artifacts[0]["allowed_consumers"]
    assert "research_coordinator" in artifacts[0]["allowed_consumers"]
    assert artifacts[0]["data"]["recommended_next_action"] == "deep_review"


def test_malformed_research_triage_note_is_rejected() -> None:
    controller = _controller()
    session = _session()
    result = TaskResult(
        task_id="task_triage_bad",
        title="Triage bad",
        executor="general",
        status="completed",
        output=json.dumps({"document": {"doc_id": "doc_bad"}, "summary": ""}),
        artifact_ref="task:task_triage_bad",
    )

    artifacts = controller._prepare_handoff_artifacts(
        session_state=session,
        task={
            "id": "task_triage_bad",
            "produces_artifacts": ["research_triage_note"],
            "handoff_schema": "research_inventory",
        },
        result=result,
    )

    assert artifacts == []
    assert "Invalid handoff payload for artifact type 'research_triage_note'." in result.warnings


def test_triage_notes_drive_selective_doc_review_fanout() -> None:
    controller = _controller()
    session = _session()
    for doc_id, relevance, action in [
        ("doc_relevant", "relevant", "deep_review"),
        ("doc_partial", "partial", "backfill"),
        ("doc_irrelevant", "irrelevant", "skip"),
    ]:
        register_handoff_artifact(
            session,
            artifact_type="research_triage_note",
            handoff_schema="research_inventory",
            producer_task_id=f"task_triage_{doc_id}",
            producer_agent="general",
            data=_triage_payload(doc_id, relevance=relevance, action=action),
            allowed_consumers=["coordinator", "general", "finalizer"],
        )
    placeholder = {
        "id": "task_6",
        "title": "Expand document review",
        "handoff_schema": "research_inventory",
        "consumes_artifacts": ["research_triage_note"],
        "controller_hints": {"dynamic_doc_review_fanout": True, "max_parallel_doc_reviews": 6},
    }
    execution_state = TaskExecutionState(
        user_request="Synthesize across all repository files.",
        planner_summary="",
        task_plan=[placeholder],
    )

    tasks = controller._build_doc_review_tasks(
        session_state=session,
        placeholder_task=placeholder,
        execution_state=execution_state,
        handoff_artifacts=list_handoff_artifacts(session, artifact_types=["research_triage_note"]),
    )

    reviewed_doc_ids = {task.doc_scope[0] for task in tasks}
    assert reviewed_doc_ids == {"doc_relevant", "doc_partial"}


def test_research_notebook_aggregates_triage_negative_evidence_and_coverage() -> None:
    controller = _controller()
    session = _session()
    register_handoff_artifact(
        session,
        artifact_type="research_triage_note",
        handoff_schema="research_inventory",
        producer_task_id="task_5_doc_a",
        producer_agent="general",
        data=_triage_payload("doc_a", relevance="irrelevant", action="skip"),
        allowed_consumers=["coordinator", "finalizer"],
    )
    register_handoff_artifact(
        session,
        artifact_type="facet_matches",
        handoff_schema="research_inventory",
        producer_task_id="task_4_facet_a",
        producer_agent="rag_worker",
        data={"facet": "Interfaces", "documents": [], "rationale": "No hits.", "supporting_citation_ids": []},
        allowed_consumers=["coordinator", "finalizer"],
    )
    execution_state = TaskExecutionState(
        user_request="Organize this repository of documents.",
        planner_summary="",
        task_plan=[{"id": "task_4_facet_a"}, {"id": "task_5_doc_a"}, {"id": "task_8"}],
    )

    notebook = controller._register_research_notebook(
        session_state=session,
        execution_state=execution_state,
        coverage_ledger={
            "coverage_state": "thin",
            "primary_source_count": 1,
            "meta_source_count": 0,
            "facets": [{"name": "Interfaces"}],
            "warnings": ["Primary source coverage is thin."],
            "artifact_id": "handoff_ledger",
        },
        producer_agent="research_coordinator",
    )

    assert notebook["coverage_status"]["coverage_state"] == "thin"
    assert len(notebook["triage_notes"]) == 1
    assert {item["kind"] for item in notebook["negative_evidence"]} == {
        "irrelevant_triage",
        "empty_facet",
        "coverage_warning",
    }
    assert list_handoff_artifacts(session, artifact_types=["research_notebook"])
