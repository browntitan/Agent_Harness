from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.rag.fanout import RagSearchTask


def _runtime_settings(tmp_path: Path, *, provider_name: str) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[1]
    return SimpleNamespace(
        runtime_dir=tmp_path / "runtime",
        workspace_dir=tmp_path / "workspaces",
        memory_dir=tmp_path / "memory",
        skills_dir=repo_root / "data" / "skills",
        agents_dir=repo_root / "data" / "agents",
        llm_provider=provider_name,
        judge_provider=provider_name,
        runtime_events_enabled=False,
        max_worker_concurrency=2,
        planner_max_tasks=4,
        rag_top_k_vector=2,
        rag_top_k_keyword=2,
        rag_max_retries=1,
        enable_coordinator_mode=False,
        agent_chat_model_overrides={},
        agent_judge_model_overrides={},
        default_tenant_id="tenant",
        default_user_id="user",
        default_conversation_id="conv",
    )


def _session_state() -> SessionState:
    return SessionState(tenant_id="tenant", user_id="user", conversation_id="conv")


def test_kernel_rag_runtime_bridge_runs_parallel_evidence_workers(monkeypatch, tmp_path: Path) -> None:
    kernel = RuntimeKernel(_runtime_settings(tmp_path, provider_name="openai"), providers=SimpleNamespace(), stores=SimpleNamespace())
    bridge = kernel.build_rag_runtime_bridge(_session_state())
    captured = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        del agent, session_state, user_text, callbacks
        worker_request = dict((task_payload or {}).get("worker_request") or {})
        captured.append(worker_request)
        payload = {
            "task_id": worker_request["task_id"],
            "evidence_entries": [],
            "candidate_docs": [],
            "graded_chunks": [],
            "warnings": [],
            "doc_focus": [{"doc_id": "doc-1", "title": "policy.md"}],
        }
        return SimpleNamespace(text=json.dumps(payload), messages=[], metadata={"rag_search_result": payload})

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    batch = bridge.run_search_tasks(
        [
            RagSearchTask(
                task_id="task_1",
                title="Search A",
                query="workflow",
                doc_scope=["doc-1"],
                research_profile="corpus_discovery",
                coverage_goal="corpus_wide",
                result_mode="inventory",
                controller_hints={"prefer_inventory_output": True},
            ),
            RagSearchTask(task_id="task_2", title="Search B", query="approval", doc_scope=["doc-2"]),
        ]
    )

    assert bridge.can_run_parallel(task_count=2) is True
    assert batch.parallel_workers_used is True
    assert len(batch.results) == 2
    assert {item["task_id"] for item in captured} == {"task_1", "task_2"}
    assert all(item["metadata"]["answer_mode"] == "evidence_only" for item in captured)
    assert any(item["doc_scope"] == ["doc-1"] for item in captured)
    assert any(item["research_profile"] == "corpus_discovery" for item in captured)
    assert any(item["controller_hints"] == {"prefer_inventory_output": True} for item in captured)


def test_kernel_rag_runtime_bridge_falls_back_to_serial_for_local_ollama(monkeypatch, tmp_path: Path) -> None:
    kernel = RuntimeKernel(_runtime_settings(tmp_path, provider_name="ollama"), providers=SimpleNamespace(), stores=SimpleNamespace())
    bridge = kernel.build_rag_runtime_bridge(_session_state())
    call_order = []

    def fake_run_agent(agent, session_state, *, user_text, callbacks, task_payload=None):
        del agent, session_state, user_text, callbacks
        worker_request = dict((task_payload or {}).get("worker_request") or {})
        call_order.append(worker_request["task_id"])
        payload = {
            "task_id": worker_request["task_id"],
            "evidence_entries": [],
            "candidate_docs": [],
            "graded_chunks": [],
            "warnings": [],
            "doc_focus": [],
        }
        return SimpleNamespace(text=json.dumps(payload), messages=[], metadata={"rag_search_result": payload})

    monkeypatch.setattr(kernel, "run_agent", fake_run_agent)

    batch = bridge.run_search_tasks(
        [
            RagSearchTask(task_id="task_1", title="Search A", query="workflow"),
            RagSearchTask(task_id="task_2", title="Search B", query="approval"),
        ]
    )

    assert bridge.can_run_parallel(task_count=2) is False
    assert batch.parallel_workers_used is False
    assert call_order == ["task_1", "task_2"]
