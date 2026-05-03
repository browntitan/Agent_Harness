from __future__ import annotations

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agentic_chatbot_next.general_agent import _synthesize_tool_results
from agentic_chatbot_next.runtime.context_budget import ContextBudgetManager
from agentic_chatbot_next.runtime.context_compaction import ContextCompactionService


def _settings(**overrides):
    payload = {
        "context_budget_enabled": True,
        "context_tool_result_max_tokens": 128,
        "context_tool_results_total_tokens": 512,
        "context_microcompact_target_tokens": 256,
        "context_smart_compaction_enabled": True,
        "context_smart_compaction_llm_enabled": False,
        "context_smart_compaction_target_tokens": 512,
        "rerank_enabled": False,
        "tiktoken_enabled": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_graph_payload_compacts_to_relevant_evidence_under_budget() -> None:
    payload = {
        "graph_id": "program_graph",
        "evidence_status": "grounded_graph_evidence",
        "requires_source_read": False,
        "results": [
            {
                "doc_id": f"DOC-{index}",
                "title": f"Irrelevant result {index}",
                "summary": "background context " + ("x" * 1200),
            }
            for index in range(20)
        ]
        + [
            {
                "doc_id": "DOC-NEEDLE",
                "title": "Supplier relationship",
                "summary": "Nova supplier approval caused the launch risk and links to the mitigation owner.",
                "relationships": [{"source": "Nova", "target": "Launch risk", "type": "caused"}],
                "citation_ids": ["DOC-NEEDLE#chunk0001"],
            }
        ],
        "citations": [{"citation_id": "DOC-NEEDLE#chunk0001", "doc_id": "DOC-NEEDLE", "title": "Supplier relationship"}],
    }
    service = ContextCompactionService(_settings())

    result = service.compact_tool_content(
        query="Which supplier approval caused the launch risk?",
        tool_name="search_graph_index",
        content=json.dumps(payload),
        target_tokens=360,
    )

    rendered = json.dumps(result.to_budgeted_payload(budget_tokens=360), ensure_ascii=False)
    assert result.compacted_tokens <= 360
    assert result.source_resolution_status == "not_required"
    assert "DOC-NEEDLE" in rendered
    assert "Nova supplier approval caused the launch risk" in rendered
    assert "relationships" in rendered
    assert len(result.dropped_atoms) > 0


def test_graph_source_candidates_require_resolution_plan() -> None:
    payload = {
        "graph_id": "program_graph",
        "evidence_status": "source_candidates_only",
        "requires_source_read": True,
        "results": [
            {
                "doc_id": "DOC-1",
                "title": "Candidate source",
                "summary": "This source may contain the relationship evidence.",
            }
        ],
    }
    service = ContextCompactionService(_settings())

    result = service.compact_tool_content(
        query="Resolve the relationship.",
        tool_name="search_graph_index",
        content=json.dumps(payload),
        target_tokens=320,
    )

    assert result.source_resolution_status == "required"
    assert result.source_resolution_plan["next_step"] == "resolve_with_rag_or_document_read"
    assert result.source_resolution_plan["preferred_doc_ids"] == ["DOC-1"]


def test_budget_tool_content_uses_evidence_compaction_metadata() -> None:
    manager = ContextBudgetManager(_settings(context_tool_result_max_tokens=80))
    payload = {
        "graph_id": "program_graph",
        "results": [
            {
                "doc_id": "DOC-1",
                "title": "Relationship",
                "summary": "The graph result contains relationship evidence." + (" x" * 500),
            }
        ],
    }

    budgeted = manager.budget_tool_content(
        json.dumps(payload),
        tool_name="search_graph_index",
        max_tokens=80,
        full_result_ref="session:conv:tool_result:abc",
    )
    parsed = json.loads(budgeted)

    assert parsed["object"] == "budgeted_tool_result"
    assert parsed["tool_name"] == "search_graph_index"
    assert parsed["full_result_ref"] == "session:conv:tool_result:abc"
    assert parsed["context_compaction"]["selected_evidence_count"] >= 1
    assert parsed["context_compaction"]["source_resolution_status"] == "not_required"


class _CaptureLLM:
    model_name = "capture-llm"

    def __init__(self) -> None:
        self.calls = []

    def invoke(self, messages, config=None):  # noqa: ANN001
        del config
        self.calls.append(messages)
        return AIMessage(content="The compacted evidence identifies DOC-NEEDLE.")


def test_synthesis_receives_compacted_tool_results_not_raw_payload() -> None:
    payload = {
        "graph_id": "program_graph",
        "results": [
            {
                "doc_id": "DOC-NEEDLE",
                "title": "Relevant relationship",
                "summary": "Needle evidence answers the supplier launch risk question.",
            },
            *[
                {
                    "doc_id": f"DOC-DROP-{index}",
                    "title": "Dropped distractor",
                    "summary": "RAW_DROP_SENTINEL " + ("x" * 5000),
                }
                for index in range(8)
            ],
        ],
    }
    llm = _CaptureLLM()
    manager = ContextBudgetManager(_settings(context_smart_compaction_target_tokens=512))

    answer, metadata = _synthesize_tool_results(
        llm,
        user_text="Answer the supplier launch risk question.",
        tool_results=[{"tool": "search_graph_index", "args": {"graph_id": "program_graph"}, "output": json.dumps(payload)}],
        callbacks=[],
        system_prompt="",
        recovery_reason="test",
        context_budget_manager=manager,
    )

    final_prompt = str(llm.calls[-1][-1].content)
    assert answer == "The compacted evidence identifies DOC-NEEDLE."
    assert "compacted_tool_results" in final_prompt
    assert "Needle evidence answers" in final_prompt
    assert "RAW_DROP_SENTINEL" not in final_prompt
    assert metadata["selected_evidence_count"] >= 1
    assert metadata["dropped_evidence_count"] >= 1


class _BrokenLLM:
    def invoke(self, messages, config=None):  # noqa: ANN001
        del messages, config
        raise RuntimeError("packing failed")


def test_llm_pack_failure_falls_back_to_semantic_mmr() -> None:
    payload = {
        "answer": "The structured answer should survive.",
        "citations": [{"doc_id": "DOC-1", "summary": "supporting citation"}],
    }
    service = ContextCompactionService(_settings(context_smart_compaction_llm_enabled=True))

    result = service.compact_tool_content(
        query="What should survive?",
        tool_name="rag_agent_tool",
        content=json.dumps(payload),
        target_tokens=400,
        llm=_BrokenLLM(),
        enable_llm=True,
    )

    assert any(item.get("method") == "llm_pack_failed" for item in result.method_trace)
    assert any("structured answer should survive" in atom.text for atom in result.selected_atoms)
