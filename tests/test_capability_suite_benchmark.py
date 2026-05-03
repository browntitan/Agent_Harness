from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.benchmark import capability_suite as suite


def _write_suite(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "test_id",
        "difficulty",
        "capability_area",
        "query",
        "guided_query",
        "expected_answer",
        "expected_sources",
        "collection_id",
        "graph_id",
        "success_criteria",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_cases_filters_easy_and_guided_query_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "suite.csv"
    _write_suite(
        path,
        [
            {
                "test_id": "KB-E01",
                "difficulty": "Easy",
                "capability_area": "Default KB",
                "query": "Original question?",
                "guided_query": "Search the default KB: Original question?",
                "expected_answer": "Gold",
                "expected_sources": "source.md",
                "collection_id": "default",
                "graph_id": "",
                "success_criteria": "",
            },
            {
                "test_id": "KB-M01",
                "difficulty": "Medium",
                "capability_area": "Default KB",
                "query": "Medium question?",
                "guided_query": "",
                "expected_answer": "Gold",
                "expected_sources": "source.md",
                "collection_id": "default",
                "graph_id": "",
                "success_criteria": "",
            },
        ],
    )

    cases = suite.load_cases(path, difficulty="Easy")

    assert [case.test_id for case in cases] == ["KB-E01"]
    assert cases[0].prompt_for("guided_query") == "Search the default KB: Original question?"
    assert cases[0].prompt_for("query") == "Original question?"


def test_canonical_scope_mismatch_rows_are_aligned() -> None:
    with Path("rag_system_capability_test_suite.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["test_id"]: row for row in csv.DictReader(handle)}

    assert "at least two" in rows["KB-E07"]["expected_answer"]
    assert rows["REQ-H01"]["guided_query"]
    assert "all legal and control documents" in rows["REQ-H01"]["guided_query"]


def test_build_gateway_payload_sets_collection_and_graph_metadata() -> None:
    case = suite.CapabilityCase(
        test_id="GRAPH-E02",
        difficulty="Easy",
        capability_area="Graph inventory",
        query="Original",
        guided_query="Guided",
        expected_answer="Gold",
        expected_sources="graph_index_sources table",
        collection_id="defense-rag-v2",
        graph_id="defense_rag_v2_graph",
        success_criteria="",
    )

    payload = suite.build_gateway_payload(case, prompt="Guided", model="enterprise-agent", prompt_variant="guided_query")
    headers = suite.build_gateway_headers(case, run_id="run1", prompt_variant="guided_query", token="secret")

    assert payload["messages"][0]["content"] == "Guided"
    assert payload["metadata"]["upload_collection_id"] == suite.BENCHMARK_UPLOAD_COLLECTION_ID
    assert payload["metadata"]["kb_collection_confirmed"] is True
    assert payload["metadata"]["kb_collection_id"] == "defense-rag-v2"
    assert payload["metadata"]["requested_kb_collection_id"] == "defense-rag-v2"
    assert payload["metadata"]["selected_kb_collection_id"] == "defense-rag-v2"
    assert payload["metadata"]["available_kb_collection_ids"] == ["defense-rag-v2"]
    assert payload["metadata"]["search_collection_ids"] == ["defense-rag-v2"]
    assert payload["metadata"]["active_graph_ids"] == ["defense_rag_v2_graph"]
    assert headers["X-Collection-ID"] == suite.BENCHMARK_UPLOAD_COLLECTION_ID
    assert headers["Authorization"] == "Bearer secret"


def test_extract_rag_diagnostics_from_gateway_metadata() -> None:
    diagnostics = suite.extract_rag_diagnostics(
        {
            "metadata": {
                "retrieval_mode": "fast",
                "tool_calls_used": 2,
                "rag_retrieval_summary": {
                    "stage_timings_ms": {"vector_search": 10.5, "synthesis": 42.0},
                    "budget_exhausted": True,
                    "search_mode": "deep",
                },
            }
        }
    )

    assert diagnostics["retrieval_mode"] == "fast"
    assert diagnostics["tool_calls_used"] == 2
    assert diagnostics["budget_exhausted"] is True
    assert diagnostics["slowest_stage"] == "synthesis:42ms"
    assert "vector_search" in diagnostics["stage_timings_ms"]


def test_memory_rows_share_conversation_and_user_ids() -> None:
    mem_save = suite.CapabilityCase("MEM-E01", "Easy", "Memory", "q", "g", "gold", "none", "", "", "")
    mem_recall = suite.CapabilityCase("MEM-E02", "Easy", "Memory", "q", "g", "gold", "none", "", "", "")
    other = suite.CapabilityCase("KB-E01", "Easy", "KB", "q", "g", "gold", "none", "", "", "")

    assert suite.conversation_id_for(mem_save, run_id="r", prompt_variant="guided") == suite.conversation_id_for(
        mem_recall,
        run_id="r",
        prompt_variant="guided",
    )
    assert suite.user_id_for(mem_save, run_id="r", prompt_variant="guided") == suite.user_id_for(
        mem_recall,
        run_id="r",
        prompt_variant="guided",
    )
    assert suite.conversation_id_for(other, run_id="r", prompt_variant="guided") != suite.conversation_id_for(
        mem_save,
        run_id="r",
        prompt_variant="guided",
    )


def test_deterministic_scoring_checks_answer_sources_and_no_citations() -> None:
    answer = "The Pro plan is $49/user/month. Citations:\n- [c1] 02_pricing_and_plans.md"

    assert suite.deterministic_answer_match(answer, "$49/user/month for Pro.")
    assert suite.deterministic_source_match(answer, "02_pricing_and_plans.md")
    assert suite.deterministic_source_match("Hello there", "none")
    assert not suite.deterministic_source_match("Hello there\n\nCitations:\n- [c1] source.md", "none")


class _FakeJudge:
    def invoke(self, prompt, config=None):  # noqa: D401 - test double
        del prompt, config
        return SimpleNamespace(content='{"answer_correct": true, "source_correct": true, "reason": "matches"}')


def test_judge_score_parses_json() -> None:
    case = suite.CapabilityCase("KB-E01", "Easy", "KB", "q", "g", "gold", "source.md", "default", "", "criteria")

    score = suite.score_with_judge(_FakeJudge(), case, "answer")

    assert score.answer_correct is True
    assert score.source_correct is True
    assert score.reason == "matches"


def test_summary_compares_deterministic_and_judge_results() -> None:
    deterministic_only = suite.CapabilityResult(
        test_id="A",
        prompt_variant="guided_query",
        difficulty="Easy",
        capability_area="Area",
        prompt="q",
        expected_answer="gold",
        expected_sources="source.md",
        deterministic_answer_correct=True,
        deterministic_source_correct=True,
        passed=True,
    )
    judge_rescue = suite.CapabilityResult(
        test_id="B",
        prompt_variant="guided_query",
        difficulty="Easy",
        capability_area="Area",
        prompt="q",
        expected_answer="gold",
        expected_sources="source.md",
        deterministic_answer_correct=False,
        deterministic_source_correct=True,
        judge_answer_correct=True,
        judge_source_correct=True,
        passed=True,
    )
    judge_downgrade = suite.CapabilityResult(
        test_id="C",
        prompt_variant="query",
        difficulty="Easy",
        capability_area="Area",
        prompt="q",
        expected_answer="gold",
        expected_sources="source.md",
        deterministic_answer_correct=True,
        deterministic_source_correct=True,
        judge_answer_correct=False,
        judge_source_correct=True,
        passed=False,
    )

    summary = suite.summarize_results([deterministic_only, judge_rescue, judge_downgrade])

    assert summary.deterministic_passed == 2
    assert summary.judge_scored == 2
    assert summary.judge_passed == 1
    assert summary.judge_delta_passes == -1
    assert summary.judge_new_passes == ["guided_query:B"]
    assert summary.judge_new_failures == ["query:C"]
    assert summary.final_scorer == "mixed"
    assert summary.by_area["Area"]["judge_delta_passes"] == -1


def test_run_suite_with_mocked_gateway_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "suite.csv"
    _write_suite(
        csv_path,
        [
            {
                "test_id": "KB-E01",
                "difficulty": "Easy",
                "capability_area": "Default KB",
                "query": "What is the Pro price?",
                "guided_query": "Search the default knowledge base: What is the Pro price?",
                "expected_answer": "$49/user/month for Pro.",
                "expected_sources": "02_pricing_and_plans.md",
                "collection_id": "default",
                "graph_id": "",
                "success_criteria": "",
            }
        ],
    )

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "The Pro plan is $49/user/month. Citations:\n- [c1] 02_pricing_and_plans.md"
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            del args
            assert kwargs["json"]["metadata"]["upload_collection_id"] == suite.BENCHMARK_UPLOAD_COLLECTION_ID
            assert kwargs["json"]["metadata"]["kb_collection_id"] == "default"
            assert kwargs["json"]["metadata"]["search_collection_ids"] == ["default"]
            return FakeResponse()

    monkeypatch.setattr(suite.httpx, "Client", FakeClient)
    output_dir = tmp_path / "out"

    summary = suite.run_suite(
        suite_path=csv_path,
        difficulty="Easy",
        prompt_field="guided_query",
        compare_original=False,
        api_base="http://testserver",
        model="enterprise-agent",
        token="",
        judge="off",
        output_dir=output_dir,
        test_ids=[],
        limit=0,
        timeout_seconds=1,
        fail_fast=False,
    )

    assert summary.total == 1
    assert summary.passed == 1
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "raw" / "guided_query_KB-E01.json").exists()
