from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from agentic_chatbot_next import cli
from agentic_chatbot_next.benchmark import public_suite as suite


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_public_fixture(root: Path) -> None:
    _write_jsonl(root / "beir" / "scifact" / "queries.jsonl", [{"_id": "q1", "text": "What supports claim one?"}])
    qrels = root / "beir" / "scifact" / "qrels"
    qrels.mkdir(parents=True, exist_ok=True)
    (qrels / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\tdoc-1\t1\n", encoding="utf-8")

    hotpot = root / "hotpotqa"
    hotpot.mkdir(parents=True, exist_ok=True)
    (hotpot / "dev.json").write_text(
        json.dumps(
            [
                {
                    "_id": "hp1",
                    "question": "Where was the author of Example Book born?",
                    "answer": "Paris",
                    "level": "medium",
                    "type": "bridge",
                    "supporting_facts": [["Example Book", 0], ["Example Author", 1]],
                }
            ]
        ),
        encoding="utf-8",
    )

    _write_jsonl(
        root / "ragbench" / "test.jsonl",
        [
            {
                "id": "rb1",
                "question": "What does the policy require?",
                "response": "The policy requires annual review.",
                "documents": [{"id": "policy.md"}],
                "faithfulness_label": True,
            }
        ],
    )
    _write_jsonl(
        root / "bfcl" / "test.jsonl",
        [
            {
                "id": "bfcl1",
                "question": "Book a flight to Boston.",
                "ground_truth": {"name": "book_flight", "arguments": {"destination": "Boston"}},
            }
        ],
    )
    _write_jsonl(
        root / "tau-bench" / "test.jsonl",
        [{"id": "tau1", "instruction": "Resolve the retail order issue.", "domain": "retail"}],
    )


def test_public_adapters_normalize_benchmark_rows(tmp_path: Path) -> None:
    _write_public_fixture(tmp_path)

    cases = suite.load_public_cases(
        benchmarks="beir:scifact,hotpotqa,ragbench,bfcl,tau-bench,gaia",
        data_root=tmp_path,
        profile="smoke",
        collection_prefix="public",
    )

    by_id = {(case.benchmark_id, case.test_id): case for case in cases}
    assert by_id[("beir:scifact", "q1")].collection_id == "public-beir-scifact"
    assert by_id[("beir:scifact", "q1")].gold_sources == ("doc-1",)
    assert by_id[("hotpotqa", "hp1")].task_type == "multi_hop_qa"
    assert by_id[("hotpotqa", "hp1")].gold_sources == ("Example Book", "Example Author")
    assert by_id[("ragbench", "rb1")].metadata["faithfulness_label"] is True
    assert by_id[("bfcl", "bfcl1")].metadata["expected_tool_name"] == "book_flight"
    assert by_id[("tau-bench", "tau1")].requires_capabilities == ("tool_environment_shim",)
    assert by_id[("gaia", "gaia-deferred")].requires_capabilities == ("browser_action_environment",)


def test_public_scoring_reports_answer_source_and_failure_categories() -> None:
    case = suite.PublicBenchmarkCase(
        benchmark_id="hotpotqa",
        test_id="hp1",
        task_type="multi_hop_qa",
        prompt="Question?",
        gold_answer="Paris",
        gold_sources=("source.md",),
        tags=("multi-hop",),
        scorer="answer_source",
        requires_capabilities=("rag",),
    )
    response_payload = {"metadata": {"rag_retrieval_summary": {"retrieved_sources": ["source.md"]}}}

    passed = suite.score_public_case(
        case,
        answer="Paris. Citations:\n- source.md",
        response_payload=response_payload,
    )
    failed = suite.score_public_case(
        case,
        answer="London. Citations:\n- other.md",
        response_payload={"metadata": {"rag_retrieval_summary": {"retrieved_sources": ["other.md"]}}},
    )

    assert passed.passed is True
    assert passed.answer_correct is True
    assert passed.source_hit is True
    assert passed.retrieval_recall_at_k == 1.0
    assert failed.passed is False
    assert "retrieval_miss" in failed.failure_categories
    assert "wrong_source" in failed.failure_categories
    assert "multi_hop_failure" in failed.failure_categories


def test_public_tool_call_scoring_matches_name_and_arguments() -> None:
    case = suite.PublicBenchmarkCase(
        benchmark_id="bfcl",
        test_id="bfcl1",
        task_type="tool_calling",
        prompt="Return a tool call",
        gold_answer='{"name": "book_flight", "arguments": {"destination": "Boston"}}',
        scorer="tool_call",
        requires_capabilities=("tool_calling",),
        metadata={"expected_tool_name": "book_flight", "expected_arguments": {"destination": "Boston"}},
    )

    result = suite.score_public_case(case, answer='{"name": "book_flight", "arguments": {"destination": "Boston"}}')

    assert result.passed is True
    assert result.tool_call_match is True
    assert result.tool_argument_match is True


def test_run_public_suite_with_mocked_gateway_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    _write_public_fixture(tmp_path)

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "Evidence supports it. Citations:\n- doc-1"}}],
                "metadata": {"rag_retrieval_summary": {"retrieved_sources": ["doc-1"]}},
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
            assert kwargs["json"]["metadata"]["kb_collection_id"] == "public-beir-scifact"
            assert kwargs["headers"]["X-Collection-ID"] == "public-beir-scifact"
            return FakeResponse()

    monkeypatch.setattr(suite.httpx, "Client", FakeClient)
    output_dir = tmp_path / "out"

    summary = suite.run_public_benchmark_suite(
        benchmarks="beir:scifact",
        profile="smoke",
        data_root=tmp_path,
        collection_prefix="public",
        api_base="http://testserver",
        model="enterprise-agent",
        token="",
        judge="off",
        output_dir=output_dir,
        limit=1,
        timeout_seconds=1,
        fail_fast=False,
        available_capabilities=("chat", "rag"),
    )

    assert summary.total == 1
    assert summary.passed == 1
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "results.md").exists()
    assert (output_dir / "failure_slices.json").exists()
    assert (output_dir / "raw" / "beir_scifact_q1.json").exists()


def test_cli_evaluate_public_benchmarks_passes_arguments(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    def fake_run_public_benchmark_suite(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"total": 0, "passed": 0})

    monkeypatch.setattr(suite, "run_public_benchmark_suite", fake_run_public_benchmark_suite)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "evaluate-public-benchmarks",
            "--profile",
            "smoke",
            "--benchmarks",
            "beir:scifact",
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--judge",
            "off",
        ],
    )

    assert result.exit_code == 0
    assert calls["benchmarks"] == "beir:scifact"
    assert calls["profile"] == "smoke"
    assert calls["data_root"] == tmp_path
