from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_chatbot_next.rag import reranker


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        rerank_enabled=True,
        rerank_provider="ollama",
        rerank_model=reranker.DEFAULT_RERANK_MODEL,
        rerank_top_n=2,
        rerank_timeout_seconds=3,
        rerank_fallback_to_heuristics=True,
        ollama_base_url="http://ollama:11434",
    )


def test_ollama_graph_reranker_reorders_candidates_and_attaches_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(**kwargs):
        assert kwargs["model"] == reranker.DEFAULT_RERANK_MODEL
        assert [candidate_id for candidate_id, _text in kwargs["candidates"]] == ["alpha", "beta"]
        return {"alpha": 0.2, "beta": 0.97}

    monkeypatch.setattr(reranker, "_post_ollama_rerank", fake_post)

    candidates = [
        {"candidate_id": "alpha", "title": "Alpha", "summary": "Generic background evidence.", "score": 0.9},
        {"candidate_id": "beta", "title": "Beta", "summary": "The direct answer evidence.", "score": 0.1},
        {"candidate_id": "tail", "title": "Tail", "summary": "Outside top_n.", "score": 0.0},
    ]

    reranked, decision = reranker.rerank_graph_candidates(_settings(), query="direct answer", candidates=candidates)

    assert decision["status"] == "reranked"
    assert reranked[0]["candidate_id"] == "beta"
    assert reranked[0]["rerank_score"] == 0.97
    assert reranked[0]["metadata"]["rerank"]["model"] == reranker.DEFAULT_RERANK_MODEL
    assert reranked[2]["candidate_id"] == "tail"


def test_ollama_graph_reranker_falls_back_to_input_order_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(**kwargs):
        del kwargs
        raise TimeoutError("model did not answer")

    monkeypatch.setattr(reranker, "_post_ollama_rerank", fake_post)

    candidates = [
        {"candidate_id": "alpha", "summary": "Alpha evidence."},
        {"candidate_id": "beta", "summary": "Beta evidence."},
    ]

    reranked, decision = reranker.rerank_graph_candidates(_settings(), query="beta", candidates=candidates)

    assert [item["candidate_id"] for item in reranked] == ["alpha", "beta"]
    assert decision["status"] == "fallback"
    assert "TimeoutError" in decision["error"]
