from __future__ import annotations

import json
import socket
from urllib.error import HTTPError

from agentic_chatbot_next.benchmark import ollama_throughput


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _FakeHTTPError(HTTPError):
    def __init__(self, url: str, code: int, payload: dict[str, object]):
        super().__init__(url, code, "error", hdrs=None, fp=None)
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_candidate_base_urls_adds_loopback_fallback_for_localhost() -> None:
    assert ollama_throughput.candidate_base_urls("http://localhost:11434") == [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ]


def test_benchmark_ollama_model_falls_back_to_loopback_on_model_not_found(monkeypatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout=0):
        calls.append(request.full_url)
        if request.full_url.startswith("http://localhost:11434"):
            raise _FakeHTTPError(request.full_url, 404, {"error": "model 'demo:1' not found"})
        return _FakeHTTPResponse(
            {
                "prompt_eval_count": 1000,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 200,
                "eval_duration": 2_000_000_000,
                "total_duration": 3_200_000_000,
                "load_duration": 100_000_000,
                "response": "token " * 200,
                "done": True,
            }
        )

    monkeypatch.setattr(ollama_throughput, "urlopen", fake_urlopen)

    report = ollama_throughput.benchmark_ollama_model(
        model="demo:1",
        base_url="http://localhost:11434",
        runs=2,
        warmup=True,
        context_words=20,
        num_predict=32,
    )

    assert report.base_url_used == "http://127.0.0.1:11434"
    assert report.avg_gen_tps == 100.0
    assert [item.run for item in report.runs] == [1, 2]
    assert any(url.startswith("http://localhost:11434") for url in calls)
    assert any(url.startswith("http://127.0.0.1:11434") for url in calls)


def test_benchmark_ollama_model_falls_back_to_loopback_without_warmup(monkeypatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout=0):
        calls.append(request.full_url)
        if request.full_url.startswith("http://localhost:11434"):
            raise _FakeHTTPError(request.full_url, 404, {"error": "model 'demo:1' not found"})
        return _FakeHTTPResponse(
            {
                "prompt_eval_count": 200,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 50,
                "eval_duration": 1_000_000_000,
                "total_duration": 2_200_000_000,
                "load_duration": 50_000_000,
                "response": "token " * 50,
                "done": True,
            }
        )

    monkeypatch.setattr(ollama_throughput, "urlopen", fake_urlopen)

    report = ollama_throughput.benchmark_ollama_model(
        model="demo:1",
        base_url="http://localhost:11434",
        runs=1,
        warmup=False,
        context_words=20,
        num_predict=32,
    )

    assert report.base_url_used == "http://127.0.0.1:11434"
    assert report.avg_gen_tps == 50.0
    assert calls[0].startswith("http://localhost:11434")
    assert calls[1].startswith("http://127.0.0.1:11434")


def test_post_json_wraps_socket_timeout(monkeypatch) -> None:
    def fake_urlopen(request, timeout=0):
        del request, timeout
        raise socket.timeout("timed out")

    monkeypatch.setattr(ollama_throughput, "urlopen", fake_urlopen)

    try:
        ollama_throughput._post_json(  # noqa: SLF001
            "http://localhost:11434",
            "/api/generate",
            {"model": "demo:1"},
            timeout_seconds=3,
        )
    except ollama_throughput.OllamaBenchmarkError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("Expected OllamaBenchmarkError for socket timeout")


def test_run_ollama_throughput_benchmark_returns_partial_results(monkeypatch) -> None:
    def fake_benchmark_ollama_model(**kwargs):
        model = kwargs["model"]
        if model == "stuck:1":
            raise ollama_throughput.OllamaBenchmarkError("request timed out")
        return ollama_throughput.OllamaModelThroughputReport(
            model=model,
            base_url_requested="http://localhost:11434",
            base_url_used="http://127.0.0.1:11434",
            runs=[],
            avg_prompt_tps=100.0,
            avg_gen_tps=50.0,
            avg_end_to_end_tps=75.0,
            stdev_gen_tps=0.0,
        )

    monkeypatch.setattr(ollama_throughput, "benchmark_ollama_model", fake_benchmark_ollama_model)

    report = ollama_throughput.run_ollama_throughput_benchmark(
        models=["fast:1", "stuck:1", "fast:2"],
        base_url="http://localhost:11434",
        runs=1,
    )

    assert [item.model for item in report.models] == ["fast:1", "fast:2"]
    assert [item.model for item in report.failures] == ["stuck:1"]
    assert report.failures[0].error == "request timed out"
    assert report.to_dict()["failures"] == [
        {"model": "stuck:1", "error": "request timed out"},
    ]


def test_run_ollama_throughput_benchmark_raises_when_all_models_fail(monkeypatch) -> None:
    def fake_benchmark_ollama_model(**kwargs):
        raise ollama_throughput.OllamaBenchmarkError(f"Unable to reach model '{kwargs['model']}'")

    monkeypatch.setattr(ollama_throughput, "benchmark_ollama_model", fake_benchmark_ollama_model)

    try:
        ollama_throughput.run_ollama_throughput_benchmark(
            models=["fail:1", "fail:2"],
            base_url="http://localhost:11434",
            runs=1,
        )
    except ollama_throughput.OllamaBenchmarkError as exc:
        text = str(exc)
        assert "fail:1" in text
        assert "fail:2" in text
    else:
        raise AssertionError("Expected OllamaBenchmarkError when all models fail")
