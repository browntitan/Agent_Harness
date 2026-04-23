from __future__ import annotations

import json
import socket
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


class OllamaBenchmarkError(RuntimeError):
    """Raised when the throughput benchmark cannot complete."""


@dataclass(frozen=True)
class OllamaThroughputRun:
    run: int
    prompt_tokens: int
    prompt_seconds: float
    prompt_tps: Optional[float]
    gen_tokens: int
    gen_seconds: float
    gen_tps: Optional[float]
    total_seconds: float
    end_to_end_tps: Optional[float]
    load_seconds: float
    wall_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": self.run,
            "prompt_tokens": self.prompt_tokens,
            "prompt_seconds": self.prompt_seconds,
            "prompt_tps": self.prompt_tps,
            "gen_tokens": self.gen_tokens,
            "gen_seconds": self.gen_seconds,
            "gen_tps": self.gen_tps,
            "total_seconds": self.total_seconds,
            "end_to_end_tps": self.end_to_end_tps,
            "load_seconds": self.load_seconds,
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True)
class OllamaModelThroughputReport:
    model: str
    base_url_requested: str
    base_url_used: str
    runs: List[OllamaThroughputRun]
    avg_prompt_tps: Optional[float]
    avg_gen_tps: Optional[float]
    avg_end_to_end_tps: Optional[float]
    stdev_gen_tps: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url_requested": self.base_url_requested,
            "base_url_used": self.base_url_used,
            "runs": [item.to_dict() for item in self.runs],
            "avg_prompt_tps": self.avg_prompt_tps,
            "avg_gen_tps": self.avg_gen_tps,
            "avg_end_to_end_tps": self.avg_end_to_end_tps,
            "stdev_gen_tps": self.stdev_gen_tps,
        }


@dataclass(frozen=True)
class OllamaBenchmarkFailure:
    model: str
    error: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "error": self.error,
        }


@dataclass(frozen=True)
class OllamaThroughputReport:
    benchmark_config: Dict[str, Any]
    models: List[OllamaModelThroughputReport]
    failures: List[OllamaBenchmarkFailure] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_config": dict(self.benchmark_config),
            "models": [item.to_dict() for item in self.models],
            "failures": [item.to_dict() for item in self.failures],
        }


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def _nanoseconds_to_seconds(raw: Any) -> float:
    value = float(raw or 0)
    return round(value / 1_000_000_000, 3)


def _safe_tps(token_count: int, duration_seconds: float) -> Optional[float]:
    if duration_seconds <= 0:
        return None
    return _round(token_count / duration_seconds)


def _stats(values: Iterable[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    usable = [float(item) for item in values if item is not None]
    if not usable:
        return None, None
    average = _round(statistics.mean(usable))
    stdev = _round(statistics.pstdev(usable)) if len(usable) > 1 else 0.0
    return average, stdev


def _replace_localhost_with_loopback(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.hostname != "localhost":
        return base_url
    host = "127.0.0.1"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def candidate_base_urls(base_url: str, *, localhost_fallback: bool = True) -> List[str]:
    candidates = [base_url.rstrip("/")]
    fallback = _replace_localhost_with_loopback(base_url).rstrip("/")
    if localhost_fallback and fallback not in candidates:
        candidates.append(fallback)
    return candidates


def _extract_error_message(exc: HTTPError) -> str:
    body = ""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    if body:
        try:
            payload = json.loads(body)
            if isinstance(payload, dict) and payload.get("error"):
                return str(payload["error"])
        except Exception:
            return body.strip() or str(exc)
    return str(exc)


def _post_json(base_url: str, path: str, payload: Dict[str, Any], *, timeout_seconds: int) -> Dict[str, Any]:
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise OllamaBenchmarkError(_extract_error_message(exc)) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise OllamaBenchmarkError("request timed out") from exc
    except URLError as exc:
        raise OllamaBenchmarkError(str(exc.reason or exc)) from exc
    try:
        payload_obj = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise OllamaBenchmarkError(f"Invalid JSON response from Ollama: {raw[:200]}") from exc
    if isinstance(payload_obj, dict) and payload_obj.get("error"):
        raise OllamaBenchmarkError(str(payload_obj["error"]))
    if not isinstance(payload_obj, dict):
        raise OllamaBenchmarkError("Unexpected Ollama response payload.")
    return payload_obj


def _build_prompt(*, context_words: int, num_predict: int) -> str:
    context = " ".join(f"ctx{i % 20}" for i in range(max(1, context_words)))
    return (
        "You are participating in a throughput benchmark. "
        "Ignore the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Task: Output only the word token separated by single spaces until you hit the generation limit of {num_predict} tokens. "
        "Do not add punctuation, explanations, headings, or reasoning.\n"
        "Answer:"
    )


def _build_run_metrics(*, run_number: int, response: Dict[str, Any], wall_seconds: float) -> OllamaThroughputRun:
    prompt_tokens = int(response.get("prompt_eval_count") or 0)
    prompt_seconds = _nanoseconds_to_seconds(response.get("prompt_eval_duration"))
    gen_tokens = int(response.get("eval_count") or 0)
    gen_seconds = _nanoseconds_to_seconds(response.get("eval_duration"))
    total_seconds = _nanoseconds_to_seconds(response.get("total_duration"))
    load_seconds = _nanoseconds_to_seconds(response.get("load_duration"))
    total_tokens = prompt_tokens + gen_tokens
    return OllamaThroughputRun(
        run=run_number,
        prompt_tokens=prompt_tokens,
        prompt_seconds=prompt_seconds,
        prompt_tps=_safe_tps(prompt_tokens, prompt_seconds),
        gen_tokens=gen_tokens,
        gen_seconds=gen_seconds,
        gen_tps=_safe_tps(gen_tokens, gen_seconds),
        total_seconds=total_seconds,
        end_to_end_tps=_safe_tps(total_tokens, total_seconds),
        load_seconds=load_seconds,
        wall_seconds=round(wall_seconds, 3),
    )


def _run_generate_request(
    *,
    base_url: str,
    model: str,
    prompt: str,
    num_predict: int,
    keep_alive: str,
    timeout_seconds: int,
    num_ctx: Optional[int] = None,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {"temperature": 0, "num_predict": num_predict}
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options,
    }
    started = time.time()
    response = _post_json(base_url, "/api/generate", payload, timeout_seconds=timeout_seconds)
    response["_wall_seconds"] = time.time() - started
    return response


def _run_generate_with_candidates(
    *,
    candidate_base_urls: List[str],
    selected_base_url: Optional[str],
    model: str,
    prompt: str,
    num_predict: int,
    keep_alive: str,
    timeout_seconds: int,
    num_ctx: Optional[int] = None,
) -> tuple[Dict[str, Any], str]:
    attempts: List[str] = []
    ordered_candidates = [item for item in [selected_base_url, *candidate_base_urls] if item]
    deduped_candidates = list(dict.fromkeys(ordered_candidates))
    for candidate in deduped_candidates:
        try:
            return (
                _run_generate_request(
                    base_url=candidate,
                    model=model,
                    prompt=prompt,
                    num_predict=num_predict,
                    keep_alive=keep_alive,
                    timeout_seconds=timeout_seconds,
                    num_ctx=num_ctx,
                ),
                candidate,
            )
        except OllamaBenchmarkError as exc:
            attempts.append(f"{candidate}: {exc}")
    detail = "; ".join(attempts) or "no candidate base URL succeeded"
    raise OllamaBenchmarkError(f"Unable to reach model '{model}': {detail}")


def benchmark_ollama_model(
    *,
    model: str,
    base_url: str,
    runs: int = 3,
    num_predict: int = 256,
    context_words: int = 2500,
    keep_alive: str = "10m",
    timeout_seconds: int = 900,
    warmup: bool = True,
    num_ctx: Optional[int] = None,
    localhost_fallback: bool = True,
) -> OllamaModelThroughputReport:
    prompt = _build_prompt(context_words=context_words, num_predict=num_predict)
    candidates = candidate_base_urls(base_url, localhost_fallback=localhost_fallback)
    selected_base_url: Optional[str] = None

    if warmup:
        _, selected_base_url = _run_generate_with_candidates(
            candidate_base_urls=candidates,
            selected_base_url=selected_base_url,
            model=model,
            prompt="Warm up the model. Reply with the single word warmup.",
            num_predict=8,
            keep_alive=keep_alive,
            timeout_seconds=timeout_seconds,
            num_ctx=num_ctx,
        )

    results: List[OllamaThroughputRun] = []
    for index in range(runs):
        response, selected_base_url = _run_generate_with_candidates(
            candidate_base_urls=candidates,
            selected_base_url=selected_base_url,
            model=model,
            prompt=prompt,
            num_predict=num_predict,
            keep_alive=keep_alive,
            timeout_seconds=timeout_seconds,
            num_ctx=num_ctx,
        )
        results.append(
            _build_run_metrics(
                run_number=index + 1,
                response=response,
                wall_seconds=float(response.get("_wall_seconds") or 0),
            )
        )

    avg_prompt_tps, _ = _stats(item.prompt_tps for item in results)
    avg_gen_tps, stdev_gen_tps = _stats(item.gen_tps for item in results)
    avg_end_to_end_tps, _ = _stats(item.end_to_end_tps for item in results)
    return OllamaModelThroughputReport(
        model=model,
        base_url_requested=base_url.rstrip("/"),
        base_url_used=selected_base_url,
        runs=results,
        avg_prompt_tps=avg_prompt_tps,
        avg_gen_tps=avg_gen_tps,
        avg_end_to_end_tps=avg_end_to_end_tps,
        stdev_gen_tps=stdev_gen_tps,
    )


def run_ollama_throughput_benchmark(
    *,
    models: List[str],
    base_url: str,
    runs: int = 3,
    num_predict: int = 256,
    context_words: int = 2500,
    keep_alive: str = "10m",
    timeout_seconds: int = 900,
    warmup: bool = True,
    num_ctx: Optional[int] = None,
    localhost_fallback: bool = True,
) -> OllamaThroughputReport:
    reports: List[OllamaModelThroughputReport] = []
    failures: List[OllamaBenchmarkFailure] = []
    for model in models:
        try:
            reports.append(
                benchmark_ollama_model(
                    model=model,
                    base_url=base_url,
                    runs=runs,
                    num_predict=num_predict,
                    context_words=context_words,
                    keep_alive=keep_alive,
                    timeout_seconds=timeout_seconds,
                    warmup=warmup,
                    num_ctx=num_ctx,
                    localhost_fallback=localhost_fallback,
                )
            )
        except OllamaBenchmarkError as exc:
            failures.append(OllamaBenchmarkFailure(model=model, error=str(exc)))

    if not reports:
        detail = "; ".join(f"{item.model}: {item.error}" for item in failures) or "no benchmark runs completed"
        raise OllamaBenchmarkError(f"Ollama throughput benchmark failed for all models: {detail}")

    return OllamaThroughputReport(
        benchmark_config={
            "base_url_requested": base_url.rstrip("/"),
            "runs": runs,
            "num_predict": num_predict,
            "context_words": context_words,
            "keep_alive": keep_alive,
            "timeout_seconds": timeout_seconds,
            "warmup": warmup,
            "num_ctx": num_ctx,
            "localhost_fallback": localhost_fallback,
        },
        models=reports,
        failures=failures,
    )
