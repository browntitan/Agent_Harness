from __future__ import annotations

from typing import Any

import pytest

from agentic_chatbot_next.providers.circuit_breaker import (
    BreakerWrappedRunnable,
    CircuitBreakerOpenError,
    CircuitBreakerPolicy,
    ProviderCircuitBreaker,
)
from agentic_chatbot_next.providers import circuit_breaker as breaker_module


def _make_breaker(
    *,
    policy: CircuitBreakerPolicy | None = None,
    event_callback=None,
) -> ProviderCircuitBreaker:
    return ProviderCircuitBreaker(
        key="chat:test:model",
        provider_role="chat",
        provider_name="test",
        model_name="model",
        policy=policy or CircuitBreakerPolicy(),
        event_callback=event_callback,
    )


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.structured_schema: Any | None = None

    def invoke(self, messages, config=None):
        self.calls.append({"messages": list(messages), "config": dict(config or {})})
        return "ok"

    def with_structured_output(self, schema):
        self.structured_schema = schema
        return self


def test_breaker_wrapped_runnable_preserves_invoke_and_structured_output() -> None:
    model = RecordingModel()
    wrapped = BreakerWrappedRunnable(model, breaker=_make_breaker())

    assert wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}}) == "ok"

    structured = wrapped.with_structured_output(dict)
    assert isinstance(structured, BreakerWrappedRunnable)
    assert structured.breaker_key == wrapped.breaker_key
    assert structured.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}}) == "ok"
    assert model.structured_schema is dict
    assert len(model.calls) == 2


def test_breaker_opens_after_repeated_provider_failures() -> None:
    class FailingModel:
        def invoke(self, messages, config=None):
            del messages, config
            raise TimeoutError("provider timeout")

    policy = CircuitBreakerPolicy(min_samples=10, consecutive_failures=2, open_seconds=30)
    wrapped = BreakerWrappedRunnable(FailingModel(), breaker=_make_breaker(policy=policy))

    with pytest.raises(TimeoutError):
        wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}})
    with pytest.raises(TimeoutError):
        wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}})
    with pytest.raises(CircuitBreakerOpenError):
        wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}})

    assert wrapped.breaker_snapshot().state == "open"


def test_breaker_does_not_trip_on_local_parsing_errors() -> None:
    class LocalFailureModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages, config=None):
            del messages, config
            self.calls += 1
            if self.calls == 1:
                raise ValueError("bad json")
            return "ok"

    wrapped = BreakerWrappedRunnable(
        LocalFailureModel(),
        breaker=_make_breaker(policy=CircuitBreakerPolicy(consecutive_failures=2, open_seconds=30)),
    )

    with pytest.raises(ValueError):
        wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}})

    snapshot = wrapped.breaker_snapshot()
    assert snapshot.state == "closed"
    assert snapshot.sample_count == 0
    assert snapshot.consecutive_failures == 0
    assert wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}}) == "ok"


def test_breaker_half_open_probe_closes_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(breaker_module.time, "monotonic", lambda: clock[0])

    class RecoveringModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages, config=None):
            del messages, config
            self.calls += 1
            if self.calls <= 2:
                raise TimeoutError("provider timeout")
            return "ok"

    breaker = _make_breaker(
        policy=CircuitBreakerPolicy(min_samples=10, consecutive_failures=2, open_seconds=5),
        event_callback=lambda event_type, session_id, payload: events.append((event_type, session_id)),
    )
    wrapped = BreakerWrappedRunnable(RecoveringModel(), breaker=breaker)

    with pytest.raises(TimeoutError):
        wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}})
    with pytest.raises(TimeoutError):
        wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}})
    with pytest.raises(CircuitBreakerOpenError):
        wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}})

    clock[0] = 10.0
    assert wrapped.invoke(["hello"], config={"metadata": {"session_id": "sess-1"}}) == "ok"
    assert wrapped.breaker_snapshot().state == "closed"
    assert [name for name, _ in events] == [
        "llm_circuit_breaker_opened",
        "llm_circuit_breaker_half_opened",
        "llm_circuit_breaker_closed",
    ]
