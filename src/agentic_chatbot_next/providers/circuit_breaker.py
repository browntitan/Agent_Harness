from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable


BreakerEventCallback = Callable[[str, str, Dict[str, Any]], None]


@dataclass(frozen=True)
class CircuitBreakerPolicy:
    enabled: bool = True
    window_size: int = 20
    min_samples: int = 6
    error_rate_threshold: float = 0.50
    consecutive_failures: int = 3
    open_seconds: int = 30


class CircuitBreakerOpenError(RuntimeError):
    def __init__(self, *, key: str, provider_role: str, provider_name: str, model_name: str) -> None:
        self.key = key
        self.provider_role = provider_role
        self.provider_name = provider_name
        self.model_name = model_name
        super().__init__(
            f"Circuit breaker is open for {provider_role} provider '{provider_name}' model '{model_name}'."
        )


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    key: str
    state: str
    failure_rate: float
    sample_count: int
    consecutive_failures: int
    provider_role: str
    provider_name: str
    model_name: str


class ProviderCircuitBreaker:
    def __init__(
        self,
        *,
        key: str,
        provider_role: str,
        provider_name: str,
        model_name: str,
        policy: CircuitBreakerPolicy,
        event_callback: BreakerEventCallback | None = None,
    ) -> None:
        self.key = key
        self.provider_role = provider_role
        self.provider_name = provider_name
        self.model_name = model_name
        self.policy = policy
        self.event_callback = event_callback
        self._state = "closed"
        self._opened_until = 0.0
        self._probe_in_flight = False
        self._recent_failures: Deque[bool] = deque(maxlen=max(1, int(policy.window_size)))
        self._consecutive_failures = 0
        self._lock = threading.Lock()

    def before_call(self, *, session_id: str = "") -> None:
        if not self.policy.enabled:
            return
        event_type = ""
        payload: Dict[str, Any] = {}
        transitioned_from_open = False
        with self._lock:
            now = time.monotonic()
            if self._state == "open":
                if now < self._opened_until:
                    raise CircuitBreakerOpenError(
                        key=self.key,
                        provider_role=self.provider_role,
                        provider_name=self.provider_name,
                        model_name=self.model_name,
                    )
                self._state = "half_open"
                self._probe_in_flight = True
                event_type = "llm_circuit_breaker_half_opened"
                payload = {"state": self._state}
                transitioned_from_open = True
            if self._state == "half_open" and self._probe_in_flight and not transitioned_from_open:
                raise CircuitBreakerOpenError(
                    key=self.key,
                    provider_role=self.provider_role,
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                )
            if self._state == "half_open":
                self._probe_in_flight = True
        if event_type:
            self._emit(event_type, session_id=session_id, extra=payload)

    def record_success(self, *, session_id: str = "") -> None:
        if not self.policy.enabled:
            return
        event_type = ""
        payload: Dict[str, Any] = {}
        with self._lock:
            previous = self._state
            self._recent_failures.append(False)
            self._consecutive_failures = 0
            self._probe_in_flight = False
            if previous in {"open", "half_open"}:
                self._state = "closed"
                self._opened_until = 0.0
                event_type = "llm_circuit_breaker_closed"
                payload = {"state": self._state}
        if event_type:
            self._emit(event_type, session_id=session_id, extra=payload)

    def record_provider_failure(self, *, session_id: str = "", error: BaseException | None = None) -> None:
        if not self.policy.enabled:
            return
        event_type = ""
        payload: Dict[str, Any] = {}
        with self._lock:
            self._recent_failures.append(True)
            self._consecutive_failures += 1
            self._probe_in_flight = False
            should_open = False
            reason = "error_rate"
            if self._state == "half_open":
                should_open = True
                reason = "half_open_probe_failed"
            elif self._consecutive_failures >= int(self.policy.consecutive_failures):
                should_open = True
                reason = "consecutive_failures"
            else:
                sample_count = len(self._recent_failures)
                if sample_count >= int(self.policy.min_samples):
                    failures = sum(1 for item in self._recent_failures if item)
                    failure_rate = failures / max(1, sample_count)
                    if failure_rate >= float(self.policy.error_rate_threshold):
                        should_open = True
                        reason = "error_rate"
            if should_open:
                self._state = "open"
                self._opened_until = time.monotonic() + max(1, int(self.policy.open_seconds))
                event_type = "llm_circuit_breaker_opened"
                payload = {
                    "state": self._state,
                    "reason": reason,
                    "error": str(error or "")[:1000],
                }
        if event_type:
            self._emit(event_type, session_id=session_id, extra=payload)

    def snapshot(self) -> CircuitBreakerSnapshot:
        with self._lock:
            sample_count = len(self._recent_failures)
            failures = sum(1 for item in self._recent_failures if item)
            return CircuitBreakerSnapshot(
                key=self.key,
                state=self._state,
                failure_rate=(failures / sample_count) if sample_count else 0.0,
                sample_count=sample_count,
                consecutive_failures=self._consecutive_failures,
                provider_role=self.provider_role,
                provider_name=self.provider_name,
                model_name=self.model_name,
            )

    def _emit(self, event_type: str, *, session_id: str, extra: Dict[str, Any]) -> None:
        if self.event_callback is None or not session_id:
            return
        snapshot = self.snapshot()
        payload = {
            "breaker_key": self.key,
            "provider_role": self.provider_role,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "failure_rate": snapshot.failure_rate,
            "sample_count": snapshot.sample_count,
            "consecutive_failures": snapshot.consecutive_failures,
            **dict(extra or {}),
        }
        self.event_callback(event_type, session_id, payload)


class CircuitBreakerRegistry:
    def __init__(self, policy: CircuitBreakerPolicy, *, event_callback: BreakerEventCallback | None = None) -> None:
        self.policy = policy
        self.event_callback = event_callback
        self._breakers: Dict[str, ProviderCircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        *,
        key: str,
        provider_role: str,
        provider_name: str,
        model_name: str,
    ) -> ProviderCircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(key)
            if breaker is None:
                breaker = ProviderCircuitBreaker(
                    key=key,
                    provider_role=provider_role,
                    provider_name=provider_name,
                    model_name=model_name,
                    policy=self.policy,
                    event_callback=self.event_callback,
                )
                self._breakers[key] = breaker
            return breaker

    def is_open(self, key: str) -> bool:
        breaker = self._breakers.get(key)
        if breaker is None:
            return False
        return breaker.snapshot().state == "open"


def _status_code_from_error(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    direct = getattr(error, "status_code", None)
    if isinstance(direct, int):
        return direct
    return None


def is_provider_availability_error(error: BaseException) -> bool:
    if isinstance(error, CircuitBreakerOpenError):
        return False
    class_name = str(error.__class__.__name__ or "").strip().lower()
    module_name = str(error.__class__.__module__ or "").strip().lower()
    status_code = _status_code_from_error(error)
    if status_code == 429:
        return True
    if isinstance(status_code, int) and status_code >= 500:
        return True
    availability_tokens = (
        "timeout",
        "timedout",
        "connection",
        "connect",
        "transport",
        "protocol",
        "network",
        "remoteprotocol",
        "ratelimit",
        "serviceunavailable",
        "internalserver",
    )
    if any(token in class_name for token in availability_tokens):
        return True
    if "httpx" in module_name and any(token in class_name for token in availability_tokens):
        return True
    if "openai" in module_name and any(token in class_name for token in availability_tokens):
        return True
    if isinstance(error, TimeoutError):
        return True
    return False


def _session_id_from_callbacks(callbacks: Iterable[Any]) -> str:
    for callback in callbacks:
        session_id = str(getattr(callback, "session_id", "") or "").strip()
        if session_id:
            return session_id
    return ""


def session_id_from_invoke_kwargs(kwargs: Dict[str, Any]) -> str:
    config = dict(kwargs.get("config") or {})
    metadata = dict(config.get("metadata") or {})
    session_id = str(metadata.get("session_id") or config.get("session_id") or "").strip()
    if session_id:
        return session_id
    callbacks = list(config.get("callbacks") or [])
    return _session_id_from_callbacks(callbacks)


class BreakerWrappedRunnable:
    def __init__(
        self,
        model: Any,
        *,
        breaker: ProviderCircuitBreaker,
    ) -> None:
        self._wrapped_model = model
        self._breaker = breaker

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        session_id = session_id_from_invoke_kwargs(kwargs)
        self._breaker.before_call(session_id=session_id)
        try:
            result = self._wrapped_model.invoke(*args, **kwargs)
        except Exception as exc:
            if is_provider_availability_error(exc):
                self._breaker.record_provider_failure(session_id=session_id, error=exc)
            raise
        self._breaker.record_success(session_id=session_id)
        return result

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        session_id = session_id_from_invoke_kwargs(kwargs)
        self._breaker.before_call(session_id=session_id)
        try:
            result = await self._wrapped_model.ainvoke(*args, **kwargs)
        except Exception as exc:
            if is_provider_availability_error(exc):
                self._breaker.record_provider_failure(session_id=session_id, error=exc)
            raise
        self._breaker.record_success(session_id=session_id)
        return result

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "BreakerWrappedRunnable":
        structured = self._wrapped_model.with_structured_output(*args, **kwargs)
        return BreakerWrappedRunnable(structured, breaker=self._breaker)

    def breaker_snapshot(self) -> CircuitBreakerSnapshot:
        return self._breaker.snapshot()

    @property
    def breaker_key(self) -> str:
        return self._breaker.key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_model, name)


def unwrap_model(model: Any) -> Any:
    current = model
    seen: set[int] = set()
    while hasattr(current, "_wrapped_model") and id(current) not in seen:
        seen.add(id(current))
        current = getattr(current, "_wrapped_model")
    return current


__all__ = [
    "BreakerWrappedRunnable",
    "CircuitBreakerOpenError",
    "CircuitBreakerPolicy",
    "CircuitBreakerRegistry",
    "CircuitBreakerSnapshot",
    "ProviderCircuitBreaker",
    "is_provider_availability_error",
    "session_id_from_invoke_kwargs",
    "unwrap_model",
]
