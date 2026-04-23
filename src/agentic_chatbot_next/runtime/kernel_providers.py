from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Tuple

from agentic_chatbot_next.observability.events import RuntimeEvent
from agentic_chatbot_next.providers.circuit_breaker import (
    BreakerWrappedRunnable,
    CircuitBreakerPolicy,
    CircuitBreakerRegistry,
    unwrap_model,
)
from agentic_chatbot_next.providers.factory import AgentProviderResolver, ProviderBundle


class KernelProviderController:
    def __init__(self, settings: Any, base_providers: ProviderBundle | None, *, event_controller: Any | None = None) -> None:
        self.settings = settings
        self.base_providers = base_providers
        self.event_controller = event_controller
        self.agent_resolver = AgentProviderResolver(settings, base_providers)
        self._wrapped_cache: Dict[int, ProviderBundle] = {}
        self._breaker_registry = CircuitBreakerRegistry(
            CircuitBreakerPolicy(
                enabled=bool(getattr(settings, "llm_circuit_breaker_enabled", True)),
                window_size=int(getattr(settings, "llm_circuit_breaker_window_size", 20)),
                min_samples=int(getattr(settings, "llm_circuit_breaker_min_samples", 6)),
                error_rate_threshold=float(getattr(settings, "llm_circuit_breaker_error_rate_threshold", 0.50)),
                consecutive_failures=int(getattr(settings, "llm_circuit_breaker_consecutive_failures", 3)),
                open_seconds=int(getattr(settings, "llm_circuit_breaker_open_seconds", 30)),
            ),
            event_callback=self._emit_breaker_event,
        )

    def resolve_base_providers(self) -> ProviderBundle | None:
        return self._wrap_bundle(self.base_providers)

    def resolve_for_agent(self, agent_name: str, *, chat_max_output_tokens: int | None = None) -> ProviderBundle | None:
        raw = self.agent_resolver.for_agent(agent_name, chat_max_output_tokens=chat_max_output_tokens)
        return self._wrap_bundle(raw)

    def resolve_for_skill(
        self,
        agent_name: str,
        *,
        model_override: str = "",
        effort: str = "",
        chat_max_output_tokens: int | None = None,
    ) -> ProviderBundle | None:
        raw = self.agent_resolver.for_skill(
            agent_name,
            model_override=model_override,
            chat_max_output_tokens=chat_max_output_tokens,
        )
        raw = self._apply_reasoning_effort(raw, effort=effort)
        return self._wrap_bundle(raw)

    def is_bundle_role_open(self, bundle: ProviderBundle | None, role: str) -> bool:
        if bundle is None:
            return False
        model = getattr(bundle, role, None)
        key = str(getattr(model, "breaker_key", "") or "")
        if not key:
            return False
        return self._breaker_registry.is_open(key)

    def bundle_role_identity(self, bundle: ProviderBundle | None, role: str) -> Tuple[str, str, str]:
        if bundle is None:
            return ("", "", "")
        model = getattr(bundle, role, None)
        key = str(getattr(model, "breaker_key", "") or "")
        if key:
            return (role, key, self._model_runtime_name(model))
        provider_name = self._provider_name_for_role(role)
        model_name = self._extract_model_name(model, fallback_role=role)
        return (role, provider_name, model_name)

    def uses_local_ollama_workers(self) -> bool:
        provider_names = {
            self._normalize_provider_name(getattr(self.settings, attr, ""))
            for attr in ("llm_provider", "judge_provider")
        }
        provider_names.discard("")
        if "ollama" in provider_names:
            return True
        bundle = self.resolve_base_providers()
        if bundle is None:
            return False
        return any(
            self._model_runtime_name(getattr(bundle, attr, None)) == "ollama"
            for attr in ("chat", "judge")
        )

    def can_identify_non_ollama_worker_runtime(self) -> bool:
        provider_names = {
            self._normalize_provider_name(getattr(self.settings, attr, ""))
            for attr in ("llm_provider", "judge_provider")
        }
        provider_names.discard("")
        if provider_names:
            return True
        bundle = self.resolve_base_providers()
        if bundle is None:
            return False
        return any(
            bool(self._model_runtime_name(getattr(bundle, attr, None)))
            for attr in ("chat", "judge")
        )

    @staticmethod
    def _normalize_provider_name(value: Any) -> str:
        return str(value or "").strip().lower()

    @classmethod
    def _model_runtime_name(cls, model: Any) -> str:
        base = unwrap_model(model)
        if base is None:
            return ""
        module_name = cls._normalize_provider_name(getattr(base.__class__, "__module__", ""))
        class_name = cls._normalize_provider_name(getattr(base.__class__, "__name__", ""))
        runtime_label = f"{module_name} {class_name}"
        if "ollama" in runtime_label:
            return "ollama"
        if "azure" in runtime_label:
            return "azure"
        if "openai" in runtime_label:
            return "openai"
        return ""

    def _wrap_bundle(self, bundle: ProviderBundle | None) -> ProviderBundle | None:
        if bundle is None:
            return None
        cache_key = id(bundle)
        cached = self._wrapped_cache.get(cache_key)
        if cached is not None:
            return cached
        chat_model = getattr(bundle, "chat", None)
        judge_model = getattr(bundle, "judge", None)
        embeddings_model = getattr(bundle, "embeddings", None)
        wrapped = ProviderBundle(
            chat=self._wrap_model(chat_model, role="chat"),
            judge=self._wrap_model(judge_model, role="judge"),
            embeddings=embeddings_model,
        )
        self._wrapped_cache[cache_key] = wrapped
        return wrapped

    def _wrap_model(self, model: Any, *, role: str) -> Any:
        if model is None:
            return None
        if isinstance(model, BreakerWrappedRunnable):
            return model
        provider_name = self._provider_name_for_role(role)
        model_name = self._extract_model_name(model, fallback_role=role)
        breaker_key = f"{role}:{provider_name}:{model_name}"
        breaker = self._breaker_registry.get_or_create(
            key=breaker_key,
            provider_role=role,
            provider_name=provider_name,
            model_name=model_name,
        )
        return BreakerWrappedRunnable(model, breaker=breaker)

    def _apply_reasoning_effort(self, bundle: ProviderBundle | None, *, effort: str = "") -> ProviderBundle | None:
        clean_effort = str(effort or "").strip().lower()
        if bundle is None or clean_effort not in {"low", "medium", "high", "xhigh"}:
            return bundle
        chat_model = getattr(bundle, "chat", None)
        bind = getattr(chat_model, "bind", None)
        if not callable(bind):
            return bundle
        try:
            bound_chat = bind(reasoning_effort=clean_effort, reasoning={"effort": clean_effort})
        except Exception:
            return bundle
        return replace(bundle, chat=bound_chat)

    def _provider_name_for_role(self, role: str) -> str:
        if role == "chat":
            return self._normalize_provider_name(getattr(self.settings, "llm_provider", ""))
        if role == "judge":
            return self._normalize_provider_name(getattr(self.settings, "judge_provider", ""))
        return ""

    @classmethod
    def _extract_model_name(cls, model: Any, *, fallback_role: str) -> str:
        base = unwrap_model(model)
        for attr in ("model_name", "model", "azure_deployment", "deployment_name"):
            value = str(getattr(base, attr, "") or "").strip()
            if value:
                return value
        runtime_name = cls._model_runtime_name(base)
        return runtime_name or fallback_role

    def _emit_breaker_event(self, event_type: str, session_id: str, payload: Dict[str, Any]) -> None:
        if self.event_controller is None or not session_id:
            return
        self.event_controller.emit(
            event_type,
            session_id,
            agent_name="provider",
            payload=payload,
        )


__all__ = ["KernelProviderController"]
