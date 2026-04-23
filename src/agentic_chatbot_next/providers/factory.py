from __future__ import annotations

import re
from typing import Any

from agentic_chatbot_next.providers import (
    ProviderBundle,
    build_embeddings,
    build_providers,
    validate_provider_configuration,
    validate_provider_dependencies,
)
from agentic_chatbot_next.providers.output_limits import (
    resolve_chat_output_cap,
    resolve_judge_output_cap,
)


def normalise_agent_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def resolve_agent_model_overrides(settings: Any, agent_name: str) -> tuple[str | None, str | None]:
    normalized = normalise_agent_name(agent_name)
    chat_overrides = dict(getattr(settings, "agent_chat_model_overrides", {}) or {})
    judge_overrides = dict(getattr(settings, "agent_judge_model_overrides", {}) or {})
    return chat_overrides.get(normalized), judge_overrides.get(normalized)


def _default_provider_model_name(settings: Any, *, provider_role: str) -> str:
    provider_name = str(getattr(settings, provider_role, "") or "").strip().lower()
    if provider_role == "llm_provider":
        if provider_name == "ollama":
            return str(getattr(settings, "ollama_chat_model", "") or "")
        if provider_name == "azure":
            return str(getattr(settings, "azure_openai_chat_deployment", "") or "")
        if provider_name == "nvidia":
            return str(getattr(settings, "nvidia_chat_model", "") or "")
        return ""

    if provider_name == "ollama":
        return str(getattr(settings, "ollama_judge_model", "") or "")
    if provider_name == "azure":
        return str(getattr(settings, "azure_openai_judge_deployment", "") or "")
    if provider_name == "nvidia":
        return str(getattr(settings, "nvidia_judge_model", "") or "")
    return ""


class AgentProviderResolver:
    def __init__(self, settings: Any, base_providers: ProviderBundle | None) -> None:
        self.settings = settings
        self.base_providers = base_providers
        self._cache: dict[tuple[str, str, str, str], ProviderBundle] = {}

    def for_agent(self, agent_name: str, *, chat_max_output_tokens: int | None = None) -> ProviderBundle | None:
        if self.base_providers is None:
            return None

        chat_override, judge_override = resolve_agent_model_overrides(self.settings, agent_name)
        default_chat = _default_provider_model_name(self.settings, provider_role="llm_provider")
        default_judge = _default_provider_model_name(self.settings, provider_role="judge_provider")
        effective_chat = chat_override or default_chat
        effective_judge = judge_override or default_judge
        default_chat_cap = resolve_chat_output_cap(self.settings)
        effective_chat_cap = resolve_chat_output_cap(
            self.settings,
            agent_name=agent_name,
            request_max_tokens=chat_max_output_tokens,
        )
        effective_judge_cap = resolve_judge_output_cap(self.settings)

        if (
            effective_chat == default_chat
            and effective_judge == default_judge
            and effective_chat_cap == default_chat_cap
        ):
            return self.base_providers

        key = (
            str(getattr(self.settings, "llm_provider", "") or "").strip().lower(),
            str(getattr(self.settings, "judge_provider", "") or "").strip().lower(),
            effective_chat,
            effective_judge,
            str(effective_chat_cap or ""),
            str(effective_judge_cap or ""),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        resolved = build_providers(
            self.settings,
            embeddings=self.base_providers.embeddings,
            chat_model_override=effective_chat,
            judge_model_override=effective_judge,
            chat_max_output_tokens=effective_chat_cap,
            judge_max_output_tokens=effective_judge_cap,
        )
        self._cache[key] = resolved
        return resolved

    def for_skill(
        self,
        agent_name: str,
        *,
        model_override: str = "",
        chat_max_output_tokens: int | None = None,
    ) -> ProviderBundle | None:
        clean_model = str(model_override or "").strip()
        if not clean_model or clean_model.lower() == "inherit":
            return self.for_agent(agent_name, chat_max_output_tokens=chat_max_output_tokens)
        if self.base_providers is None:
            return None

        _chat_override, judge_override = resolve_agent_model_overrides(self.settings, agent_name)
        default_judge = _default_provider_model_name(self.settings, provider_role="judge_provider")
        effective_judge = judge_override or default_judge
        effective_chat_cap = resolve_chat_output_cap(
            self.settings,
            agent_name=agent_name,
            request_max_tokens=chat_max_output_tokens,
        )
        effective_judge_cap = resolve_judge_output_cap(self.settings)
        key = (
            str(getattr(self.settings, "llm_provider", "") or "").strip().lower(),
            str(getattr(self.settings, "judge_provider", "") or "").strip().lower(),
            clean_model,
            effective_judge,
            str(effective_chat_cap or ""),
            str(effective_judge_cap or ""),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        resolved = build_providers(
            self.settings,
            embeddings=self.base_providers.embeddings,
            chat_model_override=clean_model,
            judge_model_override=effective_judge,
            chat_max_output_tokens=effective_chat_cap,
            judge_max_output_tokens=effective_judge_cap,
        )
        self._cache[key] = resolved
        return resolved

__all__ = [
    "AgentProviderResolver",
    "ProviderBundle",
    "build_embeddings",
    "build_providers",
    "normalise_agent_name",
    "resolve_agent_model_overrides",
    "validate_provider_configuration",
    "validate_provider_dependencies",
]
