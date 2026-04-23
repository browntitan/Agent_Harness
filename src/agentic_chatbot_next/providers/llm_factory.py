from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.providers.dependency_checks import (
    raise_if_invalid_provider_configuration,
    raise_if_missing_provider_dependencies,
)
from agentic_chatbot_next.providers.output_limits import (
    resolve_chat_output_cap,
    resolve_judge_output_cap,
)


@dataclass(frozen=True)
class ProviderBundle:
    chat: object
    judge: object
    embeddings: object


def _validate_provider_selection(*, context: str, provider: str, supported: set[str]) -> None:
    if provider not in supported:
        env_name = {
            "llm": "LLM_PROVIDER",
            "judge": "JUDGE_PROVIDER",
            "embeddings": "EMBEDDINGS_PROVIDER",
        }[context]
        supported_text = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported {env_name}: {provider!r}. Supported: {supported_text}")


def _build_httpx_client(settings: Settings):
    import httpx

    verify: bool | str = True
    if not settings.ssl_verify:
        verify = False
    elif settings.ssl_cert_file:
        verify = str(settings.ssl_cert_file)
    request_timeout = float(getattr(settings, "llm_http_timeout_seconds", 120.0) or 120.0)
    connect_timeout = float(getattr(settings, "llm_http_connect_timeout_seconds", 20.0) or 20.0)

    return httpx.Client(
        http2=settings.http2_enabled,
        verify=verify,
        timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
    )


def _build_httpx_async_client(settings: Settings):
    import httpx

    verify: bool | str = True
    if not settings.ssl_verify:
        verify = False
    elif settings.ssl_cert_file:
        verify = str(settings.ssl_cert_file)
    request_timeout = float(getattr(settings, "llm_http_timeout_seconds", 120.0) or 120.0)
    connect_timeout = float(getattr(settings, "llm_http_connect_timeout_seconds", 20.0) or 20.0)

    return httpx.AsyncClient(
        http2=settings.http2_enabled,
        verify=verify,
        timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
    )


def _normalize_openai_base_url(value: str) -> str:
    """Normalize OpenAI-compatible endpoint URLs to include /v1."""
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return value.strip()
    base = value.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _build_chat_model(
    settings: Settings,
    *,
    llm_provider: str,
    http_client,
    http_async_client=None,
    model_override: str | None = None,
    temperature_override: float | None = None,
    max_output_tokens: int | None = None,
):
    temperature = temperature_override if temperature_override is not None else (
        settings.ollama_temperature if llm_provider == "ollama"
        else settings.azure_temperature if llm_provider == "azure"
        else settings.nvidia_temperature
    )
    if llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = dict(
            base_url=settings.ollama_base_url,
            model=model_override or settings.ollama_chat_model,
            temperature=temperature,
            validate_model_on_init=True,
        )
        if max_output_tokens is not None:
            kwargs["num_predict"] = max_output_tokens
        return ChatOllama(**kwargs)

    if llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        kwargs = dict(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=model_override or settings.azure_openai_chat_deployment,
            temperature=temperature,
            http_client=http_client,
            http_async_client=http_async_client,
        )
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        return AzureChatOpenAI(**kwargs)

    from langchain_openai import ChatOpenAI

    base_url = _normalize_openai_base_url(settings.nvidia_openai_endpoint or "")
    nvidia_token = settings.nvidia_api_token or ""
    kwargs = dict(
        base_url=base_url,
        api_key="not-required",
        model=model_override or settings.nvidia_chat_model,
        temperature=temperature,
        http_client=http_client,
        http_async_client=http_async_client,
        default_headers={"Authorization": f"Bearer {nvidia_token}"},
    )
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    return ChatOpenAI(**kwargs)


def _build_judge_model(
    settings: Settings,
    *,
    judge_provider: str,
    http_client,
    http_async_client=None,
    model_override: str | None = None,
    max_output_tokens: int | None = None,
):
    if judge_provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = dict(
            base_url=settings.ollama_base_url,
            model=model_override or settings.ollama_judge_model,
            temperature=settings.judge_temperature,
            validate_model_on_init=True,
        )
        if max_output_tokens is not None:
            kwargs["num_predict"] = max_output_tokens
        return ChatOllama(**kwargs)

    if judge_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        kwargs = dict(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_deployment=model_override or settings.azure_openai_judge_deployment,
            temperature=settings.judge_temperature,
            http_client=http_client,
            http_async_client=http_async_client,
        )
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        return AzureChatOpenAI(**kwargs)

    from langchain_openai import ChatOpenAI

    base_url = _normalize_openai_base_url(settings.nvidia_openai_endpoint or "")
    nvidia_token = settings.nvidia_api_token or ""
    kwargs = dict(
        base_url=base_url,
        api_key="not-required",
        model=model_override or settings.nvidia_judge_model,
        temperature=settings.nvidia_temperature,
        http_client=http_client,
        http_async_client=http_async_client,
        default_headers={"Authorization": f"Bearer {nvidia_token}"},
    )
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    return ChatOpenAI(**kwargs)


def _build_embeddings_model(
    settings: Settings,
    *,
    emb_provider: str,
    http_client,
    http_async_client=None,
):
    if emb_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )

    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embed_deployment,
        http_client=http_client,
        http_async_client=http_async_client,
        tiktoken_enabled=settings.tiktoken_enabled,
        check_embedding_ctx_length=settings.tiktoken_enabled,
    )


def build_embeddings(settings: Settings) -> object:
    emb_provider = settings.embeddings_provider.lower()
    _validate_provider_selection(context="embeddings", provider=emb_provider, supported={"azure", "ollama"})

    raise_if_missing_provider_dependencies(settings, contexts=("embeddings",))
    raise_if_invalid_provider_configuration(settings, contexts=("embeddings",))

    http_client = _build_httpx_client(settings)
    http_async_client = _build_httpx_async_client(settings)
    return _build_embeddings_model(
        settings,
        emb_provider=emb_provider,
        http_client=http_client,
        http_async_client=http_async_client,
    )


def build_providers(
    settings: Settings,
    *,
    embeddings: object | None = None,
    chat_model_override: str | None = None,
    judge_model_override: str | None = None,
    chat_temperature_override: float | None = None,
    chat_max_output_tokens: int | None = None,
    judge_max_output_tokens: int | None = None,
) -> ProviderBundle:
    """Factory that builds the chat model, judge model, and embeddings.

    - Default: Ollama (ChatOllama + OllamaEmbeddings)
    - Optional: Azure OpenAI (AzureChatOpenAI + AzureOpenAIEmbeddings)
    - Optional: NVIDIA OpenAI-compatible endpoint for chat/judge (ChatOpenAI)

    We keep return types as `object` to avoid hard-coupling to a specific
    LangChain version/type hierarchy; callers typically treat these as
    Runnable chat models and Embeddings.
    """

    llm_provider = settings.llm_provider.lower()
    emb_provider = settings.embeddings_provider.lower()
    judge_provider = settings.judge_provider.lower()

    _validate_provider_selection(context="llm", provider=llm_provider, supported={"azure", "nvidia", "ollama"})
    _validate_provider_selection(context="embeddings", provider=emb_provider, supported={"azure", "ollama"})
    _validate_provider_selection(context="judge", provider=judge_provider, supported={"azure", "nvidia", "ollama"})

    # Fail fast with actionable instructions before provider imports.
    raise_if_missing_provider_dependencies(settings)
    raise_if_invalid_provider_configuration(settings)
    http_client = _build_httpx_client(settings)
    http_async_client = _build_httpx_async_client(settings)

    chat = _build_chat_model(
        settings,
        llm_provider=llm_provider,
        http_client=http_client,
        http_async_client=http_async_client,
        model_override=chat_model_override,
        temperature_override=chat_temperature_override,
        max_output_tokens=(
            chat_max_output_tokens
            if chat_max_output_tokens is not None
            else resolve_chat_output_cap(settings)
        ),
    )
    judge = _build_judge_model(
        settings,
        judge_provider=judge_provider,
        http_client=http_client,
        http_async_client=http_async_client,
        model_override=judge_model_override,
        max_output_tokens=(
            judge_max_output_tokens
            if judge_max_output_tokens is not None
            else resolve_judge_output_cap(settings)
        ),
    )
    resolved_embeddings = embeddings if embeddings is not None else _build_embeddings_model(
        settings,
        emb_provider=emb_provider,
        http_client=http_client,
        http_async_client=http_async_client,
    )

    return ProviderBundle(chat=chat, judge=judge, embeddings=resolved_embeddings)
