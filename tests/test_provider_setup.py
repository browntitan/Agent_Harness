from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.config import load_settings
from agentic_chatbot_next.providers import validate_provider_configuration
from agentic_chatbot_next.providers import llm_factory


_ENV_KEYS = [
    "DATA_DIR",
    "LLM_PROVIDER",
    "JUDGE_PROVIDER",
    "EMBEDDINGS_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_CHAT_MODEL",
    "OLLAMA_JUDGE_MODEL",
    "OLLAMA_EMBED_MODEL",
    "OLLAMA_TEMPERATURE",
    "CHAT_MAX_OUTPUT_TOKENS",
    "DEMO_CHAT_MAX_OUTPUT_TOKENS",
    "JUDGE_MAX_OUTPUT_TOKENS",
    "OLLAMA_NUM_PREDICT",
    "DEMO_OLLAMA_NUM_PREDICT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_JUDGE_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
    "AZURE_OPENAI_EMBED_DEPLOYMENT",
    "AZURE_OPENAI_DEPLOYMENT",
    "EMBEDDING_DIM",
    "SSL_CERT_FILE",
    "NVIDIA_OPENAI_ENDPOINT",
    "NVIDIA_API_TOKEN",
    "NVIDIA_CHAT_MODEL",
    "NVIDIA_JUDGE_MODEL",
]


def _load_test_settings(tmp_path: Path, monkeypatch, lines: list[str]):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    env_path = tmp_path / ".env.test"
    env_path.write_text("\n".join([f"DATA_DIR={tmp_path / 'data'}", *lines]) + "\n", encoding="utf-8")
    return load_settings(dotenv_path=str(env_path))


def test_validate_provider_configuration_accepts_ollama_only_with_blank_azure(tmp_path: Path, monkeypatch):
    settings = _load_test_settings(
        tmp_path,
        monkeypatch,
        [
            "LLM_PROVIDER=ollama",
            "JUDGE_PROVIDER=ollama",
            "EMBEDDINGS_PROVIDER=ollama",
            "AZURE_OPENAI_API_KEY=",
            "AZURE_OPENAI_ENDPOINT=",
            "AZURE_OPENAI_CHAT_DEPLOYMENT=",
            "AZURE_OPENAI_JUDGE_DEPLOYMENT=",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=",
            "EMBEDDING_DIM=768",
        ],
    )

    assert validate_provider_configuration(settings) == []


def test_validate_provider_configuration_accepts_azure_only_with_blank_ollama(tmp_path: Path, monkeypatch):
    settings = _load_test_settings(
        tmp_path,
        monkeypatch,
        [
            "LLM_PROVIDER=azure",
            "JUDGE_PROVIDER=azure",
            "EMBEDDINGS_PROVIDER=azure",
            "OLLAMA_BASE_URL=",
            "OLLAMA_CHAT_MODEL=",
            "OLLAMA_JUDGE_MODEL=",
            "OLLAMA_EMBED_MODEL=",
            "AZURE_OPENAI_API_KEY=test-key",
            "AZURE_OPENAI_ENDPOINT=https://example-resource.openai.azure.com/",
            "AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o",
            "AZURE_OPENAI_JUDGE_DEPLOYMENT=gpt-4o",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-ada-002",
            "EMBEDDING_DIM=1536",
        ],
    )

    assert validate_provider_configuration(settings) == []


def test_validate_provider_configuration_accepts_mixed_azure_chat_and_ollama_embeddings(tmp_path: Path, monkeypatch):
    settings = _load_test_settings(
        tmp_path,
        monkeypatch,
        [
            "LLM_PROVIDER=azure",
            "JUDGE_PROVIDER=azure",
            "EMBEDDINGS_PROVIDER=ollama",
            "AZURE_OPENAI_API_KEY=test-key",
            "AZURE_OPENAI_ENDPOINT=https://example-resource.openai.azure.com/",
            "AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o",
            "AZURE_OPENAI_JUDGE_DEPLOYMENT=gpt-4o",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=",
            "EMBEDDING_DIM=768",
        ],
    )

    assert validate_provider_configuration(settings) == []


def test_embeddings_only_validation_ignores_missing_chat_and_judge_azure_settings(tmp_path: Path, monkeypatch):
    settings = _load_test_settings(
        tmp_path,
        monkeypatch,
        [
            "LLM_PROVIDER=azure",
            "JUDGE_PROVIDER=azure",
            "EMBEDDINGS_PROVIDER=ollama",
            "AZURE_OPENAI_API_KEY=",
            "AZURE_OPENAI_ENDPOINT=",
            "AZURE_OPENAI_CHAT_DEPLOYMENT=",
            "AZURE_OPENAI_JUDGE_DEPLOYMENT=",
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=",
            "EMBEDDING_DIM=768",
        ],
    )

    scoped_issues = validate_provider_configuration(settings, contexts=("embeddings",))
    full_issues = validate_provider_configuration(settings)

    assert scoped_issues == []
    assert any(issue.context == "azure" for issue in full_issues)
    assert any(issue.context == "llm" for issue in full_issues)
    assert any(issue.context == "judge" for issue in full_issues)


def test_build_chat_model_omits_ollama_num_predict_when_no_cap_is_configured(monkeypatch):
    recorded: dict[str, object] = {}

    class _RecordingChatOllama:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_ollama", SimpleNamespace(ChatOllama=_RecordingChatOllama))

    settings = SimpleNamespace(
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="nemotron-cascade-2:30b",
        ollama_temperature=0.2,
    )

    llm_factory._build_chat_model(
        settings,
        llm_provider="ollama",
        http_client=None,
        max_output_tokens=None,
    )

    assert "num_predict" not in recorded


def test_build_chat_model_for_openai_compatible_provider_forwards_explicit_max_tokens(monkeypatch):
    recorded: dict[str, object] = {}

    class _RecordingChatOpenAI:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=_RecordingChatOpenAI))

    settings = SimpleNamespace(
        nvidia_openai_endpoint="https://example.invalid",
        nvidia_api_token="token",
        nvidia_chat_model="nvidia/model",
        nvidia_temperature=0.0,
    )

    llm_factory._build_chat_model(
        settings,
        llm_provider="nvidia",
        http_client=object(),
        max_output_tokens=4096,
    )

    assert recorded["max_tokens"] == 4096


def test_build_chat_model_for_openai_compatible_provider_forwards_sync_and_async_clients(monkeypatch):
    recorded: dict[str, object] = {}

    class _RecordingChatOpenAI:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=_RecordingChatOpenAI))

    settings = SimpleNamespace(
        nvidia_openai_endpoint="https://example.invalid",
        nvidia_api_token="token",
        nvidia_chat_model="nvidia/model",
        nvidia_temperature=0.0,
    )
    sync_client = object()
    async_client = object()

    llm_factory._build_chat_model(
        settings,
        llm_provider="nvidia",
        http_client=sync_client,
        http_async_client=async_client,
        max_output_tokens=1024,
    )

    assert recorded["http_client"] is sync_client
    assert recorded["http_async_client"] is async_client
    assert recorded["max_tokens"] == 1024


def test_build_embeddings_model_for_azure_forwards_sync_and_async_clients(monkeypatch):
    recorded: dict[str, object] = {}

    class _RecordingAzureEmbeddings:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(AzureOpenAIEmbeddings=_RecordingAzureEmbeddings),
    )

    settings = SimpleNamespace(
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://example-resource.openai.azure.com/",
        azure_openai_api_version="2024-02-01",
        azure_openai_embed_deployment="text-embedding-3-large",
        tiktoken_enabled=True,
    )
    sync_client = object()
    async_client = object()

    llm_factory._build_embeddings_model(
        settings,
        emb_provider="azure",
        http_client=sync_client,
        http_async_client=async_client,
    )

    assert recorded["http_client"] is sync_client
    assert recorded["http_async_client"] is async_client
