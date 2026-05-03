from __future__ import annotations

from pathlib import Path


def test_readme_documents_v3_docker_first_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "Agentic Chatbot v3" in readme
    assert "docker compose up -d --build" in readme
    assert "runtime registry smoke check" in readme
    assert "Open WebUI" in readme
    assert "Langfuse" in readme
    assert "Microsoft GraphRAG" in readme
    assert "src/agentic_chatbot_next" in readme
    assert "Open WebUI is the supported chat UI" in readme
    assert (repo_root / "deployment" / "openwebui" / "enterprise_agent_pipe.py").exists()
    assert (repo_root / "control_panel" / "package.json").exists()
    assert not (repo_root / "frontend").exists()
    assert not (repo_root / "packages").exists()
    assert not (repo_root / "training").exists()


def test_local_docker_doc_matches_v3_compose_bootstrap() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    local_docker = (repo_root / "docs" / "LOCAL_DOCKER_STACK.md").read_text(encoding="utf-8")
    compose = (repo_root / "docker-compose.yml").read_text(encoding="utf-8")

    assert "v3 is Docker-first" in local_docker
    assert "ollama-bootstrap" in local_docker
    assert "app-bootstrap" in local_docker
    assert "openwebui-bootstrap" in local_docker
    assert "runtime registry smoke check" in local_docker
    assert "docker compose up -d --build app app-bootstrap openwebui-bootstrap openwebui" in local_docker
    assert "langfuse" in local_docker.lower()
    assert "http://127.0.0.1:3001" in local_docker
    assert "http://127.0.0.1:3000" in local_docker
    assert "name: agentic-chatbot-v3" in compose
    assert "http://ollama:11434" in compose
    assert "http://app:8000/v1" in compose
    assert "runtime-smoke --registry-only --json" in compose
    assert "http://127.0.0.1:8000/health/ready" in compose
    assert "graphrag_projects" in compose


def test_ollama_reranker_residency_defaults_are_documented_in_runtime_envs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    compose = (repo_root / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (repo_root / ".env.example").read_text(encoding="utf-8")
    podman_env = (repo_root / "podman_startup" / "env.podman.example").read_text(encoding="utf-8")
    reranker_model = "rjmalagon/mxbai-rerank-large-v2:1.5b-fp16"
    bootstrap_models = f"nemotron-cascade-2:30b,{reranker_model},nomic-embed-text:latest"

    assert "OLLAMA_MAX_LOADED_MODELS: ${OLLAMA_MAX_LOADED_MODELS:-3}" in compose
    assert f"OLLAMA_BOOTSTRAP_MODELS: ${{OLLAMA_BOOTSTRAP_MODELS:-{bootstrap_models}}}" in compose
    assert f"RERANK_MODEL: ${{RERANK_MODEL:-{reranker_model}}}" in compose

    for text in (env_example, podman_env):
        assert "OLLAMA_MAX_LOADED_MODELS=3" in text
        assert f"OLLAMA_BOOTSTRAP_MODELS={bootstrap_models}" in text
        assert "RERANK_ENABLED=true" in text
        assert "RERANK_PROVIDER=ollama" in text
        assert f"RERANK_MODEL={reranker_model}" in text
        assert "RERANK_TOP_N=12" in text
        assert "RERANK_TIMEOUT_SECONDS=30" in text
        assert "RERANK_FALLBACK_TO_HEURISTICS=true" in text


def test_gateway_docs_keep_openwebui_primary_and_connector_compatibility() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gateway = (repo_root / "docs" / "OPENAI_GATEWAY.md").read_text(encoding="utf-8")

    assert "Open WebUI" in gateway
    assert "Legacy Connector Endpoint" in gateway
    assert "/v1/chat/completions" in gateway
    assert "/v1/connector/chat" in gateway
    assert "/v1/upload" in gateway
    assert "/v1/graphs" in gateway
    assert "X-OpenWebUI-Chat-Id" in gateway
    assert "metadata.collection_id" in gateway
    assert "event: artifacts" in gateway


def test_seed_and_demo_assets_are_present_without_runtime_state() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")

    assert (repo_root / "data" / "agents" / "general.md").exists()
    assert (repo_root / "data" / "prompts" / "router.md").exists()
    assert (repo_root / "data" / "skills" / "rag_agent.md").exists()
    assert (repo_root / "data" / "skill_packs" / "rag").exists()
    assert (repo_root / "data" / "router" / "intent_patterns.json").exists()
    assert (repo_root / "data" / "kb").exists()
    assert (repo_root / "new_demo_notebook" / "agentic_system_showcase.ipynb").exists()
    assert (repo_root / "defense_rag_test_corpus").exists()
    assert "data/runtime/" in gitignore
    assert "data/workspaces/" in gitignore
    assert "data/uploads/" in gitignore
    assert "data/graphrag/projects/" in gitignore


def test_doc_alignment_audit_covers_every_docs_markdown_file() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"
    audit = (docs_dir / "DOC_ALIGNMENT_AUDIT.md").read_text(encoding="utf-8")

    for path in sorted(docs_dir.glob("*.md")):
        if path.name == "DOC_ALIGNMENT_AUDIT.md":
            continue
        assert path.name in audit
