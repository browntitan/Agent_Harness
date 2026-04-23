from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import agentic_chatbot_next.cli as cli

from agentic_chatbot_next.persistence.postgres import vector_schema
from agentic_chatbot_next import rag as next_rag_module
from agentic_chatbot_next.benchmark import ollama_throughput


runner = CliRunner()


class _FakeHTTPResponse:
    def __init__(self, *, status_code: int, payload: dict[str, object]):
        self._status_code = status_code
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcode(self):
        return self._status_code

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_index_skills_uses_store_context_without_full_runtime(monkeypatch):
    captured: dict[str, object] = {}
    settings = SimpleNamespace(default_tenant_id="tenant-123")
    stores = SimpleNamespace(skill_store=object())

    monkeypatch.setattr(cli, "_make_app_or_exit", lambda dotenv=None: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(
        cli,
        "_make_store_context_or_exit",
        lambda dotenv=None: cli.StoreContext(settings=settings, stores=stores),
    )

    class FakeSkillIndexSync:
        def __init__(self, settings_arg, stores_arg):
            captured["settings"] = settings_arg
            captured["stores"] = stores_arg

        def sync(self, *, tenant_id):
            captured["tenant_id"] = tenant_id
            return {"indexed": [], "count": 0}

    monkeypatch.setattr(next_rag_module, "SkillIndexSync", FakeSkillIndexSync)

    result = runner.invoke(cli.app, ["index-skills"])

    assert result.exit_code == 0
    assert captured == {
        "settings": settings,
        "stores": stores,
        "tenant_id": "tenant-123",
    }


def test_sync_kb_uses_store_context_without_full_runtime(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    doc_path = kb_dir / "sample.md"
    doc_path.write_text("# Sample\n", encoding="utf-8")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    extra_path = docs_dir / "ARCHITECTURE.md"
    extra_path.write_text("# Architecture\n", encoding="utf-8")

    settings = SimpleNamespace(
        default_tenant_id="tenant-123",
        default_collection_id="default",
        kb_dir=kb_dir,
        kb_extra_dirs=(docs_dir,),
    )
    stores = SimpleNamespace(doc_store=SimpleNamespace(list_documents=lambda **kwargs: []))

    monkeypatch.setattr(cli, "_make_app_or_exit", lambda dotenv=None: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(
        cli,
        "_make_store_context_or_exit",
        lambda dotenv=None: cli.StoreContext(settings=settings, stores=stores),
    )

    def fake_ingest_paths(settings_arg, stores_arg, paths, *, source_type, tenant_id, collection_id):
        captured["settings"] = settings_arg
        captured["stores"] = stores_arg
        captured["paths"] = [str(path) for path in paths]
        captured["source_type"] = source_type
        captured["tenant_id"] = tenant_id
        captured["collection_id"] = collection_id
        return ["doc-001"]

    monkeypatch.setattr(next_rag_module, "ingest_paths", fake_ingest_paths)

    result = runner.invoke(cli.app, ["sync-kb"])

    assert result.exit_code == 0
    assert captured == {
        "settings": settings,
        "stores": stores,
        "paths": [str(doc_path), str(extra_path)],
        "source_type": "kb",
        "tenant_id": "tenant-123",
        "collection_id": "default",
    }


def _doctor_settings():
    return SimpleNamespace(
        llm_provider="ollama",
        judge_provider="ollama",
        embeddings_provider="ollama",
        embedding_dim=768,
        pg_dsn="postgresql://user:pass@localhost:5432/ragdb",
        kb_dir=Path("/tmp/data/kb"),
        kb_extra_dirs=(Path("/tmp/docs"),),
        default_tenant_id="tenant-123",
        default_collection_id="default",
        http2_enabled=True,
        ssl_verify=True,
        ssl_cert_file=None,
        tiktoken_enabled=False,
        tiktoken_cache_dir=None,
        ollama_base_url="http://ollama:11434",
        ollama_chat_model="gpt-oss:20b",
        ollama_judge_model="gpt-oss:20b",
        ollama_embed_model="nomic-embed-text:latest",
        sandbox_docker_image="agentic-chatbot-sandbox:py312",
    )


def test_doctor_fails_when_selected_ollama_model_is_missing(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: _doctor_settings())
    monkeypatch.setattr(cli, "validate_provider_dependencies", lambda settings: [])
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda settings: [])
    monkeypatch.setattr(cli, "check_docker_availability", lambda: SimpleNamespace(ok=True, detail="docker daemon reachable", remediation=""))
    monkeypatch.setattr(cli, "probe_sandbox_image", lambda image: SimpleNamespace(ok=True, detail=f"{image} ready", remediation=""))
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(
            status_code=200,
            payload={"models": [{"name": "gpt-oss:20b"}]},
        ),
    )

    result = runner.invoke(cli.app, ["doctor", "--skip-db"])

    assert result.exit_code == 1
    assert "nomic-embed-text" in result.output


def test_doctor_passes_when_selected_ollama_models_exist(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: _doctor_settings())
    monkeypatch.setattr(cli, "validate_provider_dependencies", lambda settings: [])
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda settings: [])
    monkeypatch.setattr(cli, "check_docker_availability", lambda: SimpleNamespace(ok=True, detail="docker daemon reachable", remediation=""))
    monkeypatch.setattr(cli, "probe_sandbox_image", lambda image: SimpleNamespace(ok=True, detail=f"{image} ready", remediation=""))
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(
            status_code=200,
            payload={"models": [{"name": "gpt-oss:20b"}, {"name": "nomic-embed-text:latest"}]},
        ),
    )

    result = runner.invoke(cli.app, ["doctor", "--skip-db"])

    assert result.exit_code == 0
    assert "Doctor checks passed." in result.output
    assert "nomic-embed-text" in result.output


def test_doctor_fails_when_sandbox_image_is_not_ready(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: _doctor_settings())
    monkeypatch.setattr(cli, "validate_provider_dependencies", lambda settings: [])
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda settings: [])
    monkeypatch.setattr(cli, "check_docker_availability", lambda: SimpleNamespace(ok=True, detail="docker daemon reachable", remediation=""))
    monkeypatch.setattr(
        cli,
        "probe_sandbox_image",
        lambda image: SimpleNamespace(ok=False, detail=f"{image} missing", remediation="Run build-sandbox-image"),
    )
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(
            status_code=200,
            payload={"models": [{"name": "gpt-oss:20b"}, {"name": "nomic-embed-text:latest"}]},
        ),
    )

    result = runner.invoke(cli.app, ["doctor", "--skip-db"])

    assert result.exit_code == 1
    assert "Sandbox image readiness" in result.output
    assert "build-sandbox-image" in result.output


def test_benchmark_ollama_throughput_uses_default_model_from_settings(tmp_path: Path, monkeypatch):
    settings = SimpleNamespace(
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="nemotron-cascade-2:30b",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: settings)

    class _FakeReport:
        def to_dict(self):
            return {"models": [{"model": "nemotron-cascade-2:30b"}]}

    def fake_run_ollama_throughput_benchmark(**kwargs):
        captured.update(kwargs)
        return _FakeReport()

    monkeypatch.setattr(ollama_throughput, "run_ollama_throughput_benchmark", fake_run_ollama_throughput_benchmark)

    output_path = tmp_path / "ollama-benchmark.json"
    result = runner.invoke(cli.app, ["benchmark-ollama-throughput", "--output", str(output_path)])

    assert result.exit_code == 0
    assert captured["models"] == ["nemotron-cascade-2:30b"]
    assert captured["base_url"] == "http://localhost:11434"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"models": [{"model": "nemotron-cascade-2:30b"}]}


def test_build_sandbox_image_command_reports_success(monkeypatch):
    monkeypatch.setattr(
        cli,
        "build_sandbox_image",
        lambda repo_root, image, timeout_seconds: SimpleNamespace(
            ok=True,
            detail=f"Built {image}",
            command="docker build -t image .",
            remediation="",
        ),
    )

    result = runner.invoke(cli.app, ["build-sandbox-image"])

    assert result.exit_code == 0
    assert "Built agentic-chatbot-sandbox:py312" in result.output


def test_doctor_passes_when_selected_ollama_models_use_latest_alias(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: _doctor_settings())
    monkeypatch.setattr(cli, "validate_provider_dependencies", lambda settings: [])
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda settings: [])
    monkeypatch.setattr(cli, "check_docker_availability", lambda: SimpleNamespace(ok=True, detail="docker daemon reachable", remediation=""))
    monkeypatch.setattr(cli, "probe_sandbox_image", lambda image: SimpleNamespace(ok=True, detail=f"{image} ready", remediation=""))
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(
            status_code=200,
            payload={"models": [{"name": "gpt-oss:20b"}, {"name": "nomic-embed-text"}]},
        ),
    )

    result = runner.invoke(cli.app, ["doctor", "--skip-db"])

    assert result.exit_code == 0
    assert "Doctor checks passed." in result.output
    assert "nomic-embed-text" in result.output


class _FakeDbCursor:
    def __init__(
        self,
        dimensions: dict[str, int | None],
        executed: list[tuple[str, object]] | None = None,
        source_paths: list[str] | None = None,
    ):
        self.dimensions = dimensions
        self.executed = executed if executed is not None else []
        self.table_name: str | None = None
        self._last_query: str = ""
        self.source_paths = list(source_paths or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.executed.append((query, params))
        self._last_query = str(query)
        if params:
            self.table_name = params[0]

    def fetchone(self):
        if self.table_name is None:
            return None
        dim = self.dimensions.get(self.table_name)
        if dim is None:
            return None
        return (f"vector({dim})",)

    def fetchall(self):
        if "SELECT source_path" in self._last_query:
            return [(path, Path(path).name) for path in self.source_paths]
        return []


class _FakeDbConnection:
    def __init__(
        self,
        dimensions: dict[str, int | None],
        executed: list[tuple[str, object]] | None = None,
        source_paths: list[str] | None = None,
    ):
        self.dimensions = dimensions
        self.executed = executed if executed is not None else []
        self.source_paths = list(source_paths or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakeDbCursor(self.dimensions, self.executed, self.source_paths)

    def commit(self):
        return None


def test_doctor_fails_when_skill_chunks_dimension_is_misaligned(monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: _doctor_settings())
    monkeypatch.setattr(cli, "validate_provider_dependencies", lambda settings: [])
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda settings: [])
    monkeypatch.setattr(cli, "check_docker_availability", lambda: SimpleNamespace(ok=True, detail="docker daemon reachable", remediation=""))
    monkeypatch.setattr(cli, "probe_sandbox_image", lambda image: SimpleNamespace(ok=True, detail=f"{image} ready", remediation=""))
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(
            status_code=200,
            payload={"models": [{"name": "gpt-oss:20b"}, {"name": "nomic-embed-text:latest"}]},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "psycopg2",
        SimpleNamespace(
            connect=lambda **kwargs: _FakeDbConnection({"chunks": 768, "skill_chunks": 1536})
        ),
    )

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "skill_chunks.embedding is" in result.output
    assert "EMBEDDING_DIM=768" in result.output
    assert "migrate-embedding-dim --yes" in result.output


def test_doctor_warns_when_configured_kb_sources_are_not_indexed(monkeypatch):
    settings = _doctor_settings()
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: settings)
    monkeypatch.setattr(cli, "validate_provider_dependencies", lambda settings: [])
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda settings: [])
    monkeypatch.setattr(cli, "check_docker_availability", lambda: SimpleNamespace(ok=True, detail="docker daemon reachable", remediation=""))
    monkeypatch.setattr(cli, "probe_sandbox_image", lambda image: SimpleNamespace(ok=True, detail=f"{image} ready", remediation=""))
    monkeypatch.setattr(
        "agentic_chatbot_next.rag.ingest.iter_kb_source_paths",
        lambda settings: [Path("/tmp/data/kb/a.md"), Path("/tmp/docs/ARCHITECTURE.md")],
    )
    monkeypatch.setattr(
        cli,
        "urlopen",
        lambda req, timeout=0: _FakeHTTPResponse(
            status_code=200,
            payload={"models": [{"name": "gpt-oss:20b"}, {"name": "nomic-embed-text:latest"}]},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "psycopg2",
        SimpleNamespace(
            connect=lambda **kwargs: _FakeDbConnection(
                {"chunks": 768, "skill_chunks": 768},
                source_paths=["/tmp/data/kb/a.md"],
            )
        ),
    )

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "KB corpus coverage" in result.output
    assert "ARCHITECTURE.md" in result.output


def test_migrate_embedding_dim_realigns_chunks_and_skill_chunks(monkeypatch):
    settings = SimpleNamespace(
        embedding_dim=768,
        default_tenant_id="tenant-123",
    )
    monkeypatch.setattr(cli, "load_settings", lambda dotenv=None: settings)

    import agentic_chatbot_next.persistence.postgres.connection as db_connection

    apply_calls: list[object] = []
    monkeypatch.setattr(db_connection, "apply_schema", lambda effective_settings: apply_calls.append(effective_settings))
    monkeypatch.setattr(db_connection, "init_pool", lambda effective_settings: None)

    executed: list[tuple[str, object]] = []
    monkeypatch.setattr(db_connection, "get_conn", lambda: _FakeDbConnection({}, executed))

    monkeypatch.setattr(vector_schema, "get_chunks_embedding_dim", lambda: 1536)
    monkeypatch.setattr(vector_schema, "get_skill_chunks_embedding_dim", lambda: 1536)

    changed: list[tuple[str, int]] = []
    monkeypatch.setattr(
        vector_schema,
        "set_chunks_embedding_dim",
        lambda target_dim: changed.append(("chunks", target_dim)) or True,
    )
    monkeypatch.setattr(
        vector_schema,
        "set_skill_chunks_embedding_dim",
        lambda target_dim: changed.append(("skill_chunks", target_dim)) or True,
    )

    result = runner.invoke(cli.app, ["migrate-embedding-dim", "--yes", "--skip-reindex-kb"])

    assert result.exit_code == 0
    assert ("chunks", 768) in changed
    assert ("skill_chunks", 768) in changed
    assert any("TRUNCATE TABLE skill_chunks, skills, chunks, documents" in query for query, _ in executed)
    assert len(apply_calls) == 2
