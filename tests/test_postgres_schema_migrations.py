from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import psycopg2
import pytest

from agentic_chatbot_next.persistence.postgres import connection as db_connection


def test_graph_indexes_backfill_columns_precede_query_ready_index() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agentic_chatbot_next"
        / "persistence"
        / "postgres"
        / "schema.sql"
    )
    sql = schema_path.read_text(encoding="utf-8")

    query_ready_alter = sql.index(
        "ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS query_ready BOOLEAN NOT NULL DEFAULT FALSE;"
    )
    query_backend_alter = sql.index(
        "ALTER TABLE graph_indexes ADD COLUMN IF NOT EXISTS query_backend TEXT NOT NULL DEFAULT '';"
    )
    query_ready_index = sql.index("CREATE INDEX IF NOT EXISTS graph_indexes_query_ready_idx")

    assert query_ready_alter < query_ready_index
    assert query_backend_alter < query_ready_index


def test_documents_table_precedes_collection_policy_backfill() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agentic_chatbot_next"
        / "persistence"
        / "postgres"
        / "schema.sql"
    )
    sql = schema_path.read_text(encoding="utf-8")

    documents_table = sql.index("CREATE TABLE IF NOT EXISTS documents")
    collection_policy_backfill = sql.index("UPDATE collections AS c")

    assert documents_table < collection_policy_backfill


def test_requirement_statement_backfill_columns_precede_indexes() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agentic_chatbot_next"
        / "persistence"
        / "postgres"
        / "schema.sql"
    )
    sql = schema_path.read_text(encoding="utf-8")

    requirement_statements_table = sql.index("CREATE TABLE IF NOT EXISTS requirement_statements")
    source_excerpt_alter = sql.index(
        "ALTER TABLE requirement_statements ADD COLUMN IF NOT EXISTS source_excerpt TEXT NOT NULL DEFAULT '';"
    )
    merged_locations_alter = sql.index(
        "ALTER TABLE requirement_statements ADD COLUMN IF NOT EXISTS merged_source_locations TEXT NOT NULL DEFAULT '';"
    )
    requirement_index = sql.index("CREATE INDEX IF NOT EXISTS requirement_statements_tenant_doc_idx")

    assert requirement_statements_table < source_excerpt_alter < requirement_index
    assert merged_locations_alter < requirement_index


def test_chunks_metadata_json_backfill_precedes_gin_index() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agentic_chatbot_next"
        / "persistence"
        / "postgres"
        / "schema.sql"
    )
    sql = schema_path.read_text(encoding="utf-8")

    chunks_table = sql.index("CREATE TABLE IF NOT EXISTS chunks")
    metadata_alter = sql.index("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS metadata_json JSONB;")
    metadata_index = sql.index("CREATE INDEX IF NOT EXISTS chunks_metadata_json_gin_idx")

    assert chunks_table < metadata_alter < metadata_index


def test_skills_collection_id_backfill_precedes_collection_index() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "agentic_chatbot_next"
        / "persistence"
        / "postgres"
        / "schema.sql"
    )
    sql = schema_path.read_text(encoding="utf-8")

    skills_table = sql.index("CREATE TABLE IF NOT EXISTS skills")
    collection_alter = sql.index("ALTER TABLE skills ADD COLUMN IF NOT EXISTS collection_id TEXT DEFAULT '';")
    collection_index = sql.index("CREATE INDEX IF NOT EXISTS skills_collection_idx")

    assert skills_table < collection_alter < collection_index


def test_apply_schema_reports_legacy_graph_indexes_upgrade_error(monkeypatch, tmp_path: Path) -> None:
    db_connection.close_pool()
    schema_path = tmp_path / "schema.sql"
    schema_path.write_text("SELECT 1;", encoding="utf-8")
    settings = SimpleNamespace(embedding_dim=768, pg_dsn="postgresql://unused")

    class _FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            del sql, params
            raise psycopg2.errors.UndefinedColumn('column "query_ready" does not exist')

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def rollback(self) -> None:
            return None

        def commit(self) -> None:
            raise AssertionError("commit should not be reached when execute fails")

    monkeypatch.setattr(db_connection, "init_pool", lambda settings_arg: None)

    @contextmanager
    def _fake_get_conn():
        yield _FakeConn()

    monkeypatch.setattr(db_connection, "get_conn", _fake_get_conn)

    with pytest.raises(RuntimeError) as exc_info:
        db_connection.apply_schema(settings, schema_path=str(schema_path))

    message = str(exc_info.value)
    assert "legacy `graph_indexes` table" in message
    assert "rag-postgres" in message
    assert str(schema_path) in message


def test_apply_schema_uses_advisory_lock_and_skips_repeat_calls(monkeypatch, tmp_path: Path) -> None:
    db_connection.close_pool()
    schema_path = tmp_path / "schema.sql"
    schema_path.write_text("SELECT 1;", encoding="utf-8")
    settings = SimpleNamespace(embedding_dim=768, pg_dsn="postgresql://unused")
    executed: list[tuple[str, object]] = []
    connections_opened = {"count": 0}

    class _FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            executed.append((sql, params))

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    monkeypatch.setattr(db_connection, "init_pool", lambda settings_arg: None)

    @contextmanager
    def _fake_get_conn():
        connections_opened["count"] += 1
        yield _FakeConn()

    monkeypatch.setattr(db_connection, "get_conn", _fake_get_conn)

    db_connection.apply_schema(settings, schema_path=str(schema_path))
    db_connection.apply_schema(settings, schema_path=str(schema_path))

    assert connections_opened["count"] == 1
    assert executed[0][0] == "SELECT pg_advisory_lock(%s)"
    assert executed[1][0] == "SELECT 1;"
    assert executed[2][0] == "SELECT pg_advisory_unlock(%s)"
