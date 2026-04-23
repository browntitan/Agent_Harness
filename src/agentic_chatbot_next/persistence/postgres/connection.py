from __future__ import annotations

import contextlib
import hashlib
import threading
from pathlib import Path
from typing import Generator

import psycopg2
from psycopg2 import pool as pg_pool

from agentic_chatbot_next.config import Settings

_pool: pg_pool.ThreadedConnectionPool | None = None
_applied_schema_keys: set[tuple[str, str, int]] = set()
_schema_apply_lock = threading.Lock()


def _legacy_graph_indexes_schema_message(*, schema_path: str) -> str:
    return (
        "Failed to apply PostgreSQL schema because the database appears to have a "
        "legacy `graph_indexes` table without the newer graph backfill columns such "
        "as `query_ready`/`query_backend`. Upgrade to the patched schema ordering and, "
        "if the app cannot boot far enough to rerun `python run.py migrate`, apply the "
        "one-time `graph_indexes` column backfill against `rag-postgres` first. "
        f"Schema file: {schema_path}"
    )


def init_pool(settings: Settings, minconn: int = 1, maxconn: int = 10) -> None:
    """Initialise the singleton connection pool. Safe to call multiple times."""
    global _pool
    if _pool is not None:
        return
    _pool = pg_pool.ThreadedConnectionPool(
        minconn=minconn,
        maxconn=maxconn,
        dsn=settings.pg_dsn,
    )


def _schema_cache_key(settings: Settings, *, schema_path: str) -> tuple[str, str, int]:
    return (
        str(getattr(settings, "pg_dsn", "") or ""),
        str(Path(schema_path).resolve()),
        int(getattr(settings, "embedding_dim", 0) or 0),
    )


def _schema_lock_id(cache_key: tuple[str, str, int]) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in cache_key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


def _schema_already_applied(cache_key: tuple[str, str, int]) -> bool:
    with _schema_apply_lock:
        return cache_key in _applied_schema_keys


def _mark_schema_applied(cache_key: tuple[str, str, int]) -> None:
    with _schema_apply_lock:
        _applied_schema_keys.add(cache_key)


def _release_schema_lock(conn: psycopg2.extensions.connection, *, lock_id: int, lock_acquired: bool) -> None:
    if not lock_acquired:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
        conn.commit()
    except Exception:
        conn.rollback()


def get_pool() -> pg_pool.ThreadedConnectionPool:
    if _pool is None:
        raise RuntimeError(
            "PostgreSQL connection pool is not initialised. "
            "Call db.connection.init_pool(settings) at startup."
        )
    return _pool


@contextlib.contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager that borrows a connection from the pool and returns it on exit."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def apply_schema(settings: Settings, schema_path: str | None = None) -> None:
    """Apply schema.sql to the target database.

    Idempotent — all CREATE statements use IF NOT EXISTS.
    Call once at startup or from the CLI migrate command.
    """
    if schema_path is None:
        schema_path = str(Path(__file__).parent / "schema.sql")
    cache_key = _schema_cache_key(settings, schema_path=schema_path)
    if _schema_already_applied(cache_key):
        return

    sql_template = Path(schema_path).read_text(encoding="utf-8")
    sql = sql_template.replace("__EMBEDDING_DIM__", str(int(settings.embedding_dim)))
    init_pool(settings)
    lock_id = _schema_lock_id(cache_key)
    with get_conn() as conn:
        lock_acquired = False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_lock(%s)", (lock_id,))
                lock_acquired = True
                cur.execute(sql)
                cur.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
                lock_acquired = False
            conn.commit()
        except psycopg2.errors.UndefinedColumn as exc:
            conn.rollback()
            _release_schema_lock(conn, lock_id=lock_id, lock_acquired=lock_acquired)
            message = str(exc)
            if "query_ready" in message or "query_backend" in message:
                raise RuntimeError(
                    _legacy_graph_indexes_schema_message(schema_path=schema_path)
                ) from exc
            raise
        except Exception:
            conn.rollback()
            _release_schema_lock(conn, lock_id=lock_id, lock_acquired=lock_acquired)
            raise
    _mark_schema_applied(cache_key)


def close_pool() -> None:
    """Close all connections. Call on application shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
    with _schema_apply_lock:
        _applied_schema_keys.clear()
