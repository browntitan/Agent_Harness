from __future__ import annotations

import re
from typing import Optional

from agentic_chatbot_next.persistence.postgres.connection import get_conn

_VECTOR_TYPE_RE = re.compile(r"^vector\((\d+)\)$", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_vector_dimension(type_name: str) -> Optional[int]:
    match = _VECTOR_TYPE_RE.match((type_name or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _validated_identifier(identifier: str) -> str:
    name = (identifier or "").strip()
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"invalid SQL identifier: {identifier!r}")
    return name


def get_vector_column_dim(table_name: str, column_name: str = "embedding") -> Optional[int]:
    """Return the configured vector dimension for the requested table column, if discoverable."""
    safe_table_name = _validated_identifier(table_name)
    safe_column_name = _validated_identifier(column_name)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema()
                  AND c.relname = %s
                  AND a.attname = %s
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """,
                (safe_table_name, safe_column_name),
            )
            row = cur.fetchone()

    if not row:
        return None
    return parse_vector_dimension(str(row[0]))


def get_table_embedding_dim(table_name: str) -> Optional[int]:
    """Return the configured `<table>.embedding` vector dimension, if discoverable."""
    return get_vector_column_dim(table_name, "embedding")


def get_chunks_embedding_dim() -> Optional[int]:
    return get_table_embedding_dim("chunks")


def get_skill_chunks_embedding_dim() -> Optional[int]:
    return get_table_embedding_dim("skill_chunks")


def get_graph_indexes_embedding_dim() -> Optional[int]:
    return get_vector_column_dim("graph_indexes", "summary_embedding")


def set_table_embedding_dim(table_name: str, target_dim: int) -> bool:
    """Alter `<table>.embedding` to vector(target_dim), clearing old vector values.

    Returns True when a schema change was applied, False when already aligned.
    """
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")

    safe_table_name = _validated_identifier(table_name)
    current_dim = get_table_embedding_dim(safe_table_name)
    if current_dim == target_dim:
        return False

    index_name = f"{safe_table_name}_embedding_hnsw_idx"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {index_name}")
            cur.execute(
                f"ALTER TABLE {safe_table_name} ALTER COLUMN embedding TYPE vector({int(target_dim)}) USING NULL"
            )
        conn.commit()
    return True


def set_chunks_embedding_dim(target_dim: int) -> bool:
    return set_table_embedding_dim("chunks", target_dim)


def set_skill_chunks_embedding_dim(target_dim: int) -> bool:
    return set_table_embedding_dim("skill_chunks", target_dim)
