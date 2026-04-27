from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import psycopg2.extras

from agentic_chatbot_next.persistence.postgres.connection import get_conn


@dataclass
class RequirementStatementRecord:
    requirement_id: str
    doc_id: str
    statement_text: str
    tenant_id: str = "local-dev"
    collection_id: str = "default"
    source_type: str = ""
    document_title: str = ""
    statement_index: int = 0
    chunk_id: str = ""
    chunk_index: int = 0
    normalized_statement_text: str = ""
    modality: str = ""
    page_number: Optional[int] = None
    clause_number: str = ""
    section_title: str = ""
    char_start: int = 0
    char_end: int = 0
    multi_requirement: bool = False
    source_excerpt: str = ""
    source_structure: str = ""
    binding_strength: str = ""
    confidence: float = 0.0
    risk_label: str = ""
    risk_rationale: str = ""
    dedupe_group_id: str = ""
    duplicate_count: int = 1
    merged_chunk_ids: str = ""
    merged_source_locations: str = ""
    extractor_version: str = "requirements_v2"
    extractor_mode: str = "mandatory"
    created_at: str = ""


class RequirementStatementStore:
    """CRUD helpers for persisted requirement statement inventories."""

    def replace_doc_statements(
        self,
        doc_id: str,
        tenant_id: str,
        *,
        statements: Iterable[RequirementStatementRecord],
    ) -> None:
        rows = list(statements or [])
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM requirement_statements WHERE doc_id = %s AND tenant_id = %s",
                    (doc_id, tenant_id),
                )
                if rows:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO requirement_statements (
                            requirement_id,
                            doc_id,
                            tenant_id,
                            collection_id,
                            source_type,
                            document_title,
                            statement_index,
                            chunk_id,
                            chunk_index,
                            statement_text,
                            normalized_statement_text,
                            modality,
                            page_number,
                            clause_number,
                            section_title,
                            char_start,
                            char_end,
                            multi_requirement,
                            source_excerpt,
                            source_structure,
                            binding_strength,
                            confidence,
                            risk_label,
                            risk_rationale,
                            dedupe_group_id,
                            duplicate_count,
                            merged_chunk_ids,
                            merged_source_locations,
                            extractor_version,
                            extractor_mode
                        ) VALUES %s
                        ON CONFLICT (requirement_id) DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            tenant_id = EXCLUDED.tenant_id,
                            collection_id = EXCLUDED.collection_id,
                            source_type = EXCLUDED.source_type,
                            document_title = EXCLUDED.document_title,
                            statement_index = EXCLUDED.statement_index,
                            chunk_id = EXCLUDED.chunk_id,
                            chunk_index = EXCLUDED.chunk_index,
                            statement_text = EXCLUDED.statement_text,
                            normalized_statement_text = EXCLUDED.normalized_statement_text,
                            modality = EXCLUDED.modality,
                            page_number = EXCLUDED.page_number,
                            clause_number = EXCLUDED.clause_number,
                            section_title = EXCLUDED.section_title,
                            char_start = EXCLUDED.char_start,
                            char_end = EXCLUDED.char_end,
                            multi_requirement = EXCLUDED.multi_requirement,
                            source_excerpt = EXCLUDED.source_excerpt,
                            source_structure = EXCLUDED.source_structure,
                            binding_strength = EXCLUDED.binding_strength,
                            confidence = EXCLUDED.confidence,
                            risk_label = EXCLUDED.risk_label,
                            risk_rationale = EXCLUDED.risk_rationale,
                            dedupe_group_id = EXCLUDED.dedupe_group_id,
                            duplicate_count = EXCLUDED.duplicate_count,
                            merged_chunk_ids = EXCLUDED.merged_chunk_ids,
                            merged_source_locations = EXCLUDED.merged_source_locations,
                            extractor_version = EXCLUDED.extractor_version,
                            extractor_mode = EXCLUDED.extractor_mode
                        """,
                        [
                            (
                                record.requirement_id,
                                record.doc_id,
                                record.tenant_id,
                                record.collection_id,
                                record.source_type,
                                record.document_title,
                                int(record.statement_index),
                                record.chunk_id,
                                int(record.chunk_index),
                                record.statement_text,
                                record.normalized_statement_text,
                                record.modality,
                                record.page_number,
                                record.clause_number,
                                record.section_title,
                                int(record.char_start),
                                int(record.char_end),
                                bool(record.multi_requirement),
                                record.source_excerpt,
                                record.source_structure,
                                record.binding_strength,
                                float(record.confidence),
                                record.risk_label,
                                record.risk_rationale,
                                record.dedupe_group_id,
                                max(1, int(record.duplicate_count)),
                                record.merged_chunk_ids,
                                record.merged_source_locations,
                                record.extractor_version,
                                record.extractor_mode,
                            )
                            for record in rows
                        ],
                    )
            conn.commit()

    def delete_doc_statements(self, doc_id: str, tenant_id: str) -> None:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM requirement_statements WHERE doc_id = %s AND tenant_id = %s",
                    (doc_id, tenant_id),
                )
            conn.commit()

    def has_doc_statements(self, doc_id: str, tenant_id: str, *, extractor_version: str = "") -> bool:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if extractor_version:
                    cur.execute(
                        """
                        SELECT 1
                        FROM requirement_statements
                        WHERE doc_id = %s AND tenant_id = %s AND extractor_version = %s
                        LIMIT 1
                        """,
                        (doc_id, tenant_id, extractor_version),
                    )
                else:
                    cur.execute(
                        "SELECT 1 FROM requirement_statements WHERE doc_id = %s AND tenant_id = %s LIMIT 1",
                        (doc_id, tenant_id),
                    )
                return cur.fetchone() is not None

    def document_inventory_status(self, doc_id: str, tenant_id: str) -> dict[str, Any]:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS row_count,
                        COALESCE(MAX(extractor_version), '') AS extractor_version,
                        COALESCE(MAX(extractor_mode), '') AS extractor_mode
                    FROM requirement_statements
                    WHERE doc_id = %s AND tenant_id = %s
                    """,
                    (doc_id, tenant_id),
                )
                row = dict(cur.fetchone() or {})
        return {
            "row_count": int(row.get("row_count") or 0),
            "extractor_version": str(row.get("extractor_version") or ""),
            "extractor_mode": str(row.get("extractor_mode") or ""),
        }

    def list_statements(
        self,
        *,
        tenant_id: str = "local-dev",
        collection_id: str = "",
        source_type: str = "",
        doc_ids: Iterable[str] | None = None,
        modalities: Iterable[str] | None = None,
        limit: int = 0,
    ) -> List[RequirementStatementRecord]:
        sql = ["SELECT * FROM requirement_statements WHERE tenant_id = %s"]
        params: List[Any] = [tenant_id]

        normalized_doc_ids = [str(item) for item in (doc_ids or []) if str(item)]
        normalized_modalities = [str(item) for item in (modalities or []) if str(item)]
        if collection_id:
            sql.append("AND collection_id = %s")
            params.append(collection_id)
        if source_type:
            sql.append("AND source_type = %s")
            params.append(source_type)
        if normalized_doc_ids:
            sql.append("AND doc_id = ANY(%s)")
            params.append(normalized_doc_ids)
        if normalized_modalities:
            sql.append("AND modality = ANY(%s)")
            params.append(normalized_modalities)
        sql.append("ORDER BY lower(document_title), chunk_index, statement_index")
        if limit and int(limit) > 0:
            sql.append("LIMIT %s")
            params.append(max(1, int(limit)))

        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(" ".join(sql), params)
                rows = cur.fetchall()
        return [_row_to_record(dict(row)) for row in rows]

    def list_document_statements(
        self,
        doc_id: str,
        *,
        tenant_id: str = "local-dev",
        modalities: Iterable[str] | None = None,
    ) -> List[RequirementStatementRecord]:
        return self.list_statements(
            tenant_id=tenant_id,
            doc_ids=[doc_id],
            modalities=modalities,
        )


def _row_to_record(row: dict[str, Any]) -> RequirementStatementRecord:
    return RequirementStatementRecord(
        requirement_id=str(row.get("requirement_id") or ""),
        doc_id=str(row.get("doc_id") or ""),
        tenant_id=str(row.get("tenant_id") or "local-dev"),
        collection_id=str(row.get("collection_id") or "default"),
        source_type=str(row.get("source_type") or ""),
        document_title=str(row.get("document_title") or ""),
        statement_index=int(row.get("statement_index") or 0),
        chunk_id=str(row.get("chunk_id") or ""),
        chunk_index=int(row.get("chunk_index") or 0),
        statement_text=str(row.get("statement_text") or ""),
        normalized_statement_text=str(row.get("normalized_statement_text") or ""),
        modality=str(row.get("modality") or ""),
        page_number=row.get("page_number"),
        clause_number=str(row.get("clause_number") or ""),
        section_title=str(row.get("section_title") or ""),
        char_start=int(row.get("char_start") or 0),
        char_end=int(row.get("char_end") or 0),
        multi_requirement=bool(row.get("multi_requirement") or False),
        source_excerpt=str(row.get("source_excerpt") or ""),
        source_structure=str(row.get("source_structure") or ""),
        binding_strength=str(row.get("binding_strength") or ""),
        confidence=float(row.get("confidence") or 0.0),
        risk_label=str(row.get("risk_label") or ""),
        risk_rationale=str(row.get("risk_rationale") or ""),
        dedupe_group_id=str(row.get("dedupe_group_id") or ""),
        duplicate_count=max(1, int(row.get("duplicate_count") or 1)),
        merged_chunk_ids=str(row.get("merged_chunk_ids") or ""),
        merged_source_locations=str(row.get("merged_source_locations") or ""),
        extractor_version=str(row.get("extractor_version") or "requirements_v2"),
        extractor_mode=str(row.get("extractor_mode") or "mandatory"),
        created_at=str(row.get("created_at") or ""),
    )
