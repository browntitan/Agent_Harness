from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agentic_chatbot_next.contracts.agents import AgentDefinition
from agentic_chatbot_next.contracts.messages import SessionState
from agentic_chatbot_next.persistence.postgres.chunks import ChunkRecord
from agentic_chatbot_next.rag.ingest import backfill_requirement_statements
from agentic_chatbot_next.rag.requirements import (
    BROAD_REQUIREMENT_MODE,
    MANDATORY_MODE,
    REQUIREMENT_EXTRACTOR_VERSION,
    STRICT_SHALL_MODE,
)
from agentic_chatbot_next.rag.requirements_service import RequirementExtractionService, infer_requirement_mode
from agentic_chatbot_next.runtime.context import RuntimePaths
from agentic_chatbot_next.runtime.kernel import RuntimeKernel
from agentic_chatbot_next.sandbox.workspace import SessionWorkspace
from agentic_chatbot_next.tools.requirements import make_requirement_tools


class FakeRequirementStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], list[object]] = {}

    def replace_doc_statements(self, doc_id: str, tenant_id: str, *, statements):
        self._rows[(tenant_id, doc_id)] = list(statements)

    def delete_doc_statements(self, doc_id: str, tenant_id: str) -> None:
        self._rows.pop((tenant_id, doc_id), None)

    def has_doc_statements(self, doc_id: str, tenant_id: str, *, extractor_version: str = "") -> bool:
        rows = list(self._rows.get((tenant_id, doc_id), []))
        if not extractor_version:
            return bool(rows)
        return any(str(getattr(row, "extractor_version", "") or "") == extractor_version for row in rows)

    def document_inventory_status(self, doc_id: str, tenant_id: str) -> dict[str, object]:
        rows = list(self._rows.get((tenant_id, doc_id), []))
        return {
            "row_count": len(rows),
            "extractor_version": str(getattr(rows[0], "extractor_version", "") or "") if rows else "",
            "extractor_mode": str(getattr(rows[0], "extractor_mode", "") or "") if rows else "",
        }

    def list_statements(
        self,
        *,
        tenant_id: str = "tenant",
        collection_id: str = "",
        source_type: str = "",
        doc_ids=None,
        modalities=None,
        limit: int = 0,
    ):
        rows = []
        allowed_doc_ids = {str(item) for item in (doc_ids or []) if str(item)}
        allowed_modalities = {str(item) for item in (modalities or []) if str(item)}
        for (row_tenant, doc_id), values in self._rows.items():
            if row_tenant != tenant_id:
                continue
            if allowed_doc_ids and doc_id not in allowed_doc_ids:
                continue
            for record in values:
                if collection_id and str(getattr(record, "collection_id", "") or "") != collection_id:
                    continue
                if source_type and str(getattr(record, "source_type", "") or "") != source_type:
                    continue
                if allowed_modalities and str(getattr(record, "modality", "") or "") not in allowed_modalities:
                    continue
                rows.append(record)
        rows.sort(key=lambda item: (str(getattr(item, "document_title", "")).lower(), int(getattr(item, "chunk_index", 0)), int(getattr(item, "statement_index", 0))))
        return rows[:limit] if limit else rows


def _record(
    *,
    doc_id: str,
    title: str,
    source_type: str,
    collection_id: str,
    file_type: str,
    source_path: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        doc_id=doc_id,
        tenant_id="tenant",
        collection_id=collection_id,
        title=title,
        source_type=source_type,
        source_path=source_path,
        source_identity=f"path:{source_path}",
        content_hash=f"hash-{doc_id}",
        ingested_at="2026-04-20T00:00:00Z",
        file_type=file_type,
        doc_structure_type="requirements_doc",
        source_display_path=title,
        num_chunks=1,
    )


def _stores(records: list[SimpleNamespace], chunks_by_doc_id: dict[str, list[ChunkRecord]]):
    record_map = {record.doc_id: record for record in records}
    requirement_store = FakeRequirementStore()
    return SimpleNamespace(
        doc_store=SimpleNamespace(
            list_documents=lambda tenant_id="tenant", source_type="", collection_id="": [
                record
                for record in records
                if (not source_type or record.source_type == source_type)
                and (not collection_id or record.collection_id == collection_id)
            ],
            fuzzy_search_title=lambda hint, tenant_id, limit=5, collection_id="": [
                {"doc_id": record.doc_id, "title": record.title, "score": 0.5}
                for record in records
                if hint.lower() in record.title.lower()
            ][:limit],
            get_document=lambda doc_id, tenant_id="tenant": record_map.get(doc_id),
        ),
        chunk_store=SimpleNamespace(
            list_document_chunks=lambda doc_id, tenant_id="tenant": list(chunks_by_doc_id.get(doc_id, []))
        ),
        requirement_store=requirement_store,
    )


def _session(tmp_path: Path, *, uploaded_doc_ids=None, metadata=None) -> SimpleNamespace:
    workspace = SessionWorkspace.for_session("req-session", tmp_path / "workspaces")
    workspace.open()
    return SimpleNamespace(
        tenant_id="tenant",
        session_id="req-session",
        conversation_id="req-conv",
        workspace=workspace,
        metadata=dict(metadata or {}),
        uploaded_doc_ids=list(uploaded_doc_ids or []),
    )


def test_extract_requirement_statements_prefers_single_supported_upload_and_returns_preview(tmp_path: Path) -> None:
    upload_record = _record(
        doc_id="upload-spec",
        title="SPEC_A.docx",
        source_type="upload",
        collection_id="upload-col",
        file_type="docx",
        source_path="/tmp/SPEC_A.docx",
    )
    stores = _stores(
        [upload_record],
        {
            "upload-spec": [
                ChunkRecord(
                    chunk_id="upload-spec#chunk0001",
                    doc_id="upload-spec",
                    tenant_id="tenant",
                    collection_id="upload-col",
                    chunk_index=0,
                    content=(
                        "3.1 The subsystem shall log audit events. "
                        "The reviewer should add a note. "
                        "The gateway must not expose debug routes."
                    ),
                    page_number=4,
                    clause_number="3.1",
                    section_title="Audit",
                    chunk_type="requirement",
                )
            ]
        },
    )
    session = _session(
        tmp_path,
        uploaded_doc_ids=["upload-spec"],
        metadata={"upload_collection_id": "upload-col", "collection_id": "upload-col"},
    )
    tools = {tool.name: tool for tool in make_requirement_tools(SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"), stores, session)}

    result = tools["extract_requirement_statements"].invoke(
        {"source_scope": "uploads", "mode": MANDATORY_MODE}
    )

    assert result["statement_count"] == 2
    assert result["document_count"] == 1
    assert result["extractor_version"] == REQUIREMENT_EXTRACTOR_VERSION
    assert result["preview_columns"] == [
        "document_title",
        "source_location",
        "source_structure",
        "requirement_text",
        "source_excerpt",
        "risk_rationale",
        "confidence",
    ]
    assert result["preview_rows"][0]["document_title"] == "SPEC_A.docx"
    assert "Extracted 2 requirement statements from 1 document." == result["summary_text"]


def test_extract_requirement_statements_does_not_filter_uploads_by_openwebui_source_ids(tmp_path: Path) -> None:
    upload_record = _record(
        doc_id="UPLOAD_abc123",
        title="SPEC_UPLOAD.docx",
        source_type="upload",
        collection_id="owui-chat-1",
        file_type="docx",
        source_path="/tmp/SPEC_UPLOAD.docx",
    )
    stores = _stores(
        [upload_record],
        {
            "UPLOAD_abc123": [
                ChunkRecord(
                    chunk_id="UPLOAD_abc123#chunk0001",
                    doc_id="UPLOAD_abc123",
                    tenant_id="tenant",
                    collection_id="owui-chat-1",
                    chunk_index=0,
                    content="The system shall retain audit logs. The contractor must not share credentials.",
                    chunk_type="requirement",
                )
            ]
        },
    )
    session = _session(
        tmp_path,
        uploaded_doc_ids=[],
        metadata={
            "upload_collection_id": "owui-chat-1",
            "collection_id": "owui-chat-1",
            "source_upload_ids": ["openwebui-file-uuid"],
        },
    )
    tools = {tool.name: tool for tool in make_requirement_tools(SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"), stores, session)}

    result = tools["extract_requirement_statements"].invoke({"source_scope": "uploads"})

    assert result["statement_count"] == 2
    assert result["documents"][0]["doc_id"] == "UPLOAD_abc123"


def test_extract_requirement_statements_thin_mode_requires_internal_upload_doc_ids(tmp_path: Path) -> None:
    upload_record = _record(
        doc_id="UPLOAD_abc123",
        title="SPEC_UPLOAD.docx",
        source_type="upload",
        collection_id="owui-chat-1",
        file_type="docx",
        source_path="/tmp/SPEC_UPLOAD.docx",
    )
    stores = _stores(
        [upload_record],
        {
            "UPLOAD_abc123": [
                ChunkRecord(
                    chunk_id="UPLOAD_abc123#chunk0001",
                    doc_id="UPLOAD_abc123",
                    tenant_id="tenant",
                    collection_id="owui-chat-1",
                    chunk_index=0,
                    content="The system shall retain audit logs.",
                    chunk_type="requirement",
                )
            ]
        },
    )
    session = _session(
        tmp_path,
        uploaded_doc_ids=[],
        metadata={
            "upload_collection_id": "owui-chat-1",
            "collection_id": "owui-chat-1",
            "source_upload_ids": ["openwebui-file-uuid"],
            "openwebui_thin_mode": True,
            "document_source_policy": "agent_repository_only",
        },
    )
    tools = {tool.name: tool for tool in make_requirement_tools(SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"), stores, session)}

    result = tools["extract_requirement_statements"].invoke({"source_scope": "uploads"})

    assert result["handled"] is False
    assert result["error"] == "No indexed documents are available in the requested scope."


def test_requirements_service_unwraps_openwebui_rag_prompt_for_single_upload(tmp_path: Path) -> None:
    upload_record = _record(
        doc_id="UPLOAD_wrapped",
        title="Wrapped_SPEC.docx",
        source_type="upload",
        collection_id="owui-chat-wrapped",
        file_type="docx",
        source_path="/tmp/Wrapped_SPEC.docx",
    )
    stores = _stores(
        [upload_record],
        {
            "UPLOAD_wrapped": [
                ChunkRecord(
                    chunk_id="UPLOAD_wrapped#chunk0001",
                    doc_id="UPLOAD_wrapped",
                    tenant_id="tenant",
                    collection_id="owui-chat-wrapped",
                    chunk_index=0,
                    content=(
                        "The system shall retain audit logs. "
                        "The system shall not expose debug ports. "
                        "The operator must review alerts."
                    ),
                    chunk_type="requirement",
                )
            ]
        },
    )
    session = _session(
        tmp_path,
        uploaded_doc_ids=["UPLOAD_wrapped"],
        metadata={
            "upload_collection_id": "owui-chat-wrapped",
            "collection_id": "owui-chat-wrapped",
            "openwebui_thin_mode": True,
            "document_source_policy": "agent_repository_only",
        },
    )
    wrapped = """### Task:
Respond to the user query using the provided context.

### Guidelines:
- If you don't know the answer, say so.
- Don't present information that's not present in the context.
- Examples such as e.g. should not be treated as document names.

<context>
OpenWebUI retrieval snippet that is not trusted evidence.
</context>

### Output:
Provide a clear and direct response.
extract all requirements/ shall statements from the uploaded document
"""

    service = RequirementExtractionService(
        SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"),
        stores,
        session,
    )

    result = service.extract_for_user_request(wrapped)

    assert result["handled"] is True
    assert result["mode"] == BROAD_REQUIREMENT_MODE
    assert result["statement_count"] == 3
    assert result["ignored_document_targets"] == []
    assert result["selected_doc_ids"] == ["UPLOAD_wrapped"]
    assert result["preview_rows"][0]["source_excerpt"]
    assert result["sanitized_user_query"] == "extract all requirements/ shall statements from the uploaded document"


def test_requirements_service_uses_strict_mode_only_for_explicit_shall_requests(tmp_path: Path) -> None:
    upload_record = _record(
        doc_id="UPLOAD_strict",
        title="Strict_SPEC.docx",
        source_type="upload",
        collection_id="owui-chat-strict",
        file_type="docx",
        source_path="/tmp/Strict_SPEC.docx",
    )
    stores = _stores(
        [upload_record],
        {
            "UPLOAD_strict": [
                ChunkRecord(
                    chunk_id="UPLOAD_strict#chunk0001",
                    doc_id="UPLOAD_strict",
                    tenant_id="tenant",
                    collection_id="owui-chat-strict",
                    chunk_index=0,
                    content=(
                        "The system shall retain audit logs. "
                        "The system shall not expose debug ports. "
                        "The operator must review alerts."
                    ),
                    chunk_type="requirement",
                )
            ]
        },
    )
    session = _session(
        tmp_path,
        uploaded_doc_ids=["UPLOAD_strict"],
        metadata={"upload_collection_id": "owui-chat-strict", "collection_id": "owui-chat-strict"},
    )
    service = RequirementExtractionService(
        SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"),
        stores,
        session,
    )

    assert infer_requirement_mode("extract all requirements/ shall statements from the uploaded document") == BROAD_REQUIREMENT_MODE
    assert infer_requirement_mode("extract only shall statements from the uploaded document") == STRICT_SHALL_MODE

    result = service.extract_for_user_request("extract only shall statements from the uploaded document")

    assert result["mode"] == STRICT_SHALL_MODE
    assert result["statement_count"] == 2
    assert [row["modality"] for row in result["preview_rows"]] == ["shall", "shall_not"]


def test_export_requirement_statements_writes_csv_artifact(tmp_path: Path) -> None:
    kb_record = _record(
        doc_id="kb-spec",
        title="SPEC_B.md",
        source_type="kb",
        collection_id="rfp-corpus",
        file_type="md",
        source_path="/repo/SPEC_B.md",
    )
    stores = _stores(
        [kb_record],
        {
            "kb-spec": [
                ChunkRecord(
                    chunk_id="kb-spec#chunk0001",
                    doc_id="kb-spec",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    chunk_index=0,
                    content="The platform shall encrypt data at rest. The supplier shall not expose shared credentials.",
                    chunk_type="requirement",
                )
            ]
        },
    )
    session = _session(tmp_path, metadata={"kb_collection_id": "rfp-corpus", "available_kb_collection_ids": ["rfp-corpus"]})
    tools = {tool.name: tool for tool in make_requirement_tools(SimpleNamespace(default_collection_id="default", default_tenant_id="tenant", kb_dir=str(tmp_path / 'missing'), kb_extra_dirs=()), stores, session)}

    result = tools["export_requirement_statements"].invoke(
        {
            "source_scope": "kb",
            "collection_id": "rfp-corpus",
            "all_documents": True,
            "mode": STRICT_SHALL_MODE,
        }
    )

    assert result["statement_count"] == 2
    assert result["artifact"]["filename"] == "rfp-corpus__requirement_statements.csv"
    exported = session.workspace.read_text("rfp-corpus__requirement_statements.csv")
    assert "document_title,modality,location,statement_text" in exported
    assert "source_excerpt,source_location,source_structure,binding_strength,risk_label,risk_rationale" in exported
    assert "SPEC_B.md,shall" in exported


def test_extract_requirement_statements_reports_unsupported_spreadsheets(tmp_path: Path) -> None:
    workbook_record = _record(
        doc_id="upload-xlsx",
        title="Requirements.xlsx",
        source_type="upload",
        collection_id="upload-col",
        file_type="xlsx",
        source_path="/tmp/Requirements.xlsx",
    )
    stores = _stores([workbook_record], {"upload-xlsx": []})
    session = _session(
        tmp_path,
        uploaded_doc_ids=["upload-xlsx"],
        metadata={"upload_collection_id": "upload-col", "collection_id": "upload-col"},
    )
    tools = {tool.name: tool for tool in make_requirement_tools(SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"), stores, session)}

    result = tools["extract_requirement_statements"].invoke({"source_scope": "uploads"})

    assert "not supported" in result["error"].lower()
    assert result["unsupported_documents"][0]["file_type"] == "xlsx"


def test_extract_requirement_statements_requires_explicit_file_when_scope_is_ambiguous(tmp_path: Path) -> None:
    records = [
        _record(
            doc_id="upload-a",
            title="SPEC_A.docx",
            source_type="upload",
            collection_id="upload-col",
            file_type="docx",
            source_path="/tmp/SPEC_A.docx",
        ),
        _record(
            doc_id="upload-b",
            title="SPEC_B.pdf",
            source_type="upload",
            collection_id="upload-col",
            file_type="pdf",
            source_path="/tmp/SPEC_B.pdf",
        ),
    ]
    stores = _stores(records, {"upload-a": [], "upload-b": []})
    session = _session(
        tmp_path,
        uploaded_doc_ids=["upload-a", "upload-b"],
        metadata={"upload_collection_id": "upload-col", "collection_id": "upload-col"},
    )
    tools = {tool.name: tool for tool in make_requirement_tools(SimpleNamespace(default_collection_id="default", default_tenant_id="tenant"), stores, session)}

    result = tools["extract_requirement_statements"].invoke({"source_scope": "uploads"})

    assert "multiple documents" in result["error"].lower()
    assert len(result["candidate_documents"]) == 2


def test_extract_requirement_statements_all_documents_in_collection_returns_titles_and_locations(tmp_path: Path) -> None:
    records = [
        _record(
            doc_id="kb-a",
            title="SPEC_A.docx",
            source_type="kb",
            collection_id="rfp-corpus",
            file_type="docx",
            source_path="/repo/SPEC_A.docx",
        ),
        _record(
            doc_id="kb-b",
            title="SPEC_B.txt",
            source_type="kb",
            collection_id="rfp-corpus",
            file_type="txt",
            source_path="/repo/SPEC_B.txt",
        ),
    ]
    stores = _stores(
        records,
        {
            "kb-a": [
                ChunkRecord(
                    chunk_id="kb-a#chunk0001",
                    doc_id="kb-a",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    chunk_index=0,
                    content="The payload shall include a checksum.",
                    page_number=1,
                    clause_number="2.1",
                    section_title="Interface",
                    chunk_type="requirement",
                )
            ],
            "kb-b": [
                ChunkRecord(
                    chunk_id="kb-b#chunk0001",
                    doc_id="kb-b",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    chunk_index=2,
                    content="The service must not emit plaintext secrets.",
                    chunk_type="requirement",
                )
            ],
        },
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant",
        kb_dir=str(tmp_path / "missing"),
        kb_extra_dirs=(),
    )
    session = _session(tmp_path, metadata={"kb_collection_id": "rfp-corpus", "available_kb_collection_ids": ["rfp-corpus"]})
    tools = {tool.name: tool for tool in make_requirement_tools(settings, stores, session)}

    result = tools["extract_requirement_statements"].invoke(
        {"source_scope": "kb", "collection_id": "rfp-corpus", "all_documents": True}
    )

    assert result["statement_count"] == 2
    assert {row["document_title"] for row in result["preview_rows"]} == {"SPEC_A.docx", "SPEC_B.txt"}
    assert any("Page 1" in row["location"] for row in result["preview_rows"])


def test_backfill_requirement_statements_populates_existing_kb_docs_without_reingest(tmp_path: Path) -> None:
    kb_record = _record(
        doc_id="kb-req",
        title="SPEC_BACKFILL.md",
        source_type="kb",
        collection_id="rfp-corpus",
        file_type="md",
        source_path="/repo/SPEC_BACKFILL.md",
    )
    stores = _stores(
        [kb_record],
        {
            "kb-req": [
                ChunkRecord(
                    chunk_id="kb-req#chunk0001",
                    doc_id="kb-req",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    chunk_index=0,
                    content="The subsystem shall log all mode transitions.",
                    chunk_type="requirement",
                )
            ]
        },
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant",
        kb_dir=str(tmp_path / "missing"),
        kb_extra_dirs=(),
    )

    result = backfill_requirement_statements(
        settings,
        stores,
        tenant_id="tenant",
        collection_id="rfp-corpus",
        source_type="kb",
    )

    assert result.processed_doc_ids == ("kb-req",)
    assert result.statement_count == 1
    assert stores.requirement_store.has_doc_statements("kb-req", "tenant") is True


def test_runtime_kernel_runs_requirements_workflow_before_react(tmp_path: Path) -> None:
    upload_record = _record(
        doc_id="UPLOAD_req",
        title="Runtime_SPEC.docx",
        source_type="upload",
        collection_id="owui-chat-runtime",
        file_type="docx",
        source_path="/tmp/Runtime_SPEC.docx",
    )
    stores = _stores(
        [upload_record],
        {
            "UPLOAD_req": [
                ChunkRecord(
                    chunk_id="UPLOAD_req#chunk0001",
                    doc_id="UPLOAD_req",
                    tenant_id="tenant",
                    collection_id="owui-chat-runtime",
                    chunk_index=0,
                    content="The relay shall authenticate operators. The relay must not expose debug ports.",
                    chunk_type="requirement",
                )
            ]
        },
    )
    paths = RuntimePaths(
        runtime_root=tmp_path / "runtime",
        workspace_root=tmp_path / "workspaces",
        memory_root=tmp_path / "memory",
    )
    settings = SimpleNamespace(
        default_collection_id="default",
        default_tenant_id="tenant",
        agents_dir=Path("data") / "agents",
        skills_dir=Path("data") / "skills",
        runtime_dir=paths.runtime_root,
        workspace_dir=paths.workspace_root,
        memory_dir=paths.memory_root,
        kb_dir=str(tmp_path / "missing"),
        kb_extra_dirs=(),
        memory_enabled=False,
        session_hydrate_window_messages=40,
        session_transcript_page_size=100,
        max_worker_concurrency=1,
    )
    kernel = RuntimeKernel(settings, providers=SimpleNamespace(), stores=stores, paths=paths)
    state = SessionState(
        tenant_id="tenant",
        user_id="user",
        conversation_id="conv",
        session_id="tenant:user:conv",
        uploaded_doc_ids=["UPLOAD_req"],
        workspace_root=str(paths.workspace_dir("tenant:user:conv")),
        metadata={"upload_collection_id": "owui-chat-runtime", "collection_id": "owui-chat-runtime"},
    )
    agent = AgentDefinition(
        name="general",
        mode="react",
        description="General",
        prompt_file="general.md",
        skill_scope="general",
        allowed_tools=[],
    )

    result = kernel._maybe_run_requirements_extraction(
        agent,
        state,
        user_text="extract all requirements/ shall statements from the uploaded document",
    )

    assert result is not None
    assert result.metadata["turn_outcome"] == "requirements_extraction"
    assert result.metadata["sanitized_user_query"] == "extract all requirements/ shall statements from the uploaded document"
    assert result.metadata["selected_requirement_doc_ids"] == ["UPLOAD_req"]
    assert result.metadata["requirements_extraction"]["statement_count"] == 2
    assert "Runtime_SPEC.docx" in result.text
    assert "Source Text" in result.text
    assert len(result.metadata["artifacts"]) == 2
    assert result.messages[-1].artifact_refs
