from __future__ import annotations

from datetime import datetime, timezone
import json
from types import SimpleNamespace

from agentic_chatbot_next.persistence.postgres.graphs import GraphIndexRecord, GraphIndexSourceRecord
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_FILE,
    INVENTORY_QUERY_GRAPH_INDEXES,
    INVENTORY_QUERY_KB_FILE,
    INVENTORY_QUERY_KB_COLLECTIONS,
    INVENTORY_QUERY_NONE,
    INVENTORY_QUERY_SESSION_ACCESS,
    build_graph_document_inventory_payload,
    classify_inventory_query,
    dispatch_authoritative_inventory,
    sync_session_kb_collection_state,
)
from agentic_chatbot_next.tools.groups.graph_gateway import build_graph_gateway_tools
from agentic_chatbot_next.tools.list_docs import make_list_docs_tool


class _DocStore:
    def __init__(self, records, *, collection_summaries=None, hidden_collection_ids=None):
        self._records = list(records)
        self._collection_summaries = list(collection_summaries or [])
        self._hidden_collection_ids = {
            str(item or "").strip()
            for item in (hidden_collection_ids or [])
            if str(item or "").strip()
        }

    def list_documents(self, *, source_type: str = "", tenant_id: str = "tenant", collection_id: str = ""):
        if collection_id and str(collection_id or "").strip() in self._hidden_collection_ids:
            return []
        matches = []
        for record in self._records:
            if str(getattr(record, "tenant_id", "tenant") or "tenant") != tenant_id:
                continue
            if source_type and str(getattr(record, "source_type", "") or "") != source_type:
                continue
            if collection_id and str(getattr(record, "collection_id", "") or "") != collection_id:
                continue
            matches.append(record)
        return matches

    def list_collections(self, tenant_id: str = "tenant"):
        if self._collection_summaries:
            return list(self._collection_summaries)
        grouped = {}
        for record in self._records:
            if str(getattr(record, "tenant_id", "tenant") or "tenant") != tenant_id:
                continue
            collection_id = str(getattr(record, "collection_id", "") or "")
            if not collection_id:
                continue
            item = grouped.setdefault(
                collection_id,
                {
                    "collection_id": collection_id,
                    "document_count": 0,
                    "latest_ingested_at": None,
                    "source_type_counts": {},
                },
            )
            item["document_count"] = int(item.get("document_count") or 0) + 1
            counts = dict(item.get("source_type_counts") or {})
            source_type = str(getattr(record, "source_type", "") or "unknown")
            counts[source_type] = int(counts.get(source_type) or 0) + 1
            item["source_type_counts"] = counts
        return [grouped[key] for key in sorted(grouped)]


class _GraphIndexStore:
    def __init__(self, records):
        self._records = list(records)

    def list_indexes(self, *, tenant_id="tenant", user_id="", collection_id="", status="", backend="", limit=100):
        del user_id, status, backend
        matches = [
            record
            for record in self._records
            if str(getattr(record, "tenant_id", "tenant") or "tenant") == tenant_id
            and (not collection_id or str(getattr(record, "collection_id", "") or "") == collection_id)
        ]
        return matches[:limit]

    def get_index(self, graph_id, tenant_id="tenant", user_id=""):
        del user_id
        for record in self._records:
            if str(getattr(record, "tenant_id", "tenant") or "tenant") != tenant_id:
                continue
            if str(getattr(record, "graph_id", "") or "") == str(graph_id or ""):
                return record
        return None


class _GraphSourceStore:
    def __init__(self, records):
        self._records = list(records)

    def list_sources(self, graph_id, *, tenant_id="tenant"):
        return [
            record
            for record in self._records
            if str(getattr(record, "tenant_id", "tenant") or "tenant") == tenant_id
            and str(getattr(record, "graph_id", "") or "") == str(graph_id or "")
        ]


def _settings() -> SimpleNamespace:
    return SimpleNamespace(default_collection_id="default", default_tenant_id="tenant")


def test_classify_inventory_query_distinguishes_session_access_and_kb_inventory() -> None:
    assert classify_inventory_query("what documents do we have access to") == INVENTORY_QUERY_SESSION_ACCESS
    assert classify_inventory_query("what docs do you have") == INVENTORY_QUERY_SESSION_ACCESS
    assert classify_inventory_query("what knowledge bases do you have access to") == INVENTORY_QUERY_KB_COLLECTIONS
    assert classify_inventory_query("list out the kbs i have access to") == INVENTORY_QUERY_KB_COLLECTIONS
    assert classify_inventory_query("what docs are in the knowledge base") == INVENTORY_QUERY_KB_FILE
    assert classify_inventory_query("what knowledge base documents are available") == INVENTORY_QUERY_KB_FILE
    assert classify_inventory_query("what's indexed") == INVENTORY_QUERY_KB_FILE
    assert classify_inventory_query("can you list out all of the documents in the default collection") == INVENTORY_QUERY_KB_FILE
    assert classify_inventory_query("what documents are in the default collection") == INVENTORY_QUERY_KB_FILE
    assert classify_inventory_query("show the files in collection default") == INVENTORY_QUERY_KB_FILE
    assert classify_inventory_query("list the individual files inside the default KB") == INVENTORY_QUERY_KB_FILE
    assert (
        classify_inventory_query("Use the indexed knowledge base inventory tools to list the knowledge base collections this session can access.")
        == INVENTORY_QUERY_KB_COLLECTIONS
    )


def test_classify_inventory_query_distinguishes_graph_inventory() -> None:
    assert classify_inventory_query("what knowledge graphs do i have available to me") == INVENTORY_QUERY_GRAPH_INDEXES
    assert classify_inventory_query("which knowledge graphs are available") == INVENTORY_QUERY_GRAPH_INDEXES
    assert classify_inventory_query("list my graph indexes") == INVENTORY_QUERY_GRAPH_INDEXES
    assert classify_inventory_query("what graphs exist") == INVENTORY_QUERY_GRAPH_INDEXES
    assert (
        classify_inventory_query("Use the knowledge graph inventory tools to list the knowledge graphs this session can access.")
        == INVENTORY_QUERY_GRAPH_INDEXES
    )


def test_classify_inventory_query_routes_bare_namespace_doc_list_queries_to_inventory() -> None:
    assert (
        classify_inventory_query("can you list out the documents in rfp-corpus. I just want to know the titles of the documents")
        == INVENTORY_QUERY_KB_FILE
    )


def test_classify_inventory_query_distinguishes_graph_document_inventory() -> None:
    assert classify_inventory_query("what documents are in the rfp-corpus graph") == INVENTORY_QUERY_GRAPH_FILE
    assert (
        classify_inventory_query("Search the defense_rag_v2_graph knowledge graph inventory and list the source documents in that graph.")
        == INVENTORY_QUERY_GRAPH_FILE
    )


def test_classify_inventory_query_rejects_filtered_discovery_prompts() -> None:
    assert classify_inventory_query("which documents contain onboarding workflows?") == INVENTORY_QUERY_NONE
    assert classify_inventory_query("identify all documents that mention onboarding") == INVENTORY_QUERY_NONE
    assert classify_inventory_query("what docs are available about onboarding") == INVENTORY_QUERY_NONE
    assert classify_inventory_query("which knowledge base documents mention onboarding?") == INVENTORY_QUERY_NONE
    assert classify_inventory_query("which documents in the default collection mention onboarding?") == INVENTORY_QUERY_NONE
    assert classify_inventory_query("which documents in the default collection describe the major subsystems of this repo?") == INVENTORY_QUERY_NONE
    assert classify_inventory_query("show the files in collection default that discuss routing") == INVENTORY_QUERY_NONE
    assert classify_inventory_query("what knowledge graphs are available about onboarding?") == INVENTORY_QUERY_NONE


def test_list_indexed_docs_session_access_view_scopes_to_current_chat() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"upload_collection_id": "owui-chat-1", "kb_collection_id": "default"},
        uploaded_doc_ids=["upload-1"],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-arch",
                    title="ARCHITECTURE.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                ),
                SimpleNamespace(
                    doc_id="upload-1",
                    title="contract.pdf",
                    source_type="upload",
                    collection_id="owui-chat-1",
                    tenant_id="tenant",
                    num_chunks=4,
                    file_type="pdf",
                    doc_structure_type="general",
                    source_path="/uploads/contract.pdf",
                ),
                SimpleNamespace(
                    doc_id="upload-other",
                    title="other.pdf",
                    source_type="upload",
                    collection_id="owui-chat-2",
                    tenant_id="tenant",
                    num_chunks=3,
                    file_type="pdf",
                    doc_structure_type="general",
                    source_path="/uploads/other.pdf",
                ),
            ]
        )
    )
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"view": "session_access"}))

    assert result["view"] == "session_access"
    assert result["kb_collection_id"] == "default"
    assert result["kb_doc_count"] == 1
    assert result["upload_collection_id"] == "owui-chat-1"
    assert result["has_uploads"] is True
    assert [item["doc_id"] for item in result["uploaded_documents"]] == ["upload-1"]
    assert result["next_actions"] == [
        "search uploaded docs",
        "search the KB",
        "search both",
        "list KB files",
    ]


def test_list_indexed_docs_kb_collections_view_returns_all_visible_collections() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"kb_collection_id": "default"},
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-arch",
                    title="Architecture Overview.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                ),
                SimpleNamespace(
                    doc_id="doc-pricing",
                    title="Pricing.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=5,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/Pricing.md",
                ),
                SimpleNamespace(
                    doc_id="doc-rfp",
                    title="RFP Overview.docx",
                    source_type="host_path",
                    collection_id="rfp-corpus",
                    tenant_id="tenant",
                    num_chunks=3,
                    file_type="docx",
                    doc_structure_type="general",
                    source_path="/repo/docs/RFP Overview.docx",
                ),
            ]
        ),
        graph_index_store=SimpleNamespace(
            list_indexes=lambda **kwargs: [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                    domain_summary="Graph index for cross-document RFP entity and requirement analysis",
                    source_doc_ids=["doc-rfp"],
                )
            ]
        ),
    )
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"view": "kb_collections"}))

    assert result["view"] == "kb_collections"
    assert [item["collection_id"] for item in result["collections"]] == ["default", "rfp-corpus"]
    assert result["collections"][0]["kb_doc_count"] == 2
    assert result["collections"][0]["graph_count"] == 0
    assert result["collections"][0]["summary_mode"] == "indexed"
    assert result["collections"][0]["summary_topics"] == ["product overviews", "pricing", "architecture"]
    assert (
        result["collections"][0]["summary"]
        == "The default knowledge-base collection - 2 indexed documents covering product overviews, pricing, and architecture."
    )
    assert result["collections"][1]["kb_doc_count"] == 1
    assert result["collections"][1]["graph_count"] == 1
    assert result["collections"][1]["summary_mode"] == "indexed"
    assert result["collections"][1]["summary"] == (
        "A knowledge-base collection - 1 indexed document covering product overviews."
    )
    assert result["graphs"] == [
        {
            "graph_id": "rfp_corpus",
            "display_name": "RFP Corpus Graph",
            "collection_id": "rfp-corpus",
            "status": "ready",
            "backend": "microsoft_graphrag",
            "query_ready": True,
            "domain_summary": "Graph index for cross-document RFP entity and requirement analysis",
            "source_document_count": 1,
            "summary_mode": "domain_summary",
            "summary": "Graph index for cross-document RFP entity and requirement analysis.",
        }
    ]


def test_list_indexed_docs_kb_collections_view_serializes_datetime_collection_summaries() -> None:
    latest_ingested_at = datetime(2026, 4, 20, 12, 34, 56, tzinfo=timezone.utc)
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"kb_collection_id": "default"},
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-arch",
                    title="Architecture Overview.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                )
            ],
            collection_summaries=[
                {
                    "collection_id": "default",
                    "document_count": 1,
                    "latest_ingested_at": latest_ingested_at,
                    "source_type_counts": {"kb": 1},
                }
            ],
        ),
        graph_index_store=_GraphIndexStore([]),
    )
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"view": "kb_collections"}))

    assert result["collections"][0]["latest_ingested_at"] == latest_ingested_at.isoformat()


def test_dispatch_authoritative_inventory_returns_json_safe_payloads_for_kb_access() -> None:
    latest_ingested_at = datetime(2026, 4, 20, 12, 34, 56, tzinfo=timezone.utc)
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"kb_collection_id": "default"},
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-arch",
                    title="Architecture Overview.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                )
            ],
            collection_summaries=[
                {
                    "collection_id": "default",
                    "document_count": 1,
                    "latest_ingested_at": latest_ingested_at,
                    "source_type_counts": {"kb": 1},
                }
            ],
        ),
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="default",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                    domain_summary="Graph index metadata",
                    source_doc_ids=["doc-arch"],
                )
            ]
        ),
        graph_source_store=_GraphSourceStore([]),
    )

    dispatched = dispatch_authoritative_inventory(
        _settings(),
        stores,
        session,
        query="what knowledge bases do i have access to",
        query_type=INVENTORY_QUERY_KB_COLLECTIONS,
    )

    assert dispatched["handled"] is True
    assert dispatched["payload"]["collections"][0]["latest_ingested_at"] == latest_ingested_at.isoformat()
    json.dumps(dispatched, ensure_ascii=False)


def test_list_indexed_docs_namespace_search_groups_collection_and_graph_matches() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"kb_collection_id": "default"},
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-default",
                    title="ARCHITECTURE.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                ),
                SimpleNamespace(
                    doc_id="doc-rfp",
                    title="RFP Overview.docx",
                    source_type="host_path",
                    collection_id="rfp-corpus",
                    tenant_id="tenant",
                    num_chunks=9,
                    file_type="docx",
                    doc_structure_type="general",
                    source_path="/corpora/rfp/RFP Overview.docx",
                ),
            ]
        ),
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                )
            ]
        ),
    )
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"view": "namespace_search", "query": "rfp-corpus"}))

    assert result["view"] == "namespace_search"
    assert result["namespace_query"] == "rfp-corpus"
    assert result["collections"][0]["namespace_id"] == "rfp-corpus"
    assert result["graphs"][0]["graph_id"] == "rfp_corpus"


def test_list_indexed_docs_kb_collections_view_reports_visible_zero_doc_collection() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"kb_collection_id": "default"},
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [],
            collection_summaries=[],
        ),
        collection_store=SimpleNamespace(
            list_collections=lambda tenant_id="tenant": [
                SimpleNamespace(collection_id="default"),
                SimpleNamespace(collection_id="defense-rag-test"),
            ]
        ),
    )
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"view": "kb_collections"}))

    defense = next(item for item in result["collections"] if item["collection_id"] == "defense-rag-test")
    assert defense["kb_doc_count"] == 0
    assert defense["summary_mode"] == "empty"
    assert defense["summary"] == (
        "A test knowledge-base collection. It is listed as available but currently has no indexed documents shown in the inventory."
    )


def test_list_indexed_docs_kb_collections_view_reports_nonenumerable_collection_files() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={
            "kb_collection_id": "rfp-corpus",
            "access_summary": {
                "authz_enabled": True,
                "session_upload_collection_id": "",
                "resources": {
                    "collection": {"use": ["rfp-corpus"], "manage": [], "use_all": False, "manage_all": False},
                    "graph": {"use": [], "manage": [], "use_all": False, "manage_all": False},
                    "tool": {"use": [], "manage": [], "use_all": False, "manage_all": False},
                    "skill_family": {"use": [], "manage": [], "use_all": False, "manage_all": False},
                },
            },
        },
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [],
            collection_summaries=[
                {
                    "collection_id": "rfp-corpus",
                    "document_count": 3,
                    "latest_ingested_at": "2026-04-20T00:00:00Z",
                    "source_type_counts": {"host_path": 3},
                }
            ],
            hidden_collection_ids=["rfp-corpus"],
        ),
        collection_store=SimpleNamespace(
            list_collections=lambda tenant_id="tenant": [SimpleNamespace(collection_id="rfp-corpus")]
        ),
    )
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"view": "kb_collections"}))

    assert result["collections"] == [
        {
            "collection_id": "rfp-corpus",
            "maintenance_policy": "indexed_documents",
            "kb_available": False,
            "kb_doc_count": 3,
            "latest_ingested_at": "2026-04-20T00:00:00Z",
            "source_type_counts": {"host_path": 3},
            "graph_count": 0,
            "title_samples": [],
            "summary_topics": [],
            "summary_mode": "not_enumerated",
            "summary": (
                "A knowledge-base collection. "
                "It is listed as available, but the inventory payload does not enumerate its individual files."
            ),
        }
    ]


def test_list_indexed_docs_kb_source_scopes_to_requested_collection() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"kb_collection_id": "default"},
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-arch",
                    title="ARCHITECTURE.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                ),
                SimpleNamespace(
                    doc_id="doc-other",
                    title="Other.md",
                    source_type="kb",
                    collection_id="other",
                    tenant_id="tenant",
                    num_chunks=3,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/Other.md",
                ),
            ]
        )
    )
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"source_type": "kb", "collection_id": "default"}))

    assert result["view"] == "kb_file_inventory"
    assert result["kb_collection_id"] == "default"
    assert result["requested_collection_available"] is True
    assert [item["doc_id"] for item in result["documents"]] == ["doc-arch"]


def test_list_indexed_docs_kb_source_reports_unavailable_collection() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"kb_collection_id": "default"},
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(doc_store=_DocStore([]))
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"source_type": "kb", "collection_id": "other"}))

    assert result["view"] == "kb_file_inventory"
    assert result["kb_collection_id"] == "other"
    assert result["requested_collection_available"] is False
    assert result["session_kb_collection_id"] == "default"
    assert result["available_kb_collection_ids"] == ["default"]
    assert result["documents"] == []


def test_list_indexed_docs_kb_source_scopes_to_visible_host_path_collection() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"kb_collection_id": "default"},
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-default",
                    title="ARCHITECTURE.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                    source_display_path="ARCHITECTURE.md",
                ),
                SimpleNamespace(
                    doc_id="doc-rfp",
                    title="asterion_ecp_04_rev_c.docx",
                    source_type="host_path",
                    collection_id="rfp-corpus",
                    tenant_id="tenant",
                    num_chunks=9,
                    file_type="docx",
                    doc_structure_type="general",
                    source_path="/corpora/rfp/asterion_ecp_04_rev_c.docx",
                    source_display_path="docx/asterion_ecp_04_rev_c.docx",
                ),
            ]
        )
    )
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"source_type": "kb", "collection_id": "rfp-corpus"}))

    assert result["view"] == "kb_file_inventory"
    assert result["kb_collection_id"] == "rfp-corpus"
    assert result["requested_collection_available"] is True
    assert [item["doc_id"] for item in result["documents"]] == ["doc-rfp"]
    assert result["documents"][0]["source_type"] == "host_path"


def test_list_indexed_docs_kb_collections_view_only_lists_graphs_allowed_by_access_grants() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        user_id="user-1",
        metadata={
            "kb_collection_id": "default",
            "access_summary": {
                "authz_enabled": True,
                "session_upload_collection_id": "",
                "resources": {
                    "collection": {"use": ["default"], "manage": [], "use_all": False, "manage_all": False},
                    "graph": {"use": ["allowed_graph", "wrong_collection_graph"], "manage": [], "use_all": False, "manage_all": False},
                    "tool": {"use": [], "manage": [], "use_all": False, "manage_all": False},
                    "skill_family": {"use": [], "manage": [], "use_all": False, "manage_all": False},
                },
            },
        },
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-default",
                    title="Architecture.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                ),
                SimpleNamespace(
                    doc_id="doc-rfp",
                    title="RFP Overview.docx",
                    source_type="host_path",
                    collection_id="rfp-corpus",
                    tenant_id="tenant",
                    num_chunks=4,
                    file_type="docx",
                    doc_structure_type="general",
                    source_path="/repo/docs/RFP Overview.docx",
                ),
            ]
        ),
        graph_index_store=SimpleNamespace(
            list_indexes=lambda **kwargs: [
                GraphIndexRecord(
                    graph_id="allowed_graph",
                    tenant_id="tenant",
                    collection_id="default",
                    display_name="Allowed Graph",
                    status="ready",
                    query_ready=True,
                    source_doc_ids=["doc-default"],
                ),
                GraphIndexRecord(
                    graph_id="missing_graph_grant",
                    tenant_id="tenant",
                    collection_id="default",
                    display_name="Missing Graph Grant",
                    status="draft",
                    query_ready=False,
                ),
                GraphIndexRecord(
                    graph_id="wrong_collection_graph",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="Wrong Collection Graph",
                    status="ready",
                    query_ready=True,
                ),
            ]
        ),
    )
    tool = make_list_docs_tool(_settings(), stores, session)

    result = json.loads(tool.invoke({"view": "kb_collections"}))

    assert [item["collection_id"] for item in result["collections"]] == ["default"]
    assert result["graphs"] == [
        {
            "graph_id": "allowed_graph",
            "display_name": "Allowed Graph",
            "collection_id": "default",
            "status": "ready",
            "backend": "microsoft_graphrag",
            "query_ready": True,
            "domain_summary": "",
            "source_document_count": 1,
            "summary_mode": "fallback",
            "summary": "Graph index over default, query-ready, covering 1 source document.",
        }
    ]


def test_kb_collection_access_inventory_excludes_current_and_previous_chat_uploads() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        user_id="user-1",
        metadata={
            "collection_id": "owui-chat-1",
            "upload_collection_id": "owui-chat-1",
            "kb_collection_id": "default",
            "access_summary": {
                "authz_enabled": True,
                "session_upload_collection_id": "owui-chat-1",
                "resources": {
                    "collection": {
                        "use": ["default", "owui-chat-1", "owui-chat-old"],
                        "manage": [],
                        "use_all": False,
                        "manage_all": False,
                    },
                    "graph": {
                        "use": ["kb_graph", "current_upload_graph", "old_upload_graph"],
                        "manage": [],
                        "use_all": False,
                        "manage_all": False,
                    },
                    "tool": {"use": [], "manage": [], "use_all": False, "manage_all": False},
                    "skill_family": {"use": [], "manage": [], "use_all": False, "manage_all": False},
                },
            },
        },
        uploaded_doc_ids=["upload-current"],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-default",
                    title="Architecture.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                ),
                SimpleNamespace(
                    doc_id="upload-current",
                    title="current-contract.pdf",
                    source_type="upload",
                    collection_id="owui-chat-1",
                    tenant_id="tenant",
                    num_chunks=4,
                    file_type="pdf",
                    doc_structure_type="general",
                    source_path="/uploads/current-contract.pdf",
                ),
                SimpleNamespace(
                    doc_id="upload-old",
                    title="old-contract.pdf",
                    source_type="upload",
                    collection_id="owui-chat-old",
                    tenant_id="tenant",
                    num_chunks=3,
                    file_type="pdf",
                    doc_structure_type="general",
                    source_path="/uploads/old-contract.pdf",
                ),
            ]
        ),
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="kb_graph",
                    tenant_id="tenant",
                    collection_id="default",
                    display_name="KB Graph",
                    status="ready",
                    query_ready=True,
                    source_doc_ids=["doc-default"],
                ),
                GraphIndexRecord(
                    graph_id="current_upload_graph",
                    tenant_id="tenant",
                    collection_id="owui-chat-1",
                    display_name="Current Upload Graph",
                    status="ready",
                    query_ready=True,
                    source_doc_ids=["upload-current"],
                ),
                GraphIndexRecord(
                    graph_id="old_upload_graph",
                    tenant_id="tenant",
                    collection_id="owui-chat-old",
                    display_name="Old Upload Graph",
                    status="ready",
                    query_ready=True,
                    source_doc_ids=["upload-old"],
                ),
            ]
        ),
    )

    dispatched = dispatch_authoritative_inventory(
        _settings(),
        stores,
        session,
        query="what knowledge bases do i have access to",
        query_type=INVENTORY_QUERY_KB_COLLECTIONS,
    )

    assert dispatched["handled"] is True
    assert [item["collection_id"] for item in dispatched["payload"]["collections"]] == ["default"]
    assert session.metadata["available_kb_collection_ids"] == ["default"]
    assert [item["graph_id"] for item in dispatched["payload"]["graphs"]] == ["kb_graph"]
    answer = dispatched["answer"]["answer"]
    assert "KB Graph (`kb_graph`)" in answer
    assert "current-contract.pdf" not in answer
    assert "old-contract.pdf" not in answer
    assert "owui-chat-1" not in answer
    assert "owui-chat-old" not in answer
    assert "Current Upload Graph" not in answer
    assert "Old Upload Graph" not in answer

    session_access = json.loads(make_list_docs_tool(_settings(), stores, session).invoke({"view": "session_access"}))
    assert session_access["view"] == "session_access"
    assert session_access["available_kb_collection_ids"] == ["default"]
    assert [item["doc_id"] for item in session_access["uploaded_documents"]] == ["upload-current"]


def test_sync_session_kb_collection_state_promotes_clarified_collection_choice() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={
            "kb_collection_id": "default",
            "kb_collection_confirmed": False,
            "pending_clarification": {
                "reason": "kb_collection_selection",
                "options": ["default", "rfp-corpus"],
            },
        },
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-default",
                    title="ARCHITECTURE.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                ),
                SimpleNamespace(
                    doc_id="doc-rfp",
                    title="asterion_ecp_04_rev_c.docx",
                    source_type="host_path",
                    collection_id="rfp-corpus",
                    tenant_id="tenant",
                    num_chunks=9,
                    file_type="docx",
                    doc_structure_type="general",
                    source_path="/corpora/rfp/asterion_ecp_04_rev_c.docx",
                ),
            ]
        )
    )

    payload = sync_session_kb_collection_state(_settings(), stores, session, query="rfp-corpus")

    assert payload["kb_collection_id"] == "rfp-corpus"
    assert payload["kb_collection_confirmed"] is True
    assert payload["selected_kb_collection_id"] == "rfp-corpus"
    assert session.metadata["kb_collection_id"] == "rfp-corpus"
    assert session.metadata["search_collection_ids"] == ["rfp-corpus"]


def test_sync_session_kb_collection_state_records_pending_namespace_candidates_for_ambiguous_bare_namespace() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={"kb_collection_id": "default", "kb_collection_confirmed": False},
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-default",
                    title="ARCHITECTURE.md",
                    source_type="kb",
                    collection_id="default",
                    tenant_id="tenant",
                    num_chunks=12,
                    file_type="md",
                    doc_structure_type="general",
                    source_path="/repo/docs/ARCHITECTURE.md",
                ),
                SimpleNamespace(
                    doc_id="doc-rfp",
                    title="RFP Overview.docx",
                    source_type="host_path",
                    collection_id="rfp-corpus",
                    tenant_id="tenant",
                    num_chunks=9,
                    file_type="docx",
                    doc_structure_type="general",
                    source_path="/corpora/rfp/RFP Overview.docx",
                ),
            ]
        ),
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                )
            ]
        ),
    )

    payload = sync_session_kb_collection_state(
        _settings(),
        stores,
        session,
        query="can you list out the documents in rfp-corpus. I just want to know the titles of the documents",
    )

    assert payload["namespace_mode"] == "clarify"
    assert payload["namespace_query"] == "rfp-corpus"
    assert session.metadata["pending_namespace_candidates"]["collections"][0]["namespace_id"] == "rfp-corpus"
    assert session.metadata["pending_namespace_candidates"]["graphs"][0]["graph_id"] == "rfp_corpus"


def test_sync_session_kb_collection_state_resolves_use_all_namespace_selection() -> None:
    session = SimpleNamespace(
        tenant_id="tenant",
        metadata={
            "kb_collection_id": "default",
            "kb_collection_confirmed": False,
            "pending_clarification": {
                "reason": "namespace_scope_selection",
                "options": ["rfp-corpus", "rfp_corpus", "collections only", "graphs only", "use all"],
            },
            "pending_namespace_candidates": {
                "namespace_query": "rfp-corpus",
                "collections": [
                    {"namespace_id": "rfp-corpus", "display_name": "rfp-corpus", "score": 1.0, "exactness": "exact"}
                ],
                "graphs": [
                    {
                        "namespace_id": "rfp_corpus",
                        "graph_id": "rfp_corpus",
                        "display_name": "RFP Corpus Graph",
                        "collection_id": "rfp-corpus",
                        "score": 0.96,
                        "exactness": "normalized_exact",
                    }
                ],
            },
        },
        uploaded_doc_ids=[],
    )
    stores = SimpleNamespace(
        doc_store=_DocStore(
            [
                SimpleNamespace(
                    doc_id="doc-rfp",
                    title="RFP Overview.docx",
                    source_type="host_path",
                    collection_id="rfp-corpus",
                    tenant_id="tenant",
                    num_chunks=9,
                    file_type="docx",
                    doc_structure_type="general",
                    source_path="/corpora/rfp/RFP Overview.docx",
                ),
            ]
        ),
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                )
            ]
        ),
    )

    payload = sync_session_kb_collection_state(_settings(), stores, session, query="use all")

    assert payload["selected_kb_collection_ids"] == ["rfp-corpus"]
    assert payload["selected_graph_ids"] == ["rfp_corpus"]
    assert session.metadata["search_collection_ids"] == ["rfp-corpus"]
    assert session.metadata["active_graph_ids"] == ["rfp_corpus"]


def test_build_graph_document_inventory_payload_sorts_and_dedupes_sources() -> None:
    session = SimpleNamespace(tenant_id="tenant", metadata={"kb_collection_id": "default"}, uploaded_doc_ids=[])
    stores = SimpleNamespace(
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                )
            ]
        ),
        graph_source_store=_GraphSourceStore(
            [
                GraphIndexSourceRecord(
                    graph_source_id="src-1",
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    source_doc_id="DOC-2",
                    source_path="/corpora/rfp/B.pdf",
                    source_title="B.pdf",
                    source_type="host_path",
                ),
                GraphIndexSourceRecord(
                    graph_source_id="src-2",
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    source_doc_id="DOC-1",
                    source_path="/corpora/rfp/A.pdf",
                    source_title="A.pdf",
                    source_type="host_path",
                ),
                GraphIndexSourceRecord(
                    graph_source_id="src-3",
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    source_doc_id="DOC-1",
                    source_path="/corpora/rfp/A.pdf",
                    source_title="A.pdf",
                    source_type="host_path",
                ),
            ]
        ),
    )

    payload = build_graph_document_inventory_payload(_settings(), stores, session, graph_id="rfp_corpus")

    assert payload["view"] == "graph_file_inventory"
    assert payload["graph_id"] == "rfp_corpus"
    assert [item["title"] for item in payload["documents"]] == ["A.pdf", "B.pdf"]


def test_list_graph_documents_tool_returns_graph_source_inventory() -> None:
    session = SimpleNamespace(tenant_id="tenant", metadata={"kb_collection_id": "default"}, uploaded_doc_ids=[])
    stores = SimpleNamespace(
        graph_index_store=_GraphIndexStore(
            [
                GraphIndexRecord(
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    collection_id="rfp-corpus",
                    display_name="RFP Corpus Graph",
                    status="ready",
                    query_ready=True,
                )
            ]
        ),
        graph_source_store=_GraphSourceStore(
            [
                GraphIndexSourceRecord(
                    graph_source_id="src-1",
                    graph_id="rfp_corpus",
                    tenant_id="tenant",
                    source_doc_id="DOC-1",
                    source_path="/corpora/rfp/A.pdf",
                    source_title="A.pdf",
                    source_type="host_path",
                )
            ]
        ),
    )
    ctx = SimpleNamespace(settings=_settings(), stores=stores, session_handle=session, progress_emitter=None, kernel=None, session=None)
    tools = {tool.name: tool for tool in build_graph_gateway_tools(ctx)}

    result = json.loads(tools["list_graph_documents"].invoke({"graph_id": "rfp_corpus"}))

    assert result["view"] == "graph_file_inventory"
    assert result["graph_id"] == "rfp_corpus"
    assert result["documents"][0]["title"] == "A.pdf"
