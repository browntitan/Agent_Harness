from __future__ import annotations

import json
from typing import Callable

from langchain.tools import tool

from agentic_chatbot_next.rag.inventory import (
    build_kb_collection_access_payload,
    build_kb_file_inventory_payload,
    build_namespace_search_payload,
    build_session_access_inventory_payload,
)
from agentic_chatbot_next.rag.retrieval_scope import (
    resolve_kb_collection_id,
    resolve_upload_collection_id,
)


def _demo_group_for_title(title: str) -> str:
    lower = title.lower()
    if "runbook" in lower or "playbook" in lower:
        return "runbooks"
    if lower.startswith("api_") or "api" in lower:
        return "api_references"
    if any(token in lower for token in ("agreement", "contract", "addendum", "schedule", "msa", "dpa")):
        return "contracts"
    if any(token in lower for token in ("security", "privacy", "compliance", "control", "incident")):
        return "security_compliance"
    return "other"


def make_list_docs_tool(settings: object, stores: object, session: object) -> Callable:
    @tool
    def list_indexed_docs(source_type: str = "", view: str = "", collection_id: str = "", query: str = "") -> str:
        """List indexed documents, KB collections, session-scoped access, or namespace matches, including prompts like "what's indexed", "what docs are in the knowledge base", "what knowledge bases do you have access to", "what documents do we have access to", or "what documents are in the default collection"."""

        tenant_id = getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))
        normalized_view = str(view or "").strip().lower()
        if normalized_view == "session_access":
            return json.dumps(
                build_session_access_inventory_payload(settings, stores, session),
                ensure_ascii=False,
                indent=2,
            )
        if normalized_view == "kb_collections":
            return json.dumps(
                build_kb_collection_access_payload(settings, stores, session),
                ensure_ascii=False,
                indent=2,
            )
        if normalized_view == "namespace_search":
            return json.dumps(
                build_namespace_search_payload(
                    settings,
                    stores,
                    session,
                    namespace_query=str(query or "").strip(),
                ),
                ensure_ascii=False,
                indent=2,
            )
        normalized_source_type = str(source_type or "").strip().lower()
        requested_collection_id = str(collection_id or "").strip()
        collection_id = ""
        if normalized_source_type == "kb":
            payload = build_kb_file_inventory_payload(
                settings,
                stores,
                session,
                collection_id=requested_collection_id or resolve_kb_collection_id(settings, session),
            )
            if getattr(session, "demo_mode", False) and payload.get("requested_collection_available", True):
                grouped = {
                    "contracts": [],
                    "security_compliance": [],
                    "runbooks": [],
                    "api_references": [],
                    "other": [],
                }
                for record in payload.get("documents") or []:
                    grouped[_demo_group_for_title(str(record.get("title") or ""))].append(
                        {"doc_id": record.get("doc_id"), "title": record.get("title")}
                    )
                for key in grouped:
                    grouped[key] = sorted(grouped[key], key=lambda item: str(item.get("title") or "").lower())
                return json.dumps(
                    {
                        "view": "kb_file_inventory",
                        "kb_collection_id": payload.get("kb_collection_id"),
                        "requested_collection_available": payload.get("requested_collection_available", True),
                        "session_kb_collection_id": payload.get("session_kb_collection_id"),
                        "total_documents": len(payload.get("documents") or []),
                        "groups": grouped,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(payload, ensure_ascii=False, indent=2)
        elif normalized_source_type == "upload":
            collection_id = requested_collection_id or resolve_upload_collection_id(settings, session)
        records = stores.doc_store.list_documents(
            source_type=normalized_source_type,
            tenant_id=tenant_id,
            collection_id=collection_id,
        )
        if getattr(session, "demo_mode", False):
            grouped = {
                "contracts": [],
                "security_compliance": [],
                "runbooks": [],
                "api_references": [],
                "other": [],
            }
            for record in records:
                grouped[_demo_group_for_title(record.title)].append(
                    {"doc_id": record.doc_id, "title": record.title}
                )
            for key in grouped:
                grouped[key] = sorted(grouped[key], key=lambda item: item["title"].lower())
            return json.dumps(
                {
                    "total_documents": len(records),
                    "source_type_filter": source_type or "all",
                    "groups": grouped,
                },
                ensure_ascii=False,
            )

        docs = [
            {
                "doc_id": record.doc_id,
                "title": record.title,
                "source_type": record.source_type,
                "num_chunks": record.num_chunks,
                "file_type": record.file_type,
                "doc_structure_type": record.doc_structure_type,
            }
            for record in records
        ]
        return json.dumps(sorted(docs, key=lambda item: item["doc_id"]), ensure_ascii=False, indent=2)

    return list_indexed_docs
