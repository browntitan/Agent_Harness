from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


class StructuredSearchAdapter:
    """Read-only allowlisted structured lookup surface over PostgreSQL-backed stores."""

    def __init__(self, stores: Any, *, settings: Any, session: Any) -> None:
        self.stores = stores
        self.settings = settings
        self.session = session
        self.tenant_id = str(getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev")) or "local-dev")
        self.user_id = str(getattr(session, "user_id", getattr(settings, "default_user_id", "")) or "")

    def allowed_views(self) -> List[str]:
        configured = [str(item) for item in getattr(self.settings, "graph_sql_allowed_views", ()) if str(item)]
        if configured:
            return configured
        return ["documents", "graph_indexes"]

    def search_documents(self, query: str, *, collection_id: str = "", limit: int = 6) -> List[Dict[str, Any]]:
        if "documents" not in set(self.allowed_views()):
            return []
        clean_query = str(query or "").strip()
        if not clean_query:
            return []

        doc_store = getattr(self.stores, "doc_store", None)
        if doc_store is None:
            return []

        matches: List[Dict[str, Any]] = []
        seen: set[str] = set()

        try:
            fuzzy = doc_store.fuzzy_search_title(
                clean_query,
                self.tenant_id,
                limit=max(1, int(limit)),
                collection_id=collection_id,
            )
        except Exception:
            fuzzy = []
        for item in fuzzy or []:
            doc_id = str(item.get("doc_id") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            matches.append(
                {
                    "doc_id": doc_id,
                    "title": str(item.get("title") or doc_id),
                    "source_type": str(item.get("source_type") or ""),
                    "doc_structure_type": str(item.get("doc_structure_type") or ""),
                    "match_reason": "fuzzy_title",
                    "score": float(item.get("score") or 0.0),
                }
            )
            if len(matches) >= limit:
                return matches

        lower_query = clean_query.casefold()
        try:
            records = doc_store.list_documents(tenant_id=self.tenant_id, collection_id=collection_id)
        except Exception:
            records = []
        for record in records:
            title = str(getattr(record, "title", "") or "")
            source_path = str(getattr(record, "source_path", "") or "")
            source_name = Path(source_path).name if source_path else ""
            if lower_query not in title.casefold() and lower_query not in source_name.casefold():
                continue
            doc_id = str(getattr(record, "doc_id", "") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            matches.append(
                {
                    "doc_id": doc_id,
                    "title": title or doc_id,
                    "source_type": str(getattr(record, "source_type", "") or ""),
                    "doc_structure_type": str(getattr(record, "doc_structure_type", "") or ""),
                    "match_reason": "metadata_scan",
                    "score": 0.45,
                }
            )
            if len(matches) >= limit:
                break
        return matches

    def search_graph_metadata(self, query: str, *, collection_id: str = "", limit: int = 4) -> List[Dict[str, Any]]:
        if "graph_indexes" not in set(self.allowed_views()):
            return []
        graph_index_store = getattr(self.stores, "graph_index_store", None)
        if graph_index_store is None:
            return []
        try:
            try:
                matches = graph_index_store.search_indexes(
                    query,
                    tenant_id=self.tenant_id,
                    user_id=self.user_id,
                    collection_id=collection_id,
                    limit=max(1, int(limit)),
                )
            except TypeError:
                matches = graph_index_store.search_indexes(
                    query,
                    tenant_id=self.tenant_id,
                    collection_id=collection_id,
                    limit=max(1, int(limit)),
                )
        except Exception:
            matches = []
        return [
            {
                "graph_id": item.graph_id,
                "display_name": item.display_name or item.graph_id,
                "backend": item.backend,
                "status": item.status,
                "domain_summary": item.domain_summary,
                "supported_query_methods": list(item.supported_query_methods),
                "source_doc_ids": list(item.source_doc_ids),
            }
            for item in matches
        ]


__all__ = ["StructuredSearchAdapter"]
