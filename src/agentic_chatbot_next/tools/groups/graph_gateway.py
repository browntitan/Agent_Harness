from __future__ import annotations

import json
from typing import Any, List

from langchain_core.tools import tool

from agentic_chatbot_next.graph.service import GraphService
from agentic_chatbot_next.rag.inventory import build_graph_document_inventory_payload


def _parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def build_graph_gateway_tools(ctx: Any) -> List[Any]:
    service = GraphService(ctx.settings, ctx.stores, session=ctx.session_handle)

    def _emit_progress(label: str, *, detail: str = "", stage: str = "graph") -> None:
        emitter = getattr(ctx, "progress_emitter", None)
        if emitter is not None and hasattr(emitter, "emit_progress"):
            emitter.emit_progress(
                "phase_update",
                label=label,
                detail=detail,
                agent=str(getattr(ctx, "active_agent", "") or "graph_manager"),
                stage=stage,
            )

    def _emit_runtime_event(event_type: str, payload: dict[str, Any]) -> None:
        kernel = getattr(ctx, "kernel", None)
        session = getattr(ctx, "session", None)
        if kernel is not None and session is not None and hasattr(kernel, "_emit"):
            kernel._emit(
                event_type,
                session.session_id,
                agent_name=str(getattr(ctx, "active_agent", "") or "graph_manager"),
                payload=payload,
            )

    @tool
    def list_graph_indexes(collection_id: str = "", limit: int = 20) -> str:
        """List managed graph indexes available to the current tenant."""
        return json.dumps(
            {
                "graphs": service.list_indexes(collection_id=collection_id, limit=max(1, min(int(limit), 100))),
            },
            ensure_ascii=False,
        )

    @tool
    def inspect_graph_index(graph_id: str) -> str:
        """Inspect one managed graph index, including sources and recent runs."""
        return json.dumps(service.inspect_index(graph_id), ensure_ascii=False)

    @tool
    def list_graph_documents(graph_id: str) -> str:
        """List the source documents recorded for one managed graph index."""
        return json.dumps(
            build_graph_document_inventory_payload(
                ctx.settings,
                ctx.stores,
                ctx.session_handle,
                graph_id=graph_id,
            ),
            ensure_ascii=False,
        )

    @tool
    def search_graph_index(
        query: str,
        graph_id: str = "",
        collection_id: str = "",
        methods_csv: str = "",
        top_k_graphs: int = 3,
        limit: int = 8,
    ) -> str:
        """Search one graph or a shortlist of relevant graphs for graph-backed evidence candidates."""
        methods = _parse_csv(methods_csv)
        if graph_id.strip():
            payload = service.query_index(
                graph_id.strip(),
                query=query,
                methods=methods,
                limit=max(1, min(int(limit), 20)),
            )
        else:
            payload = service.query_across_graphs(
                query,
                collection_id=collection_id,
                methods=methods,
                limit=max(1, min(int(limit), 20)),
                top_k_graphs=max(1, min(int(top_k_graphs), 8)),
            )
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def explain_source_plan(
        query: str,
        collection_id: str = "",
        preferred_doc_ids_csv: str = "",
    ) -> str:
        """Explain how the retrieval controller would choose among graph, vector, keyword, and SQL sources."""
        payload = service.explain_source_plan(
            query,
            collection_id=collection_id,
            preferred_doc_ids=_parse_csv(preferred_doc_ids_csv),
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def index_graph_corpus(
        graph_id: str = "",
        display_name: str = "",
        collection_id: str = "",
        source_doc_ids_csv: str = "",
        source_paths_csv: str = "",
        backend: str = "",
        refresh: bool = False,
    ) -> str:
        """Create or refresh a managed graph index for a set of indexed source documents."""
        _emit_progress("Indexing graph", detail=display_name or graph_id or collection_id)
        _emit_runtime_event(
            "graph_index_started",
            {
                "graph_id": graph_id,
                "display_name": display_name,
                "collection_id": collection_id,
                "refresh": bool(refresh),
            },
        )
        payload = service.index_corpus(
            graph_id=graph_id,
            display_name=display_name,
            collection_id=collection_id,
            source_doc_ids=_parse_csv(source_doc_ids_csv),
            source_paths=_parse_csv(source_paths_csv),
            refresh=refresh,
            backend=backend,
        )
        _emit_runtime_event("graph_index_completed", dict(payload))
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def import_existing_graph(
        graph_id: str = "",
        display_name: str = "",
        collection_id: str = "",
        import_backend: str = "neo4j",
        artifact_path: str = "",
        source_doc_ids_csv: str = "",
        source_paths_csv: str = "",
    ) -> str:
        """Register an existing graph artifact or Neo4j-backed graph in the managed catalog."""
        _emit_progress("Importing graph", detail=display_name or graph_id or artifact_path)
        _emit_runtime_event(
            "graph_import_started",
            {
                "graph_id": graph_id,
                "display_name": display_name,
                "collection_id": collection_id,
                "artifact_path": artifact_path,
                "import_backend": import_backend,
            },
        )
        payload = service.import_existing_graph(
            graph_id=graph_id,
            display_name=display_name,
            collection_id=collection_id,
            import_backend=import_backend,
            artifact_path=artifact_path,
            source_doc_ids=_parse_csv(source_doc_ids_csv),
            source_paths=_parse_csv(source_paths_csv),
        )
        _emit_runtime_event("graph_import_completed", dict(payload))
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def refresh_graph_index(graph_id: str) -> str:
        """Refresh a managed graph index using its previously recorded sources."""
        _emit_progress("Refreshing graph", detail=graph_id)
        _emit_runtime_event("graph_refresh_started", {"graph_id": graph_id})
        payload = service.refresh_graph_index(graph_id)
        _emit_runtime_event("graph_refresh_completed", dict(payload))
        return json.dumps(payload, ensure_ascii=False)

    return [
        list_graph_indexes,
        inspect_graph_index,
        list_graph_documents,
        search_graph_index,
        explain_source_plan,
        index_graph_corpus,
        import_existing_graph,
        refresh_graph_index,
    ]
