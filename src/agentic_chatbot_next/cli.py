from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_chatbot_next.config import Settings, load_settings
from agentic_chatbot_next.graph.service import GraphService
from agentic_chatbot_next.demo import (
    DemoScenario,
    DemoTurn,
    evaluate_response,
    load_demo_scenarios,
    render_scenario_summary,
)
from agentic_chatbot_next.providers import (
    ProviderConfigurationError,
    ProviderDependencyError,
    build_embeddings,
    build_providers,
    validate_provider_configuration,
    validate_provider_dependencies,
)
from agentic_chatbot_next.providers.output_limits import resolve_chat_output_cap
from agentic_chatbot_next.app.cli_adapter import CliAdapter
from agentic_chatbot_next.app.service import RuntimeService
from agentic_chatbot_next.context import build_local_context
from agentic_chatbot_next.runtime.registry_diagnostics import build_runtime_error_payload
from agentic_chatbot_next.sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    build_sandbox_image,
    check_docker_availability,
    probe_sandbox_image,
)
from agentic_chatbot_next.session import ChatSession


app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True)
class DoctorCheckResult:
    name: str
    status: str
    details: str
    remediation: str = ""


@dataclass(frozen=True)
class StoreContext:
    settings: Settings
    stores: Any


def _build_bot(settings: Settings) -> RuntimeService:
    providers = build_providers(settings)
    return CliAdapter.create_service(settings, providers)


def _build_store_context(settings: Settings) -> StoreContext:
    from agentic_chatbot_next.rag import load_stores

    embeddings = build_embeddings(settings)
    stores = load_stores(settings, embeddings)
    return StoreContext(settings=settings, stores=stores)


def _make_app(dotenv: Optional[str] = None) -> RuntimeService:
    settings = load_settings(dotenv)
    return _build_bot(settings)


def _make_store_context(dotenv: Optional[str] = None) -> StoreContext:
    settings = load_settings(dotenv)
    return _build_store_context(settings)


def _make_local_session(
    dotenv: Optional[str] = None,
    conversation_id: Optional[str] = None,
    collection_id: Optional[str] = None,
) -> ChatSession:
    settings = load_settings(dotenv)
    ctx = build_local_context(settings, conversation_id=conversation_id)
    session = ChatSession.from_context(ctx)
    session.metadata["collection_id"] = str(collection_id or settings.default_collection_id)
    return session


def _make_graph_service(
    dotenv: Optional[str] = None,
    *,
    collection_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> GraphService:
    store_ctx = _make_store_context_or_exit(dotenv)
    session = _make_local_session(
        dotenv,
        conversation_id=conversation_id or "graph-cli",
        collection_id=collection_id,
    )
    return GraphService(store_ctx.settings, store_ctx.stores, session=session)


def _render_demo_notes(scenario_obj: DemoScenario) -> str:
    if not scenario_obj.notes:
        return ""
    return f"Notes:\n{scenario_obj.notes}"


def _coerce_force_agent(global_force_agent: bool, turn: DemoTurn) -> bool:
    if global_force_agent:
        return True
    if turn.force_agent is None:
        return False
    return bool(turn.force_agent)


def _verify_status_style(status: str) -> str:
    style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
    }
    return style.get(status, "white")


def _doctor_status_style(status: str) -> str:
    style = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
        "SKIP": "cyan",
    }
    return style.get(status, "white")


def _mask_dsn_password(dsn: str) -> str:
    try:
        parsed = urlsplit(dsn)
    except Exception:
        return dsn

    if not parsed.scheme or "@" not in parsed.netloc:
        return dsn

    userinfo, hostinfo = parsed.netloc.rsplit("@", 1)
    if ":" in userinfo:
        username = userinfo.split(":", 1)[0]
        masked_userinfo = f"{username}:***"
    else:
        masked_userinfo = userinfo

    return urlunsplit((parsed.scheme, f"{masked_userinfo}@{hostinfo}", parsed.path, parsed.query, parsed.fragment))


def _read_table_embedding_dim(conn, table_name: str) -> Optional[int]:
    from agentic_chatbot_next.persistence.postgres.vector_schema import parse_vector_dimension

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = current_schema()
              AND c.relname = %s
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (table_name,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return parse_vector_dimension(str(row[0]))


def _exit_provider_error(exc: Exception) -> None:
    title = "Provider Configuration Error" if isinstance(exc, ProviderConfigurationError) else "Provider Dependency Error"
    console.print(Panel(str(exc), title=title, border_style="red"))
    console.print("Run `python run.py doctor` to validate providers, database, and connectivity checks.")
    raise typer.Exit(code=2)


def _make_app_or_exit(dotenv: Optional[str] = None) -> RuntimeService:
    try:
        return _make_app(dotenv)
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
        _exit_provider_error(exc)
        raise


def _build_bot_or_exit(settings: Settings) -> RuntimeService:
    try:
        return _build_bot(settings)
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
        _exit_provider_error(exc)
        raise


def _make_store_context_or_exit(dotenv: Optional[str] = None) -> StoreContext:
    try:
        return _make_store_context(dotenv)
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
        _exit_provider_error(exc)
        raise


def _build_store_context_or_exit(settings: Settings) -> StoreContext:
    try:
        return _build_store_context(settings)
    except (ProviderDependencyError, ProviderConfigurationError) as exc:
        _exit_provider_error(exc)
        raise


def _with_demo_settings(settings: Settings) -> Settings:
    if settings.llm_provider.lower() != "ollama":
        return settings
    base_cap = resolve_chat_output_cap(settings)
    demo_cap = resolve_chat_output_cap(settings, demo_mode=True)
    if demo_cap == base_cap:
        return settings
    return replace(settings, chat_max_output_tokens=demo_cap)


def _selected_ollama_models(settings: Settings) -> dict[str, str]:
    models: dict[str, str] = {}
    if settings.llm_provider.lower() == "ollama":
        models["llm"] = settings.ollama_chat_model
    if settings.judge_provider.lower() == "ollama":
        models["judge"] = settings.ollama_judge_model
    if settings.embeddings_provider.lower() == "ollama":
        models["embeddings"] = settings.ollama_embed_model
    return models


def _ollama_model_aliases(model_name: str) -> set[str]:
    normalized = str(model_name).strip()
    if not normalized:
        return set()
    aliases = {normalized}
    if ":" in normalized:
        base_name, tag = normalized.rsplit(":", 1)
        if tag == "latest":
            aliases.add(base_name)
    else:
        aliases.add(f"{normalized}:latest")
    return aliases


def _ollama_missing_models(required_models: dict[str, str], available_models: set[str]) -> list[str]:
    missing: list[str] = []
    for model_name in required_models.values():
        if not any(candidate in available_models for candidate in _ollama_model_aliases(model_name)):
            missing.append(model_name)
    return sorted(set(missing))


def _iter_kb_source_paths(settings: Settings) -> list[Path]:
    roots = [Path(settings.kb_dir), *(Path(path) for path in getattr(settings, "kb_extra_dirs", ()))]
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.glob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
    return paths


def _normalized_source_paths(paths: List[Path]) -> set[str]:
    return {str(path.resolve()) for path in paths}


def _normalized_file_names(paths: List[Path]) -> set[str]:
    return {path.name for path in paths}


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q"),
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    force_agent: bool = typer.Option(False, "--force-agent"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Run a single-turn query."""

    bot = _make_app_or_exit(dotenv)
    session = _make_local_session(dotenv, collection_id=collection_id)

    response = bot.process_turn(session, user_text=question, upload_paths=upload, force_agent=force_agent)

    console.print(Panel(response, title="Assistant"))


@app.command()
def chat(
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Start an interactive chat session. Use /upload PATH to ingest docs mid-chat."""

    bot = _make_app_or_exit(dotenv)
    session = _make_local_session(dotenv, collection_id=collection_id)

    if upload:
        console.print("[bold]Ingesting uploads...[/bold]")
        bot.ingest_and_summarize_uploads(session, upload)

    console.print(Panel("Type your message. Commands: /upload PATH, /exit", title="Agentic Chatbot"))

    while True:
        try:
            user_text = console.input("\n[bold cyan]You>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break

        if user_text.startswith("/upload"):
            parts = user_text.split(maxsplit=1)
            if len(parts) < 2:
                console.print("Usage: /upload PATH")
                continue
            path = Path(parts[1]).expanduser()
            if not path.exists():
                console.print(f"File not found: {path}")
                continue
            doc_ids, summary = bot.ingest_and_summarize_uploads(session, [path])
            console.print(Panel(summary, title=f"Ingested: {doc_ids}"))
            continue

        response = bot.process_turn(session, user_text=user_text, upload_paths=[])
        console.print(Panel(response, title="Assistant"))


@app.command()
def init_kb(dotenv: Optional[str] = typer.Option(None, "--dotenv")):
    """Deprecated alias: seed the built-in demo KB into the default collection."""

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    console.print(
        "[yellow]`init-kb` is deprecated.[/yellow] "
        "Use `python run.py sync-kb` for normal DB-first ingestion. "
        "This command now only seeds the bundled demo KB."
    )
    from agentic_chatbot_next.rag import ingest_paths  # noqa: PLC0415

    kb_paths = _iter_kb_source_paths(store_ctx.settings)
    ingest_paths(
        store_ctx.settings,
        store_ctx.stores,
        kb_paths,
        source_type="kb",
        tenant_id=tenant_id,
        collection_id=store_ctx.settings.default_collection_id,
    )
    records = store_ctx.stores.doc_store.list_documents(
        tenant_id=tenant_id,
        source_type="kb",
        collection_id=store_ctx.settings.default_collection_id,
    )
    docs = [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "source_type": r.source_type,
            "collection_id": r.collection_id,
            "num_chunks": r.num_chunks,
            "doc_structure_type": r.doc_structure_type,
        }
        for r in records
    ]
    console.print(json.dumps(docs, indent=2, ensure_ascii=False)[:4000])


@app.command("sync-kb")
def sync_kb(
    path: List[Path] = typer.Option([], "--path", "-p", help="File(s) to ingest. Defaults to all files in data/kb."),
    source_type: str = typer.Option("kb", "--source-type"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Ingest a corpus into PostgreSQL + pgvector using an explicit collection ID."""

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    paths = [item.resolve() for item in path] if path else _iter_kb_source_paths(store_ctx.settings)
    from agentic_chatbot_next.rag import ingest_paths  # noqa: PLC0415

    effective_collection = collection_id or store_ctx.settings.default_collection_id
    doc_ids = ingest_paths(
        store_ctx.settings,
        store_ctx.stores,
        paths,
        source_type=source_type,
        tenant_id=tenant_id,
        collection_id=effective_collection,
    )
    requested_source_paths = _normalized_source_paths(paths)
    requested_titles = _normalized_file_names(paths)
    all_records = store_ctx.stores.doc_store.list_documents(
        tenant_id=tenant_id,
        source_type=source_type,
        collection_id=effective_collection,
    )
    resolved_records_by_key = {}
    covered_titles: set[str] = set()
    for record in all_records:
        resolved_source_path = str(Path(record.source_path).resolve())
        if resolved_source_path in requested_source_paths:
            resolved_records_by_key.setdefault(resolved_source_path, record)
            covered_titles.add(record.title)
    for record in all_records:
        if record.title in covered_titles:
            continue
        if record.title in requested_titles:
            resolved_records_by_key.setdefault(record.title, record)
    resolved_records = list(resolved_records_by_key.values())
    resolved_doc_ids = [record.doc_id for record in resolved_records]
    console.print(
        json.dumps(
            {
                "ingested_doc_ids": doc_ids,
                "resolved_doc_ids": resolved_doc_ids,
                "requested_count": len(paths),
                "count": len(doc_ids),
                "already_indexed_count": max(0, len(resolved_doc_ids) - len(doc_ids)),
                "missing_count": max(0, len(paths) - len(resolved_doc_ids)),
                "collection_id": effective_collection,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("inspect-kb")
def inspect_kb(
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    title: str = typer.Option("", "--title", help="Filter to source groups whose title contains this text."),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Inspect KB source health, including missing files, duplicates, and drift."""

    from agentic_chatbot_next.rag.ingest import build_kb_health_report  # noqa: PLC0415

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    effective_collection = collection_id or store_ctx.settings.default_collection_id
    report = build_kb_health_report(
        store_ctx.settings,
        store_ctx.stores,
        tenant_id=tenant_id,
        collection_id=effective_collection,
    )
    payload = report.to_dict()
    title_filter = str(title or "").strip().casefold()
    if title_filter:
        payload["source_groups"] = [
            group
            for group in payload.get("source_groups", [])
            if title_filter in str(group.get("title") or "").casefold()
        ]
        payload["duplicate_groups"] = [
            group
            for group in payload.get("duplicate_groups", [])
            if title_filter in str(group.get("title") or "").casefold()
        ]
        payload["drifted_groups"] = [
            group
            for group in payload.get("drifted_groups", [])
            if title_filter in str(group.get("title") or "").casefold()
        ]
    console.print(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("backfill-requirements")
def backfill_requirements(
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    source_type: str = typer.Option("kb", "--source-type"),
    title: str = typer.Option("", "--title", help="Optional title substring filter."),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Populate persisted requirement statements for already indexed prose documents."""

    from agentic_chatbot_next.rag import backfill_requirement_statements  # noqa: PLC0415

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    effective_collection = collection_id or store_ctx.settings.default_collection_id
    result = backfill_requirement_statements(
        store_ctx.settings,
        store_ctx.stores,
        tenant_id=tenant_id,
        collection_id=effective_collection,
        source_type=source_type,
        title_hint=title,
    )
    console.print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


@app.command("graph-list")
def graph_list(
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    limit: int = typer.Option(20, "--limit"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """List managed graph indexes for the current tenant."""

    service = _make_graph_service(dotenv, collection_id=collection_id)
    console.print(
        json.dumps(
            {
                "graphs": service.list_indexes(
                    collection_id=str(collection_id or ""),
                    limit=max(1, min(int(limit), 100)),
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("graph-inspect")
def graph_inspect(
    graph_id: str = typer.Argument(...),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Inspect a managed graph index, including sources and recent runs."""

    service = _make_graph_service(dotenv)
    payload = service.inspect_index(graph_id)
    console.print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload.get("error"):
        raise typer.Exit(code=1)


@app.command("graph-index")
def graph_index(
    graph_id: str = typer.Option("", "--graph-id"),
    display_name: str = typer.Option("", "--display-name"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    source_doc_id: List[str] = typer.Option([], "--source-doc-id"),
    source_path: List[Path] = typer.Option([], "--source-path"),
    backend: str = typer.Option("", "--backend"),
    refresh: bool = typer.Option(False, "--refresh"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Create or refresh a managed graph index from indexed source documents."""

    service = _make_graph_service(dotenv, collection_id=collection_id)
    payload = service.index_corpus(
        graph_id=graph_id,
        display_name=display_name,
        collection_id=str(collection_id or ""),
        source_doc_ids=list(source_doc_id),
        source_paths=[str(item.resolve()) for item in source_path],
        refresh=refresh,
        backend=backend,
    )
    console.print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload.get("error"):
        raise typer.Exit(code=1)


@app.command("graph-import")
def graph_import(
    artifact_path: str = typer.Option(..., "--artifact-path"),
    graph_id: str = typer.Option("", "--graph-id"),
    display_name: str = typer.Option("", "--display-name"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    import_backend: str = typer.Option("neo4j", "--import-backend"),
    source_doc_id: List[str] = typer.Option([], "--source-doc-id"),
    source_path: List[Path] = typer.Option([], "--source-path"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Register an existing graph artifact or Neo4j graph in the managed catalog."""

    service = _make_graph_service(dotenv, collection_id=collection_id)
    payload = service.import_existing_graph(
        graph_id=graph_id,
        display_name=display_name,
        collection_id=str(collection_id or ""),
        import_backend=import_backend,
        artifact_path=artifact_path,
        source_doc_ids=list(source_doc_id),
        source_paths=[str(item.resolve()) for item in source_path],
    )
    console.print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload.get("error"):
        raise typer.Exit(code=1)


@app.command("graph-refresh")
def graph_refresh(
    graph_id: str = typer.Argument(...),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Refresh a managed graph index using its recorded sources."""

    service = _make_graph_service(dotenv)
    payload = service.refresh_graph_index(graph_id)
    console.print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload.get("error"):
        raise typer.Exit(code=1)


@app.command("graph-query")
def graph_query(
    query: str = typer.Argument(...),
    graph_id: str = typer.Option("", "--graph-id"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    method: List[str] = typer.Option([], "--method"),
    limit: int = typer.Option(8, "--limit"),
    top_k_graphs: int = typer.Option(3, "--top-k-graphs"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Query one graph or search across the most relevant managed graphs."""

    service = _make_graph_service(dotenv, collection_id=collection_id)
    if graph_id.strip():
        payload = service.query_index(
            graph_id.strip(),
            query=query,
            methods=list(method),
            limit=max(1, min(int(limit), 20)),
        )
    else:
        payload = service.query_across_graphs(
            query,
            collection_id=str(collection_id or ""),
            methods=list(method),
            limit=max(1, min(int(limit), 20)),
            top_k_graphs=max(1, min(int(top_k_graphs), 8)),
        )
    console.print(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("repair-kb")
def repair_kb(
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    title: str = typer.Option("", "--title", help="Limit repair to documents whose title contains this text."),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Repair KB duplicates, drifted files, and missing configured sources."""

    from agentic_chatbot_next.rag.ingest import repair_kb_collection  # noqa: PLC0415

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    effective_collection = collection_id or store_ctx.settings.default_collection_id
    result = repair_kb_collection(
        store_ctx.settings,
        store_ctx.stores,
        tenant_id=tenant_id,
        collection_id=effective_collection,
        title_hint=title,
    )
    console.print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


@app.command("sync-defense-corpus")
def sync_defense_corpus(
    path: Optional[Path] = typer.Option(None, "--path", help="Corpus root. Defaults to defense_rag_test_corpus/documents."),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    source_type: str = typer.Option("host_path", "--source-type"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Recursively ingest the defense benchmark corpus into a persistent collection."""

    from agentic_chatbot_next.benchmark.defense_corpus import DEFENSE_COLLECTION_ID  # noqa: PLC0415
    from agentic_chatbot_next.rag import ingest_paths  # noqa: PLC0415

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    corpus_root = (path or (store_ctx.settings.project_root / "defense_rag_test_corpus" / "documents")).resolve()
    effective_collection = collection_id or DEFENSE_COLLECTION_ID
    file_paths = [corpus_root] if corpus_root.is_file() else [item.resolve() for item in sorted(corpus_root.rglob("*")) if item.is_file()]
    source_display_paths = {
        str(item): (item.relative_to(corpus_root).as_posix() if corpus_root.is_dir() else item.name)
        for item in file_paths
    }
    source_identities = {str(item): f"path:{str(item)}" for item in file_paths}
    doc_ids = ingest_paths(
        store_ctx.settings,
        store_ctx.stores,
        file_paths,
        source_type=source_type,
        tenant_id=tenant_id,
        collection_id=effective_collection,
        source_display_paths=source_display_paths,
        source_identities=source_identities,
    )
    console.print(
        json.dumps(
            {
                "corpus_root": str(corpus_root),
                "collection_id": effective_collection,
                "ingested_count": len(doc_ids),
                "doc_ids": doc_ids,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("reindex-document")
def reindex_document(
    path: Path = typer.Argument(..., exists=True, readable=True),
    source_type: str = typer.Option("kb", "--source-type"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Ingest a new active version for a source path while retaining old rows."""

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    effective_collection = collection_id or store_ctx.settings.default_collection_id
    existing = [
        record
        for record in store_ctx.stores.doc_store.list_documents(tenant_id=tenant_id, collection_id=effective_collection)
        if Path(record.source_path) == path and record.source_type == source_type
    ]
    from agentic_chatbot_next.rag import ingest_paths  # noqa: PLC0415

    doc_ids = ingest_paths(
        store_ctx.settings,
        store_ctx.stores,
        [path],
        source_type=source_type,
        tenant_id=tenant_id,
        collection_id=effective_collection,
    )
    console.print(
        json.dumps(
            {
                "superseded_doc_ids": [record.doc_id for record in existing],
                "ingested_doc_ids": doc_ids,
                "collection_id": effective_collection,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("evaluate-defense-corpus")
def evaluate_defense_corpus(
    answer_key: Optional[Path] = typer.Option(None, "--answer-key"),
    collection_id: Optional[str] = typer.Option(None, "--collection-id"),
    question_id: List[str] = typer.Option([], "--question-id"),
    limit: int = typer.Option(0, "--limit"),
    sync_first: bool = typer.Option(False, "--sync-first"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Run the defense benchmark against the live RAG contract and report grounded diagnostics."""

    from agentic_chatbot_next.benchmark.defense_corpus import (  # noqa: PLC0415
        DEFENSE_COLLECTION_ID,
        load_defense_answer_key,
        run_defense_benchmark,
    )
    from agentic_chatbot_next.rag.engine import run_rag_contract  # noqa: PLC0415

    bot = _make_app_or_exit(dotenv)
    effective_collection = collection_id or DEFENSE_COLLECTION_ID
    answer_key_path = (answer_key or (bot.ctx.settings.project_root / "defense_rag_test_corpus" / "evaluation" / "answer_key.json")).resolve()
    if sync_first:
        corpus_root = (bot.ctx.settings.project_root / "defense_rag_test_corpus" / "documents").resolve()
        from agentic_chatbot_next.rag import ingest_paths  # noqa: PLC0415

        ingest_paths(
            bot.ctx.settings,
            bot.ctx.stores,
            [corpus_root],
            source_type="kb",
            tenant_id=bot.ctx.settings.default_tenant_id,
            collection_id=effective_collection,
        )

    questions = load_defense_answer_key(answer_key_path)
    if question_id:
        wanted = {item.strip() for item in question_id if item.strip()}
        questions = [item for item in questions if item.question_id in wanted]
    if limit > 0:
        questions = questions[:limit]

    rag_providers = bot.kernel.resolve_providers_for_agent("rag_worker") or bot.ctx.providers

    def _answer_fn(question):
        session = _make_local_session(
            dotenv,
            conversation_id=f"defense-benchmark-{question.question_id.lower()}",
            collection_id=effective_collection,
        )
        return run_rag_contract(
            bot.ctx.settings,
            bot.ctx.stores,
            providers=rag_providers,
            session=session,
            query=question.question_text,
            conversation_context="Defense benchmark evaluation.",
            preferred_doc_ids=[],
            must_include_uploads=False,
            top_k_vector=bot.ctx.settings.rag_top_k_vector,
            top_k_keyword=bot.ctx.settings.rag_top_k_keyword,
            max_retries=bot.ctx.settings.rag_max_retries,
            callbacks=[],
            search_mode="auto",
            max_search_rounds=max(2, int(bot.ctx.settings.rag_max_retries) + 1),
        )

    report = run_defense_benchmark(
        questions,
        answer_fn=_answer_fn,
        collection_id=effective_collection,
    )
    console.print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


@app.command("benchmark-ollama-throughput")
def benchmark_ollama_throughput(
    model: List[str] = typer.Option([], "--model", help="Repeat to benchmark multiple models sequentially."),
    runs: int = typer.Option(3, "--runs", min=1, help="Measured runs per model."),
    num_predict: int = typer.Option(256, "--num-predict", min=1, help="Requested output tokens per run."),
    context_words: int = typer.Option(2500, "--context-words", min=1, help="Approximate prompt size before tokenization."),
    num_ctx: Optional[int] = typer.Option(None, "--num-ctx", min=1, help="Optional Ollama context window override."),
    keep_alive: str = typer.Option("10m", "--keep-alive", help="How long Ollama should keep the model resident."),
    timeout_seconds: int = typer.Option(900, "--timeout-seconds", min=1, help="Per-request timeout."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override the Ollama base URL for the benchmark."),
    warmup: bool = typer.Option(True, "--warmup/--no-warmup", help="Run one excluded warm-up request before measuring."),
    localhost_fallback: bool = typer.Option(
        True,
        "--localhost-fallback/--no-localhost-fallback",
        help="Try 127.0.0.1 if localhost resolves to the wrong listener.",
    ),
    output: Optional[Path] = typer.Option(None, "--output", help="Optional JSON output path."),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Measure Ollama prompt and decode throughput for one or more local models."""

    from agentic_chatbot_next.benchmark.ollama_throughput import (  # noqa: PLC0415
        OllamaBenchmarkError,
        run_ollama_throughput_benchmark,
    )

    settings = load_settings(dotenv)
    selected_models = [item.strip() for item in model if item.strip()]
    if not selected_models:
        selected_models = [str(settings.ollama_chat_model or "").strip()]
    selected_models = [item for item in selected_models if item]
    if not selected_models:
        console.print("No Ollama model was provided and OLLAMA_CHAT_MODEL is empty.")
        raise typer.Exit(code=1)

    effective_base_url = str(base_url or settings.ollama_base_url or "").strip()
    if not effective_base_url:
        console.print("No Ollama base URL was provided and OLLAMA_BASE_URL is empty.")
        raise typer.Exit(code=1)

    try:
        report = run_ollama_throughput_benchmark(
            models=selected_models,
            base_url=effective_base_url,
            runs=runs,
            num_predict=num_predict,
            context_words=context_words,
            keep_alive=keep_alive,
            timeout_seconds=timeout_seconds,
            warmup=warmup,
            num_ctx=num_ctx,
            localhost_fallback=localhost_fallback,
        )
    except OllamaBenchmarkError as exc:
        console.print(Panel(str(exc), title="Ollama Benchmark Error", border_style="red"))
        raise typer.Exit(code=1) from exc

    failures = list(getattr(report, "failures", []) or [])
    if failures:
        failure_lines = "\n".join(f"{item.model}: {item.error}" for item in failures)
        console.print(
            Panel(
                failure_lines,
                title="Partial Ollama Benchmark Results",
                border_style="yellow",
            )
        )

    payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if output is not None:
        output.write_text(payload + "\n", encoding="utf-8")
    console.print(payload)


@app.command("evaluate-public-benchmarks")
def evaluate_public_benchmarks(
    profile: str = typer.Option("smoke", "--profile", help="Benchmark profile: smoke, diagnostic, or full."),
    benchmarks: str = typer.Option(
        "beir:scifact,hotpotqa,ragbench",
        "--benchmarks",
        help="Comma-separated benchmark specs such as beir:scifact,hotpotqa,ragbench,bfcl.",
    ),
    data_root: Path = typer.Option(Path("data/public_benchmarks"), "--data-root"),
    collection_prefix: str = typer.Option("public", "--collection-prefix"),
    api_base: str = typer.Option("http://127.0.0.1:18000", "--api-base"),
    model: str = typer.Option("enterprise-agent", "--model"),
    token_env: str = typer.Option("GATEWAY_SHARED_BEARER_TOKEN", "--token-env"),
    token: str = typer.Option("", "--token"),
    judge: str = typer.Option("auto", "--judge", help="Use auto or off."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    limit: int = typer.Option(0, "--limit"),
    timeout_seconds: float = typer.Option(180.0, "--timeout-seconds"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    available_capability: Optional[List[str]] = typer.Option(
        None,
        "--available-capability",
        help="Repeat to mark benchmark capabilities as available, e.g. --available-capability tool_calling.",
    ),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Run public benchmark adapters through the OpenAI-compatible live gateway."""

    from datetime import datetime, timezone

    from agentic_chatbot_next.benchmark.public_suite import run_public_benchmark_suite  # noqa: PLC0415

    if profile not in {"smoke", "diagnostic", "full"}:
        console.print("Profile must be one of: smoke, diagnostic, full.")
        raise typer.Exit(code=2)
    if judge not in {"auto", "off"}:
        console.print("Judge must be one of: auto, off.")
        raise typer.Exit(code=2)
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv or None)
    except Exception:
        pass

    effective_token = token or os.environ.get(token_env, "")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    effective_output_dir = output_dir or Path("data/runtime/public_benchmark_runs") / timestamp
    summary = run_public_benchmark_suite(
        benchmarks=benchmarks,
        profile=profile,
        data_root=data_root,
        collection_prefix=collection_prefix,
        api_base=api_base,
        model=model,
        token=effective_token,
        judge=judge,
        output_dir=effective_output_dir,
        limit=limit,
        timeout_seconds=timeout_seconds,
        fail_fast=fail_fast,
        available_capabilities=available_capability or ["chat", "rag"],
        dotenv_path=dotenv or "",
    )
    console.print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))


@app.command("delete-document")
def delete_document(
    doc_id: str = typer.Argument(...),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Delete one indexed document and its chunks from the database."""

    store_ctx = _make_store_context_or_exit(dotenv)
    tenant_id = store_ctx.settings.default_tenant_id
    record = store_ctx.stores.doc_store.get_document(doc_id, tenant_id=tenant_id)
    if record is None:
        console.print(json.dumps({"deleted": False, "doc_id": doc_id, "reason": "not_found"}, indent=2))
        raise typer.Exit(code=1)
    store_ctx.stores.doc_store.delete_document(doc_id, tenant_id=tenant_id)
    console.print(
        json.dumps(
            {
                "deleted": True,
                "doc_id": doc_id,
                "title": record.title,
                "collection_id": record.collection_id,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("index-skills")
def index_skills(dotenv: Optional[str] = typer.Option(None, "--dotenv")):
    """Index repo-authored skill packs into the DB-backed skill store."""

    store_ctx = _make_store_context_or_exit(dotenv)
    from agentic_chatbot_next.rag import SkillIndexSync  # noqa: PLC0415

    result = SkillIndexSync(store_ctx.settings, store_ctx.stores).sync(
        tenant_id=store_ctx.settings.default_tenant_id,
    )
    console.print(json.dumps(result, indent=2, ensure_ascii=False)[:6000])


@app.command("list-skills")
def list_skills(
    agent_scope: str = typer.Option("", "--agent-scope"),
    enabled_only: bool = typer.Option(False, "--enabled-only"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """List DB-indexed skill packs."""

    store_ctx = _make_store_context_or_exit(dotenv)
    records = store_ctx.stores.skill_store.list_skill_packs(
        tenant_id=store_ctx.settings.default_tenant_id,
        agent_scope=agent_scope,
        enabled_only=enabled_only,
    )
    console.print(
        json.dumps(
            [
                {
                    "skill_id": record.skill_id,
                    "name": record.name,
                    "agent_scope": record.agent_scope,
                    "tool_tags": record.tool_tags,
                    "task_tags": record.task_tags,
                    "version": record.version,
                    "enabled": record.enabled,
                    "source_path": record.source_path,
                    "kind": getattr(record, "kind", "retrievable"),
                }
                for record in records
            ],
            indent=2,
            ensure_ascii=False,
        )[:6000]
    )


@app.command("inspect-skill")
def inspect_skill(
    skill_id: str = typer.Argument(...),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Show one indexed skill pack and its stored chunks."""

    store_ctx = _make_store_context_or_exit(dotenv)
    record = store_ctx.stores.skill_store.get_skill_pack(
        skill_id,
        tenant_id=store_ctx.settings.default_tenant_id,
    )
    if record is None:
        console.print(json.dumps({"found": False, "skill_id": skill_id}, indent=2))
        raise typer.Exit(code=1)
    chunks = store_ctx.stores.skill_store.get_skill_chunks(
        skill_id,
        tenant_id=store_ctx.settings.default_tenant_id,
    )
    console.print(
        json.dumps(
            {
                "found": True,
                "record": {
                    "skill_id": record.skill_id,
                    "name": record.name,
                    "agent_scope": record.agent_scope,
                    "tool_tags": record.tool_tags,
                    "task_tags": record.task_tags,
                    "version": record.version,
                    "enabled": record.enabled,
                    "source_path": record.source_path,
                    "description": record.description,
                    "kind": getattr(record, "kind", "retrievable"),
                    "execution_config": dict(getattr(record, "execution_config", {}) or {}),
                },
                "chunks": chunks,
            },
            indent=2,
            ensure_ascii=False,
        )[:8000]
    )


@app.command()
def reset_indexes(
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Truncate all indexed data from PostgreSQL (documents, chunks, memory, skills)."""

    settings = load_settings(dotenv)

    if not confirm:
        typer.confirm(
            "This will DELETE all documents, chunks, memory, and indexed skills from the database. Continue?",
            abort=True,
        )

    from agentic_chatbot_next.persistence.postgres.connection import apply_schema, get_conn, init_pool

    init_pool(settings)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE memory, skill_chunks, skills, chunks, documents RESTART IDENTITY CASCADE")
        conn.commit()

    console.print("All indexes cleared. Run sync-kb and index-skills to rebuild.")


@app.command()
def migrate(dotenv: Optional[str] = typer.Option(None, "--dotenv")):
    """Apply the database schema (idempotent — safe to run multiple times)."""

    settings = load_settings(dotenv)
    from agentic_chatbot_next.persistence.postgres.connection import apply_schema

    apply_schema(settings)
    console.print("Schema applied successfully.")


@app.command("migrate-embedding-dim")
def migrate_embedding_dim(
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
    target_dim: int = typer.Option(0, "--target-dim", help="Target embedding vector dimension (0 uses EMBEDDING_DIM)."),
    reindex_kb: bool = typer.Option(True, "--reindex-kb/--skip-reindex-kb"),
    reset_memory: bool = typer.Option(False, "--reset-memory/--keep-memory"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Align chunk vector dimension, reset indexed docs/chunks, and optionally reindex KB."""

    settings = load_settings(dotenv)
    desired_dim = target_dim if target_dim > 0 else settings.embedding_dim
    if desired_dim <= 0:
        raise typer.BadParameter("--target-dim must be positive.")

    if not confirm:
        typer.confirm(
            "This will clear indexed documents/chunks to rebuild embeddings at the target dimension. Continue?",
            abort=True,
        )

    from agentic_chatbot_next.persistence.postgres.connection import apply_schema, get_conn, init_pool
    from agentic_chatbot_next.persistence.postgres.vector_schema import (
        get_chunks_embedding_dim,
        get_skill_chunks_embedding_dim,
        set_chunks_embedding_dim,
        set_skill_chunks_embedding_dim,
    )

    effective_settings = settings if desired_dim == settings.embedding_dim else replace(settings, embedding_dim=desired_dim)

    apply_schema(effective_settings)
    init_pool(effective_settings)
    before_chunks_dim = get_chunks_embedding_dim()
    before_skill_chunks_dim = get_skill_chunks_embedding_dim()
    if before_chunks_dim is None or before_skill_chunks_dim is None:
        missing = []
        if before_chunks_dim is None:
            missing.append("chunks.embedding")
        if before_skill_chunks_dim is None:
            missing.append("skill_chunks.embedding")
        console.print(
            "[red]Unable to detect "
            + ", ".join(missing)
            + " dimension(s). Ensure schema is applied and the required tables exist.[/red]"
        )
        raise typer.Exit(code=1)
    changed_chunks = set_chunks_embedding_dim(desired_dim)
    changed_skill_chunks = set_skill_chunks_embedding_dim(desired_dim)

    with get_conn() as conn:
        with conn.cursor() as cur:
            if reset_memory:
                cur.execute("TRUNCATE TABLE memory, skill_chunks, skills, chunks, documents RESTART IDENTITY CASCADE")
            else:
                cur.execute("TRUNCATE TABLE skill_chunks, skills, chunks, documents RESTART IDENTITY CASCADE")
        conn.commit()

    # Recreate dropped vector index(es) if the embedding column type was altered.
    apply_schema(effective_settings)

    after_chunks_dim = get_chunks_embedding_dim()
    after_skill_chunks_dim = get_skill_chunks_embedding_dim()
    console.print(
        "Embedding schema alignment complete "
        "(chunks: "
        f"{before_chunks_dim}->{after_chunks_dim}, "
        "skill_chunks: "
        f"{before_skill_chunks_dim}->{after_skill_chunks_dim}, "
        f"schema_changed={'yes' if changed_chunks or changed_skill_chunks else 'no'})."
    )

    if desired_dim != settings.embedding_dim:
        console.print(
            f"[yellow]Note:[/yellow] Settings currently use EMBEDDING_DIM={settings.embedding_dim}. "
            f"Update `.env` to EMBEDDING_DIM={desired_dim} before normal runs."
        )

    if reindex_kb:
        store_ctx = _build_store_context_or_exit(effective_settings)
        tenant_id = effective_settings.default_tenant_id
        from agentic_chatbot_next.rag import SkillIndexSync, ingest_paths  # noqa: PLC0415

        kb_paths = sorted(Path(effective_settings.kb_dir).glob("*"))
        ingest_paths(
            effective_settings,
            store_ctx.stores,
            kb_paths,
            source_type="kb",
            tenant_id=tenant_id,
            collection_id=effective_settings.default_collection_id,
        )
        SkillIndexSync(effective_settings, store_ctx.stores).sync(tenant_id=tenant_id)
        kb_docs = store_ctx.stores.doc_store.list_documents(
            source_type="kb",
            tenant_id=tenant_id,
            collection_id=effective_settings.default_collection_id,
        )
        console.print(f"Reindex complete. Demo KB documents: {len(kb_docs)}; skill packs re-synced.")
    else:
        console.print("Skipped KB reindex (--skip-reindex-kb). Run `python run.py sync-kb` and `python run.py index-skills` when ready.")


@app.command()
def doctor(
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero if any WARN checks are present."),
    timeout_seconds: float = typer.Option(3.0, "--timeout-seconds", min=0.5, help="Timeout used for connectivity checks."),
    check_db: bool = typer.Option(True, "--check-db/--skip-db", help="Enable or skip PostgreSQL connectivity check."),
    check_ollama: bool = typer.Option(
        True,
        "--check-ollama/--skip-ollama",
        help="Enable or skip Ollama API check when Ollama providers are selected.",
    ),
):
    """Run provider/runtime preflight checks for local or Docker execution."""

    settings = load_settings(dotenv)
    provider_set = {
        settings.llm_provider.lower(),
        settings.judge_provider.lower(),
        settings.embeddings_provider.lower(),
    }
    needs_ollama = "ollama" in provider_set
    needs_azure = "azure" in provider_set
    needs_nvidia = "nvidia" in provider_set

    config_lines = [
        f"LLM_PROVIDER={settings.llm_provider}",
        f"JUDGE_PROVIDER={settings.judge_provider}",
        f"EMBEDDINGS_PROVIDER={settings.embeddings_provider}",
        f"EMBEDDING_DIM={settings.embedding_dim}",
        f"PG_DSN={_mask_dsn_password(settings.pg_dsn)}",
        f"SANDBOX_DOCKER_IMAGE={settings.sandbox_docker_image}",
        f"KB_DIR={settings.kb_dir}",
        "KB_EXTRA_DIRS=" + (
            ",".join(str(path) for path in getattr(settings, "kb_extra_dirs", ())) or "<unset>"
        ),
        f"HTTP2_ENABLED={settings.http2_enabled}",
        f"SSL_VERIFY={settings.ssl_verify}",
        f"SSL_CERT_FILE={settings.ssl_cert_file or '<unset>'}",
        f"TIKTOKEN_ENABLED={settings.tiktoken_enabled}",
        f"TIKTOKEN_CACHE_DIR={settings.tiktoken_cache_dir or '<unset>'}",
    ]
    if needs_ollama:
        config_lines.append(f"OLLAMA_BASE_URL={settings.ollama_base_url}")
    if needs_azure:
        config_lines.extend(
            [
                f"AZURE_OPENAI_ENDPOINT={settings.azure_openai_endpoint or '<unset>'}",
                f"AZURE_OPENAI_CHAT_DEPLOYMENT={settings.azure_openai_chat_deployment or '<unset>'}",
                f"AZURE_OPENAI_JUDGE_DEPLOYMENT={settings.azure_openai_judge_deployment or '<unset>'}",
                f"AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT={settings.azure_openai_embed_deployment or '<unset>'}",
            ]
        )
    if needs_nvidia:
        config_lines.extend(
            [
                f"NVIDIA_OPENAI_ENDPOINT={settings.nvidia_openai_endpoint or '<unset>'}",
                f"NVIDIA_CHAT_MODEL={settings.nvidia_chat_model or '<unset>'}",
                f"NVIDIA_JUDGE_MODEL={settings.nvidia_judge_model or '<unset>'}",
                f"NVIDIA_API_TOKEN={'<set>' if settings.nvidia_api_token else '<unset>'}",
            ]
        )
    console.print(Panel("\n".join(config_lines), title="Selected Configuration"))

    checks: List[DoctorCheckResult] = []

    issues = validate_provider_dependencies(settings)
    if issues:
        details = "; ".join(f"{issue.module} ({', '.join(issue.contexts)})" for issue in issues)
        remediation = "Install dependencies with `python -m pip install -r requirements.txt`, then rerun `python run.py doctor`."
        checks.append(
            DoctorCheckResult(
                name="Provider dependency packages",
                status="FAIL",
                details=details,
                remediation=remediation,
            )
        )
    else:
        checks.append(
            DoctorCheckResult(
                name="Provider dependency packages",
                status="PASS",
                details="All required provider packages are importable.",
            )
        )

    config_issues = validate_provider_configuration(settings)
    if config_issues:
        details = "; ".join(f"({issue.context}) {issue.message}" for issue in config_issues)
        remediation = (
            "Fix provider variables in .env "
            "(Azure Gov endpoints like https://<resource>.openai.azure.us are supported; "
            "NVIDIA endpoints should be OpenAI-compatible base URLs)."
        )
        checks.append(
            DoctorCheckResult(
                name="Provider runtime configuration",
                status="FAIL",
                details=details,
                remediation=remediation,
            )
        )
    else:
        checks.append(
            DoctorCheckResult(
                name="Provider runtime configuration",
                status="PASS",
                details="Provider env/settings are internally consistent.",
            )
        )

    docker_check = check_docker_availability()
    checks.append(
        DoctorCheckResult(
            name="Docker daemon reachability",
            status="PASS" if docker_check.ok else "FAIL",
            details=docker_check.detail,
            remediation=docker_check.remediation,
        )
    )
    sandbox_probe = probe_sandbox_image(settings.sandbox_docker_image)
    checks.append(
        DoctorCheckResult(
            name="Sandbox image readiness",
            status="PASS" if sandbox_probe.ok else "FAIL",
            details=sandbox_probe.detail,
            remediation=sandbox_probe.remediation,
        )
    )

    if check_db:
        try:
            import psycopg2

            with psycopg2.connect(dsn=settings.pg_dsn, connect_timeout=max(1, int(timeout_seconds))) as conn:
                checks.append(
                    DoctorCheckResult(
                        name="PostgreSQL connectivity",
                        status="PASS",
                        details="Connected to PG_DSN successfully.",
                    )
                )
                chunks_dim = _read_table_embedding_dim(conn, "chunks")
                skill_chunks_dim = _read_table_embedding_dim(conn, "skill_chunks")
                if chunks_dim is None or skill_chunks_dim is None:
                    missing = []
                    if chunks_dim is None:
                        missing.append("chunks.embedding")
                    if skill_chunks_dim is None:
                        missing.append("skill_chunks.embedding")
                    checks.append(
                        DoctorCheckResult(
                            name="Embedding schema alignment",
                            status="WARN",
                            details=(
                                "Could not detect "
                                + ", ".join(missing)
                                + " dimension(s) (table may not exist yet)."
                            ),
                            remediation="Run `python run.py migrate` first, then rerun doctor.",
                        )
                    )
                elif chunks_dim != settings.embedding_dim or skill_chunks_dim != settings.embedding_dim:
                    mismatches = []
                    if chunks_dim != settings.embedding_dim:
                        mismatches.append(
                            f"chunks.embedding is vector({chunks_dim})"
                        )
                    if skill_chunks_dim != settings.embedding_dim:
                        mismatches.append(
                            f"skill_chunks.embedding is vector({skill_chunks_dim})"
                        )
                    checks.append(
                        DoctorCheckResult(
                            name="Embedding schema alignment",
                            status="FAIL",
                            details=(
                                "; ".join(mismatches)
                                + f" but EMBEDDING_DIM={settings.embedding_dim}."
                            ),
                            remediation="Run `python run.py migrate-embedding-dim --yes` to realign and rebuild vectors.",
                        )
                    )
                else:
                    checks.append(
                        DoctorCheckResult(
                            name="Embedding schema alignment",
                            status="PASS",
                            details=(
                                "chunks.embedding and skill_chunks.embedding dimensions match "
                                f"settings ({chunks_dim})."
                            ),
                        )
                    )

                from agentic_chatbot_next.rag.ingest import build_kb_health_report  # noqa: PLC0415

                class _DoctorDocStore:
                    def __init__(self, live_conn: Any) -> None:
                        self._conn = live_conn

                    def list_documents(
                        self,
                        source_type: str = "",
                        tenant_id: str = "local-dev",
                        collection_id: str = "",
                    ) -> List[dict[str, Any]]:
                        with self._conn.cursor() as cur:
                            if source_type and collection_id:
                                cur.execute(
                                    """
                                    SELECT source_path, title
                                    FROM documents
                                    WHERE tenant_id = %s AND source_type = %s AND collection_id = %s
                                    ORDER BY ingested_at
                                    """,
                                    (tenant_id, source_type, collection_id),
                                )
                            elif source_type:
                                cur.execute(
                                    """
                                    SELECT source_path, title
                                    FROM documents
                                    WHERE tenant_id = %s AND source_type = %s
                                    ORDER BY ingested_at
                                    """,
                                    (tenant_id, source_type),
                                )
                            elif collection_id:
                                cur.execute(
                                    """
                                    SELECT source_path, title
                                    FROM documents
                                    WHERE tenant_id = %s AND collection_id = %s
                                    ORDER BY ingested_at
                                    """,
                                    (tenant_id, collection_id),
                                )
                            else:
                                cur.execute(
                                    """
                                    SELECT source_path, title
                                    FROM documents
                                    WHERE tenant_id = %s
                                    ORDER BY ingested_at
                                    """,
                                    (tenant_id,),
                                )
                            rows = cur.fetchall() or []
                        records: List[dict[str, Any]] = []
                        for row in rows:
                            if isinstance(row, dict):
                                source_path = str(row.get("source_path") or "")
                                title = str(row.get("title") or "")
                            else:
                                source_path = str(row[0] if len(row) > 0 else "")
                                title = str(row[1] if len(row) > 1 else Path(source_path).name)
                            records.append(
                                {
                                    "source_path": source_path,
                                    "title": title,
                                    "source_type": source_type,
                                    "collection_id": collection_id,
                                }
                            )
                        return records

                class _DoctorStores:
                    def __init__(self, doc_store: Any) -> None:
                        self.doc_store = doc_store

                kb_health = build_kb_health_report(
                    settings,
                    _DoctorStores(_DoctorDocStore(conn)),
                    tenant_id=settings.default_tenant_id,
                    collection_id=settings.default_collection_id,
                )
                missing_source_paths = list(kb_health.missing_source_paths)
                if missing_source_paths:
                    preview = ", ".join(Path(path).name for path in missing_source_paths[:4])
                    if len(missing_source_paths) > 4:
                        preview += ", ..."
                    checks.append(
                        DoctorCheckResult(
                            name="KB corpus coverage",
                            status="WARN",
                            details=(
                                f"{len(missing_source_paths)} configured source file(s) are not indexed "
                                f"for collection '{settings.default_collection_id}': {preview}"
                            ),
                            remediation=f"Run `python run.py repair-kb --collection-id {settings.default_collection_id}` to repair KB coverage.",
                        )
                    )
                elif kb_health.duplicate_groups:
                    preview = ", ".join(group.title for group in kb_health.duplicate_groups[:4])
                    if len(kb_health.duplicate_groups) > 4:
                        preview += ", ..."
                    checks.append(
                        DoctorCheckResult(
                            name="KB corpus coverage",
                            status="WARN",
                            details=(
                                f"{len(kb_health.duplicate_groups)} KB source group(s) have duplicate indexed copies "
                                f"for collection '{settings.default_collection_id}': {preview}"
                            ),
                            remediation=f"Run `python run.py repair-kb --collection-id {settings.default_collection_id}` to prune stale duplicates.",
                        )
                    )
                elif kb_health.drifted_groups:
                    preview = ", ".join(group.title for group in kb_health.drifted_groups[:4])
                    if len(kb_health.drifted_groups) > 4:
                        preview += ", ..."
                    checks.append(
                        DoctorCheckResult(
                            name="KB corpus coverage",
                            status="WARN",
                            details=(
                                f"{len(kb_health.drifted_groups)} KB source file(s) differ from the active indexed copy "
                                f"for collection '{settings.default_collection_id}': {preview}"
                            ),
                            remediation=f"Run `python run.py repair-kb --collection-id {settings.default_collection_id}` to reindex drifted files.",
                        )
                    )
                else:
                    checks.append(
                        DoctorCheckResult(
                            name="KB corpus coverage",
                            status="PASS",
                            details=(
                                f"Indexed KB covers all {len(kb_health.configured_source_paths)} configured source files "
                                f"for collection '{settings.default_collection_id}'."
                            ),
                        )
                    )
        except Exception as exc:  # pragma: no cover - environment dependent
            checks.append(
                DoctorCheckResult(
                    name="PostgreSQL connectivity",
                    status="FAIL",
                    details=str(exc),
                    remediation="Ensure PostgreSQL is running and PG_DSN points to a reachable instance.",
                )
                )
            checks.append(
                DoctorCheckResult(
                    name="Embedding schema alignment",
                    status="SKIP",
                    details="Skipped because database connectivity failed.",
                )
            )
            checks.append(
                DoctorCheckResult(
                    name="KB corpus coverage",
                    status="SKIP",
                    details="Skipped because database connectivity failed.",
                )
            )
    else:
        checks.append(
            DoctorCheckResult(
                name="PostgreSQL connectivity",
                status="SKIP",
                details="Skipped by --skip-db.",
            )
        )
        checks.append(
            DoctorCheckResult(
                name="Embedding schema alignment",
                status="SKIP",
                details="Skipped by --skip-db.",
            )
        )
        checks.append(
            DoctorCheckResult(
                name="KB corpus coverage",
                status="SKIP",
                details="Skipped by --skip-db.",
            )
        )

    if not needs_ollama:
        checks.append(
            DoctorCheckResult(
                name="Ollama API reachability",
                status="SKIP",
                details="No Ollama provider selected.",
            )
        )
    elif not check_ollama:
        checks.append(
            DoctorCheckResult(
                name="Ollama API reachability",
                status="SKIP",
                details="Skipped by --skip-ollama.",
            )
        )
    else:
        url = settings.ollama_base_url.rstrip("/") + "/api/tags"
        required_models = _selected_ollama_models(settings)
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=timeout_seconds) as resp:
                code = int(resp.getcode() or 0)
                body = resp.read().decode("utf-8")
            if 200 <= code < 300:
                payload = json.loads(body or "{}")
                available_models = {
                    str(item.get("name")).strip()
                    for item in payload.get("models", [])
                    if isinstance(item, dict) and item.get("name")
                }
                missing_models = _ollama_missing_models(required_models, available_models)
                if missing_models:
                    checks.append(
                        DoctorCheckResult(
                            name="Ollama API reachability",
                            status="FAIL",
                            details=(
                                f"HTTP {code} from {url}, but missing configured Ollama model(s): "
                                f"{', '.join(missing_models)}."
                            ),
                            remediation=(
                                "Pull or create the missing Ollama models, or update the "
                                "OLLAMA_*_MODEL settings to models available at OLLAMA_BASE_URL."
                            ),
                        )
                    )
                else:
                    checks.append(
                        DoctorCheckResult(
                            name="Ollama API reachability",
                            status="PASS",
                            details=(
                                f"HTTP {code} from {url}. Available configured models: "
                                f"{', '.join(sorted(set(required_models.values())))}."
                            ),
                        )
                    )
            else:
                checks.append(
                    DoctorCheckResult(
                        name="Ollama API reachability",
                        status="FAIL",
                        details=f"Received HTTP {code} from {url}.",
                        remediation="Ensure Ollama is running and OLLAMA_BASE_URL is correct.",
                    )
                )
        except json.JSONDecodeError as exc:  # pragma: no cover - environment dependent
            checks.append(
                DoctorCheckResult(
                    name="Ollama API reachability",
                    status="FAIL",
                    details=f"Invalid JSON from {url}: {exc}",
                    remediation="Ensure OLLAMA_BASE_URL points to a working Ollama API endpoint.",
                )
            )
        except URLError as exc:  # pragma: no cover - environment dependent
            checks.append(
                DoctorCheckResult(
                    name="Ollama API reachability",
                    status="FAIL",
                    details=str(exc.reason),
                    remediation="Start Ollama or update OLLAMA_BASE_URL to a reachable endpoint.",
                )
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            checks.append(
                DoctorCheckResult(
                    name="Ollama API reachability",
                    status="FAIL",
                    details=str(exc),
                    remediation="Start Ollama or update OLLAMA_BASE_URL to a reachable endpoint.",
                )
            )

    table = Table(title="Doctor Preflight Results")
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")
    for row in checks:
        status_style = _doctor_status_style(row.status)
        table.add_row(row.name, f"[{status_style}]{row.status}[/{status_style}]", row.details)
    console.print(table)

    remediations = [row.remediation for row in checks if row.remediation]
    if remediations:
        unique_remediations: List[str] = []
        for remediation in remediations:
            if remediation not in unique_remediations:
                unique_remediations.append(remediation)
        console.print(Panel("\n".join(f"- {item}" for item in unique_remediations), title="Suggested Fixes"))

    fail_count = sum(1 for row in checks if row.status == "FAIL")
    warn_count = sum(1 for row in checks if row.status == "WARN")

    if fail_count > 0 or (strict and warn_count > 0):
        raise typer.Exit(code=1)

    console.print("[green]Doctor checks passed.[/green]")


@app.command("build-sandbox-image")
def build_sandbox_image_command(
    image: str = typer.Option(
        DEFAULT_SANDBOX_IMAGE,
        "--image",
        help="Docker image tag to build for the offline data analyst sandbox.",
    ),
    timeout_seconds: float = typer.Option(
        900.0,
        "--timeout-seconds",
        min=30.0,
        help="Maximum time to allow the Docker build to run.",
    ),
):
    """Build the dedicated offline Docker image used by the data analyst sandbox."""

    repo_root = Path(__file__).resolve().parents[2]
    result = build_sandbox_image(
        repo_root,
        image=image,
        timeout_seconds=timeout_seconds,
    )
    if result.ok:
        console.print(f"[green]{result.detail}[/green]")
        console.print(f"Command: {result.command}")
        return

    console.print(f"[red]{result.detail}[/red]")
    if result.command:
        console.print(f"Command: {result.command}")
    if result.remediation:
        console.print(Panel(result.remediation, title="Suggested Fix"))
    raise typer.Exit(code=1)


@app.command()
def demo(
    scenario: str = typer.Option("all", "--scenario", "-s", help="Scenario name, or 'all'."),
    list_scenarios: bool = typer.Option(False, "--list-scenarios", help="List available demo scenarios and exit."),
    max_turns: int = typer.Option(0, "--max-turns", help="Max prompts per scenario (0 = all)."),
    force_agent: bool = typer.Option(False, "--force-agent", help="Force AGENT path for every demo prompt."),
    session_mode: str = typer.Option(
        "scenario",
        "--session-mode",
        help="Session isolation mode: scenario (new session per scenario) or suite (one shared session).",
    ),
    verify: bool = typer.Option(False, "--verify", help="Run heuristic response checks and print PASS/WARN/FAIL."),
    show_notes: bool = typer.Option(False, "--show-notes", help="Show scenario briefing notes before execution."),
    upload: List[Path] = typer.Option([], "--upload", "-u"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--stop-on-error"),
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
):
    """Run a curated demo suite across multiple capabilities."""

    settings = load_settings(dotenv)
    scenarios = load_demo_scenarios(settings.data_dir)
    if list_scenarios:
        console.print("[bold]Available scenarios:[/bold]")
        for scenario_obj in scenarios.values():
            console.print(render_scenario_summary(scenario_obj))
        return

    session_mode = session_mode.strip().lower()
    if session_mode not in {"scenario", "suite"}:
        raise typer.BadParameter("--session-mode must be one of: scenario, suite")

    selected_names = list(scenarios.keys()) if scenario == "all" else [scenario]
    missing = [name for name in selected_names if name not in scenarios]
    if missing:
        raise typer.BadParameter(
            f"Unknown scenario(s): {', '.join(missing)}. "
            f"Use --list-scenarios to see valid names."
        )

    demo_settings = _with_demo_settings(settings)
    if demo_settings.chat_max_output_tokens != settings.chat_max_output_tokens:
        console.print(
            f"[cyan]Demo mode override:[/cyan] CHAT_MAX_OUTPUT_TOKENS={demo_settings.chat_max_output_tokens}"
        )

    bot = _build_bot_or_exit(demo_settings)
    shared_session = _make_local_session(dotenv, conversation_id="demo-suite") if session_mode == "suite" else None
    if shared_session is not None:
        shared_session.demo_mode = True
    if shared_session is not None and upload:
        console.print("[bold]Ingesting demo uploads for suite session...[/bold]")
        bot.ingest_and_summarize_uploads(shared_session, upload)

    verify_pass = 0
    verify_warn = 0
    verify_fail = 0

    for name in selected_names:
        scenario_obj = scenarios[name]
        turns = list(scenario_obj.turns)
        if max_turns > 0:
            turns = turns[:max_turns]

        if shared_session is None:
            session = _make_local_session(dotenv, conversation_id=f"demo-{name}")
            session.demo_mode = True
            if upload:
                console.print(f"[bold]Ingesting demo uploads for scenario '{name}'...[/bold]")
                bot.ingest_and_summarize_uploads(session, upload)
        else:
            session = shared_session

        header = f"{scenario_obj.id} - {scenario_obj.title} ({len(turns)} turn(s), difficulty={scenario_obj.difficulty})"
        console.print(Panel(header, title="Scenario"))
        console.print(f"[bold]Goal:[/bold] {scenario_obj.goal}")
        if scenario_obj.tool_focus:
            console.print(f"[bold]Tool focus:[/bold] {', '.join(scenario_obj.tool_focus)}")
        if show_notes and scenario_obj.notes:
            console.print(Panel(_render_demo_notes(scenario_obj), title="Scenario Notes"))

        for i, turn in enumerate(turns, start=1):
            console.print(f"[bold cyan]You[{i}]>[/bold cyan] {turn.prompt}")
            try:
                response = bot.process_turn(
                    session,
                    user_text=turn.prompt,
                    upload_paths=[],
                    force_agent=_coerce_force_agent(force_agent, turn),
                )
                console.print(Panel(response, title="Assistant"))

                if verify:
                    result = evaluate_response(
                        response,
                        scenario=scenario_obj,
                        turn=turn,
                    )
                    style = _verify_status_style(result.status)
                    console.print(f"[{style}]VERIFY {result.status}[/{style}] [{name} turn {i}]")
                    for message in result.messages:
                        console.print(f"- {message}")

                    if result.status == "PASS":
                        verify_pass += 1
                    elif result.status == "WARN":
                        verify_warn += 1
                    else:
                        verify_fail += 1
                        if not continue_on_error:
                            raise typer.Exit(code=1)
            except Exception as e:
                console.print(Panel(f"Demo prompt failed: {e}", title="Error"))
                if verify:
                    verify_fail += 1
                if not continue_on_error:
                    raise typer.Exit(code=1)

    if verify:
        summary = (
            f"PASS={verify_pass}  "
            f"WARN={verify_warn}  "
            f"FAIL={verify_fail}"
        )
        console.print(Panel(summary, title="Verification Summary"))


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Run the OpenAI-compatible FastAPI gateway."""
    import uvicorn

    uvicorn.run("agentic_chatbot_next.api.main:app", host=host, port=port, reload=reload, factory=False)


@app.command("runtime-smoke")
def runtime_smoke(
    dotenv: Optional[str] = typer.Option(None, "--dotenv"),
    registry_only: bool = typer.Option(False, "--registry-only", help="Validate agent/tool registry without opening stores."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable diagnostics."),
) -> None:
    """Validate the live runtime registry and required OpenWebUI upload tools."""

    from agentic_chatbot_next.runtime.kernel import RuntimeKernel
    from agentic_chatbot_next.tools.registry import build_tool_definitions

    settings = load_settings(dotenv)
    required_tools = {"list_worker_requests", "respond_worker_request"}
    tool_names = set(build_tool_definitions(None))
    missing_required = sorted(required_tools - tool_names)
    payload = {
        "status": "ready",
        "registry_valid": True,
        "required_tools_present": not missing_required,
        "missing_required_tools": missing_required,
    }
    if missing_required:
        payload.update(
            {
                "status": "not_ready",
                "registry_valid": False,
                "error_code": "runtime_registry_invalid",
                "remediation": "Rebuild/recreate the app image so the runtime tool registry includes worker-request tools.",
            }
        )
        if json_output:
            console.print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            console.print(Panel(json.dumps(payload, ensure_ascii=False, indent=2), title="Runtime Smoke Failed"))
        raise typer.Exit(code=1)

    try:
        if registry_only:
            RuntimeKernel(settings, providers=None, stores=None).validate_registry()
        else:
            bot = _build_bot(settings)
            bot.kernel.validate_registry()
    except Exception as exc:
        error_payload = build_runtime_error_payload(exc)
        payload.update({"status": "not_ready", "registry_valid": False, **error_payload})
        if json_output:
            console.print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            console.print(Panel(json.dumps(payload, ensure_ascii=False, indent=2), title="Runtime Smoke Failed"))
        raise typer.Exit(code=1) from exc

    if json_output:
        console.print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        console.print(Panel("Runtime registry smoke check passed.", title="Runtime Smoke"))
