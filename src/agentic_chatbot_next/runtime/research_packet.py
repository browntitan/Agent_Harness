from __future__ import annotations

from typing import Any, Dict, Iterable, List

from agentic_chatbot_next.runtime.doc_focus import active_doc_focus_from_metadata
from agentic_chatbot_next.runtime.turn_contracts import message_is_context_eligible


def _message_role(message: Any) -> str:
    role = str(getattr(message, "role", "") or "").strip().lower()
    if role:
        return role
    role = str(getattr(message, "type", "") or "").strip().lower()
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return role or "assistant"


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return " ".join(str(item) for item in content if str(item).strip()).strip()
    return str(content or "").strip()


def _message_metadata(message: Any) -> Dict[str, Any]:
    raw = getattr(message, "metadata", None)
    if isinstance(raw, dict):
        return dict(raw)
    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, dict):
        return dict(additional)
    return {}


def _summarize_retrieval_summary(summary: Dict[str, Any]) -> str:
    query_used = str(summary.get("query_used") or "").strip()
    search_mode = str(summary.get("search_mode") or "").strip()
    rounds = int(summary.get("rounds") or 0)
    strategies = [str(item) for item in (summary.get("strategies_used") or []) if str(item)]
    candidate_counts = dict(summary.get("candidate_counts") or {})
    selected_docs = int(candidate_counts.get("selected_docs") or 0)
    strong_chunks = int(candidate_counts.get("strong_chunks") or 0)
    parts: List[str] = []
    if query_used:
        parts.append(f"query={query_used[:120]}")
    if search_mode:
        parts.append(f"mode={search_mode}")
    if rounds:
        parts.append(f"rounds={rounds}")
    if strategies:
        parts.append(f"strategies={','.join(strategies[:4])}")
    if selected_docs:
        parts.append(f"selected_docs={selected_docs}")
    if strong_chunks:
        parts.append(f"strong_chunks={strong_chunks}")
    return ", ".join(parts)


def _recent_retrieval_rows(messages: Iterable[Any], *, limit: int) -> List[str]:
    rows: List[str] = []
    for message in reversed(list(messages)):
        if not message_is_context_eligible(message):
            continue
        metadata = _message_metadata(message)
        rag_contract = dict(metadata.get("rag_contract") or {})
        retrieval_summary = dict(rag_contract.get("retrieval_summary") or {})
        if not retrieval_summary:
            continue
        rendered = _summarize_retrieval_summary(retrieval_summary)
        if not rendered:
            continue
        rows.append(f"- {rendered}")
        if len(rows) >= max(1, int(limit)):
            break
    return list(reversed(rows))


def _recent_message_rows(messages: Iterable[Any], *, limit: int, char_limit: int) -> List[str]:
    rows: List[str] = []
    for message in reversed(list(messages)):
        if not message_is_context_eligible(message):
            continue
        role = _message_role(message)
        if role not in {"user", "assistant"}:
            continue
        content = _message_content(message)
        if not content:
            continue
        rows.append(f"- {role}: {content[: max(40, int(char_limit))]}")
        if len(rows) >= max(1, int(limit)):
            break
    return list(reversed(rows))


def build_research_packet(
    session_like: Any,
    *,
    recent_messages: int = 8,
    message_char_limit: int = 280,
    retrieval_limit: int = 3,
) -> str:
    messages = list(getattr(session_like, "messages", []) or [])
    metadata = dict(getattr(session_like, "metadata", {}) or {})
    uploaded_doc_ids = [str(item) for item in list(getattr(session_like, "uploaded_doc_ids", []) or []) if str(item)]
    blocks: List[str] = []

    recent_rows = _recent_message_rows(
        messages,
        limit=max(1, int(recent_messages)),
        char_limit=max(80, int(message_char_limit)),
    )
    if recent_rows:
        blocks.append("## Recent Conversation\n" + "\n".join(recent_rows))

    scope_rows: List[str] = []
    kb_collection_id = str(
        metadata.get("kb_collection_id")
        or metadata.get("collection_id")
        or metadata.get("requested_kb_collection_id")
        or "default"
    ).strip()
    if kb_collection_id:
        scope_rows.append(f"- kb_collection_id: {kb_collection_id}")
    if uploaded_doc_ids:
        scope_rows.append(f"- uploaded_doc_ids: {', '.join(uploaded_doc_ids[:6])}")
    elif metadata.get("upload_collection_id"):
        scope_rows.append(f"- upload_collection_id: {metadata.get('upload_collection_id')}")
    retrieval_scope_mode = str(metadata.get("retrieval_scope_mode") or "").strip()
    if retrieval_scope_mode:
        scope_rows.append(f"- retrieval_scope_mode: {retrieval_scope_mode}")
    if scope_rows:
        blocks.append("## Retrieval Scope\n" + "\n".join(scope_rows))

    active_doc_focus = active_doc_focus_from_metadata(metadata)
    if active_doc_focus is not None:
        focus_rows: List[str] = []
        for item in list(active_doc_focus.get("documents") or [])[:6]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("doc_id") or "").strip()
            doc_id = str(item.get("doc_id") or "").strip()
            if title and doc_id and title != doc_id:
                focus_rows.append(f"- {title} ({doc_id})")
            elif title:
                focus_rows.append(f"- {title}")
        if focus_rows:
            blocks.append(
                "## Active Document Focus\n"
                f"Collection: {active_doc_focus.get('collection_id')}\n"
                + "\n".join(focus_rows)
            )

    pending = dict(metadata.get("pending_clarification") or {})
    pending_question = str(pending.get("question") or "").strip()
    if pending_question:
        blocks.append(
            "## Pending Clarification\n"
            f"- question: {pending_question}\n"
            f"- reason: {str(pending.get('reason') or '').strip()}"
        )

    retrieval_rows = _recent_retrieval_rows(messages, limit=max(1, int(retrieval_limit)))
    if retrieval_rows:
        blocks.append("## Prior Retrieval Summaries\n" + "\n".join(retrieval_rows))

    route_context = dict(metadata.get("route_context") or {})
    deep_rag = dict(route_context.get("deep_rag") or {})
    if deep_rag:
        deep_rows: List[str] = []
        for key in ("mode", "search_mode", "preferred_agent", "background_recommended", "reasoning"):
            value = deep_rag.get(key)
            if value in (None, "", []):
                continue
            deep_rows.append(f"- {key}: {value}")
        if deep_rows:
            blocks.append("## Deep RAG Policy\n" + "\n".join(deep_rows))

    return "\n\n".join(block for block in blocks if block.strip()).strip()


__all__ = ["build_research_packet"]
