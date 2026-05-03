from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable
from urllib.parse import urlparse


_EXPLICIT_MCP_RE = re.compile(
    r"\b(?:mcp|model\s+context\s+protocol)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "about",
    "active",
    "after",
    "again",
    "agent",
    "also",
    "available",
    "before",
    "between",
    "call",
    "could",
    "find",
    "from",
    "give",
    "have",
    "include",
    "into",
    "list",
    "look",
    "matching",
    "need",
    "open",
    "please",
    "return",
    "search",
    "server",
    "service",
    "show",
    "that",
    "them",
    "this",
    "tool",
    "tools",
    "using",
    "with",
    "would",
}


def empty_mcp_intent(*, discover_query: str = "") -> dict[str, Any]:
    return {
        "detected": False,
        "trigger": "",
        "discover_query": str(discover_query or "").strip(),
        "matched_connections": [],
        "matched_tools": [],
    }


def mcp_intent_detected(metadata: dict[str, Any] | None) -> bool:
    data = dict((metadata or {}).get("mcp_intent") or {})
    return bool(data.get("detected"))


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[_\-/.:]+", " ", text.casefold())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalise_text(value))
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _schema_text(value: Any, *, limit: int = 120) -> str:
    parts: list[str] = []

    def walk(item: Any) -> None:
        if len(parts) >= limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if len(parts) >= limit:
                    return
                parts.append(str(key))
                walk(child)
            return
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if isinstance(item, str):
            parts.append(item)

    walk(value)
    return " ".join(parts)


def _host_terms(server_url: str) -> str:
    try:
        parsed = urlparse(str(server_url or ""))
    except Exception:
        return ""
    host = str(parsed.hostname or "").strip()
    return host.replace(".", " ")


def _conceptual_mcp_question(query: str) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    if not re.search(r"^\s*(?:what\s+(?:is|are)|explain|define|tell\s+me\s+about)\b", text, re.IGNORECASE):
        return False
    return not bool(
        re.search(
            r"\b(?:use|using|call|invoke|search|find|get|pull|fetch|query|list|show|available|connect|register|inspect|refresh)\b",
            text,
            re.IGNORECASE,
        )
    )


def _is_connection_allowed_for_general(connection: Any) -> bool:
    allowed = [
        str(item or "").strip().casefold()
        for item in list(getattr(connection, "allowed_agents", []) or [])
        if str(item or "").strip()
    ]
    return not allowed or "*" in allowed or "general" in allowed


def _connection_search_text(connection: Any) -> str:
    metadata = getattr(connection, "metadata_json", None)
    metadata_text = _schema_text(metadata if isinstance(metadata, dict) else {})
    return " ".join(
        part
        for part in (
            getattr(connection, "display_name", ""),
            getattr(connection, "connection_slug", ""),
            _host_terms(getattr(connection, "server_url", "")),
            metadata_text,
        )
        if str(part or "").strip()
    )


def _tool_search_text(tool: Any, connection: Any | None = None) -> str:
    parts = [
        getattr(tool, "registry_name", ""),
        getattr(tool, "raw_tool_name", ""),
        getattr(tool, "tool_slug", ""),
        getattr(tool, "description", ""),
        getattr(tool, "search_hint", ""),
        _schema_text(getattr(tool, "input_schema", None)),
        _schema_text(getattr(tool, "metadata_json", None)),
    ]
    if connection is not None:
        parts.extend(
            [
                getattr(connection, "display_name", ""),
                getattr(connection, "connection_slug", ""),
            ]
        )
    return " ".join(str(part) for part in parts if str(part or "").strip())


def _phrase_match(query_normalised: str, candidates: Iterable[Any]) -> bool:
    for candidate in candidates:
        phrase = _normalise_text(candidate)
        if len(phrase) >= 3 and phrase in query_normalised:
            return True
    return False


def _connection_payload(connection: Any) -> dict[str, str]:
    return {
        "connection_id": str(getattr(connection, "connection_id", "") or ""),
        "display_name": str(getattr(connection, "display_name", "") or ""),
        "connection_slug": str(getattr(connection, "connection_slug", "") or ""),
    }


def _tool_payload(tool: Any) -> dict[str, str]:
    return {
        "registry_name": str(getattr(tool, "registry_name", "") or ""),
        "raw_tool_name": str(getattr(tool, "raw_tool_name", "") or ""),
        "connection_id": str(getattr(tool, "connection_id", "") or ""),
    }


def detect_mcp_intent(
    user_text: str,
    *,
    settings: Any | None,
    stores: Any | None,
    tenant_id: str,
    user_id: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Detect MCP service intent from explicit language or active catalog rows.

    The returned payload is intentionally small and contains no server URLs,
    tokens, or environment values. It is safe to put into route metadata.
    """

    query = str(user_text or "").strip()
    if not query:
        return empty_mcp_intent(discover_query=query)

    explicit = bool(_EXPLICIT_MCP_RE.search(query))
    tool_discovery_required = not _conceptual_mcp_question(query)
    if not bool(getattr(settings, "mcp_tool_plane_enabled", False)):
        if explicit:
            return {
                **empty_mcp_intent(discover_query=query),
                "detected": True,
                "trigger": "explicit_mcp",
                "tool_discovery_required": tool_discovery_required,
            }
        return empty_mcp_intent(discover_query=query)

    store = getattr(stores, "mcp_connection_store", None) if stores is not None else None
    if store is None:
        if explicit:
            return {
                **empty_mcp_intent(discover_query=query),
                "detected": True,
                "trigger": "explicit_mcp",
                "tool_discovery_required": tool_discovery_required,
            }
        return empty_mcp_intent(discover_query=query)

    try:
        connections = list(
            store.list_connections(
                tenant_id=tenant_id,
                owner_user_id=user_id,
                include_disabled=False,
            )
        )
    except TypeError:
        connections = list(store.list_connections())
    except Exception:
        connections = []

    visible_connections = {
        str(getattr(connection, "connection_id", "") or ""): connection
        for connection in connections
        if str(getattr(connection, "connection_id", "") or "")
        and str(getattr(connection, "status", "active") or "active").casefold() == "active"
        and _is_connection_allowed_for_general(connection)
    }

    try:
        tools = list(
            store.list_tool_catalog(
                tenant_id=tenant_id,
                owner_user_id=user_id,
                include_disabled=False,
            )
        )
    except TypeError:
        tools = list(store.list_tool_catalog())
    except Exception:
        tools = []

    active_tools: list[Any] = []
    for tool in tools:
        connection_id = str(getattr(tool, "connection_id", "") or "")
        if connection_id not in visible_connections:
            continue
        if not bool(getattr(tool, "enabled", True)):
            continue
        if str(getattr(tool, "status", "active") or "active").casefold() != "active":
            continue
        active_tools.append(tool)

    active_tool_connection_ids = {str(getattr(tool, "connection_id", "") or "") for tool in active_tools}
    query_normalised = _normalise_text(query)
    query_tokens = _tokens(query)
    connection_scores: list[tuple[int, Any]] = []
    matched_connection_ids: set[str] = set()

    for connection in visible_connections.values():
        connection_id = str(getattr(connection, "connection_id", "") or "")
        if connection_id not in active_tool_connection_ids:
            continue
        phrases = [
            getattr(connection, "display_name", ""),
            getattr(connection, "connection_slug", ""),
            _host_terms(getattr(connection, "server_url", "")),
        ]
        phrase_hit = _phrase_match(query_normalised, phrases)
        overlap = len(query_tokens & _tokens(_connection_search_text(connection)))
        score = (12 if phrase_hit else 0) + (overlap * 3)
        if phrase_hit or overlap >= 1:
            connection_scores.append((score, connection))
            matched_connection_ids.add(connection_id)

    tool_scores: list[tuple[int, Any]] = []
    for tool in active_tools:
        connection_id = str(getattr(tool, "connection_id", "") or "")
        connection = visible_connections.get(connection_id)
        if connection is None:
            continue
        phrases = [
            getattr(tool, "registry_name", ""),
            getattr(tool, "raw_tool_name", ""),
            getattr(tool, "tool_slug", ""),
        ]
        phrase_hit = _phrase_match(query_normalised, phrases)
        overlap = len(query_tokens & _tokens(_tool_search_text(tool, connection)))
        connection_hit = connection_id in matched_connection_ids
        score = (10 if phrase_hit else 0) + (4 if connection_hit else 0) + overlap
        if phrase_hit or connection_hit or overlap >= 3:
            tool_scores.append((score, tool))

    connection_scores.sort(key=lambda item: item[0], reverse=True)
    tool_scores.sort(key=lambda item: item[0], reverse=True)
    matched_connections = [_connection_payload(connection) for _, connection in connection_scores[:top_k]]
    matched_tools = [_tool_payload(tool) for _, tool in tool_scores[:top_k]]

    if explicit:
        return {
            "detected": True,
            "trigger": "explicit_mcp",
            "discover_query": query,
            "matched_connections": matched_connections,
            "matched_tools": matched_tools,
            "tool_discovery_required": tool_discovery_required,
        }
    if matched_connections:
        return {
            "detected": True,
            "trigger": "connection_match",
            "discover_query": query,
            "matched_connections": matched_connections,
            "matched_tools": matched_tools,
            "tool_discovery_required": True,
        }
    if matched_tools:
        return {
            "detected": True,
            "trigger": "tool_match",
            "discover_query": query,
            "matched_connections": matched_connections,
            "matched_tools": matched_tools,
            "tool_discovery_required": True,
        }
    return empty_mcp_intent(discover_query=query)


__all__ = ["detect_mcp_intent", "empty_mcp_intent", "mcp_intent_detected"]
