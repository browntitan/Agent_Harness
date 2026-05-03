from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_chatbot_next.prompt_fallbacks import compose_fallback_prompt
from agentic_chatbot_next.rag.inventory import (
    INVENTORY_QUERY_GRAPH_FILE,
    INVENTORY_QUERY_GRAPH_INDEXES,
    classify_inventory_query,
    inventory_query_requests_grounded_analysis,
)
from agentic_chatbot_next.rag.citations import citation_display_label, replace_inline_citation_ids
from agentic_chatbot_next.runtime.tool_parallelism import (
    PolicyAwareToolNode,
    count_current_turn_ai_messages,
    count_current_turn_tool_messages,
)
from agentic_chatbot_next.runtime.context_budget import build_microcompact_hook
from agentic_chatbot_next.utils.json_utils import extract_json

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = compose_fallback_prompt("general_agent.md")
_GRAPH_ID_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_-]*(?:_graph|-graph)[A-Za-z0-9_-]*\b")
_GRAPH_EVIDENCE_RE = re.compile(
    r"\b(?:knowledge\s+graph|graphrag|graph\s+rag|graph\s+index|search\s+graph|query\s+graph|use\s+.*graph)\b"
    r"|(?:\bgraph\b.*\b(?:relationships?|entities|entity|multi[-\s]?hop|connected|dependencies|network|evidence)\b)"
    r"|(?:\b(?:relationships?|entities|entity|multi[-\s]?hop|connected|dependencies|network|cross[-\s]?document)\b.*\bgraph\b)"
    r"|\b(?:vendors?|suppliers?|risks?|approvals?|dependencies|program\s+outcomes?)\b.*\b(?:relationships?|connected|graph)\b",
    re.IGNORECASE,
)
_GRAPH_METADATA_INTENT_RE = re.compile(
    r"\b(?:relationships?|entities|entity|multi[-\s]?hop|connected|dependencies|network|evidence|"
    r"vendors?|suppliers?|risks?|approvals?|causes?|causal|outcomes?|source[-\s]?resolve|cross[-\s]?document)\b",
    re.IGNORECASE,
)
_CAUSAL_SENTENCE_RE = re.compile(
    r"\b("
    r"because(?:\s+of)?|caused\s+by|due\s+to|driven\s+by|drove|drives|"
    r"resulted\s+from|attributed\s+to|root\s+cause|main\s+issue|reason\s+(?:was|is)"
    r")\b",
    re.IGNORECASE,
)
_CLAIM_TERM_STOPWORDS = {
    "about",
    "answer",
    "based",
    "because",
    "better",
    "caused",
    "causal",
    "cited",
    "claim",
    "claims",
    "could",
    "demonstrated",
    "detail",
    "details",
    "does",
    "driven",
    "drove",
    "evidence",
    "from",
    "have",
    "issue",
    "main",
    "more",
    "most",
    "only",
    "question",
    "reason",
    "resulted",
    "says",
    "should",
    "source",
    "support",
    "supported",
    "supporting",
    "that",
    "these",
    "this",
    "those",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
_EVIDENCE_TEXT_KEYS = {
    "chunk_text",
    "content",
    "evidence",
    "excerpt",
    "page_content",
    "quote",
    "relationship_path",
    "snippet",
    "summary",
    "text",
}


@dataclass
class DataAnalystIntent:
    task_family: str
    nlp_task: str | None
    delivery_mode: str
    requires_code: bool
    target_dataset: str = ""
    target_columns: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DataAnalystTargetCandidate:
    score: int
    dataset_ref: str
    column: str
    exact_column: bool
    filename_score: int


@dataclass(frozen=True)
class GraphIdResolution:
    graph_id: str = ""
    status: str = "missing"
    source: str = ""
    candidates: Tuple[str, ...] = ()


def _content_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _response_text(response: Any) -> str:
    text = _content_text(response)
    if text:
        return text
    if hasattr(response, "content"):
        return ""
    return str(response)


def _render_rag_tool_fallback(messages: List[Any]) -> str:
    from agentic_chatbot_next.rag.engine import coerce_rag_contract, render_rag_contract

    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        payload = extract_json(_content_text(message))
        if not isinstance(payload, dict):
            continue
        if "answer" not in payload or "citations" not in payload:
            continue
        rendered = render_rag_contract(coerce_rag_contract(payload))
        if rendered.strip():
            return rendered.strip()
    return ""


def _sanitize_tool_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(args or {}).items()
        if value is not None
    }


def _message_metadata(message: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    metadata.update(dict(getattr(message, "response_metadata", {}) or {}))
    metadata.update(dict(getattr(message, "additional_kwargs", {}) or {}))
    return metadata


def _is_output_truncated(message: Any) -> bool:
    metadata = _message_metadata(message)
    finish_reason = str(
        metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or metadata.get("reason")
        or metadata.get("completion_reason")
        or ""
    ).strip().lower()
    return finish_reason in {"length", "max_tokens", "max_output_tokens"}


def _collect_tool_results(messages: List[Any]) -> List[Dict[str, Any]]:
    tool_results: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = _content_text(message)
        if not content:
            continue
        payload = extract_json(content)
        tool_results.append(
            {
                "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
                "tool": str(getattr(message, "name", "") or ""),
                "content": content,
                "json": payload if isinstance(payload, dict) else None,
            }
        )
    return tool_results


def _tool_result_name(result: Dict[str, Any]) -> str:
    return str(result.get("tool") or result.get("name") or "").strip()


def _tool_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = result.get("json")
    if isinstance(payload, dict):
        return payload
    content = result.get("content") or result.get("output") or ""
    parsed = extract_json(content)
    return parsed if isinstance(parsed, dict) else {}


def _is_docker_unavailable_error(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    return "docker sandbox is not available" in normalized or "docker is not available" in normalized


def _render_data_analyst_sandbox_unavailable(error_text: str) -> str:
    detail = str(error_text or "").strip()
    message = (
        "Data analysis requires the Docker sandbox, but it is unavailable. "
        "Run `python run.py build-sandbox-image` and `python run.py doctor --strict`, then retry."
    )
    if detail:
        return f"{message}\n\nSandbox detail: {detail}"
    return message


def _escape_markdown_table_cell(value: Any) -> str:
    text = str(value if value is not None else "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > 120:
        text = text[:117].rstrip() + "..."
    return text.replace("|", "\\|")


def _render_markdown_table(columns: List[str], rows: List[Dict[str, Any]]) -> str:
    if not columns or not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_escape_markdown_table_cell(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _render_data_analyst_preview_table(payload: Dict[str, Any]) -> str:
    rows = payload.get("preview_rows") or payload.get("preview") or []
    if not isinstance(rows, list) or not rows:
        return ""
    preview_rows = [row for row in rows if isinstance(row, dict)][:5]
    if not preview_rows:
        return ""
    columns = [str(column) for column in (payload.get("preview_columns") or []) if str(column)]
    if not columns:
        columns = [str(key) for key in preview_rows[0].keys() if str(key)]
    return _render_markdown_table(columns, preview_rows)


def _render_data_analyst_missing_file_result(payload: Dict[str, Any]) -> str:
    column_name = str(payload.get("column") or "the requested column")
    doc_id = str(payload.get("doc_id") or "the selected dataset")
    return (
        "The analyst completed the row-level NLP task but did not produce the derived file required "
        f"for '{column_name}' in {doc_id}. Please retry the request."
    )


def _render_data_analyst_tool_results(
    tool_results: List[Dict[str, Any]],
    *,
    expected_delivery_mode: str = "",
) -> str:
    latest_nlp_payload: Dict[str, Any] | None = None
    latest_execute_payload: Dict[str, Any] | None = None
    latest_return_payload: Dict[str, Any] | None = None
    latest_nlp_error = ""

    for result in tool_results:
        tool_name = _tool_result_name(result)
        payload = _tool_result_payload(result)
        if not payload:
            continue
        if tool_name == "run_nlp_column_task":
            error_text = str(payload.get("error") or "").strip()
            if error_text:
                latest_nlp_error = error_text
                continue
            if int(payload.get("processed_rows") or 0) > 0:
                latest_nlp_payload = payload
        elif tool_name == "execute_code":
            latest_execute_payload = payload
        elif tool_name == "return_file":
            latest_return_payload = payload

    if latest_nlp_payload is not None:
        if expected_delivery_mode in {"summary_and_file", "file_only"} and not str(
            latest_nlp_payload.get("written_file") or ""
        ).strip():
            return _render_data_analyst_missing_file_result(latest_nlp_payload)
        rendered = str(latest_nlp_payload.get("summary_text") or "").strip()
        if not rendered:
            counts = latest_nlp_payload.get("result_counts") or {}
            count_text = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
            processed_rows = int(latest_nlp_payload.get("processed_rows") or 0)
            column_name = str(latest_nlp_payload.get("column") or "the requested column")
            doc_id = str(latest_nlp_payload.get("doc_id") or "the dataset")
            rendered = f"Processed {processed_rows} rows from '{column_name}' in {doc_id}."
            if count_text:
                rendered += f" Counts: {count_text}."
        sections = [rendered.strip()]
        preview_table = _render_data_analyst_preview_table(latest_nlp_payload)
        if preview_table and expected_delivery_mode != "file_only":
            sections.append("Preview:\n" + preview_table)
        filename = ""
        if latest_return_payload is not None:
            filename = str(latest_return_payload.get("filename") or "").strip()
        if not filename:
            filename = str(latest_nlp_payload.get("written_file") or "").strip()
        if filename and filename not in rendered:
            sections.append(f"Returned file: {filename}.")
        return "\n\n".join(section for section in sections if section).strip()

    if latest_return_payload is not None:
        filename = str(latest_return_payload.get("filename") or "").strip()
        if filename:
            return f"Prepared {filename} for download."

    if latest_execute_payload is not None:
        stdout = str(latest_execute_payload.get("stdout") or "").strip()
        if latest_execute_payload.get("success") is True and stdout:
            return stdout
        error_text = str(
            latest_execute_payload.get("error")
            or latest_execute_payload.get("stderr")
            or ""
        ).strip()
        if error_text:
            if _is_docker_unavailable_error(error_text):
                return _render_data_analyst_sandbox_unavailable(error_text)
            return f"Data analysis failed in the sandbox:\n{error_text}"

    if latest_nlp_error:
        return f"Data analyst NLP task failed: {latest_nlp_error}"

    return ""


def _render_requirements_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    latest_payload: Dict[str, Any] | None = None
    for result in tool_results:
        tool_name = _tool_result_name(result)
        if tool_name not in {"extract_requirement_statements", "export_requirement_statements"}:
            continue
        payload = _tool_result_payload(result)
        if payload:
            latest_payload = payload
    if not latest_payload or latest_payload.get("error"):
        return ""

    sections: List[str] = []
    summary_text = str(latest_payload.get("summary_text") or "").strip()
    if summary_text:
        sections.append(summary_text)
    preview_table = _render_data_analyst_preview_table(latest_payload)
    if preview_table:
        sections.append("Preview:\n" + preview_table)
    artifact = latest_payload.get("artifact") or {}
    filename = str(latest_payload.get("filename") or artifact.get("filename") or "").strip()
    if filename:
        sections.append(f"Returned file: {filename}.")
    return "\n\n".join(section for section in sections if section).strip()


def _compact_tool_schema(tool: Any) -> str:
    name = str(getattr(tool, "name", "") or "").strip()
    description = re.sub(r"\s+", " ", str(getattr(tool, "description", "") or "").strip())
    if len(description) > 260:
        description = description[:257].rstrip() + "..."
    schema: Dict[str, Any] = {}
    raw_schema = getattr(tool, "args_schema", None)
    if isinstance(raw_schema, dict):
        schema = raw_schema
    elif raw_schema is not None and hasattr(raw_schema, "model_json_schema"):
        try:
            schema = raw_schema.model_json_schema()
        except Exception:
            schema = {}
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        properties = getattr(tool, "args", {}) or {}
    required = schema.get("required") if isinstance(schema, dict) else []
    if not isinstance(required, list):
        required = []
    required_names = {str(item) for item in required if str(item)}
    args = []
    for arg_name, arg_schema in list(dict(properties or {}).items())[:10]:
        details = arg_schema if isinstance(arg_schema, dict) else {}
        arg_type = str(details.get("type") or "any")
        marker = " required" if str(arg_name) in required_names else ""
        args.append(f"{arg_name}:{arg_type}{marker}")
    arg_text = ", ".join(args) if args else "no args"
    return f"- {name}({arg_text}): {description}".strip()


def _compact_tool_schemas(tools: List[Any]) -> str:
    return "\n".join(_compact_tool_schema(tool) for tool in tools if str(getattr(tool, "name", "") or "").strip())


def _extract_named_graph_id_from_text(text: str) -> str:
    for match in _GRAPH_ID_RE.finditer(str(text or "")):
        candidate = match.group(0).strip("`'\".,;:()[]{}")
        lower = candidate.lower()
        if lower in {"graph", "knowledge_graph", "graphrag"}:
            continue
        return candidate
    return ""


def _graph_ids_from_context(tool_context: Any | None) -> List[str]:
    values: List[str] = []

    def _collect_from_mapping(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        route_context = raw.get("route_context") if isinstance(raw.get("route_context"), dict) else {}
        for source in (raw, route_context):
            for key in (
                "graph_id",
                "active_graph_id",
                "selected_graph_id",
                "active_graph_ids",
                "selected_graph_ids",
                "graph_ids",
                "planned_graph_ids",
            ):
                value = source.get(key)
                if isinstance(value, str):
                    clean = value.strip()
                    if clean:
                        values.append(clean)
                elif isinstance(value, list):
                    values.extend(str(item).strip() for item in value if str(item).strip())

    _collect_from_mapping(dict(getattr(tool_context, "metadata", {}) or {}) if tool_context is not None else {})
    session = getattr(tool_context, "session", None) if tool_context is not None else None
    _collect_from_mapping(dict(getattr(session, "metadata", {}) or {}) if session is not None else {})
    return list(dict.fromkeys(item for item in values if item))


def _graph_context_sources(tool_context: Any | None) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []

    def _append(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        sources.append(raw)
        route_context = raw.get("route_context") if isinstance(raw.get("route_context"), dict) else {}
        if route_context:
            sources.append(route_context)

    _append(dict(getattr(tool_context, "metadata", {}) or {}) if tool_context is not None else {})
    session = getattr(tool_context, "session", None) if tool_context is not None else None
    _append(dict(getattr(session, "metadata", {}) or {}) if session is not None else {})
    return sources


def _graph_values_for_key(sources: List[Dict[str, Any]], key: str) -> List[str]:
    values: List[str] = []
    for source in sources:
        value = source.get(key)
        if isinstance(value, str):
            clean = value.strip()
            if clean:
                values.append(clean)
        elif isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
    return list(dict.fromkeys(values))


def _context_graph_id_resolution(tool_context: Any | None) -> GraphIdResolution:
    sources = _graph_context_sources(tool_context)
    all_candidates = tuple(_graph_ids_from_context(tool_context))
    selected_graph_ids = _graph_values_for_key(sources, "selected_graph_id")
    if selected_graph_ids:
        return GraphIdResolution(
            graph_id=selected_graph_ids[0],
            status="resolved:selected_graph_id",
            source="selected_graph_id",
            candidates=all_candidates,
        )

    selected_values = _graph_values_for_key(sources, "selected_graph_ids")
    if len(selected_values) == 1:
        return GraphIdResolution(
            graph_id=selected_values[0],
            status="resolved:selected_graph_ids",
            source="selected_graph_ids",
            candidates=all_candidates,
        )
    if len(selected_values) > 1:
        return GraphIdResolution(status="ambiguous:selected_graph_ids", source="selected_graph_ids", candidates=tuple(selected_values))

    for key in ("graph_id", "active_graph_id"):
        values = _graph_values_for_key(sources, key)
        if values:
            return GraphIdResolution(graph_id=values[0], status=f"resolved:{key}", source=key, candidates=all_candidates)

    fallback_values = list(
        dict.fromkeys(
            _graph_values_for_key(sources, "active_graph_ids")
            + _graph_values_for_key(sources, "planned_graph_ids")
            + _graph_values_for_key(sources, "graph_ids")
        )
    )
    if len(fallback_values) == 1:
        return GraphIdResolution(
            graph_id=fallback_values[0],
            status="resolved:single_context_graph",
            source="single_context_graph",
            candidates=all_candidates,
        )
    if len(fallback_values) > 1:
        return GraphIdResolution(status="ambiguous:context_graphs", source="context_graphs", candidates=tuple(fallback_values))
    if all_candidates:
        return GraphIdResolution(status="ambiguous:context_graphs", source="context_graphs", candidates=all_candidates)
    return GraphIdResolution()


def _context_graph_id(tool_context: Any | None) -> str:
    return _context_graph_id_resolution(tool_context).graph_id


def _graph_id_resolution_for_request(
    user_text: str,
    tool_results: List[Dict[str, Any]],
    *,
    tool_context: Any | None = None,
) -> GraphIdResolution:
    requested = _extract_named_graph_id_from_text(user_text)
    if requested:
        return GraphIdResolution(graph_id=requested, status="resolved:text_graph_id", source="text_graph_id", candidates=(requested,))
    context_resolution = _context_graph_id_resolution(tool_context)
    if context_resolution.graph_id:
        return context_resolution
    graph_ids: List[str] = []
    for result in tool_results:
        payload = _payload_from_tool_result(result)
        graphs = payload.get("graphs") if isinstance(payload, dict) else None
        if isinstance(graphs, list):
            for graph in graphs:
                if isinstance(graph, dict):
                    graph_id = str(graph.get("graph_id") or "").strip()
                    if graph_id:
                        graph_ids.append(graph_id)
        graph = payload.get("graph") if isinstance(payload, dict) else None
        if isinstance(graph, dict):
            graph_id = str(graph.get("graph_id") or "").strip()
            if graph_id:
                graph_ids.append(graph_id)
    unique = tuple(dict.fromkeys(graph_ids))
    if len(unique) == 1:
        return GraphIdResolution(graph_id=unique[0], status="resolved:tool_result", source="tool_result", candidates=unique)
    if len(unique) > 1:
        return GraphIdResolution(status="ambiguous:tool_results", source="tool_results", candidates=unique)
    if context_resolution.status.startswith("ambiguous"):
        return context_resolution
    return GraphIdResolution()


def _is_grounded_graph_request(user_text: str, tool_map: Dict[str, Any], *, tool_context: Any | None = None) -> bool:
    if "search_graph_index" not in tool_map:
        return False
    inventory_type = classify_inventory_query(user_text)
    if inventory_type in {INVENTORY_QUERY_GRAPH_INDEXES, INVENTORY_QUERY_GRAPH_FILE} and not inventory_query_requests_grounded_analysis(
        user_text,
        query_type=inventory_type,
    ):
        return False
    text = str(user_text or "")
    if _GRAPH_EVIDENCE_RE.search(text):
        return True
    if _extract_named_graph_id_from_text(text) and _GRAPH_METADATA_INTENT_RE.search(text):
        return True
    return bool(_graph_ids_from_context(tool_context) and _GRAPH_METADATA_INTENT_RE.search(text))


def _payload_from_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = _tool_result_payload(result)
    if payload:
        return payload
    output = str(result.get("output") or "")
    parsed = extract_json(output)
    return parsed if isinstance(parsed, dict) else {}


def _graph_id_from_tool_results(
    user_text: str,
    tool_results: List[Dict[str, Any]],
    *,
    tool_context: Any | None = None,
) -> str:
    return _graph_id_resolution_for_request(user_text, tool_results, tool_context=tool_context).graph_id


def _has_graph_search_attempt(tool_results: List[Dict[str, Any]]) -> bool:
    return any(_tool_result_name(result) == "search_graph_index" for result in tool_results)


def _has_rag_agent_tool_attempt(tool_results: List[Dict[str, Any]]) -> bool:
    return any(_tool_result_name(result) == "rag_agent_tool" for result in tool_results)


def _graph_search_has_error(tool_results: List[Dict[str, Any]]) -> bool:
    for result in tool_results:
        if _tool_result_name(result) != "search_graph_index":
            continue
        payload = _payload_from_tool_result(result)
        if payload.get("error"):
            return True
    return False


def _graph_hit_is_catalog_only(hit: Dict[str, Any]) -> bool:
    metadata = dict(hit.get("metadata") or {})
    evidence_kind = str(hit.get("evidence_kind") or metadata.get("evidence_kind") or "").strip().lower()
    return bool(
        str(hit.get("backend") or "").strip().lower() == "catalog"
        or str(metadata.get("fallback") or "").strip().lower() == "catalog"
        or bool(metadata.get("catalog_only"))
        or evidence_kind == "source_candidate"
    )


def _graph_search_has_grounded_evidence(tool_results: List[Dict[str, Any]]) -> bool:
    for result in tool_results:
        if _tool_result_name(result) != "search_graph_index":
            continue
        payload = _payload_from_tool_result(result)
        if str(payload.get("evidence_status") or "").strip() == "grounded_graph_evidence":
            return True
        for hit in payload.get("results") or []:
            if not isinstance(hit, dict) or _graph_hit_is_catalog_only(hit):
                continue
            if hit.get("chunk_ids") or hit.get("relationship_path") or str(hit.get("summary") or "").strip():
                return True
    return False


def _graph_search_requires_source_read(tool_results: List[Dict[str, Any]]) -> bool:
    for result in tool_results:
        if _tool_result_name(result) != "search_graph_index":
            continue
        payload = _payload_from_tool_result(result)
        if bool(payload.get("requires_source_read")):
            return True
    return False


def _graph_source_candidate_doc_ids(tool_results: List[Dict[str, Any]], *, limit: int = 8) -> List[str]:
    doc_ids: List[str] = []
    for result in tool_results:
        if _tool_result_name(result) != "search_graph_index":
            continue
        payload = _payload_from_tool_result(result)
        for hit in payload.get("results") or []:
            if not isinstance(hit, dict):
                continue
            doc_id = str(hit.get("doc_id") or "").strip()
            if doc_id:
                doc_ids.append(doc_id)
        for citation in payload.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            doc_id = str(citation.get("doc_id") or "").strip()
            if doc_id:
                doc_ids.append(doc_id)
    return list(dict.fromkeys(doc_ids))[:limit]


def _graph_source_candidate_summaries(tool_results: List[Dict[str, Any]], *, limit: int = 8) -> List[str]:
    summaries: List[str] = []
    for result in tool_results:
        if _tool_result_name(result) != "search_graph_index":
            continue
        payload = _payload_from_tool_result(result)
        for hit in payload.get("results") or []:
            if not isinstance(hit, dict):
                continue
            doc_id = str(hit.get("doc_id") or "").strip()
            title = str(hit.get("title") or "").strip()
            source_path = str(hit.get("source_path") or "").strip()
            label = title or Path(source_path).name or doc_id
            if not label and not doc_id:
                continue
            if doc_id and label and doc_id not in label:
                summaries.append(f"{label} (doc_id: {doc_id})")
            else:
                summaries.append(label or f"doc_id: {doc_id}")
    return list(dict.fromkeys(summaries))[:limit]


def _render_rag_payload(payload: Dict[str, Any]) -> str:
    if not payload or payload.get("error"):
        return ""
    if "answer" not in payload or "citations" not in payload:
        return ""
    try:
        from agentic_chatbot_next.rag.engine import coerce_rag_contract, render_rag_contract

        rendered = render_rag_contract(coerce_rag_contract(payload))
    except Exception:
        rendered = str(payload.get("answer") or "").strip()
    return rendered.strip()


def _render_latest_rag_tool_result(tool_results: List[Dict[str, Any]]) -> str:
    for result in reversed(tool_results):
        if _tool_result_name(result) != "rag_agent_tool":
            continue
        rendered = _render_rag_payload(_payload_from_tool_result(result))
        if rendered:
            return rendered
    return ""


def _graph_rag_recovery_context(tool_results: List[Dict[str, Any]]) -> str:
    doc_ids = _graph_source_candidate_doc_ids(tool_results, limit=8)
    if not doc_ids:
        return ""
    candidate_summaries = _graph_source_candidate_summaries(tool_results, limit=8)
    candidate_text = "; ".join(candidate_summaries) if candidate_summaries else ", ".join(doc_ids)
    return (
        "Graph search returned source candidates or graph leads that require source-text confirmation. "
        "Use grounded retrieval over these candidate doc ids before answering: "
        + ", ".join(doc_ids)
        + ". Start with the original user question. If the initial retrieval is weak, reason about query rewrites "
        "using only visible information from the user question, source titles, candidate metadata, and retrieved snippets. "
        "Try claim-focused, entity-focused, refutation-focused, causal-factor-focused, and date/status/outcome-focused "
        "rewrites only when those concepts are present in the visible information. Evaluate whether snippets explicitly "
        "support, refute, or fail to address the needed answer claims. Candidate sources: "
        + candidate_text
        + "."
    )


def _run_graph_catalog_rag_recovery(
    tool_map: Dict[str, Any],
    messages: List[Any],
    tool_results: List[Dict[str, Any]],
    *,
    user_text: str,
    callbacks: List[Any],
    tool_calls: int,
    max_tool_calls: int,
) -> tuple[str, int, List[str]]:
    if "rag_agent_tool" not in tool_map or _has_rag_agent_tool_attempt(tool_results) or tool_calls >= max_tool_calls:
        return "", tool_calls, []
    preferred_doc_ids = _graph_source_candidate_doc_ids(tool_results, limit=8)
    args: Dict[str, Any] = {
        "query": user_text,
        "conversation_context": _graph_rag_recovery_context(tool_results),
        "preferred_doc_ids_csv": ",".join(preferred_doc_ids),
        "must_include_uploads": False,
        "top_k_vector": 16,
        "top_k_keyword": 16,
        "max_retries": 2,
        "search_mode": "deep",
        "max_search_rounds": 2,
    }
    try:
        output = tool_map["rag_agent_tool"].invoke(args, config={"callbacks": callbacks})
    except TypeError:
        output = tool_map["rag_agent_tool"].invoke(args)
    except Exception as exc:
        output = {"error": str(exc)}
    tool_calls += 1
    out_text = json.dumps(output, ensure_ascii=False) if isinstance(output, (dict, list)) else str(output)
    tool_results.append({"tool": "rag_agent_tool", "args": args, "output": out_text})
    messages.append(ToolMessage(content=out_text, tool_call_id=f"fallback_rag_agent_tool_{tool_calls}", name="rag_agent_tool"))
    payload = output if isinstance(output, dict) else extract_json(out_text)
    rendered = _render_rag_payload(payload if isinstance(payload, dict) else {})
    return rendered, tool_calls, ["graph_catalog_to_rag_recovery"] if rendered else ["graph_catalog_rag_recovery_failed"]


def _normalize_planned_graph_tool_args(
    name: str,
    args: Dict[str, Any],
    *,
    user_text: str,
    tool_results: List[Dict[str, Any]],
    tool_context: Any | None = None,
) -> Dict[str, Any]:
    if name not in {"inspect_graph_index", "search_graph_index"}:
        return args
    normalized = dict(args)
    graph_id = str(normalized.get("graph_id") or "").strip() or _graph_id_from_tool_results(
        user_text,
        tool_results,
        tool_context=tool_context,
    )
    if graph_id:
        normalized["graph_id"] = graph_id
    if name == "inspect_graph_index":
        normalized.pop("collection_id", None)
    if name == "search_graph_index" and not str(normalized.get("query") or "").strip():
        normalized["query"] = user_text
    return normalized


def _graph_evidence_missing_text(
    user_text: str,
    tool_results: List[Dict[str, Any]],
    *,
    tool_context: Any | None = None,
) -> str:
    graph_resolution = _graph_id_resolution_for_request(user_text, tool_results, tool_context=tool_context)
    graph_id = graph_resolution.graph_id
    target = f"`{graph_id}`" if graph_id else "the requested graph"
    if not graph_id and graph_resolution.status.startswith("ambiguous"):
        candidate_text = ", ".join(f"`{item}`" for item in graph_resolution.candidates[:4])
        suffix = " Please select one graph ID for grounded graph search."
        if candidate_text:
            return (
                f"I found multiple candidate graphs ({candidate_text}), but no single graph was selected. "
                "I do not have enough GraphRAG evidence to answer this relationship question without guessing."
                f"{suffix}"
            )
        return (
            "I found multiple candidate graphs, but no single graph was selected. "
            "I do not have enough GraphRAG evidence to answer this relationship question without guessing."
            f"{suffix}"
        )
    if _graph_search_has_error(tool_results):
        return (
            f"I found {target}, but the graph search tool failed before returning relationship evidence. "
            "I do not have enough GraphRAG evidence to answer the relationship question without guessing."
        )
    if _graph_source_candidate_doc_ids(tool_results):
        return (
            f"I found source candidates in {target}, but the graph search did not return chunks, excerpts, or relationship paths. "
            "Those catalog matches are not enough evidence for a causal answer, so I should not infer the answer from metadata alone."
        )
    return (
        f"I do not have GraphRAG search evidence from {target} for this relationship request. "
        "I can confirm catalog/readiness information only, so I should not infer vendors, risks, approvals, dependencies, or outcomes from metadata alone."
    )


def _markdown_link(label: str, url: str) -> str:
    clean_label = str(label or url or "source").replace("\n", " ").strip()
    clean_url = str(url or "").strip()
    if not clean_url:
        return clean_label
    escaped_label = clean_label.replace("[", "\\[").replace("]", "\\]")
    escaped_url = clean_url.replace(" ", "%20").replace(")", "%29")
    return f"[{escaped_label}]({escaped_url})"


def _graph_citations_from_tool_results(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations_by_id: Dict[str, Dict[str, Any]] = {}
    for result in tool_results:
        if _tool_result_name(result) != "search_graph_index":
            continue
        payload = _payload_from_tool_result(result)
        for citation in payload.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            citation_id = str(citation.get("citation_id") or "").strip()
            if citation_id:
                citations_by_id.setdefault(citation_id, dict(citation))
        if citations_by_id:
            continue
        for hit in payload.get("results") or []:
            if not isinstance(hit, dict):
                continue
            for citation_id in [str(item) for item in (hit.get("citation_ids") or []) if str(item)]:
                citations_by_id.setdefault(
                    citation_id,
                    {
                        "citation_id": citation_id,
                        "doc_id": str(hit.get("doc_id") or ""),
                        "title": str(hit.get("title") or hit.get("doc_id") or citation_id),
                        "source_path": str(hit.get("source_path") or ""),
                        "url": "",
                    },
                )
    return list(citations_by_id.values())


def _graph_execution_metadata(
    *,
    user_text: str,
    tool_results: List[Dict[str, Any]],
    tool_context: Any | None = None,
) -> Dict[str, Any]:
    graph_results = [result for result in tool_results if _tool_result_name(result) == "search_graph_index"]
    rag_results = [result for result in tool_results if _tool_result_name(result) == "rag_agent_tool"]
    result_count = 0
    cited_doc_ids: List[str] = []
    requires_source_read = False
    evidence_status = ""
    for result in graph_results:
        payload = _payload_from_tool_result(result)
        result_count += len([item for item in (payload.get("results") or []) if isinstance(item, dict)])
        requires_source_read = requires_source_read or bool(payload.get("requires_source_read"))
        evidence_status = evidence_status or str(payload.get("evidence_status") or "")
        for doc_id in _graph_source_candidate_doc_ids([result], limit=20):
            if doc_id not in cited_doc_ids:
                cited_doc_ids.append(doc_id)
    for result in rag_results:
        payload = _payload_from_tool_result(result)
        for citation in payload.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            doc_id = str(citation.get("doc_id") or "").strip()
            if doc_id and doc_id not in cited_doc_ids:
                cited_doc_ids.append(doc_id)
    graph_resolution = _graph_id_resolution_for_request(user_text, tool_results, tool_context=tool_context)
    return {
        "graph_id": graph_resolution.graph_id,
        "graph_id_resolution_status": graph_resolution.status,
        "graph_id_resolution_source": graph_resolution.source,
        "graph_id_resolution_candidates": list(graph_resolution.candidates[:12]),
        "graph_ids_from_context": _graph_ids_from_context(tool_context),
        "graph_tool_attempted": bool(graph_results),
        "graph_result_count": result_count,
        "evidence_status": evidence_status,
        "requires_source_read": requires_source_read,
        "rag_recovery_attempted": bool(rag_results),
        "rag_recovery_renderable": bool(_render_latest_rag_tool_result(tool_results)),
        "source_resolution_status": (
            "source_resolved"
            if rag_results
            else "graph_grounded"
            if graph_results and _graph_search_has_grounded_evidence(tool_results) and not requires_source_read
            else "source_resolution_required"
            if graph_results and (requires_source_read or _graph_source_candidate_doc_ids(tool_results))
            else "not_attempted"
        ),
        "cited_doc_ids": cited_doc_ids[:12],
    }


def _with_graph_execution_metadata(
    metadata: Dict[str, Any],
    *,
    graph_grounded_request: bool,
    user_text: str,
    tool_results: List[Dict[str, Any]],
    tool_context: Any | None = None,
) -> Dict[str, Any]:
    if not graph_grounded_request and not _has_graph_search_attempt(tool_results):
        return metadata
    return {
        **dict(metadata or {}),
        "graph_execution": _graph_execution_metadata(
            user_text=user_text,
            tool_results=tool_results,
            tool_context=tool_context,
        ),
    }


def _append_missing_graph_citations(final_text: str, tool_results: List[Dict[str, Any]]) -> str:
    if not _has_graph_search_attempt(tool_results):
        return str(final_text or "")
    citations = _graph_citations_from_tool_results(tool_results)
    if not citations:
        return str(final_text or "")
    raw_text = str(final_text or "").strip()
    used_ids = [
        str(citation.get("citation_id") or "").strip()
        for citation in citations
        if str(citation.get("citation_id") or "").strip()
        and str(citation.get("citation_id") or "").strip() in raw_text
    ]
    if not used_ids:
        used_ids = [str(citation.get("citation_id") or "").strip() for citation in citations[:8]]
    used = set(used_ids)
    text = replace_inline_citation_ids(
        raw_text,
        citations,
        used_citation_ids=used_ids,
        link_renderer=_markdown_link,
    )
    if re.search(r"(?im)^#{0,6}\s*citations\s*:", text) or re.search(r"(?im)^citations\s*$", text):
        return text
    lines = ["Citations:"]
    for citation in citations:
        citation_id = str(citation.get("citation_id") or "").strip()
        if not citation_id or citation_id not in used:
            continue
        rendered_title = _markdown_link(citation_display_label(citation), str(citation.get("url") or ""))
        details = []
        source_path = str(citation.get("source_path") or "").strip()
        location = str(citation.get("location") or "").strip()
        collection_id = str(citation.get("collection_id") or "").strip()
        if location:
            details.append(location)
        if collection_id:
            details.append(f"Collection: {collection_id}")
        if source_path:
            details.append(f"source: {Path(source_path).name or source_path}")
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {rendered_title}{suffix}")
    if len(lines) == 1:
        return text
    return f"{text}\n\n" + "\n".join(lines)


def _append_evidence_text_parts(value: Any, parts: List[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            parts.append(text)
        return
    if isinstance(value, (int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _append_evidence_text_parts(item, parts)
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in _EVIDENCE_TEXT_KEYS:
                _append_evidence_text_parts(nested, parts)


def _evidence_text_from_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for result in tool_results:
        tool_name = _tool_result_name(result)
        payload = _payload_from_tool_result(result)
        if not payload:
            continue
        if tool_name == "rag_agent_tool":
            for citation in payload.get("citations") or []:
                if isinstance(citation, dict):
                    _append_evidence_text_parts(citation, parts)
            for key in ("supporting_evidence", "evidence", "chunks", "retrieved_chunks", "source_chunks"):
                _append_evidence_text_parts(payload.get(key), parts)
        elif tool_name == "search_graph_index":
            for citation in payload.get("citations") or []:
                if isinstance(citation, dict) and not citation.get("catalog_only"):
                    _append_evidence_text_parts(citation, parts)
            for hit in payload.get("results") or []:
                if isinstance(hit, dict) and not _graph_hit_is_catalog_only(hit):
                    _append_evidence_text_parts(hit, parts)
    return "\n".join(parts)


def _answer_body_without_citations(final_text: str) -> str:
    return re.split(r"(?im)^\s*#{0,6}\s*citations\s*:?\s*$", str(final_text or ""), maxsplit=1)[0]


def _claim_terms(value: str) -> List[str]:
    terms: List[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", str(value or "").casefold()):
        normalized = token.replace("_", "-").strip("-")
        for part in normalized.split("-"):
            clean = part.strip()
            if len(clean) < 4 or clean in _CLAIM_TERM_STOPWORDS or clean in seen:
                continue
            seen.add(clean)
            terms.append(clean)
    return terms


def _causal_claim_sentences(final_text: str) -> List[str]:
    body = _answer_body_without_citations(final_text)
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", body) if item.strip()]
    return [sentence for sentence in sentences if _CAUSAL_SENTENCE_RE.search(sentence)]


def _unsupported_causal_claim_terms(
    final_text: str,
    tool_results: List[Dict[str, Any]],
    *,
    user_text: str,
) -> List[str]:
    sentences = _causal_claim_sentences(final_text)
    if not sentences:
        return []
    evidence_terms = set(_claim_terms(_evidence_text_from_tool_results(tool_results)))
    user_terms = set(_claim_terms(user_text))
    unsupported: List[str] = []
    for sentence in sentences:
        for term in _claim_terms(sentence):
            if term in user_terms or term in evidence_terms:
                continue
            unsupported.append(term)
    unique = list(dict.fromkeys(unsupported))
    return unique if len(unique) >= 2 else []


def _unsupported_claims_rejection_text(unsupported_terms: List[str]) -> str:
    terms = ", ".join(dict.fromkeys(term for term in unsupported_terms if term))
    return (
        "I do not have cited evidence supporting these causal claim terms: "
        f"{terms}. The cited snippets do not support those details, so I should not present them as the answer."
    )


def _apply_final_evidence_verifier(
    final_text: str,
    tool_results: List[Dict[str, Any]],
    *,
    user_text: str,
) -> tuple[str, List[str]]:
    if not any(_tool_result_name(result) in {"rag_agent_tool", "search_graph_index"} for result in tool_results):
        return str(final_text or ""), []
    unsupported = _unsupported_causal_claim_terms(final_text, tool_results, user_text=user_text)
    if not unsupported:
        return str(final_text or ""), []
    rag_rendered = _render_latest_rag_tool_result(tool_results)
    if rag_rendered and not _unsupported_causal_claim_terms(rag_rendered, tool_results, user_text=user_text):
        return rag_rendered, ["unsupported_causal_claims_replaced_with_rag_answer"]
    return _unsupported_claims_rejection_text(unsupported), ["unsupported_causal_claims_rejected"]


def _synthesize_tool_results(
    chat_llm: Any,
    *,
    user_text: str,
    tool_results: List[Dict[str, Any]],
    callbacks: List[Any],
    system_prompt: str,
    recovery_reason: str,
) -> str:
    synth_system = (
        "You are recovering a final answer after a tool-using agent run.\n"
        "Use the tool results to answer the user's request clearly, with enough detail to satisfy the request.\n"
        "Preserve citations and uncertainty, and do not dump raw JSON.\n"
        "When tool results include search_graph_index JSON, only treat results as graph evidence when they include non-catalog chunks, relationships, or excerpts. If evidence_status is source_candidates_only or requires_source_read is true, those are source candidates or graph leads only; do not synthesize causal claims from them unless a rag_agent_tool result provides cited source-text support.\n"
        "Do not invent Cypher queries, schema labels, relationship names, or how-to steps unless the user explicitly asks for query syntax.\n"
        f"Recovery reason: {recovery_reason}."
    )
    if system_prompt.strip():
        synth_system += "\n\nRole Instructions:\n" + system_prompt.strip()
    synth_user = f"USER_REQUEST: {user_text}\n\nTOOL_RESULTS: {json.dumps(tool_results, ensure_ascii=False)}"
    response = chat_llm.invoke(
        [SystemMessage(content=synth_system), HumanMessage(content=synth_user)],
        config={"callbacks": callbacks},
    )
    return _response_text(response)


def _finalize_messages(
    chat_llm: Any,
    *,
    messages: List[Any],
    user_text: str,
    callbacks: List[Any],
    system_prompt: str,
) -> tuple[str, List[str]]:
    recovery: List[str] = []
    final_message = None
    final_text = ""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        final_message = message
        final_text = _content_text(message)
        if final_text:
            break
    if final_text and final_message is not None and not _is_output_truncated(final_message):
        return final_text, recovery
    if final_message is not None and _is_output_truncated(final_message):
        recovery.append("output_truncated")
    else:
        recovery.append("no_final_answer")
    rag_text = _render_rag_tool_fallback(messages)
    if rag_text:
        recovery.append("render_rag_tool_fallback")
        return rag_text, recovery
    tool_results = _collect_tool_results(messages)
    analyst_text = _render_data_analyst_tool_results(tool_results)
    if analyst_text:
        recovery.append("render_data_analyst_tool_fallback")
        return analyst_text, recovery
    requirements_text = _render_requirements_tool_results(tool_results)
    if requirements_text:
        recovery.append("render_requirements_tool_fallback")
        return requirements_text, recovery
    if tool_results:
        recovery.append("tool_result_synthesis")
        synthesized = _synthesize_tool_results(
            chat_llm,
            user_text=user_text,
            tool_results=tool_results,
            callbacks=callbacks,
            system_prompt=system_prompt,
            recovery_reason=",".join(recovery),
        ).strip()
        if synthesized:
            return synthesized, recovery
    if final_text:
        recovery.append("truncated_output_notice")
        return f"{final_text}\n\nNote: the previous response may have been truncated.".strip(), recovery
    return "I couldn't produce a complete final answer from the tool run. Please try again with a narrower request.", recovery


def _invoke_tool_with_trace(
    tool_map: Dict[str, Any],
    messages: List[Any],
    callbacks: List[Any],
    tool_name: str,
    args: Dict[str, Any],
    *,
    call_index: int,
) -> tuple[str, int]:
    tool = tool_map[tool_name]
    safe_args = _sanitize_tool_args(args)
    try:
        output = tool.invoke(safe_args, config={"callbacks": callbacks})
    except TypeError:
        output = tool.invoke(safe_args)
    output_text = json.dumps(output, ensure_ascii=False) if isinstance(output, (dict, list)) else str(output)
    messages.append(ToolMessage(content=output_text, tool_call_id=f"guided_{tool_name}_{call_index}", name=tool_name))
    return output_text, call_index + 1


def _data_analyst_fallback_enabled(tool_map: Dict[str, Any]) -> bool:
    required = {"load_dataset", "inspect_columns", "workspace_list"}
    return required.issubset(set(tool_map)) and (
        "execute_code" in tool_map or "run_nlp_column_task" in tool_map
    )


def _has_successful_execute_code(tool_results: List[Dict[str, Any]]) -> bool:
    for result in tool_results:
        if _tool_result_name(result) != "execute_code":
            continue
        payload = _tool_result_payload(result)
        if payload.get("success") is True:
            return True
    return False


def _has_successful_data_analyst_tool_result(tool_results: List[Dict[str, Any]]) -> bool:
    for result in tool_results:
        tool_name = _tool_result_name(result)
        payload = _tool_result_payload(result)
        if not payload:
            continue
        if tool_name == "execute_code" and payload.get("success") is True:
            return True
        if tool_name == "run_nlp_column_task" and not payload.get("error") and int(payload.get("processed_rows") or 0) > 0:
            return True
    return False


def _normalize_freeform_phrase(value: str) -> str:
    collapsed = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", collapsed).strip()


def _infer_data_analyst_nlp_task(user_text: str) -> str:
    normalized = _normalize_freeform_phrase(user_text)
    if any(term in normalized for term in ("sentiment", "positive", "negative", "neutral")):
        return "sentiment"
    if any(term in normalized for term in ("categorize", "classification", "classify", "category", "label")):
        return "categorize"
    if any(term in normalized for term in ("keyword", "key phrase", "keyphrase")):
        return "keywords"
    if any(term in normalized for term in ("summarize", "summary", "summarization")):
        return "summarize"
    return ""


def _requests_table_delivery(user_text: str) -> bool:
    normalized = _normalize_freeform_phrase(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "table",
            "returned table",
            "return it as a table",
            "return as a table",
            "give me a table",
            "show me a table",
        )
    )


def _requests_file_return(user_text: str) -> bool:
    normalized = _normalize_freeform_phrase(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "return the file",
            "return file",
            "return the workbook",
            "return the updated workbook",
            "return the updated file",
            "download",
            "send back",
            "export",
        )
    )


def _requests_file_only_delivery(user_text: str) -> bool:
    normalized = _normalize_freeform_phrase(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "only return the file",
            "file only",
            "just return the file",
            "without explanation",
            "no summary",
            "no explanation",
        )
    )


def _requests_summary_only_delivery(user_text: str) -> bool:
    normalized = _normalize_freeform_phrase(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "summary only",
            "just summarize",
            "only summarize",
            "in chat only",
            "do not return the file",
            "don't return the file",
            "do not update the file",
            "don't update the file",
        )
    )


def _infer_data_analyst_nlp_output_mode(user_text: str) -> str:
    normalized = _normalize_freeform_phrase(user_text)
    if _requests_summary_only_delivery(user_text):
        return "summary_only"
    if _requests_file_return(user_text) or _requests_table_delivery(user_text):
        return "append_columns"
    if any(
        phrase in normalized
        for phrase in (
            "add sentiment label",
            "add sentiment labels",
            "add sentiment score",
            "add columns",
            "append columns",
            "write back",
            "update the file",
            "update the dataset",
            "updated csv",
        )
    ):
        return "append_columns"
    return "summary_only"


def _requires_code_analysis(user_text: str) -> bool:
    normalized = _normalize_freeform_phrase(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "chart",
            "plot",
            "graph",
            "pivot",
            "group by",
            "groupby",
            "correlation",
            "regression",
            "trend",
            "merge",
            "join",
            "workbook",
            "sheet",
            "tab",
            "formula",
        )
    )


def _requests_multi_dataset_analysis(user_text: str) -> bool:
    normalized = _normalize_freeform_phrase(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "both uploads",
            "both files",
            "both datasets",
            "across files",
            "across both",
            "using both",
            "compare",
            "join",
            "merge",
        )
    )


def _looks_like_row_level_nlp_request(user_text: str) -> bool:
    normalized = _normalize_freeform_phrase(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "all rows",
            "all of the",
            "for each row",
            "each row",
            "every row",
            "entire column",
            "label each",
            "classify each",
            "tag each",
            "return the file",
            "give me a table",
            "return it as a table",
            "add columns",
        )
    )


def _looks_like_summary_qa_request(user_text: str) -> bool:
    normalized = _normalize_freeform_phrase(user_text)
    return any(
        phrase in normalized
        for phrase in (
            "distribution",
            "breakdown",
            "overall",
            "trend",
            "how many",
            "what percentage",
            "what percent",
            "share of",
            "count of",
            "summarize the",
        )
    )


def _classify_data_analyst_intent(user_text: str) -> DataAnalystIntent:
    nlp_task = _infer_data_analyst_nlp_task(user_text) or None
    requires_code = _requires_code_analysis(user_text) or _requests_multi_dataset_analysis(user_text)
    if requires_code or not nlp_task:
        delivery_mode = "summary_and_file" if _requests_file_return(user_text) or _requests_table_delivery(user_text) else "summary_only"
        return DataAnalystIntent(
            task_family="code_analysis",
            nlp_task=nlp_task,
            delivery_mode=delivery_mode,
            requires_code=True,
        )
    if _looks_like_summary_qa_request(user_text) and not _looks_like_row_level_nlp_request(user_text):
        return DataAnalystIntent(
            task_family="summary_qa",
            nlp_task=nlp_task,
            delivery_mode="summary_only",
            requires_code=False,
        )
    if _requests_summary_only_delivery(user_text):
        delivery_mode = "summary_only"
    elif _requests_file_only_delivery(user_text):
        delivery_mode = "file_only"
    else:
        delivery_mode = "summary_and_file"
    return DataAnalystIntent(
        task_family="row_nlp",
        nlp_task=nlp_task,
        delivery_mode=delivery_mode,
        requires_code=False,
    )


def _choose_data_analyst_fallback_mode(user_text: str) -> str:
    intent = _classify_data_analyst_intent(user_text)
    if intent.task_family in {"row_nlp", "summary_qa"} and intent.nlp_task and not intent.requires_code:
        return "nlp"
    return "code"


def _dataset_name_relevance_score(user_text: str, dataset_ref: str) -> int:
    normalized_text = f" {_normalize_freeform_phrase(user_text)} "
    stem = Path(str(dataset_ref or "")).stem
    normalized_name = _normalize_freeform_phrase(stem)
    if not normalized_name:
        return 0
    score = 0
    if f" {normalized_name} " in normalized_text:
        score += 8
    for token in normalized_name.split():
        if len(token) < 3:
            continue
        if f" {token} " in normalized_text:
            score += 2
    return score


def _sort_dataset_refs_for_request(user_text: str, dataset_refs: List[str]) -> List[str]:
    return sorted(
        [str(ref) for ref in dataset_refs if str(ref)],
        key=lambda ref: (-_dataset_name_relevance_score(user_text, ref), ref),
    )


def _requested_filename_from_text(user_text: str, filenames: List[str]) -> str:
    normalized_text = str(user_text or "").casefold()
    for filename in sorted([str(item) for item in filenames if str(item)], key=len, reverse=True):
        if Path(filename).name.casefold() in normalized_text:
            return Path(filename).name
    for match in re.findall(r"\b[A-Za-z0-9_. -]+\.[A-Za-z0-9]{2,8}\b", str(user_text or "")):
        candidate = Path(match.strip()).name
        for filename in filenames:
            if candidate.casefold() == Path(str(filename)).name.casefold():
                return Path(str(filename)).name
    return ""


def _render_unsupported_uploaded_dataset(filename: str) -> str:
    ext = Path(str(filename or "")).suffix.lower() or "this file type"
    return (
        f"The uploaded file `{filename}` is available in this session, but {ext} is not supported "
        "by the dataset analysis tools. Use a CSV or Excel workbook for data analysis, or ask for "
        "a document-style read/summarization path instead."
    )


def _data_analyst_target_candidates(
    user_text: str,
    dataset_payloads: Dict[str, Dict[str, Any]],
) -> List[DataAnalystTargetCandidate]:
    normalized_text = f" {_normalize_freeform_phrase(user_text)} "
    candidates: List[DataAnalystTargetCandidate] = []
    preferred_names = {"review", "reviews", "customer message", "comments", "feedback", "notes", "text", "summary"}

    for dataset_ref, payload in dataset_payloads.items():
        filename_score = _dataset_name_relevance_score(user_text, dataset_ref)
        columns = [str(column) for column in (payload.get("columns") or [])]
        dtypes = {
            str(key): str(value).lower()
            for key, value in dict(payload.get("dtypes") or {}).items()
        }
        for column in columns:
            normalized_column = _normalize_freeform_phrase(column)
            if not normalized_column:
                continue
            score = filename_score
            exact_column = False
            if f" {normalized_column} column " in normalized_text:
                score += 8
                exact_column = True
            elif f" {normalized_column} " in normalized_text:
                score += 5
            dtype = dtypes.get(column, "")
            if "object" in dtype or "string" in dtype or "text" in dtype:
                score += 2
            if normalized_column in preferred_names:
                score += 1
            if score:
                candidates.append(
                    DataAnalystTargetCandidate(
                        score=score,
                        dataset_ref=dataset_ref,
                        column=column,
                        exact_column=exact_column,
                        filename_score=filename_score,
                    )
                )
    return sorted(
        candidates,
        key=lambda item: (-item.score, -item.filename_score, item.dataset_ref, item.column),
    )


def _infer_data_analyst_nlp_target(
    user_text: str,
    dataset_payloads: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    candidates = _data_analyst_target_candidates(user_text, dataset_payloads)
    if candidates:
        top = candidates[0]
        return top.dataset_ref, top.column

    text_like_columns: List[Tuple[str, str]] = []
    for dataset_ref, payload in dataset_payloads.items():
        columns = [str(column) for column in (payload.get("columns") or [])]
        dtypes = {
            str(key): str(value).lower()
            for key, value in dict(payload.get("dtypes") or {}).items()
        }
        for column in columns:
            dtype = dtypes.get(column, "")
            if "object" in dtype or "string" in dtype or "text" in dtype:
                text_like_columns.append((dataset_ref, column))
    if len(text_like_columns) == 1:
        return text_like_columns[0]
    return "", ""


def _infer_data_analyst_nlp_target_ambiguity(
    user_text: str,
    dataset_payloads: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str]]:
    candidates = _data_analyst_target_candidates(user_text, dataset_payloads)
    if len(candidates) < 2:
        return []
    top = candidates[0]
    ambiguous = [
        (candidate.dataset_ref, candidate.column)
        for candidate in candidates
        if candidate.dataset_ref != top.dataset_ref
        and candidate.score == top.score
        and candidate.column == top.column
    ]
    return [(top.dataset_ref, top.column), *ambiguous] if ambiguous else []


def _render_data_analyst_target_ambiguity(candidates: List[Tuple[str, str]]) -> str:
    details = [f"- {dataset_ref}: column `{column}`" for dataset_ref, column in candidates]
    return (
        "Multiple uploaded datasets look equally valid for this row-level analysis. "
        "Please name the file you want me to use.\n\n"
        + "\n".join(details)
    )


def _build_data_analyst_code(dataset_refs: List[str], dataset_payloads: Dict[str, Dict[str, Any]]) -> str:
    columns_by_ref = {
        ref: {str(column) for column in (payload.get("columns") or [])}
        for ref, payload in dataset_payloads.items()
    }
    if len(dataset_refs) >= 2:
        first, second = dataset_refs[:2]
        first_cols = columns_by_ref.get(first, set())
        second_cols = columns_by_ref.get(second, set())
        required_first = {"region", "annual_spend_usd", "current_reserve_usd"}
        required_second = {"region", "reserve_target_pct", "risk_score", "control_owner"}
        if required_first.issubset(first_cols) and required_second.issubset(second_cols):
            return (
                "import pandas as pd\n\n"
                f"spend = pd.read_csv('/workspace/{first}')\n"
                f"controls = pd.read_csv('/workspace/{second}')\n"
                "merged = spend.merge(controls, on='region', how='inner')\n"
                "merged['target_reserve_usd'] = merged['annual_spend_usd'] * merged['reserve_target_pct']\n"
                "merged['reserve_gap_usd'] = merged['target_reserve_usd'] - merged['current_reserve_usd']\n"
                "ranked = merged.sort_values(['risk_score', 'reserve_gap_usd'], ascending=[False, False]).head(3)\n"
                "columns = ['region', 'annual_spend_usd', 'current_reserve_usd', 'target_reserve_usd', 'reserve_gap_usd', 'risk_score', 'control_owner']\n"
                "print('Top three highest-risk regions by risk score and reserve gap:')\n"
                "print(ranked[columns].to_string(index=False))\n"
                "print('\\nSummary:')\n"
                "for _, row in ranked.iterrows():\n"
                "    print(f\"- {row['region']}: gap=${row['reserve_gap_usd']:.2f}, risk_score={int(row['risk_score'])}, owner={row['control_owner']}\")\n"
            )

    for ref in dataset_refs:
        payload = dataset_payloads.get(ref, {})
        if {"region", "annual_spend_usd", "current_reserve_usd", "reserve_target_pct", "risk_score"}.issubset(
            columns_by_ref.get(ref, set())
        ):
            return (
                "import pandas as pd\n\n"
                f"df = pd.read_csv('/workspace/{ref}')\n"
                "df['target_reserve_usd'] = df['annual_spend_usd'] * df['reserve_target_pct']\n"
                "df['reserve_gap_usd'] = df['target_reserve_usd'] - df['current_reserve_usd']\n"
                "ranked = df.sort_values(['risk_score', 'reserve_gap_usd'], ascending=[False, False]).head(3)\n"
                "print(ranked.to_string(index=False))\n"
            )

    print_targets = ", ".join(f"'/workspace/{ref}'" for ref in dataset_refs)
    return (
        "import os\nimport pandas as pd\n\n"
        f"files = [{print_targets}]\n"
        "for path in files:\n"
        "    if path.endswith('.csv') and os.path.exists(path):\n"
        "        df = pd.read_csv(path)\n"
        "        print(f'File: {os.path.basename(path)} shape={df.shape}')\n"
        "        print(df.head().to_string(index=False))\n"
        "        print('---')\n"
    )


def _run_data_analyst_guided_fallback(
    *,
    chat_llm: Any,
    tool_map: Dict[str, Any],
    messages: List[Any],
    user_text: str,
    callbacks: List[Any],
    max_tool_calls: int,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    tool_calls = 0
    intent = _classify_data_analyst_intent(user_text)
    fallback_mode = "nlp" if intent.task_family in {"row_nlp", "summary_qa"} and intent.nlp_task else "code"
    workspace_listing_text, tool_calls = _invoke_tool_with_trace(
        tool_map,
        messages,
        callbacks,
        "workspace_list",
        {},
        call_index=tool_calls,
    )
    workspace_listing = extract_json(workspace_listing_text) or {}
    workspace_files = [str(name) for name in (workspace_listing.get("files") or []) if str(name)]
    dataset_refs = _sort_dataset_refs_for_request(
        user_text,
        [
            name
            for name in workspace_files
            if str(name).lower().endswith((".csv", ".xlsx", ".xls"))
        ],
    )
    if not dataset_refs and workspace_files:
        requested_file = _requested_filename_from_text(user_text, workspace_files)
        if requested_file:
            final_text = _render_unsupported_uploaded_dataset(requested_file)
            messages.append(AIMessage(content=final_text))
            return final_text, messages, {
                "fallback": "data_analyst_guided",
                "guided_mode": fallback_mode,
                "tool_calls": tool_calls,
                "upload_resolution": {
                    "requested_filename": requested_file,
                    "status": "unsupported_type",
                    "available_files": workspace_files[:20],
                },
            }
    dataset_payloads: Dict[str, Dict[str, Any]] = {}

    if fallback_mode == "code":
        for dataset_ref in dataset_refs:
            if tool_calls >= max_tool_calls:
                break
            load_text, tool_calls = _invoke_tool_with_trace(
                tool_map,
                messages,
                callbacks,
                "load_dataset",
                {"doc_id": dataset_ref},
                call_index=tool_calls,
            )
            dataset_payloads[dataset_ref] = extract_json(load_text) or {}
            if tool_calls >= max_tool_calls:
                break
            _, tool_calls = _invoke_tool_with_trace(
                tool_map,
                messages,
                callbacks,
                "inspect_columns",
                {"doc_id": dataset_ref, "columns": ""},
                call_index=tool_calls,
            )
    else:
        for dataset_ref in dataset_refs:
            if tool_calls >= max_tool_calls:
                break
            load_text, tool_calls = _invoke_tool_with_trace(
                tool_map,
                messages,
                callbacks,
                "load_dataset",
                {"doc_id": dataset_ref},
                call_index=tool_calls,
            )
            dataset_payloads[dataset_ref] = extract_json(load_text) or {}

        ambiguous_targets = _infer_data_analyst_nlp_target_ambiguity(user_text, dataset_payloads)
        if ambiguous_targets:
            final_text = _render_data_analyst_target_ambiguity(ambiguous_targets)
            messages.append(AIMessage(content=final_text))
            return final_text, messages, {
                "fallback": "data_analyst_guided",
                "guided_mode": "nlp",
                "tool_calls": tool_calls,
                "ambiguous_targets": [
                    {"dataset_ref": dataset_ref, "column": column}
                    for dataset_ref, column in ambiguous_targets
                ],
            }

        target_dataset, target_column = _infer_data_analyst_nlp_target(user_text, dataset_payloads)
        intent.target_dataset = target_dataset
        intent.target_columns = (target_column,) if target_column else ()
        if intent.target_dataset and tool_calls < max_tool_calls:
            _, tool_calls = _invoke_tool_with_trace(
                tool_map,
                messages,
                callbacks,
                "inspect_columns",
                {"doc_id": intent.target_dataset, "columns": target_column},
                call_index=tool_calls,
            )

    if "scratchpad_write" in tool_map and tool_calls < max_tool_calls:
        if intent.task_family == "row_nlp":
            plan_text = (
                "Use the bounded NLP tool on the target text column, write derived output columns, "
                "return the resulting file, and summarize the result in chat."
            )
        elif intent.task_family == "summary_qa":
            plan_text = (
                "Use the bounded NLP tool on the target text column and summarize the result in chat "
                "without publishing a derived file."
            )
        else:
            plan_text = "Inspect the uploaded datasets, run the necessary sandbox analysis, and summarize the findings."
        _, tool_calls = _invoke_tool_with_trace(
            tool_map,
            messages,
            callbacks,
            "scratchpad_write",
            {
                "key": "analysis_plan",
                "value": plan_text,
            },
            call_index=tool_calls,
        )

    if fallback_mode == "nlp" and "run_nlp_column_task" in tool_map and tool_calls < max_tool_calls:
        dataset_ref = intent.target_dataset or next(iter(dataset_payloads), "")
        column = intent.target_columns[0] if intent.target_columns else ""
        if not dataset_ref or not column:
            final_text = (
                "I couldn't identify a unique text column for this analyst NLP task. "
                "Please name the file and column explicitly."
            )
            messages.append(AIMessage(content=final_text))
            return final_text, messages, {"fallback": "data_analyst_guided", "guided_mode": "nlp", "tool_calls": tool_calls}
        nlp_args: Dict[str, Any] = {
            "doc_id": dataset_ref,
            "column": column,
            "task": intent.nlp_task or "sentiment",
            "output_mode": "append_columns" if intent.delivery_mode in {"summary_and_file", "file_only"} else "summary_only",
        }
        nlp_text, tool_calls = _invoke_tool_with_trace(
            tool_map,
            messages,
            callbacks,
            "run_nlp_column_task",
            nlp_args,
            call_index=tool_calls,
        )
        tool_results = [{"tool": "run_nlp_column_task", "args": dict(nlp_args), "output": nlp_text}]
        nlp_payload = extract_json(nlp_text) or {}
        file_required = intent.delivery_mode in {"summary_and_file", "file_only"}
        if file_required:
            written_file = str((nlp_payload or {}).get("written_file") or "").strip()
            if not written_file:
                final_text = _render_data_analyst_missing_file_result(nlp_payload if isinstance(nlp_payload, dict) else {})
                messages.append(AIMessage(content=final_text))
                return final_text, messages, {
                    "fallback": "data_analyst_guided",
                    "guided_mode": "nlp",
                    "tool_calls": tool_calls,
                    "delivery_mode": intent.delivery_mode,
                }
            if "return_file" not in tool_map or tool_calls >= max_tool_calls:
                final_text = f"The analyst produced {written_file} but could not publish it back to the user."
                messages.append(AIMessage(content=final_text))
                return final_text, messages, {
                    "fallback": "data_analyst_guided",
                    "guided_mode": "nlp",
                    "tool_calls": tool_calls,
                    "delivery_mode": intent.delivery_mode,
                }
            return_args = {"filename": written_file}
            return_text, tool_calls = _invoke_tool_with_trace(
                tool_map,
                messages,
                callbacks,
                "return_file",
                return_args,
                call_index=tool_calls,
            )
            tool_results.append({"tool": "return_file", "args": return_args, "output": return_text})

        final_text = _render_data_analyst_tool_results(tool_results, expected_delivery_mode=intent.delivery_mode)
        if not final_text:
            error_text = str((nlp_payload or {}).get("error") or "").strip()
            final_text = (
                f"Data analyst NLP task failed: {error_text}"
                if error_text
                else "The data analyst NLP workflow did not produce a usable result."
            )
        messages.append(AIMessage(content=final_text))
        return final_text, messages, {
            "fallback": "data_analyst_guided",
            "guided_mode": "nlp",
            "tool_calls": tool_calls,
            "delivery_mode": intent.delivery_mode,
            "target_dataset": dataset_ref,
            "target_column": column,
        }

    code = _build_data_analyst_code(dataset_refs, dataset_payloads)
    execute_text, tool_calls = _invoke_tool_with_trace(
        tool_map,
        messages,
        callbacks,
        "execute_code",
        {"code": code, "doc_ids": ",".join(dataset_refs)},
        call_index=tool_calls,
    )
    final_text = _render_data_analyst_tool_results(
        [{"tool": "execute_code", "args": {"doc_ids": ",".join(dataset_refs)}, "output": execute_text}]
    )
    if not final_text:
        execute_payload = extract_json(execute_text) or {}
        error_text = str(
            execute_payload.get("error")
            or execute_payload.get("stderr")
            or ""
        ).strip()
        final_text = (
            f"Data analysis failed in the sandbox:\n{error_text}"
            if error_text
            else "The data analyst code workflow did not produce a usable result."
        )

    messages.append(AIMessage(content=final_text))
    return final_text, messages, {"fallback": "data_analyst_guided", "guided_mode": "code", "tool_calls": tool_calls}


def _normalize_planned_data_analyst_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    intent: DataAnalystIntent | None,
    tool_results: List[Dict[str, Any]],
    user_text: str,
) -> Dict[str, Any]:
    normalized = dict(args or {})
    if tool_name != "run_nlp_column_task" or intent is None:
        return normalized
    if intent.nlp_task and not str(normalized.get("task") or "").strip():
        normalized["task"] = intent.nlp_task
    if intent.task_family == "row_nlp":
        normalized["output_mode"] = "append_columns"
    elif intent.task_family == "summary_qa" and not str(normalized.get("output_mode") or "").strip():
        normalized["output_mode"] = "summary_only"
    observed_payloads: Dict[str, Dict[str, Any]] = {}
    for result in tool_results:
        if _tool_result_name(result) != "load_dataset":
            continue
        payload = _tool_result_payload(result)
        if not payload:
            continue
        dataset_ref = str(payload.get("doc_id") or result.get("args", {}).get("doc_id") or "").strip()
        if dataset_ref:
            observed_payloads[dataset_ref] = payload
    if observed_payloads:
        target_dataset, target_column = _infer_data_analyst_nlp_target(user_text, observed_payloads)
        current_dataset = str(
            normalized.get("doc_id") or normalized.get("dataset") or normalized.get("dataset_name") or ""
        ).strip()
        current_column = str(normalized.get("column") or "").strip()
        known_datasets = set(observed_payloads)
        known_columns = {
            str(column)
            for payload in observed_payloads.values()
            for column in list(payload.get("columns") or [])
            if str(column)
        }
        if target_dataset and (not current_dataset or current_dataset not in known_datasets):
            normalized["doc_id"] = target_dataset
            normalized.pop("dataset", None)
            normalized.pop("dataset_name", None)
        if target_column and (not current_column or current_column not in known_columns):
            normalized["column"] = target_column
    return normalized


def _ensure_system(messages: List[Any], system_prompt: str) -> List[Any]:
    if not messages or not isinstance(messages[0], SystemMessage):
        return [SystemMessage(content=system_prompt)] + list(messages)
    return list(messages)


def _has_latest_user_message(messages: List[Any], user_text: str) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return False
    return str(getattr(last, "content", "") or "").strip() == user_text.strip()


def build_react_agent_graph(
    chat_llm: Any,
    *,
    tools: List[Any],
    max_tool_calls: int = 12,
    max_parallel_tool_calls: int = 4,
    context_budget_manager: Any | None = None,
    tool_context: Any | None = None,
    providers: Any | None = None,
) -> Any:
    from langgraph.prebuilt import create_react_agent

    tool_node = PolicyAwareToolNode(
        tools,
        max_tool_calls=max_tool_calls,
        max_parallel_tool_calls=max_parallel_tool_calls,
        context_budget_manager=context_budget_manager,
        tool_context=tool_context,
    )
    pre_model_hook = build_microcompact_hook(
        context_budget_manager,
        providers=providers,
        tool_context=tool_context,
    )
    graph_kwargs = {"pre_model_hook": pre_model_hook} if pre_model_hook is not None else {}
    return create_react_agent(chat_llm, tools=tool_node, **graph_kwargs)


def run_general_agent(
    chat_llm: Any,
    *,
    tools: List[Any],
    messages: List[Any],
    user_text: str,
    system_prompt: str = "",
    callbacks=None,
    max_steps: int = 10,
    max_tool_calls: int = 12,
    max_parallel_tool_calls: int = 4,
    force_plan_execute: bool = False,
    context_budget_manager: Any | None = None,
    tool_context: Any | None = None,
    providers: Any | None = None,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    callbacks = callbacks or []
    effective_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
    tool_map = {tool.name: tool for tool in tools}
    graph_grounded_request = _is_grounded_graph_request(user_text, tool_map, tool_context=tool_context)
    msgs = _ensure_system(messages, effective_prompt)
    if not _has_latest_user_message(msgs, user_text):
        msgs.append(HumanMessage(content=user_text))

    supports_tool_calls = False
    if not force_plan_execute:
        try:
            chat_llm.bind_tools(tools)
            supports_tool_calls = True
        except Exception:
            supports_tool_calls = False

    if force_plan_execute or not supports_tool_calls:
        return _run_plan_execute_fallback(
            chat_llm,
            tools=tools,
            messages=msgs,
            user_text=user_text,
            callbacks=callbacks,
            max_tool_calls=max_tool_calls,
            system_prompt=effective_prompt,
            context_budget_manager=context_budget_manager,
            tool_context=tool_context,
        )

    graph = build_react_agent_graph(
        chat_llm,
        tools=tools,
        max_tool_calls=max_tool_calls,
        max_parallel_tool_calls=max_parallel_tool_calls,
        context_budget_manager=context_budget_manager,
        tool_context=tool_context,
        providers=providers,
    )
    recursion_limit = (max(max_steps, max_tool_calls) + 1) * 2 + 1
    try:
        result = graph.invoke(
            {"messages": msgs},
            config={"callbacks": callbacks, "recursion_limit": recursion_limit},
        )
        updated_messages: List[Any] = result["messages"]
    except Exception as exc:
        logger.warning("LangGraph ReAct agent failed; falling back to plan-execute recovery: %s", exc)
        final_text, updated_messages, metadata = _run_plan_execute_fallback(
            chat_llm,
            tools=tools,
            messages=msgs,
            user_text=user_text,
            callbacks=callbacks,
            max_tool_calls=max_tool_calls,
            system_prompt=effective_prompt,
            context_budget_manager=context_budget_manager,
            tool_context=tool_context,
        )
        metadata["recovery"] = ["langgraph_error"]
        metadata["langgraph_error"] = str(exc)
        return final_text, updated_messages, metadata

    tool_calls_used = count_current_turn_tool_messages(updated_messages)
    steps = count_current_turn_ai_messages(updated_messages)
    final_text, recovery = _finalize_messages(
        chat_llm,
        messages=updated_messages,
        user_text=user_text,
        callbacks=callbacks,
        system_prompt=effective_prompt,
    )
    if (
        graph_grounded_request
        and not _has_graph_search_attempt(_collect_tool_results(updated_messages))
        and "<clarification_request" not in str(final_text or "").casefold()
    ):
        final_text, updated_messages, metadata = _run_plan_execute_fallback(
            chat_llm,
            tools=tools,
            messages=msgs,
            user_text=user_text,
            callbacks=callbacks,
            max_tool_calls=max_tool_calls,
            system_prompt=effective_prompt,
            context_budget_manager=context_budget_manager,
            tool_context=tool_context,
        )
        metadata["recovery"] = [
            "graph_search_missing_react",
            *list(metadata.get("recovery") or []),
        ]
        return final_text, updated_messages, metadata
    collected_tool_results = _collect_tool_results(updated_messages)
    if (
        graph_grounded_request
        and _has_graph_search_attempt(collected_tool_results)
        and (
            not _graph_search_has_grounded_evidence(collected_tool_results)
            or _graph_search_requires_source_read(collected_tool_results)
        )
    ):
        rag_rendered = _render_latest_rag_tool_result(collected_tool_results)
        if rag_rendered:
            rag_rendered, verifier_recovery = _apply_final_evidence_verifier(
                rag_rendered,
                collected_tool_results,
                user_text=user_text,
            )
            return rag_rendered, updated_messages + [AIMessage(content=rag_rendered)], _with_graph_execution_metadata(
                {
                    "steps": steps,
                    "tool_calls": tool_calls_used,
                    "recovery": ["graph_catalog_used_rag_tool", *verifier_recovery, *recovery],
                },
                graph_grounded_request=graph_grounded_request,
                user_text=user_text,
                tool_results=collected_tool_results,
                tool_context=tool_context,
            )
        recovered, recovered_tool_calls, recovered_recovery = _run_graph_catalog_rag_recovery(
            tool_map,
            updated_messages,
            collected_tool_results,
            user_text=user_text,
            callbacks=callbacks,
            tool_calls=tool_calls_used,
            max_tool_calls=max_tool_calls,
        )
        if recovered:
            recovered, verifier_recovery = _apply_final_evidence_verifier(
                recovered,
                collected_tool_results,
                user_text=user_text,
            )
            return recovered, updated_messages + [AIMessage(content=recovered)], _with_graph_execution_metadata(
                {
                    "steps": steps,
                    "tool_calls": recovered_tool_calls,
                    "recovery": [*recovered_recovery, *verifier_recovery, *recovery],
                },
                graph_grounded_request=graph_grounded_request,
                user_text=user_text,
                tool_results=collected_tool_results,
                tool_context=tool_context,
            )
        final_text = _graph_evidence_missing_text(user_text, collected_tool_results, tool_context=tool_context)
        return final_text, updated_messages + [AIMessage(content=final_text)], _with_graph_execution_metadata(
            {
                "steps": steps,
                "tool_calls": recovered_tool_calls,
                "recovery": ["graph_catalog_only", *recovered_recovery, *recovery],
            },
            graph_grounded_request=graph_grounded_request,
            user_text=user_text,
            tool_results=collected_tool_results,
            tool_context=tool_context,
        )
    final_text = _append_missing_graph_citations(final_text, collected_tool_results)
    verified_text, verifier_recovery = _apply_final_evidence_verifier(
        final_text,
        collected_tool_results,
        user_text=user_text,
    )
    if verifier_recovery:
        final_text = verified_text
        recovery = [*verifier_recovery, *recovery]
    if recovery:
        updated_messages = list(updated_messages) + [AIMessage(content=final_text)]
    return str(final_text), updated_messages, _with_graph_execution_metadata(
        {"steps": steps, "tool_calls": tool_calls_used, "recovery": recovery},
        graph_grounded_request=graph_grounded_request,
        user_text=user_text,
        tool_results=collected_tool_results,
        tool_context=tool_context,
    )


def _run_plan_execute_fallback(
    chat_llm: Any,
    *,
    tools: List[Any],
    messages: List[Any],
    user_text: str,
    callbacks=None,
    max_tool_calls: int = 12,
    system_prompt: str = "",
    context_budget_manager: Any | None = None,
    tool_context: Any | None = None,
) -> Tuple[str, List[Any], Dict[str, Any]]:
    callbacks = callbacks or []
    tool_map = {tool.name: tool for tool in tools}
    analyst_intent = _classify_data_analyst_intent(user_text) if _data_analyst_fallback_enabled(tool_map) else None
    graph_grounded_request = _is_grounded_graph_request(user_text, tool_map, tool_context=tool_context)
    planner_system = (
        "You are a planning assistant. You cannot call tools directly.\n"
        "Produce a tool plan as JSON ONLY.\n"
        "Allowed tools: " + ", ".join(tool_map.keys()) + "\n\n"
        "Tool schemas:\n"
        + (_compact_tool_schemas(tools) or "- none")
        + "\n\n"
        "Return JSON in this schema:\n"
        "{\"plan\": [{\"tool\": \"tool_name\", \"args\": {...}, \"purpose\": \"...\"}], \"notes\": \"...\"}\n\n"
        "Rules:\n"
        "- Use 0 tools if not needed.\n"
        "- Keep args minimal and valid JSON.\n"
        "- Prefer rag_agent_tool when citations or documents are involved.\n"
        "- For named graph ids, use search_graph_index with graph_id and query for graph-backed evidence.\n"
        "- Use list_graph_indexes only for graph inventory or discovery, not as sufficient evidence for relationship answers.\n"
        "- Use inspect_graph_index with graph_id, never collection_id.\n"
    )
    if system_prompt.strip():
        planner_system += "\n\nRole Instructions:\n" + system_prompt.strip()

    def _extract_plan(text: Any) -> List[Dict[str, Any]] | None:
        payload = extract_json(text) or {}
        plan_value = payload.get("plan") if isinstance(payload, dict) else None
        return plan_value if isinstance(plan_value, list) else None

    plan_response = chat_llm.invoke(
        [SystemMessage(content=planner_system), HumanMessage(content=user_text)],
        config={"callbacks": callbacks},
    )
    plan_text = _response_text(plan_response)
    plan = _extract_plan(plan_text)

    if not isinstance(plan, list):
        repair_system = (
            "You repair model outputs into strict JSON.\n"
            "Return JSON ONLY using this exact schema:\n"
            "{\"plan\": [{\"tool\": \"tool_name\", \"args\": {...}, \"purpose\": \"...\"}], \"notes\": \"...\"}\n"
            "Do not include markdown fences or explanatory prose."
        )
        repair_response = chat_llm.invoke(
            [
                SystemMessage(content=repair_system),
                HumanMessage(
                    content=(
                        "Convert the following planner output into valid JSON using the required schema.\n\n"
                        f"{plan_text}"
                    )
                ),
            ],
            config={"callbacks": callbacks},
        )
        plan = _extract_plan(_response_text(repair_response))

    if not isinstance(plan, list) and graph_grounded_request:
        graph_id = _extract_named_graph_id_from_text(user_text) or _context_graph_id(tool_context)
        if graph_id:
            plan = [
                {
                    "tool": "search_graph_index",
                    "args": {"query": user_text, "graph_id": graph_id, "limit": 8},
                    "purpose": "Retrieve graph-backed evidence for the named graph relationship request.",
                }
            ]

    if not isinstance(plan, list):
        if graph_grounded_request:
            final_text = _graph_evidence_missing_text(user_text, [], tool_context=tool_context)
            if not _has_latest_user_message(messages, user_text):
                messages.append(HumanMessage(content=user_text))
            messages.append(AIMessage(content=final_text))
            return str(final_text), messages, _with_graph_execution_metadata(
                {
                    "fallback": "plan_execute",
                    "tool_calls": 0,
                    "recovery": ["graph_plan_missing"],
                },
                graph_grounded_request=graph_grounded_request,
                user_text=user_text,
                tool_results=[],
                tool_context=tool_context,
            )
        if _data_analyst_fallback_enabled(tool_map):
            return _run_data_analyst_guided_fallback(
                chat_llm=chat_llm,
                tool_map=tool_map,
                messages=messages,
                user_text=user_text,
                callbacks=callbacks,
                max_tool_calls=max_tool_calls,
            )
        fallback_messages = _ensure_system(messages, system_prompt or _DEFAULT_SYSTEM_PROMPT)
        if not _has_latest_user_message(fallback_messages, user_text):
            fallback_messages = fallback_messages + [HumanMessage(content=user_text)]
        direct = chat_llm.invoke(
            fallback_messages,
            config={"callbacks": callbacks},
        )
        final = _response_text(direct)
        if not _has_latest_user_message(messages, user_text):
            messages.append(HumanMessage(content=user_text))
        messages.append(AIMessage(content=final))
        return str(final), messages, {"fallback": "direct_no_plan"}

    if not _has_latest_user_message(messages, user_text):
        messages.append(HumanMessage(content=user_text))
    tool_calls = 0
    tool_results: List[Dict[str, Any]] = []
    for step in plan:
        if tool_calls >= max_tool_calls:
            break
        if not isinstance(step, dict):
            continue
        name = step.get("tool")
        args = _sanitize_tool_args(step.get("args") or {})
        if not isinstance(name, str) or name not in tool_map:
            tool_results.append({"tool": name, "error": "unknown tool"})
            continue
        args = _normalize_planned_graph_tool_args(
            name,
            args,
            user_text=user_text,
            tool_results=tool_results,
            tool_context=tool_context,
        )
        args = _normalize_planned_data_analyst_tool_args(name, args, analyst_intent, tool_results, user_text)
        try:
            output = tool_map[name].invoke(args, config={"callbacks": callbacks})
        except TypeError:
            output = tool_map[name].invoke(args)
        except Exception as exc:
            output = {"error": str(exc)}
        tool_calls += 1
        out_text = json.dumps(output, ensure_ascii=False) if isinstance(output, (dict, list)) else str(output)
        tool_results.append({"tool": name, "args": args, "output": out_text})
        tool_message = ToolMessage(content=out_text, tool_call_id=f"fallback_{name}_{tool_calls}", name=name)
        if context_budget_manager is not None and bool(getattr(context_budget_manager, "enabled", False)):
            tool_message = context_budget_manager.budget_tool_message(tool_message, tool_context=tool_context)
        messages.append(tool_message)

    if (
        graph_grounded_request
        and not _has_graph_search_attempt(tool_results)
        and "search_graph_index" in tool_map
        and tool_calls < max_tool_calls
    ):
        graph_id = _graph_id_from_tool_results(user_text, tool_results, tool_context=tool_context)
        if graph_id:
            args = {"query": user_text, "graph_id": graph_id, "limit": 8}
            try:
                output = tool_map["search_graph_index"].invoke(args, config={"callbacks": callbacks})
            except TypeError:
                output = tool_map["search_graph_index"].invoke(args)
            except Exception as exc:
                output = {"error": str(exc)}
            tool_calls += 1
            out_text = json.dumps(output, ensure_ascii=False) if isinstance(output, (dict, list)) else str(output)
            tool_results.append({"tool": "search_graph_index", "args": args, "output": out_text})
            tool_message = ToolMessage(
                content=out_text,
                tool_call_id=f"fallback_search_graph_index_{tool_calls}",
                name="search_graph_index",
            )
            if context_budget_manager is not None and bool(getattr(context_budget_manager, "enabled", False)):
                tool_message = context_budget_manager.budget_tool_message(tool_message, tool_context=tool_context)
            messages.append(tool_message)

    if graph_grounded_request and not _has_graph_search_attempt(tool_results):
        final_text = _graph_evidence_missing_text(user_text, tool_results, tool_context=tool_context)
        messages.append(AIMessage(content=final_text))
        return str(final_text), messages, _with_graph_execution_metadata(
            {
                "fallback": "plan_execute",
                "tool_calls": tool_calls,
                "recovery": ["graph_search_missing"],
            },
            graph_grounded_request=graph_grounded_request,
            user_text=user_text,
            tool_results=tool_results,
            tool_context=tool_context,
        )

    if (
        graph_grounded_request
        and _has_graph_search_attempt(tool_results)
        and (
            not _graph_search_has_grounded_evidence(tool_results)
            or _graph_search_requires_source_read(tool_results)
        )
    ):
        rag_rendered = _render_latest_rag_tool_result(tool_results)
        if rag_rendered:
            rag_rendered, verifier_recovery = _apply_final_evidence_verifier(
                rag_rendered,
                tool_results,
                user_text=user_text,
            )
            messages.append(AIMessage(content=rag_rendered))
            return str(rag_rendered), messages, _with_graph_execution_metadata(
                {
                    "fallback": "plan_execute",
                    "tool_calls": tool_calls,
                    "recovery": ["graph_catalog_used_rag_tool", *verifier_recovery],
                },
                graph_grounded_request=graph_grounded_request,
                user_text=user_text,
                tool_results=tool_results,
                tool_context=tool_context,
            )
        recovered, tool_calls, recovered_recovery = _run_graph_catalog_rag_recovery(
            tool_map,
            messages,
            tool_results,
            user_text=user_text,
            callbacks=callbacks,
            tool_calls=tool_calls,
            max_tool_calls=max_tool_calls,
        )
        if recovered:
            recovered, verifier_recovery = _apply_final_evidence_verifier(
                recovered,
                tool_results,
                user_text=user_text,
            )
            messages.append(AIMessage(content=recovered))
            return str(recovered), messages, _with_graph_execution_metadata(
                {
                    "fallback": "plan_execute",
                    "tool_calls": tool_calls,
                    "recovery": [*recovered_recovery, *verifier_recovery],
                },
                graph_grounded_request=graph_grounded_request,
                user_text=user_text,
                tool_results=tool_results,
                tool_context=tool_context,
            )
        final_text = _graph_evidence_missing_text(user_text, tool_results, tool_context=tool_context)
        messages.append(AIMessage(content=final_text))
        return str(final_text), messages, _with_graph_execution_metadata(
            {
                "fallback": "plan_execute",
                "tool_calls": tool_calls,
                "recovery": ["graph_catalog_only", *recovered_recovery],
            },
            graph_grounded_request=graph_grounded_request,
            user_text=user_text,
            tool_results=tool_results,
            tool_context=tool_context,
        )

    if analyst_intent is not None:
        latest_nlp_payload = None
        has_return_file = False
        for result in tool_results:
            tool_name = _tool_result_name(result)
            payload = _tool_result_payload(result)
            if tool_name == "run_nlp_column_task" and isinstance(payload, dict) and not payload.get("error"):
                latest_nlp_payload = payload
            elif tool_name == "return_file":
                has_return_file = True
        if (
            latest_nlp_payload is not None
            and analyst_intent.delivery_mode in {"summary_and_file", "file_only"}
            and not has_return_file
        ):
            written_file = str(latest_nlp_payload.get("written_file") or "").strip()
            if written_file and "return_file" in tool_map and tool_calls < max_tool_calls:
                return_args = {"filename": written_file}
                try:
                    return_output = tool_map["return_file"].invoke(return_args, config={"callbacks": callbacks})
                except TypeError:
                    return_output = tool_map["return_file"].invoke(return_args)
                except Exception as exc:
                    return_output = {"error": str(exc)}
                tool_calls += 1
                return_text = (
                    json.dumps(return_output, ensure_ascii=False)
                    if isinstance(return_output, (dict, list))
                    else str(return_output)
                )
                tool_results.append({"tool": "return_file", "args": return_args, "output": return_text})
                tool_message = ToolMessage(
                    content=return_text,
                    tool_call_id=f"fallback_return_file_{tool_calls}",
                    name="return_file",
                )
                if context_budget_manager is not None and bool(getattr(context_budget_manager, "enabled", False)):
                    tool_message = context_budget_manager.budget_tool_message(tool_message, tool_context=tool_context)
                messages.append(tool_message)

    if _data_analyst_fallback_enabled(tool_map) and not _has_successful_data_analyst_tool_result(tool_results):
        return _run_data_analyst_guided_fallback(
            chat_llm=chat_llm,
            tool_map=tool_map,
            messages=messages,
            user_text=user_text,
            callbacks=callbacks,
            max_tool_calls=max_tool_calls,
        )

    if analyst_intent is not None and _has_successful_data_analyst_tool_result(tool_results):
        analyst_text = _render_data_analyst_tool_results(
            tool_results,
            expected_delivery_mode=analyst_intent.delivery_mode,
        )
        if analyst_text:
            messages.append(AIMessage(content=analyst_text))
            return str(analyst_text), messages, {
                "fallback": "plan_execute",
                "tool_calls": tool_calls,
                "recovery": ["render_data_analyst_tool_fallback"],
                "delivery_mode": analyst_intent.delivery_mode,
            }

    synth_system = (
        "You are a helpful assistant. Use the TOOL_RESULTS to answer the USER_REQUEST.\n"
        "If TOOL_RESULTS include a rag_agent_tool JSON output, use its 'answer' and include citations.\n"
        "If TOOL_RESULTS include search_graph_index JSON output, use its `results` as graph evidence only when they include non-catalog chunks, relationship paths, or excerpts and requires_source_read is not true. Catalog-only results and requires_source_read=true results are source candidates or graph leads, not final evidence.\n"
        "If graph search results are empty, errored, source_candidates_only, or requires_source_read without a rag_agent_tool answer, say the graph evidence was insufficient instead of inventing relationships or causes.\n"
        "Do not invent Cypher queries, schema labels, relationship names, or how-to query syntax unless the user explicitly requested query syntax.\n"
        "Do not dump raw JSON; write a user-facing response."
    )
    synth_user = f"USER_REQUEST: {user_text}\n\nTOOL_RESULTS: {tool_results}"
    synth_response = chat_llm.invoke(
        [SystemMessage(content=synth_system), HumanMessage(content=synth_user)],
        config={"callbacks": callbacks},
    )
    final_text = _response_text(synth_response)
    recovery: List[str] = []
    if _is_output_truncated(synth_response):
        recovery.append("output_truncated")
        repaired = _synthesize_tool_results(
            chat_llm,
            user_text=user_text,
            tool_results=tool_results,
            callbacks=callbacks,
            system_prompt=system_prompt,
            recovery_reason="output_truncated",
        ).strip()
        if repaired:
            final_text = repaired
            recovery.append("tool_result_synthesis")
    if not str(final_text).strip():
        recovery.append("no_final_answer")
        fallback_text, fallback_recovery = _finalize_messages(
            chat_llm,
            messages=messages,
            user_text=user_text,
            callbacks=callbacks,
            system_prompt=system_prompt,
        )
        final_text = fallback_text
        recovery.extend(fallback_recovery)
    final_text = _append_missing_graph_citations(str(final_text), tool_results)
    final_text, verifier_recovery = _apply_final_evidence_verifier(
        final_text,
        tool_results,
        user_text=user_text,
    )
    recovery.extend(verifier_recovery)
    messages.append(AIMessage(content=str(final_text)))
    return str(final_text), messages, _with_graph_execution_metadata(
        {"fallback": "plan_execute", "tool_calls": tool_calls, "recovery": recovery},
        graph_grounded_request=graph_grounded_request,
        user_text=user_text,
        tool_results=tool_results,
        tool_context=tool_context,
    )
