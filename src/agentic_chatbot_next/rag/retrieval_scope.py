from __future__ import annotations

from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from agentic_chatbot_next.authz import access_summary_allowed_ids, access_summary_authz_enabled

RETRIEVAL_SCOPE_MODES = {
    "uploads_only",
    "kb_only",
    "both",
    "none",
    "ambiguous",
}

_EXPLICIT_NONE_PATTERNS = (
    re.compile(r"\b(?:do not|don't|dont|without)\s+(?:look|search|use|consult)\b", re.IGNORECASE),
    re.compile(r"\b(?:don't|do not|dont)\s+look\s+anything\s+up\b", re.IGNORECASE),
    re.compile(r"\b(?:just|only)\s+(?:rewrite|rephrase|paraphrase|translate|summarize)\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:lookup|lookups|retrieval|search|searches|citations?|sources?)\b", re.IGNORECASE),
)
_EXPLICIT_BOTH_PATTERNS = (
    re.compile(r"\buse\s+both\b", re.IGNORECASE),
    re.compile(r"\bboth\s+the\s+(?:uploaded|attached)\b", re.IGNORECASE),
    re.compile(r"\bcompare\b.*\b(?:uploaded|attached|file|csv|xlsx|spreadsheet|workbook)\b.*\b(?:knowledge\s*base|kb|docs?|documentation|database|vector)\b", re.IGNORECASE),
    re.compile(r"\b(?:uploaded|attached|file|csv|xlsx|spreadsheet|workbook)\b.*\b(?:knowledge\s*base|kb|docs?|documentation|database|vector)\b", re.IGNORECASE),
    re.compile(r"\b(?:knowledge\s*base|kb|docs?|documentation|database|vector)\b.*\b(?:uploaded|attached|file|csv|xlsx|spreadsheet|workbook)\b", re.IGNORECASE),
)
_EXPLICIT_UPLOAD_PATTERNS = (
    re.compile(r"\bonly\s+(?:use|search|consult|refer(?:\s+to)?)\s+(?:the\s+)?(?:uploaded|attached)\b", re.IGNORECASE),
    re.compile(r"\b(?:uploaded|attached)\s+(?:file|files|document|documents|csv|xlsx|spreadsheet|workbook)\s+only\b", re.IGNORECASE),
    re.compile(r"\b(?:use|search|analy[sz]e|summarize|review|inspect)\b.*\b(?:uploaded|attached|attachment|csv|xlsx|spreadsheet|workbook|pdf)\b", re.IGNORECASE),
    re.compile(r"\b(?:from|in)\s+(?:the\s+)?(?:uploaded|attached|attachment)\b", re.IGNORECASE),
)
_EXPLICIT_KB_PATTERNS = (
    re.compile(r"\bonly\s+(?:use|search|consult|refer(?:\s+to)?)\s+(?:the\s+)?(?:knowledge\s*base|kb|docs?|documentation|repo)\b", re.IGNORECASE),
    re.compile(r"\b(?:knowledge\s*base|kb|docs?|documentation|repo|vector\s+database|database)\s+only\b", re.IGNORECASE),
    re.compile(r"\b(?:search|look\s+up|consult|use)\b.*\b(?:knowledge\s*base|kb|docs?|documentation|repo|vector\s+database|database)\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s|\s+is)\s+indexed\b", re.IGNORECASE),
    re.compile(r"\bknowledge\s*base\s+inventory\b", re.IGNORECASE),
    re.compile(r"\b(?:what|which|list|show)\b.*\b(?:knowledge\s*base|kb)\b", re.IGNORECASE),
)
_CASUAL_NONE_PATTERNS = (
    re.compile(r"^\s*(?:hi|hello|hey|yo|good\s+(?:morning|afternoon|evening))[\s!.?]*$", re.IGNORECASE),
    re.compile(r"^\s*(?:thanks|thank\s+you|thx)[\s!.?]*$", re.IGNORECASE),
    re.compile(r"\b(?:joke|poem|brainstorm|story)\b", re.IGNORECASE),
)
_FILE_NAME_PATTERN = re.compile(
    r"(?<![\w/-])([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8})(?![\w/-])"
)
_QUOTED_PATTERN = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_GROUNDED_HINTS = re.compile(
    r"\b("
    r"cite|citation|source|sources|according to|document|documents|doc|docs|documentation|knowledge\s*base|kb|"
    r"repo|manual|policy|policies|architecture|pricing|security|privacy|workflow|process|compare|difference|"
    r"differences|list|identify|find|search|look up|explain|summarize|summarise|analy[sz]e|review|"
    r"uploaded|attachment|attached|file|files|csv|xlsx|spreadsheet|workbook|pdf"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalScopeDecision:
    mode: str
    reason: str
    upload_collection_id: str = ""
    kb_collection_id: str = ""
    search_collection_ids: tuple[str, ...] = ()
    has_uploads: bool = False
    kb_available: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "retrieval_scope_mode": self.mode,
            "retrieval_scope_reason": self.reason,
            "upload_collection_id": self.upload_collection_id,
            "kb_collection_id": self.kb_collection_id,
            "search_collection_ids": list(self.search_collection_ids),
            "has_uploads": self.has_uploads,
            "kb_available": self.kb_available,
        }


def _metadata(session: Any) -> dict[str, Any]:
    if isinstance(session, Mapping):
        return dict(session)
    return dict(getattr(session, "metadata", {}) or {})


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _dedupe_non_empty(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def _normalize_collection_ids(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    return _dedupe_non_empty(str(item) for item in values)


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def document_source_policy_requires_repository(session_or_metadata: Any) -> bool:
    metadata = _metadata(session_or_metadata)
    policy = str(metadata.get("document_source_policy") or "").strip().lower()
    if policy == "agent_repository_only":
        return True
    return _coerce_bool(metadata.get("openwebui_thin_mode"), default=False)


def repository_upload_doc_ids(session: Any) -> tuple[str, ...]:
    metadata = _metadata(session)
    return _dedupe_non_empty(
        [
            *list(getattr(session, "uploaded_doc_ids", []) or []),
            *list(metadata.get("uploaded_doc_ids") or []),
            *list(metadata.get("upload_doc_ids") or []),
            *list(metadata.get("active_uploaded_doc_ids") or []),
        ]
    )


def has_upload_evidence(session: Any) -> bool:
    return bool(repository_upload_doc_ids(session))


def query_requests_upload_scope(query: str) -> bool:
    text = str(query or "")
    return any(pattern.search(text) for pattern in _EXPLICIT_UPLOAD_PATTERNS)


def resolve_upload_collection_id(settings: Any, session: Any) -> str:
    metadata = _metadata(session)
    return _first_non_empty(
        metadata.get("upload_collection_id"),
        metadata.get("collection_id"),
        getattr(settings, "default_collection_id", "default"),
        "default",
    )


def resolve_kb_collection_id(settings: Any, session: Any) -> str:
    metadata = _metadata(session)
    return _first_non_empty(
        metadata.get("kb_collection_id"),
        metadata.get("collection_id"),
        getattr(settings, "default_collection_id", "default"),
        "default",
    )


def resolve_available_kb_collection_ids(settings: Any, session: Any) -> tuple[str, ...]:
    metadata = _metadata(session)
    access_summary = dict(metadata.get("access_summary") or {})
    if access_summary_authz_enabled(access_summary):
        return _dedupe_non_empty(access_summary_allowed_ids(access_summary, "collection", action="use"))
    normalized = _normalize_collection_ids(metadata.get("available_kb_collection_ids"))
    if normalized:
        return normalized
    explicit_upload = _first_non_empty(metadata.get("upload_collection_id"))
    explicit_kb = _first_non_empty(metadata.get("kb_collection_id"))
    legacy_collection = _first_non_empty(metadata.get("collection_id"))
    fallback_ids: list[str] = []
    if explicit_kb:
        fallback_ids.append(explicit_kb)
    if legacy_collection and not explicit_upload and not explicit_kb:
        fallback_ids.append(legacy_collection)
    fallback_ids.append(_first_non_empty(getattr(settings, "default_collection_id", "default"), "default"))
    return _dedupe_non_empty(
        fallback_ids
    )


def resolve_kb_collection_confirmed(session: Any) -> bool:
    metadata = _metadata(session)
    return _coerce_bool(metadata.get("kb_collection_confirmed"), default=False)


def resolve_search_collection_ids(settings: Any, session: Any) -> tuple[str, ...]:
    metadata = _metadata(session)
    access_summary = dict(metadata.get("access_summary") or {})
    if access_summary_authz_enabled(access_summary):
        return _dedupe_non_empty(
            [
                *access_summary_allowed_ids(access_summary, "collection", action="use"),
                str(access_summary.get("session_upload_collection_id") or "").strip(),
            ]
        )
    raw = metadata.get("search_collection_ids")
    if isinstance(raw, (list, tuple)):
        normalized = _dedupe_non_empty(str(item) for item in raw)
        if normalized:
            return normalized
    return _dedupe_non_empty(
        [
            metadata.get("collection_id"),
            metadata.get("upload_collection_id"),
            metadata.get("kb_collection_id"),
            getattr(settings, "default_collection_id", "default"),
        ]
    )


def resolve_requested_kb_collection_id(session: Any) -> str:
    metadata = _metadata(session)
    route_context = dict(metadata.get("route_context") or {})
    semantic_routing = dict(route_context.get("semantic_routing") or {})
    return _first_non_empty(
        metadata.get("requested_kb_collection_id"),
        metadata.get("selected_kb_collection_id"),
        semantic_routing.get("requested_collection_id"),
        route_context.get("requested_collection_id"),
    )


def resolve_collection_ids_for_source(
    settings: Any,
    session: Any,
    *,
    source_type: str = "kb",
    explicit_collection_id: str = "",
) -> tuple[str, ...]:
    explicit = _first_non_empty(explicit_collection_id)
    if explicit:
        return (explicit,)

    normalized_source = str(source_type or "kb").strip().lower()
    metadata = _metadata(session)
    upload_collection_id = resolve_upload_collection_id(settings, session)
    requested_kb_collection_id = resolve_requested_kb_collection_id(session)

    if normalized_source == "upload":
        return _dedupe_non_empty([upload_collection_id])

    if normalized_source in {"all", "*", "any", ""}:
        raw = metadata.get("search_collection_ids")
        if isinstance(raw, (list, tuple)):
            normalized = _dedupe_non_empty(str(item) for item in raw)
            if normalized:
                return normalized
        return _dedupe_non_empty(
            [
                requested_kb_collection_id,
                *resolve_available_kb_collection_ids(settings, session),
                upload_collection_id,
            ]
        )

    if normalized_source != "kb":
        return resolve_search_collection_ids(settings, session)

    if requested_kb_collection_id:
        return (requested_kb_collection_id,)

    available_kb_collection_ids = tuple(
        item
        for item in resolve_available_kb_collection_ids(settings, session)
        if item != upload_collection_id
    )
    available = set(available_kb_collection_ids)
    raw = metadata.get("search_collection_ids")
    if isinstance(raw, (list, tuple)):
        normalized = _dedupe_non_empty(str(item) for item in raw)
        filtered = [
            item
            for item in normalized
            if item != upload_collection_id and (not available or item in available)
        ]
        if filtered:
            return _dedupe_non_empty(filtered)

    return _dedupe_non_empty(
        [
            *available_kb_collection_ids,
            resolve_kb_collection_id(settings, session),
            getattr(settings, "default_collection_id", "default"),
        ]
    )


def merge_scope_metadata(settings: Any, metadata: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(metadata or {})
    legacy_collection_id = _first_non_empty(
        raw.get("collection_id"),
        getattr(settings, "default_collection_id", "default"),
        "default",
    )
    explicit_upload = _first_non_empty(raw.get("upload_collection_id"))
    explicit_kb = _first_non_empty(raw.get("kb_collection_id"))

    if explicit_upload or explicit_kb:
        upload_collection_id = _first_non_empty(explicit_upload, legacy_collection_id)
        kb_collection_id = _first_non_empty(explicit_kb, getattr(settings, "default_collection_id", "default"), legacy_collection_id)
    else:
        upload_collection_id = legacy_collection_id
        kb_collection_id = legacy_collection_id

    if "kb_collection_confirmed" in raw:
        kb_collection_confirmed = _coerce_bool(raw.get("kb_collection_confirmed"), default=False)
    else:
        kb_collection_confirmed = bool(explicit_kb or (str(raw.get("collection_id") or "").strip() and not explicit_upload))

    access_summary = dict(raw.get("access_summary") or {})
    available_kb_collection_ids = list(
        _normalize_collection_ids(raw.get("available_kb_collection_ids"))
        or (
            _dedupe_non_empty(
                [
                    *access_summary_allowed_ids(access_summary, "collection", action="use"),
                ]
            )
            if access_summary_authz_enabled(access_summary)
            else _dedupe_non_empty([kb_collection_id])
        )
    )

    return {
        **raw,
        "collection_id": legacy_collection_id,
        "upload_collection_id": upload_collection_id,
        "kb_collection_id": kb_collection_id,
        "available_kb_collection_ids": available_kb_collection_ids,
        "kb_collection_confirmed": kb_collection_confirmed,
    }


def _has_uploads(session: Any) -> bool:
    return has_upload_evidence(session)


def _looks_casual_or_non_grounded(query: str) -> bool:
    return any(pattern.search(query) for pattern in _CASUAL_NONE_PATTERNS)


def _looks_grounded(query: str) -> bool:
    return bool(_GROUNDED_HINTS.search(query))


def _explicit_upload_excludes_kb(query: str) -> bool:
    text = str(query or "")
    if not any(pattern.search(text) for pattern in _EXPLICIT_UPLOAD_PATTERNS):
        return False
    return bool(
        re.search(
            r"\b(?:not|without|ignore|excluding)\b.*\b(?:knowledge\s*base|kb|docs?|documentation|repo)\b",
            text,
            re.IGNORECASE,
        )
    )


def _explicit_current_chat_upload_request(query: str) -> bool:
    text = str(query or "")
    return bool(
        re.search(
            r"\b(?:(?:current|this)\s+chat\s+|attached\s+|attachment\s+|my\s+)(?:upload|uploads|file|files|document|documents)\b",
            text,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:in|from|use|inspect|review|summari[sz]e)\s+(?:the\s+)?(?:(?:current|this)\s+chat\s+)?(?:uploaded|attached|attachment)\b",
            text,
            re.IGNORECASE,
        )
        or re.search(
            r"\bsearch\s+(?:the\s+)?(?:(?:current|this)\s+chat\s+)?(?:attached|attachment)\b",
            text,
            re.IGNORECASE,
        )
    )


def _explicit_kb_scope_requested(session: Any) -> bool:
    metadata = _metadata(session)
    if resolve_requested_kb_collection_id(session):
        return True
    if _coerce_bool(metadata.get("kb_collection_confirmed"), default=False):
        return True
    raw_search_ids = metadata.get("search_collection_ids")
    if isinstance(raw_search_ids, (list, tuple)):
        upload_collection_id = _first_non_empty(metadata.get("upload_collection_id"))
        normalized = [str(item or "").strip() for item in raw_search_ids if str(item or "").strip()]
        if normalized and any(item != upload_collection_id for item in normalized):
            return True
    return False


def _extract_named_document_targets(query: str) -> tuple[str, ...]:
    text = str(query or "")
    candidates: list[str] = []
    seen: set[str] = set()
    for match in _FILE_NAME_PATTERN.finditer(text):
        candidate = str(match.group(1) or "").strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    for match in _QUOTED_PATTERN.finditer(text):
        candidate = str(match.group(1) or match.group(2) or "").strip()
        if not candidate:
            continue
        if "." not in candidate and "/" not in candidate and "\\" not in candidate:
            continue
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return tuple(candidates)


def _configured_kb_doc_names(settings: Any) -> set[str]:
    try:
        from agentic_chatbot_next.rag.ingest import iter_kb_source_paths

        return {
            str(path.name or "").strip().casefold()
            for path in iter_kb_source_paths(settings)
            if str(path.name or "").strip()
        }
    except Exception:
        return set()


def _query_explicitly_names_kb_doc(settings: Any, query: str) -> bool:
    configured_names = _configured_kb_doc_names(settings)
    if not configured_names:
        return False
    for target in _extract_named_document_targets(query):
        basename = Path(str(target or "").replace("\\", "/")).name.casefold()
        if basename and basename in configured_names:
            return True
    return False


def decide_retrieval_scope(
    settings: Any,
    session: Any,
    *,
    query: str,
    kb_available: bool = False,
    has_uploads: bool | None = None,
) -> RetrievalScopeDecision:
    text = str(query or "").strip()
    upload_collection_id = resolve_upload_collection_id(settings, session)
    kb_collection_id = resolve_kb_collection_id(settings, session)
    effective_has_uploads = bool(_has_uploads(session) if has_uploads is None else has_uploads)

    if not text:
        return RetrievalScopeDecision(
            mode="none",
            reason="empty_query",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if any(pattern.search(text) for pattern in _EXPLICIT_NONE_PATTERNS):
        return RetrievalScopeDecision(
            mode="none",
            reason="explicit_none",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if _explicit_upload_excludes_kb(text):
        return RetrievalScopeDecision(
            mode="uploads_only",
            reason="explicit_uploads_only",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            search_collection_ids=_dedupe_non_empty([upload_collection_id if effective_has_uploads else ""]),
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    explicit_kb_text = any(pattern.search(text) for pattern in _EXPLICIT_KB_PATTERNS)
    if (explicit_kb_text or _explicit_kb_scope_requested(session)) and not _explicit_current_chat_upload_request(text):
        return RetrievalScopeDecision(
            mode="kb_only",
            reason="explicit_kb_only",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            search_collection_ids=_dedupe_non_empty([kb_collection_id]),
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if any(pattern.search(text) for pattern in _EXPLICIT_BOTH_PATTERNS):
        return RetrievalScopeDecision(
            mode="both",
            reason="explicit_both",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            search_collection_ids=_dedupe_non_empty(
                [
                    kb_collection_id if kb_collection_id and kb_available else "",
                    upload_collection_id if effective_has_uploads else "",
                ]
            ),
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if any(pattern.search(text) for pattern in _EXPLICIT_UPLOAD_PATTERNS):
        return RetrievalScopeDecision(
            mode="uploads_only",
            reason="explicit_uploads_only",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            search_collection_ids=_dedupe_non_empty([upload_collection_id if effective_has_uploads else ""]),
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if any(pattern.search(text) for pattern in _EXPLICIT_KB_PATTERNS):
        return RetrievalScopeDecision(
            mode="kb_only",
            reason="explicit_kb_only",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            search_collection_ids=_dedupe_non_empty([kb_collection_id]),
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if _looks_casual_or_non_grounded(text):
        return RetrievalScopeDecision(
            mode="none",
            reason="casual_or_non_grounded",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if kb_available and _query_explicitly_names_kb_doc(settings, text):
        return RetrievalScopeDecision(
            mode="kb_only",
            reason="explicit_named_kb_doc",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            search_collection_ids=_dedupe_non_empty([kb_collection_id]),
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    grounded = _looks_grounded(text)

    if grounded and effective_has_uploads and kb_available:
        return RetrievalScopeDecision(
            mode="ambiguous",
            reason="grounded_without_source_preference",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if grounded and effective_has_uploads:
        return RetrievalScopeDecision(
            mode="uploads_only",
            reason="grounded_with_uploads_available",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            search_collection_ids=_dedupe_non_empty([upload_collection_id if effective_has_uploads else ""]),
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if grounded:
        return RetrievalScopeDecision(
            mode="kb_only",
            reason="grounded_with_shared_kb",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            search_collection_ids=_dedupe_non_empty([kb_collection_id]),
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if effective_has_uploads and kb_available:
        return RetrievalScopeDecision(
            mode="ambiguous",
            reason="sources_available_without_explicit_preference",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    if effective_has_uploads:
        return RetrievalScopeDecision(
            mode="uploads_only",
            reason="uploads_available_only",
            upload_collection_id=upload_collection_id,
            kb_collection_id=kb_collection_id,
            search_collection_ids=_dedupe_non_empty([upload_collection_id if effective_has_uploads else ""]),
            has_uploads=effective_has_uploads,
            kb_available=kb_available,
        )

    return RetrievalScopeDecision(
        mode="kb_only",
        reason="kb_only_fallback",
        upload_collection_id=upload_collection_id,
        kb_collection_id=kb_collection_id,
        search_collection_ids=_dedupe_non_empty([kb_collection_id]),
        has_uploads=effective_has_uploads,
        kb_available=kb_available,
    )


__all__ = [
    "RETRIEVAL_SCOPE_MODES",
    "RetrievalScopeDecision",
    "decide_retrieval_scope",
    "document_source_policy_requires_repository",
    "has_upload_evidence",
    "merge_scope_metadata",
    "query_requests_upload_scope",
    "repository_upload_doc_ids",
    "resolve_available_kb_collection_ids",
    "resolve_collection_ids_for_source",
    "resolve_kb_collection_confirmed",
    "resolve_kb_collection_id",
    "resolve_requested_kb_collection_id",
    "resolve_search_collection_ids",
    "resolve_upload_collection_id",
]
