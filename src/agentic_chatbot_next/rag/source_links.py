from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Callable
from urllib.parse import urlencode


def _session_value(session_or_state: Any, name: str, default: str = "") -> str:
    return str(getattr(session_or_state, name, default) or default or "")


def _join_base_url(base_url: str, path: str) -> str:
    clean_path = str(path or "")
    clean_base = str(base_url or "").strip().rstrip("/")
    if not clean_base:
        return clean_path
    if not clean_path.startswith("/"):
        clean_path = "/" + clean_path
    return f"{clean_base}{clean_path}"


def _sign_download_token(
    *,
    download_id: str,
    tenant_id: str,
    user_id: str,
    conversation_id: str,
    expires: int,
    secret: str,
) -> str:
    payload = "\n".join(
        [
            str(download_id or ""),
            str(tenant_id or ""),
            str(user_id or ""),
            str(conversation_id or ""),
            str(int(expires)),
        ]
    )
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _build_signed_source_url(
    *,
    doc_id: str,
    tenant_id: str,
    user_id: str,
    conversation_id: str,
    secret: str | None,
    ttl_seconds: int,
) -> str:
    path = f"/v1/documents/{doc_id}/source"
    secret_text = str(secret or "").strip()
    if not secret_text:
        return f"{path}?conversation_id={conversation_id}"
    expires = int(time.time()) + max(1, int(ttl_seconds))
    sig = _sign_download_token(
        download_id=doc_id,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        expires=expires,
        secret=secret_text,
    )
    return f"{path}?{urlencode({'tenant_id': tenant_id, 'user_id': user_id, 'conversation_id': conversation_id, 'expires': expires, 'sig': sig})}"


def build_document_source_url(
    settings: Any,
    session_or_state: Any,
    doc_id: str,
    *,
    tenant_id: str = "",
    user_id: str = "",
    conversation_id: str = "",
) -> str:
    clean_doc_id = str(doc_id or "").strip()
    if not clean_doc_id:
        return ""
    resolved_tenant_id = tenant_id or _session_value(
        session_or_state,
        "tenant_id",
        str(getattr(settings, "default_tenant_id", "local-dev") or "local-dev"),
    )
    resolved_user_id = user_id or _session_value(
        session_or_state,
        "user_id",
        str(getattr(settings, "default_user_id", "") or ""),
    )
    resolved_conversation_id = conversation_id or _session_value(
        session_or_state,
        "conversation_id",
        str(getattr(settings, "default_conversation_id", "local-session") or "local-session"),
    )
    relative = _build_signed_source_url(
        doc_id=clean_doc_id,
        tenant_id=resolved_tenant_id,
        user_id=resolved_user_id,
        conversation_id=resolved_conversation_id,
        secret=getattr(settings, "download_url_secret", None),
        ttl_seconds=int(getattr(settings, "download_url_ttl_seconds", 900) or 900),
    )
    return _join_base_url(str(getattr(settings, "gateway_public_base_url", "") or ""), relative)


def make_document_source_url_resolver(settings: Any, session_or_state: Any) -> Callable[[str], str]:
    def _resolve(doc_id: str) -> str:
        return build_document_source_url(settings, session_or_state, doc_id)

    return _resolve
