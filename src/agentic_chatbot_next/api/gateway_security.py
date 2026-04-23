from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlencode


def extract_bearer_token(authorization_header: str | None) -> str:
    raw = str(authorization_header or "").strip()
    if not raw:
        return ""
    scheme, _, token = raw.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def is_authorized_bearer_token(authorization_header: str | None, expected_token: str | None) -> bool:
    expected = str(expected_token or "").strip()
    if not expected:
        return True
    received = extract_bearer_token(authorization_header)
    if not received:
        return False
    return hmac.compare_digest(received, expected)


def sign_download_token(
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


def verify_download_token(
    *,
    download_id: str,
    tenant_id: str,
    user_id: str,
    conversation_id: str,
    expires: int | str,
    sig: str | None,
    secret: str | None,
    now: int | None = None,
) -> bool:
    secret_text = str(secret or "").strip()
    signature = str(sig or "").strip()
    if not secret_text or not signature:
        return False
    try:
        expires_at = int(expires)
    except (TypeError, ValueError):
        return False
    current = int(now if now is not None else time.time())
    if expires_at < current:
        return False
    expected = sign_download_token(
        download_id=download_id,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        expires=expires_at,
        secret=secret_text,
    )
    return hmac.compare_digest(signature, expected)


def build_signed_download_url(
    *,
    download_id: str,
    tenant_id: str,
    user_id: str,
    conversation_id: str,
    secret: str | None,
    ttl_seconds: int,
    path: str | None = None,
    now: int | None = None,
) -> str:
    secret_text = str(secret or "").strip()
    base_path = str(path or f"/v1/files/{download_id}")
    if not secret_text:
        return f"{base_path}?conversation_id={conversation_id}"

    current = int(now if now is not None else time.time())
    ttl = max(1, int(ttl_seconds))
    expires = current + ttl
    sig = sign_download_token(
        download_id=download_id,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        expires=expires,
        secret=secret_text,
    )
    query = urlencode(
        {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "expires": expires,
            "sig": sig,
        }
    )
    separator = "&" if "?" in base_path else "?"
    return f"{base_path}{separator}{query}"
