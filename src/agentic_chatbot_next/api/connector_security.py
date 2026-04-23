from __future__ import annotations

import hmac
import threading
import time
from collections import deque
from dataclasses import dataclass

from fastapi import HTTPException

from agentic_chatbot_next.api.gateway_security import extract_bearer_token
from agentic_chatbot_next.config import Settings


@dataclass(frozen=True)
class ConnectorAuthResult:
    token_type: str
    origin: str


_PUBLISHABLE_BUCKETS: dict[str, deque[float]] = {}
_PUBLISHABLE_BUCKETS_LOCK = threading.Lock()


def _matches(expected: str | None, received: str) -> bool:
    token = str(expected or "").strip()
    return bool(token and received and hmac.compare_digest(received, token))


def _require_allowed_origin(settings: Settings, origin: str) -> None:
    allowed = tuple(str(item or "").strip() for item in settings.connector_allowed_origins)
    if not allowed:
        return
    if origin and origin in allowed:
        return
    raise HTTPException(
        status_code=403,
        detail="Publishable connector keys may only be used from configured browser origins.",
    )


def _enforce_publishable_rate_limit(
    settings: Settings,
    *,
    origin: str,
    client_host: str,
) -> None:
    limit = int(getattr(settings, "connector_publishable_rate_limit_per_minute", 0) or 0)
    if limit <= 0:
        return
    key = f"{origin or '-'}|{client_host or '-'}"
    now = time.time()
    cutoff = now - 60.0
    with _PUBLISHABLE_BUCKETS_LOCK:
        bucket = _PUBLISHABLE_BUCKETS.setdefault(key, deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Connector publishable key rate limit exceeded. Retry shortly.",
            )
        bucket.append(now)


def require_connector_bearer_auth(
    settings: Settings,
    authorization: str | None,
    *,
    origin: str | None = None,
    client_host: str | None = None,
) -> ConnectorAuthResult:
    received = extract_bearer_token(authorization)
    origin_text = str(origin or "").strip()
    if _matches(settings.connector_secret_api_key, received):
        return ConnectorAuthResult(token_type="secret", origin=origin_text)
    if _matches(settings.connector_publishable_api_key, received):
        _require_allowed_origin(settings, origin_text)
        _enforce_publishable_rate_limit(
            settings,
            origin=origin_text,
            client_host=str(client_host or "").strip(),
        )
        return ConnectorAuthResult(token_type="publishable", origin=origin_text)

    configured_any = bool(
        str(settings.connector_secret_api_key or "").strip()
        or str(settings.connector_publishable_api_key or "").strip()
    )
    if not configured_any:
        return ConnectorAuthResult(token_type="anonymous", origin=origin_text)
    raise HTTPException(status_code=401, detail="Missing or invalid connector bearer token.")
