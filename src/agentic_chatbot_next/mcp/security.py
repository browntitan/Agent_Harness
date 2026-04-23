from __future__ import annotations

import base64
import hashlib
import ipaddress
import re
from typing import Any
from urllib.parse import urlparse

from cryptography.fernet import Fernet, InvalidToken


_FERNET_PREFIX = "fernet:v1:"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_mcp_name(value: str, *, fallback: str = "mcp") -> str:
    slug = _SLUG_RE.sub("_", str(value or "").strip().lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return (slug or fallback).strip("_")[:80] or fallback


def normalize_mcp_registry_name(connection_slug: str, raw_tool_name: str) -> str:
    connection = slugify_mcp_name(connection_slug, fallback="connection")
    tool = slugify_mcp_name(raw_tool_name, fallback="tool")
    return f"mcp__{connection}__{tool}"


def _fernet(settings: Any) -> Fernet:
    raw_key = str(getattr(settings, "mcp_secret_encryption_key", "") or "").strip()
    if not raw_key:
        raise ValueError("MCP_SECRET_ENCRYPTION_KEY is required for secret-bearing MCP connections.")
    key = base64.urlsafe_b64encode(hashlib.sha256(raw_key.encode("utf-8")).digest())
    return Fernet(key)


def encrypt_mcp_secret(settings: Any, secret: str) -> str:
    clean = str(secret or "")
    if not clean:
        return ""
    token = _fernet(settings).encrypt(clean.encode("utf-8")).decode("ascii")
    return f"{_FERNET_PREFIX}{token}"


def decrypt_mcp_secret(settings: Any, encrypted_secret: str) -> str:
    encrypted = str(encrypted_secret or "")
    if not encrypted:
        return ""
    if not encrypted.startswith(_FERNET_PREFIX):
        raise ValueError("Unsupported MCP secret format.")
    token = encrypted[len(_FERNET_PREFIX) :]
    try:
        return _fernet(settings).decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("MCP secret could not be decrypted with the configured key.") from exc


def _host_is_private(hostname: str) -> bool:
    host = str(hostname or "").strip().lower().strip("[]")
    if not host:
        return True
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved)


def validate_mcp_server_url(settings: Any, server_url: str) -> str:
    clean = str(server_url or "").strip()
    if not clean:
        raise ValueError("MCP server URL is required.")
    parsed = urlparse(clean)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("MCP Streamable HTTP URLs must use http or https.")
    if bool(getattr(settings, "mcp_require_https", True)) and scheme != "https":
        raise ValueError("MCP_REQUIRE_HTTPS is enabled; MCP server URLs must use https.")
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("MCP server URL must include a host.")
    if not bool(getattr(settings, "mcp_allow_private_network", False)) and _host_is_private(parsed.hostname):
        raise ValueError("MCP private-network URLs are blocked by MCP_ALLOW_PRIVATE_NETWORK=false.")
    return clean
