from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

import httpx


class GatewayClientError(RuntimeError):
    """Raised when the backend gateway returns an error response."""


@dataclass
class GatewayChatResponse:
    text: str
    raw: Dict[str, Any]


@dataclass(frozen=True)
class GatewayStreamEvent:
    kind: str
    event_name: str
    payload: Any
    text_delta: str = ""


@dataclass
class GatewayStreamResult:
    text: str
    events: List[GatewayStreamEvent]
    progress_events: List[Dict[str, Any]]
    artifacts: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    raw_chunks: List[Dict[str, Any]]


class GatewayClient:
    def __init__(self, base_url: str, *, timeout_seconds: Optional[float] = None) -> None:
        self.base_url = base_url.rstrip("/")
        effective_timeout = timeout_seconds
        if effective_timeout is None:
            effective_timeout = float(os.getenv("NEXT_RUNTIME_GATEWAY_TIMEOUT_SECONDS", "1800"))
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                timeout=effective_timeout,
                connect=min(10.0, effective_timeout),
                read=effective_timeout,
                write=effective_timeout,
                pool=effective_timeout,
            ),
        )

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health_ready(self) -> Dict[str, Any]:
        response = self._client.get("/health/ready")
        response.raise_for_status()
        return dict(response.json())

    def health_live(self) -> Dict[str, Any]:
        response = self._client.get("/health/live")
        response.raise_for_status()
        return dict(response.json())

    def get_model_id(self) -> str:
        response = self._client.get("/v1/models")
        response.raise_for_status()
        payload = dict(response.json())
        data = payload.get("data") or []
        if not data:
            raise GatewayClientError("Gateway returned no models.")
        return str(data[0].get("id") or "")

    @staticmethod
    def collect_stream(events: Iterable[GatewayStreamEvent]) -> GatewayStreamResult:
        all_events = list(events)
        raw_chunks: List[Dict[str, Any]] = []
        progress_events: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []
        metadata: Dict[str, Any] = {}
        content_parts: List[str] = []
        for event in all_events:
            if event.kind == "progress" and isinstance(event.payload, dict):
                progress_events.append(dict(event.payload))
            elif event.kind == "artifacts" and isinstance(event.payload, list):
                artifacts.extend(dict(item) for item in event.payload if isinstance(item, dict))
            elif event.kind == "metadata" and isinstance(event.payload, dict):
                metadata.update(dict(event.payload))
            elif event.kind == "content":
                if event.text_delta:
                    content_parts.append(event.text_delta)
                if isinstance(event.payload, dict):
                    raw_chunks.append(dict(event.payload))
        return GatewayStreamResult(
            text="".join(content_parts),
            events=all_events,
            progress_events=progress_events,
            artifacts=artifacts,
            metadata=metadata,
            raw_chunks=raw_chunks,
        )

    def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        model: str,
        force_agent: bool = False,
        request_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: str = "",
        user_id: str = "",
        collection_id: str = "",
    ) -> GatewayChatResponse:
        payload_metadata = dict(metadata or {})
        if force_agent:
            payload_metadata["force_agent"] = True
        if collection_id:
            payload_metadata.setdefault("collection_id", collection_id)
        response = self._client.post(
            "/v1/chat/completions",
            headers=self._chat_headers(
                conversation_id=conversation_id,
                request_id=request_id,
                tenant_id=tenant_id,
                user_id=user_id,
                collection_id=collection_id,
            ),
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "metadata": payload_metadata,
            },
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"chat failed: {response.status_code} {response.text}")
        payload = dict(response.json())
        text = str(payload["choices"][0]["message"]["content"])
        return GatewayChatResponse(text=text, raw=payload)

    def chat_turn(
        self,
        *,
        history: List[Dict[str, Any]],
        user_text: str,
        conversation_id: str,
        model: str,
        force_agent: bool = False,
        request_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: str = "",
        user_id: str = "",
        collection_id: str = "",
    ) -> GatewayChatResponse:
        return self.chat(
            messages=list(history) + [{"role": "user", "content": user_text}],
            conversation_id=conversation_id,
            model=model,
            force_agent=force_agent,
            request_id=request_id,
            metadata=metadata,
            tenant_id=tenant_id,
            user_id=user_id,
            collection_id=collection_id,
        )

    def stream_chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        model: str,
        force_agent: bool = False,
        request_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: str = "",
        user_id: str = "",
        collection_id: str = "",
    ) -> Iterator[GatewayStreamEvent]:
        payload_metadata = dict(metadata or {})
        if force_agent:
            payload_metadata["force_agent"] = True
        if collection_id:
            payload_metadata.setdefault("collection_id", collection_id)
        with self._client.stream(
            "POST",
            "/v1/chat/completions",
            headers=self._chat_headers(
                conversation_id=conversation_id,
                request_id=request_id,
                tenant_id=tenant_id,
                user_id=user_id,
                collection_id=collection_id,
            ),
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "metadata": payload_metadata,
            },
        ) as response:
            if response.status_code >= 400:
                raise GatewayClientError(f"stream chat failed: {response.status_code} {response.text}")
            yield from self._parse_sse_events(response.iter_lines())

    def stream_chat_turn(
        self,
        *,
        history: List[Dict[str, Any]],
        user_text: str,
        conversation_id: str,
        model: str,
        force_agent: bool = False,
        request_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: str = "",
        user_id: str = "",
        collection_id: str = "",
    ) -> Iterator[GatewayStreamEvent]:
        yield from self.stream_chat(
            messages=list(history) + [{"role": "user", "content": user_text}],
            conversation_id=conversation_id,
            model=model,
            force_agent=force_agent,
            request_id=request_id,
            metadata=metadata,
            tenant_id=tenant_id,
            user_id=user_id,
            collection_id=collection_id,
        )

    def ingest(
        self,
        *,
        paths: List[str],
        conversation_id: str,
        source_type: str = "upload",
        request_id: str = "",
        collection_id: str = "",
        tenant_id: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        response = self._client.post(
            "/v1/ingest/documents",
            headers=self._chat_headers(
                conversation_id=conversation_id,
                request_id=request_id,
                tenant_id=tenant_id,
                user_id=user_id,
                collection_id=collection_id,
            ),
            json={
                "paths": paths,
                "source_type": source_type,
                "collection_id": collection_id or None,
                "conversation_id": conversation_id,
            },
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"ingest failed: {response.status_code} {response.text}")
        return dict(response.json())

    def list_skills(
        self,
        *,
        agent_scope: str = "",
        enabled_only: bool = False,
        visibility: str = "",
        status: str = "",
        tenant_id: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        response = self._client.get(
            "/v1/skills",
            headers=self._scope_headers(tenant_id=tenant_id, user_id=user_id),
            params={
                "agent_scope": agent_scope,
                "enabled_only": str(bool(enabled_only)).lower(),
                "visibility": visibility,
                "status": status,
            },
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"list skills failed: {response.status_code} {response.text}")
        return dict(response.json())

    def get_skill(self, skill_id: str, *, tenant_id: str = "", user_id: str = "") -> Dict[str, Any]:
        response = self._client.get(
            f"/v1/skills/{skill_id}",
            headers=self._scope_headers(tenant_id=tenant_id, user_id=user_id),
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"get skill failed: {response.status_code} {response.text}")
        return dict(response.json())

    def create_skill(self, payload: Dict[str, Any], *, tenant_id: str = "", user_id: str = "") -> Dict[str, Any]:
        response = self._client.post(
            "/v1/skills",
            headers=self._scope_headers(tenant_id=tenant_id, user_id=user_id),
            json=payload,
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"create skill failed: {response.status_code} {response.text}")
        return dict(response.json())

    def update_skill(self, skill_id: str, payload: Dict[str, Any], *, tenant_id: str = "", user_id: str = "") -> Dict[str, Any]:
        response = self._client.put(
            f"/v1/skills/{skill_id}",
            headers=self._scope_headers(tenant_id=tenant_id, user_id=user_id),
            json=payload,
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"update skill failed: {response.status_code} {response.text}")
        return dict(response.json())

    def activate_skill(self, skill_id: str, *, tenant_id: str = "", user_id: str = "") -> Dict[str, Any]:
        response = self._client.post(
            f"/v1/skills/{skill_id}/activate",
            headers=self._scope_headers(tenant_id=tenant_id, user_id=user_id),
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"activate skill failed: {response.status_code} {response.text}")
        return dict(response.json())

    def deactivate_skill(self, skill_id: str, *, tenant_id: str = "", user_id: str = "") -> Dict[str, Any]:
        response = self._client.post(
            f"/v1/skills/{skill_id}/deactivate",
            headers=self._scope_headers(tenant_id=tenant_id, user_id=user_id),
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"deactivate skill failed: {response.status_code} {response.text}")
        return dict(response.json())

    def rollback_skill(
        self,
        skill_id: str,
        *,
        target_skill_id: str,
        tenant_id: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        response = self._client.post(
            f"/v1/skills/{skill_id}/rollback",
            headers=self._scope_headers(tenant_id=tenant_id, user_id=user_id),
            json={"target_skill_id": target_skill_id},
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"rollback skill failed: {response.status_code} {response.text}")
        return dict(response.json())

    def preview_skill_search(
        self,
        *,
        query: str,
        agent_scope: str = "",
        top_k: int = 4,
        tenant_id: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        response = self._client.post(
            "/v1/skills/preview",
            headers=self._scope_headers(tenant_id=tenant_id, user_id=user_id),
            json={"query": query, "agent_scope": agent_scope, "top_k": top_k},
        )
        if response.status_code >= 400:
            raise GatewayClientError(f"preview skill failed: {response.status_code} {response.text}")
        return dict(response.json())

    def _chat_headers(
        self,
        *,
        conversation_id: str,
        request_id: str = "",
        tenant_id: str = "",
        user_id: str = "",
        collection_id: str = "",
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "X-Conversation-ID": conversation_id,
        }
        if request_id:
            headers["X-Request-ID"] = request_id
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
        if user_id:
            headers["X-User-ID"] = user_id
        if collection_id:
            headers["X-Collection-ID"] = collection_id
        return headers

    @staticmethod
    def _scope_headers(*, tenant_id: str = "", user_id: str = "") -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
        if user_id:
            headers["X-User-ID"] = user_id
        return headers

    def _parse_sse_events(self, lines: Iterable[str]) -> Iterator[GatewayStreamEvent]:
        current_event = ""
        data_lines: List[str] = []
        for raw_line in lines:
            line = str(raw_line or "")
            if not line:
                if not data_lines:
                    current_event = ""
                    continue
                payload_text = "\n".join(data_lines)
                event_name = current_event or "message"
                if payload_text == "[DONE]":
                    yield GatewayStreamEvent(kind="done", event_name=event_name, payload="[DONE]")
                else:
                    yield self._decode_sse_payload(event_name, payload_text)
                current_event = ""
                data_lines = []
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
        if data_lines:
            payload_text = "\n".join(data_lines)
            if payload_text == "[DONE]":
                yield GatewayStreamEvent(kind="done", event_name=current_event or "message", payload="[DONE]")
            else:
                yield self._decode_sse_payload(current_event or "message", payload_text)

    @staticmethod
    def _decode_sse_payload(event_name: str, payload_text: str) -> GatewayStreamEvent:
        try:
            payload: Any = json.loads(payload_text)
        except Exception:
            payload = payload_text
        if event_name == "progress":
            return GatewayStreamEvent(kind="progress", event_name=event_name, payload=payload)
        if event_name == "artifacts":
            return GatewayStreamEvent(kind="artifacts", event_name=event_name, payload=payload)
        if event_name == "metadata":
            return GatewayStreamEvent(kind="metadata", event_name=event_name, payload=payload)
        if isinstance(payload, dict):
            choices = payload.get("choices") or []
            delta = {}
            if choices and isinstance(choices[0], dict):
                delta = dict(choices[0].get("delta") or {})
            text_delta = str(delta.get("content") or "")
            return GatewayStreamEvent(kind="content", event_name=event_name, payload=payload, text_delta=text_delta)
        return GatewayStreamEvent(kind="raw", event_name=event_name, payload=payload)
