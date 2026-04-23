from __future__ import annotations

from typing import Any, List

from langchain_core.tools import tool

from agentic_chatbot_next.memory.scope import MemoryScope


def build_memory_tools(ctx: Any) -> List[Any]:
    if not bool(getattr(getattr(ctx, "settings", None), "memory_enabled", True)):
        return []
    active_store = getattr(ctx, "memory_store", None)
    if active_store is None and ctx.file_memory_store is None:
        return []

    @tool
    def memory_save(key: str, value: str, scope: str = "conversation") -> str:
        """Save a fact to file-backed memory.

        Args:
            key: Short identifier for the memory entry.
            value: The value to remember.
            scope: conversation or user.
        """

        scope_value = MemoryScope(scope).value
        if active_store is not None:
            active_store.save_explicit(
                tenant_id=ctx.session.tenant_id,
                user_id=ctx.session.user_id,
                conversation_id=ctx.session.conversation_id,
                session_id=ctx.session.session_id,
                scope=scope_value,
                key=key,
                value=value,
                source="tool",
                evidence_turn_ids=[
                    str(message.message_id)
                    for message in list(ctx.session.messages or [])[-2:]
                    if str(getattr(message, "message_id", "") or "")
                ],
            )
            projector = getattr(getattr(ctx, "kernel", None), "memory_projector", None)
            if projector is not None:
                projector.project_session(
                    tenant_id=ctx.session.tenant_id,
                    user_id=ctx.session.user_id,
                    conversation_id=ctx.session.conversation_id,
                    session_id=ctx.session.session_id,
                )
        else:
            ctx.file_memory_store.save(
                tenant_id=ctx.session.tenant_id,
                user_id=ctx.session.user_id,
                conversation_id=ctx.session.conversation_id,
                scope=scope_value,
                key=key,
                value=value,
            )
        return f"Saved memory in {scope_value} scope: {key!r} = {value!r}"

    @tool
    def memory_load(key: str, scope: str = "conversation") -> str:
        """Load a fact from file-backed memory by key."""

        scope_value = MemoryScope(scope).value
        if active_store is not None:
            value = active_store.load_value(
                tenant_id=ctx.session.tenant_id,
                user_id=ctx.session.user_id,
                conversation_id=ctx.session.conversation_id,
                session_id=ctx.session.session_id,
                scope=scope_value,
                key=key,
            )
        else:
            value = ctx.file_memory_store.get(
                tenant_id=ctx.session.tenant_id,
                user_id=ctx.session.user_id,
                conversation_id=ctx.session.conversation_id,
                scope=scope_value,
                key=key,
            )
        if value is None:
            return f"No memory found for key {key!r} in {scope_value} scope."
        return value

    @tool
    def memory_list(scope: str = "conversation") -> str:
        """List keys in file-backed memory for the selected scope."""

        scope_value = MemoryScope(scope).value
        if active_store is not None:
            keys = active_store.list_keys(
                tenant_id=ctx.session.tenant_id,
                user_id=ctx.session.user_id,
                conversation_id=ctx.session.conversation_id,
                session_id=ctx.session.session_id,
                scope=scope_value,
            )
        else:
            keys = ctx.file_memory_store.list_keys(
                tenant_id=ctx.session.tenant_id,
                user_id=ctx.session.user_id,
                conversation_id=ctx.session.conversation_id,
                scope=scope_value,
            )
        if not keys:
            return f"No memory keys saved for {scope_value} scope."
        return ", ".join(keys)

    return [memory_save, memory_load, memory_list]
