"""Next-generation runtime package."""

__all__ = [
    "ChatSession",
    "RequestContext",
    "RuntimeKernel",
    "RuntimePaths",
    "RuntimeService",
    "build_local_context",
    "filesystem_key",
]


def __getattr__(name: str):
    if name == "RuntimeService":
        from agentic_chatbot_next.app.service import RuntimeService

        return RuntimeService
    if name == "RuntimeKernel":
        from agentic_chatbot_next.runtime.kernel import RuntimeKernel

        return RuntimeKernel
    if name in {"RequestContext", "build_local_context"}:
        from agentic_chatbot_next.context import RequestContext, build_local_context

        return {"RequestContext": RequestContext, "build_local_context": build_local_context}[name]
    if name in {"RuntimePaths", "filesystem_key"}:
        from agentic_chatbot_next.runtime.context import RuntimePaths, filesystem_key

        return {"RuntimePaths": RuntimePaths, "filesystem_key": filesystem_key}[name]
    if name == "ChatSession":
        from agentic_chatbot_next.session import ChatSession

        return ChatSession
    raise AttributeError(name)
