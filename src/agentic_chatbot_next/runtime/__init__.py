"""Runtime kernel and path helpers for the next runtime."""

from agentic_chatbot_next.runtime.context import RuntimePaths, filesystem_key

__all__ = ["RuntimeKernel", "RuntimePaths", "filesystem_key"]


def __getattr__(name: str):
    if name == "RuntimeKernel":
        from agentic_chatbot_next.runtime.kernel import RuntimeKernel

        return RuntimeKernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
