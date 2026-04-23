from __future__ import annotations

from typing import Any, List

from agentic_chatbot_next.tools.calculator import calculator
from agentic_chatbot_next.tools.list_docs import make_list_docs_tool
from agentic_chatbot_next.tools.skills_search_tool import make_skills_search_tool


def build_utility_tools(ctx: Any) -> List[Any]:
    tools: List[Any] = [
        calculator,
        make_list_docs_tool(ctx.settings, ctx.stores, ctx.session_handle),
    ]
    try:
        tools.append(
            make_skills_search_tool(
                ctx.settings,
                stores=ctx.stores,
                session=ctx.session_handle,
            )
        )
    except Exception:
        pass
    return tools
