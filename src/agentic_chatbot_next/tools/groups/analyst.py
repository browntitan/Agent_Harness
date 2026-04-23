from __future__ import annotations

from typing import Any, List

from agentic_chatbot_next.tools.data_analyst_tools import make_data_analyst_tools


def build_analyst_tools(ctx: Any) -> List[Any]:
    return make_data_analyst_tools(
        ctx.stores,
        ctx.session_handle,
        settings=ctx.settings,
        providers=ctx.providers,
    )
