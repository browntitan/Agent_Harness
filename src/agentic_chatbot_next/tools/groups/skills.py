from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.tools import tool


def build_skill_execution_tools(ctx: Any) -> List[Any]:
    if ctx.kernel is None or ctx.active_definition is None:
        return []
    if not bool(getattr(ctx.settings, "executable_skills_enabled", False)):
        return []

    @tool
    def execute_skill(skill_id: str, input: str = "", arguments: Dict[str, Any] | None = None) -> str:
        """Execute an active executable or hybrid skill by id."""

        return json.dumps(
            ctx.kernel.execute_skill_from_tool(
                ctx,
                skill_id=skill_id,
                input_text=input,
                arguments=dict(arguments or {}),
            ),
            ensure_ascii=False,
        )

    return [execute_skill]
