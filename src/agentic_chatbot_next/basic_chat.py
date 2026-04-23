from __future__ import annotations

from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_chatbot_next.prompt_fallbacks import compose_fallback_prompt

_DEFAULT_BASIC_SYSTEM_PROMPT = compose_fallback_prompt("basic_chat.md")


def run_basic_chat(
    chat_llm: Any,
    *,
    messages: List[Any],
    user_text: str,
    system_prompt: str = "",
    callbacks=None,
) -> str:
    callbacks = callbacks or []
    effective_prompt = system_prompt or _DEFAULT_BASIC_SYSTEM_PROMPT
    if not messages or not isinstance(messages[0], SystemMessage):
        msgs = [SystemMessage(content=effective_prompt)] + list(messages)
    else:
        msgs = list(messages)
    msgs.append(HumanMessage(content=user_text))
    response = chat_llm.invoke(msgs, config={"callbacks": callbacks})
    return getattr(response, "content", None) or str(response)
