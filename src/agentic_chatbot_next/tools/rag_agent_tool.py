from __future__ import annotations

from typing import Any, Callable, Dict, List

from langchain.tools import tool

from agentic_chatbot_next.rag.engine import run_rag_contract


def _parse_csv(raw: str) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def make_rag_agent_tool(
    settings: object,
    stores: object,
    *,
    providers: Any,
    session: Any,
    event_sink: Any | None = None,
) -> Callable:
    default_top_k_vector = max(1, int(getattr(settings, "rag_top_k_vector", 15)))
    default_top_k_keyword = max(1, int(getattr(settings, "rag_top_k_keyword", 15)))

    @tool
    def rag_agent_tool(
        query: str,
        conversation_context: str = "",
        preferred_doc_ids_csv: str = "",
        must_include_uploads: bool = True,
        top_k_vector: int = default_top_k_vector,
        top_k_keyword: int = default_top_k_keyword,
        max_retries: int = 2,
        search_mode: str = "auto",
        max_search_rounds: int = 0,
        scratchpad_context_key: str = "",
    ) -> Dict[str, Any]:
        """Answer grounded questions with staged retrieval across KB and uploaded docs.

        Start with a fast evidence probe, then allow the runtime to escalate into deeper
        keyword, vector/hybrid, and document-read retrieval when the first evidence wave
        is not sufficient.
        """

        preferred_doc_ids = _parse_csv(preferred_doc_ids_csv)
        effective_top_k_vector = max(1, int(top_k_vector or default_top_k_vector))
        effective_top_k_keyword = max(1, int(top_k_keyword or default_top_k_keyword))
        if scratchpad_context_key and scratchpad_context_key in getattr(session, "scratchpad", {}):
            extra = session.scratchpad[scratchpad_context_key]
            conversation_context = f"{extra}\n\n{conversation_context}".strip()

        callbacks: List[Any] = []
        try:
            from langchain_core.runnables.config import get_config

            cfg = get_config() or {}
            callbacks = cfg.get("callbacks") or []
        except Exception:
            callbacks = []

        contract = run_rag_contract(
            settings,
            stores,
            providers=providers,
            session=session,
            query=query,
            conversation_context=conversation_context,
            preferred_doc_ids=preferred_doc_ids,
            must_include_uploads=must_include_uploads,
            top_k_vector=effective_top_k_vector,
            top_k_keyword=effective_top_k_keyword,
            max_retries=max_retries,
            callbacks=callbacks,
            search_mode=search_mode,
            max_search_rounds=max_search_rounds,
            event_sink=event_sink,
        )
        return contract.to_dict()

    return rag_agent_tool
