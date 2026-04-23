"""Extended next-owned RAG specialist tools."""
from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from langchain_core.tools import tool

from agentic_chatbot_next.config import Settings
from agentic_chatbot_next.prompting import load_judge_rewrite_prompt, render_template
from agentic_chatbot_next.rag.stores import KnowledgeStores
from agentic_chatbot_next.utils.json_utils import extract_json

logger = logging.getLogger(__name__)


def make_extended_rag_tools(
    stores: KnowledgeStores,
    session: Any,
    *,
    judge_llm: Any,
    settings: Optional[Settings] = None,
) -> List[Any]:
    tenant_id = getattr(session, "tenant_id", getattr(settings, "default_tenant_id", "local-dev"))
    top_k_vector = max(1, int(getattr(settings, "rag_top_k_vector", 8)))
    web_search_enabled = bool(getattr(settings, "web_search_enabled", False))
    tavily_api_key = getattr(settings, "tavily_api_key", None)

    @tool
    def query_rewriter(query: str, reason: str = "") -> str:
        """Rewrite an ambiguous query to improve retrieval quality."""
        try:
            template = load_judge_rewrite_prompt(settings) if settings is not None else ""
            prompt = render_template(
                template,
                {
                    "ATTEMPT": 1,
                    "QUESTION": query,
                    "CONVERSATION_CONTEXT": reason or "",
                },
            )
            response = judge_llm.invoke(prompt)
            text = getattr(response, "content", None) or str(response)
            obj = extract_json(text) or {}
            rewritten = str(obj.get("rewritten_query") or query).strip()
            changed = rewritten != query.strip()
            return json.dumps({"original": query, "rewritten": rewritten, "changed": changed})
        except Exception as exc:
            logger.warning("query_rewriter failed: %s", exc)
            return json.dumps({"original": query, "rewritten": query, "changed": False, "error": str(exc)})

    @tool
    def chunk_expander(chunk_id: str, window: int = 2) -> str:
        """Fetch a chunk plus neighboring chunks."""
        try:
            chunk = stores.chunk_store.get_chunk_by_id(chunk_id, tenant_id)
            if chunk is None:
                return json.dumps({"error": f"chunk_id {chunk_id!r} not found", "chunks": []})
            neighbours = stores.chunk_store.get_chunks_by_index_range(
                chunk.doc_id,
                max(0, chunk.chunk_index - window),
                chunk.chunk_index + window,
                tenant_id,
            )
            payload = [
                {
                    "chunk_id": item.chunk_id,
                    "chunk_index": item.chunk_index,
                    "clause_number": item.clause_number,
                    "section_title": item.section_title,
                    "content": item.content,
                }
                for item in neighbours
            ]
            return json.dumps({"chunks": payload, "count": len(payload)})
        except Exception as exc:
            logger.warning("chunk_expander failed: %s", exc)
            return json.dumps({"error": str(exc), "chunks": []})

    @tool
    def document_summarizer(doc_id: str, focus: str = "") -> str:
        """Produce a dense document summary from representative chunks."""
        try:
            search_query = focus if focus.strip() else "document overview key topics summary"
            search_top_k = min(top_k_vector * 2, 20)
            chunks = stores.chunk_store.vector_search(
                search_query,
                top_k=search_top_k,
                doc_id_filter=doc_id,
                tenant_id=tenant_id,
            )
            if not chunks:
                return json.dumps({"error": f"No chunks found for doc_id {doc_id!r}", "summary": "", "chunk_ids_used": []})
            chunks_sorted = sorted(chunks, key=lambda item: item.doc.metadata.get("chunk_index", 0))
            chunk_ids_used = [item.doc.metadata.get("chunk_id", "") for item in chunks_sorted]
            excerpts = "\n\n---\n\n".join(
                f"[{item.doc.metadata.get('chunk_id', '')}]\n{item.doc.page_content}"
                for item in chunks_sorted
            )
            prompt = (
                "Summarise the following document excerpts in approximately 300 words. "
                f"{'Focus: ' + focus + '. ' if focus.strip() else ''}"
                "Be concise, factual, and preserve key terminology.\n\n"
                f"Document ID: {doc_id}\n\nExcerpts:\n{excerpts}"
            )
            response = judge_llm.invoke(prompt)
            summary = getattr(response, "content", None) or str(response)
            return json.dumps({"summary": summary.strip(), "doc_id": doc_id, "chunk_ids_used": chunk_ids_used})
        except Exception as exc:
            logger.warning("document_summarizer failed: %s", exc)
            return json.dumps({"error": str(exc), "summary": "", "chunk_ids_used": []})

    @tool
    def citation_validator(claim: str, chunk_id: str) -> str:
        """Check whether a specific chunk supports a claim."""
        try:
            chunk = stores.chunk_store.get_chunk_by_id(chunk_id, tenant_id)
            if chunk is None:
                return json.dumps({"supported": False, "confidence": 0.0, "reason": f"Chunk {chunk_id!r} not found.", "chunk_id": chunk_id})
            prompt = (
                "You are a fact-checking assistant. Given the following text excerpt and a claim, "
                "determine whether the excerpt supports the claim.\n\n"
                f"Excerpt ({chunk_id}):\n{chunk.content}\n\n"
                f"Claim: {claim}\n\n"
                'Return ONLY valid JSON in this exact schema:\n'
                '{"supported": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}'
            )
            response = judge_llm.invoke(prompt)
            text = getattr(response, "content", None) or str(response)
            obj = extract_json(text) or {}
            return json.dumps(
                {
                    "supported": bool(obj.get("supported", False)),
                    "confidence": max(0.0, min(1.0, float(obj.get("confidence", 0.5)))),
                    "reason": str(obj.get("reason", "Could not parse LLM response")),
                    "chunk_id": chunk_id,
                }
            )
        except Exception as exc:
            logger.warning("citation_validator failed: %s", exc)
            return json.dumps({"supported": False, "confidence": 0.0, "reason": str(exc), "chunk_id": chunk_id})

    @tool
    def web_search_fallback(query: str, max_results: int = 5) -> str:
        """Search the public web when KB evidence is insufficient."""
        if not web_search_enabled:
            return json.dumps({"error": "Web search is disabled. Set WEB_SEARCH_ENABLED=true to enable.", "results": [], "source": "web"})
        if not tavily_api_key:
            return json.dumps({"error": "Web search is not configured. TAVILY_API_KEY is missing.", "results": [], "source": "web"})
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore[import]

            capped = max(1, min(int(max_results), 10))
            search = TavilySearchResults(max_results=capped, tavily_api_key=tavily_api_key)
            raw = search.invoke(query)
            results = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        results.append(
                            {
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "snippet": item.get("content", "")[:400],
                            }
                        )
            return json.dumps({"results": results, "source": "web", "count": len(results)})
        except ImportError:
            return json.dumps({"error": "langchain-community[tavily] is not installed.", "results": [], "source": "web"})
        except Exception as exc:
            logger.warning("web_search_fallback failed: %s", exc)
            return json.dumps({"error": str(exc), "results": [], "source": "web"})

    return [
        query_rewriter,
        chunk_expander,
        document_summarizer,
        citation_validator,
        web_search_fallback,
    ]
