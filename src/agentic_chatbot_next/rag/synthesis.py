from __future__ import annotations

import re
from typing import Any, Dict, Sequence

from langchain_core.documents import Document

from agentic_chatbot_next.prompting import load_grounded_answer_prompt, render_template
from agentic_chatbot_next.rag.citations import render_citation_location
from agentic_chatbot_next.runtime.clarification import (
    append_clarification_policy_context,
    clarification_policy_bucket,
)
from agentic_chatbot_next.utils.json_utils import coerce_float, extract_json


def _answer_claims_missing_evidence(answer: str, warnings: Sequence[str]) -> bool:
    haystack = " ".join([answer, *warnings]).lower()
    phrases = (
        "no evidence",
        "no evidence available",
        "insufficient evidence",
        "no supporting evidence",
        "cannot provide",
    )
    return any(phrase in haystack for phrase in phrases)


def _best_summary_snippet(text: str) -> str:
    candidates = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" -*#\t")
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("graph ") or lower in {"```", "mermaid"}:
            continue
        if len(line) < 32:
            continue
        candidates.append(line)
    if candidates:
        return candidates[0][:220]

    normalized = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= 32:
            return sentence[:220]
    return normalized[:220]


def _title_overlap_score(question: str, doc: Document) -> int:
    title = str((doc.metadata or {}).get("title") or "").lower()
    q_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", question.lower()))
    t_terms = set(re.findall(r"[A-Za-z0-9_]{3,}", title.replace("_", " ")))
    overlap = len(q_terms & t_terms)
    if "architecture" in q_terms and "architecture" in t_terms:
        overlap += 2
    return overlap


def _normalized_text_fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())[:220]


def _prepare_synthesis_docs(question: str, evidence_docs: Sequence[Document], *, limit: int = 6) -> list[Document]:
    prioritized = sorted(
        list(evidence_docs),
        key=lambda doc: (
            _title_overlap_score(question, doc),
            len(str(getattr(doc, "page_content", "") or "")),
        ),
        reverse=True,
    )
    deduped: list[Document] = []
    seen: set[tuple[str, str]] = set()
    for doc in prioritized:
        metadata = doc.metadata or {}
        source_key = str(metadata.get("source_path") or metadata.get("doc_id") or metadata.get("title") or "").strip().lower()
        fingerprint = _normalized_text_fingerprint(doc.page_content)
        signature = (source_key, fingerprint)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(doc)
        if len(deduped) >= limit:
            break
    return deduped


def _extract_used_citation_ids(obj: Dict[str, Any]) -> list[str]:
    used_citation_ids = [str(item) for item in (obj.get("used_citation_ids") or []) if str(item)]
    if used_citation_ids:
        return used_citation_ids
    citations = obj.get("citations") or []
    extracted: list[str] = []
    for item in citations:
        if isinstance(item, dict):
            citation_id = str(item.get("citation_id") or "").strip()
        else:
            citation_id = str(item or "").strip()
        if citation_id and citation_id not in extracted:
            extracted.append(citation_id)
    return extracted


def _infer_fallback_themes(docs: Sequence[Document]) -> list[str]:
    combined = "\n".join(str(doc.page_content or "") for doc in docs).lower()
    themes: list[str] = []
    checks = (
        ("routing and agent selection", ("router", "routing", "basic", "agent")),
        ("runtime orchestration", ("runtimekernel", "runtimeservice", "queryloop")),
        ("gateway and control-plane surfaces", ("gateway", "/v1/skills", "control plane")),
        ("grounded retrieval and citations", ("rag", "citation", "knowledge base")),
        ("coordinator and worker execution", ("coordinator", "worker", "planner", "finalizer", "verifier", "jobmanager")),
    )
    for label, tokens in checks:
        if any(token in combined for token in tokens):
            themes.append(label)
    return themes[:3]


def _fallback_followups(question: str, source_titles: Sequence[str]) -> list[str]:
    lowered = question.lower()
    if "architecture" in lowered:
        return [
            "Would you like a per-document breakdown of the architecture docs?",
            "Should I compare runtime orchestration and gateway responsibilities next?",
        ]
    if source_titles:
        return [f"Want a deeper comparison of {source_titles[0]} against the other cited docs?"]
    return []


def _question_looks_under_specified(question: str) -> bool:
    lowered = question.lower()
    if re.search(r'["\']', question):
        return False
    if re.search(r"\b(architecture|pricing|security|privacy|release notes|msa|agreement)\b", lowered):
        return False
    generic_tokens = (
        "policy",
        "workflow",
        "process",
        "procedure",
        "approval",
        "timeline",
        "details",
        "requirements",
        "section",
    )
    return any(token in lowered for token in generic_tokens)


def _soft_ambiguity_payload(question: str, docs: Sequence[Document], *, sensitivity: Any) -> Dict[str, Any] | None:
    bucket = clarification_policy_bucket(sensitivity)
    if bucket == "low":
        return None
    titles: list[str] = []
    for doc in docs:
        title = str((doc.metadata or {}).get("title") or "").strip()
        if title and title not in titles:
            titles.append(title)
    if len(titles) < 2:
        return None
    if not _question_looks_under_specified(question):
        return None
    top_overlap = max((_title_overlap_score(question, doc) for doc in docs), default=0)
    if bucket == "balanced" and not (len(titles) >= 3 and top_overlap == 0):
        return None
    if bucket == "high" and top_overlap > 1:
        return None
    return {
        "answer": (
            "I found multiple plausible documents or interpretations for this request. "
            "Which one should I focus on first?"
        ),
        "used_citation_ids": [],
        "followups": titles[:3],
        "warnings": ["SOFT_QUERY_AMBIGUITY"],
        "confidence_hint": 0.0,
    }


def _fallback_grounded_answer(question: str, evidence_docs: Sequence[Document], *, warning: str) -> Dict[str, Any]:
    docs = _prepare_synthesis_docs(question, evidence_docs, limit=6)
    prioritized = sorted(docs, key=lambda doc: _title_overlap_score(question, doc), reverse=True)
    if prioritized and _title_overlap_score(question, prioritized[0]) > 0:
        docs = [doc for doc in prioritized if _title_overlap_score(question, doc) > 0][:4]
    else:
        docs = prioritized[:4]
    if not docs:
        return {
            "answer": (
                "I couldn't confidently answer from the retrieved evidence. "
                "Can you clarify what section or keyword you want me to focus on?"
            ),
            "used_citation_ids": [],
            "followups": ["Can you specify what part of the documents you mean?"],
            "warnings": [warning],
            "confidence_hint": 0.2,
        }

    answer_lines: list[str] = []
    used_citation_ids: list[str] = []
    source_titles: list[str] = []
    themes = _infer_fallback_themes(docs)
    if themes:
        answer_lines.append(
            "Based on the retrieved repo evidence, the main implementation details cluster around "
            + ", ".join(themes[:-1] + [f"and {themes[-1]}"] if len(themes) > 1 else themes)
            + "."
        )
    else:
        answer_lines.append("Based on the retrieved evidence, these are the key implementation details I could verify directly from the indexed docs.")
    answer_lines.append("")
    answer_lines.append("Key grounded points:")
    for doc in docs:
        metadata = doc.metadata or {}
        citation_id = str(metadata.get("chunk_id") or "")
        title = str(metadata.get("title") or "").strip()
        snippet = _best_summary_snippet(doc.page_content)
        if not snippet:
            continue
        suffix = f" ({citation_id})" if citation_id else ""
        prefix = f"- {title}: " if title else "- "
        answer_lines.append(f"{prefix}{snippet}{suffix}")
        if citation_id:
            used_citation_ids.append(citation_id)
        if title and title not in source_titles:
            source_titles.append(title)

    if len(answer_lines) <= 2:
        answer_lines.append("- The retrieved evidence did not contain enough descriptive text to summarize cleanly.")
    elif source_titles:
        answer_lines.append("")
        answer_lines.append("Sources: " + ", ".join(source_titles))

    return {
        "answer": "\n".join(answer_lines),
        "used_citation_ids": used_citation_ids,
        "followups": _fallback_followups(question, source_titles),
        "warnings": [warning],
        "confidence_hint": 0.45 if used_citation_ids else 0.25,
    }


def generate_grounded_answer(
    llm: Any,
    *,
    settings: Any,
    question: str,
    conversation_context: str,
    evidence_docs: Sequence[Document],
    max_evidence: int = 8,
    callbacks=None,
) -> Dict[str, Any]:
    docs = _prepare_synthesis_docs(question, evidence_docs, limit=max_evidence)
    soft_ambiguity = _soft_ambiguity_payload(
        question,
        docs,
        sensitivity=getattr(settings, "clarification_sensitivity", 50),
    )
    if soft_ambiguity is not None:
        return soft_ambiguity
    evidence_pack = []
    for doc in docs:
        metadata = doc.metadata or {}
        evidence_pack.append(
            {
                "citation_id": metadata.get("chunk_id"),
                "title": metadata.get("title"),
                "location": render_citation_location(metadata),
                "text": doc.page_content[:900],
            }
        )
    effective_context = append_clarification_policy_context(
        conversation_context,
        sensitivity=getattr(settings, "clarification_sensitivity", 50),
    )
    prompt = render_template(
        load_grounded_answer_prompt(settings),
        {
            "QUESTION": question,
            "CONVERSATION_CONTEXT": effective_context,
            "EVIDENCE_JSON": evidence_pack,
        },
    )
    callbacks = callbacks or []
    try:
        response = llm.invoke(prompt, config={"callbacks": callbacks})
        text = getattr(response, "content", None) or str(response)
        obj = extract_json(text)
        if obj and isinstance(obj.get("answer"), str):
            payload = {
                "answer": obj.get("answer", "").strip(),
                "used_citation_ids": _extract_used_citation_ids(obj),
                "followups": [str(item) for item in (obj.get("followups") or []) if str(item)],
                "warnings": [str(item) for item in (obj.get("warnings") or []) if str(item)],
                "confidence_hint": coerce_float(obj.get("confidence_hint"), default=0.5),
            }
            if docs and _answer_claims_missing_evidence(payload["answer"], payload["warnings"]):
                return _fallback_grounded_answer(question, docs, warning="LLM_NO_EVIDENCE_OVERRIDE")
            return payload
    except Exception:
        pass

    return _fallback_grounded_answer(question, docs, warning="LLM_JSON_PARSE_FAILED")
