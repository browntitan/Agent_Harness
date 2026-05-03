from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_chatbot_next.rag.reranker import rerank_items
from agentic_chatbot_next.utils.json_utils import extract_json, make_json_compatible


_TEXT_KEYS = {
    "answer",
    "content",
    "description",
    "excerpt",
    "final_answer",
    "matched_text",
    "relationship",
    "response",
    "source_excerpt",
    "summary",
    "summary_text",
    "text",
}
_TITLE_KEYS = {"title", "filename", "file_name", "source_path", "path", "label"}
_DOC_KEYS = {"doc_id", "document_id", "source_doc_id", "source_document_id"}
_CITATION_KEYS = {"citation_id", "citation_ids", "chunk_id", "chunk_ids"}
_STRUCTURAL_GRAPH_KEYS = {
    "edge",
    "edges",
    "entities",
    "entity",
    "node",
    "nodes",
    "path",
    "paths",
    "relationship_path",
    "relationships",
}
_STOPWORDS = {
    "about",
    "after",
    "again",
    "answer",
    "before",
    "between",
    "briefly",
    "cited",
    "cite",
    "could",
    "documents",
    "evidence",
    "from",
    "graph",
    "have",
    "knowledge",
    "please",
    "query",
    "should",
    "source",
    "sources",
    "that",
    "their",
    "then",
    "there",
    "this",
    "tool",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def _estimate_tokens(text: str) -> int:
    clean = str(text or "")
    return 0 if not clean else max(1, math.ceil(len(clean) / 4))


def _clip_text(text: str, max_chars: int) -> str:
    clean = str(text or "")
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    if max_chars < 120:
        return clean[:max_chars].rstrip()
    head = max_chars // 2
    tail = max_chars - head - 34
    return f"{clean[:head].rstrip()}\n...[truncated]...\n{clean[-tail:].lstrip()}"


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*", str(text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _label_tokens(text: str) -> set[str]:
    raw = str(text or "").lower()
    labels = set(re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", raw))
    labels.update(match.group(1).strip().lower() for match in re.finditer(r"\b(?:field|column|label|key)\s+[`'\"]?([^`'\"?,.;:]+)", raw))
    return {label for label in labels if label}


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass
class EvidenceAtom:
    atom_id: str
    tool_name: str
    kind: str
    text: str
    title: str = ""
    doc_id: str = ""
    citation_id: str = ""
    source: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.atom_id,
            "kind": self.kind,
            "text": _clip_text(_compact_whitespace(self.text), 1400),
        }
        if self.title:
            payload["title"] = self.title
        if self.doc_id:
            payload["doc_id"] = self.doc_id
        if self.citation_id:
            payload["citation_id"] = self.citation_id
        if self.source:
            payload["source"] = self.source
        if self.score:
            payload["score"] = round(self.score, 4)
        keep_meta = {
            key: value
            for key, value in self.metadata.items()
            if key
            in {
                "backend",
                "collection_id",
                "evidence_status",
                "graph_id",
                "method",
                "requires_source_read",
                "score",
                "source_type",
            }
        }
        if keep_meta:
            payload["metadata"] = keep_meta
        return payload


@dataclass
class ContextCompactionResult:
    tool_name: str
    original_tokens: int
    compacted_tokens: int
    selected_atoms: List[EvidenceAtom]
    dropped_atoms: List[EvidenceAtom]
    source_resolution_status: str
    source_resolution_plan: dict[str, Any]
    method_trace: List[dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    full_result_ref: str = ""

    def context_metadata(self) -> dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "compacted_tokens": self.compacted_tokens,
            "method": self.method,
            "selected_evidence_count": len(self.selected_atoms),
            "dropped_evidence_count": len(self.dropped_atoms),
            "full_result_ref": self.full_result_ref,
            "source_resolution_status": self.source_resolution_status,
            "warnings": list(self.warnings),
        }

    @property
    def method(self) -> str:
        for item in reversed(self.method_trace):
            method = str(item.get("method") or "").strip()
            if method:
                return method
        return "semantic_mmr"

    def to_budgeted_payload(self, *, budget_tokens: int, microcompact: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "object": "budgeted_tool_result",
            "tool_name": self.tool_name,
            "budgeted": True,
            "microcompact": bool(microcompact),
            "original_tokens": self.original_tokens,
            "budget_tokens": min(self.original_tokens, budget_tokens),
            "context_compaction": self.context_metadata(),
            "evidence_ledger": [atom.to_prompt_dict() for atom in self.selected_atoms],
            "dropped_evidence_summary": {
                "count": len(self.dropped_atoms),
                "doc_ids": _dedupe_strings(atom.doc_id for atom in self.dropped_atoms if atom.doc_id)[:12],
                "kinds": _dedupe_strings(atom.kind for atom in self.dropped_atoms if atom.kind)[:12],
            },
            "source_resolution_plan": self.source_resolution_plan,
            "warnings": [
                "Tool output was compacted for model context. Use full_result_ref to reopen complete details."
            ]
            + list(self.warnings),
        }
        if self.full_result_ref:
            payload["full_result_ref"] = self.full_result_ref
        return make_json_compatible(payload)

    def render_for_prompt(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "context_compaction": self.context_metadata(),
            "evidence_ledger": [atom.to_prompt_dict() for atom in self.selected_atoms],
            "source_resolution_plan": self.source_resolution_plan,
        }


class ContextCompactionService:
    """Corpus-neutral tool-result compaction for graph/RAG/final synthesis contexts."""

    def __init__(
        self,
        settings: Any | None = None,
        *,
        estimate_tokens: Callable[[str], int] | None = None,
    ) -> None:
        self.settings = settings
        self.estimate_tokens = estimate_tokens or _estimate_tokens

    def compact_tool_content(
        self,
        *,
        query: str = "",
        tool_name: str,
        content: str,
        target_tokens: int,
        full_result_ref: str = "",
        llm: Any | None = None,
        enable_llm: bool | None = None,
    ) -> ContextCompactionResult:
        original = str(content or "")
        parsed = extract_json(original)
        original_tokens = self.estimate_tokens(original)
        atoms = self._extract_atoms(tool_name=tool_name, parsed=parsed, original=original)
        method_trace: List[dict[str, Any]] = [
            {
                "method": "atom_extraction",
                "tool_name": tool_name,
                "atom_count": len(atoms),
            }
        ]
        if not atoms:
            atoms = [
                EvidenceAtom(
                    atom_id=f"{tool_name or 'tool'}:preview:0",
                    tool_name=tool_name,
                    kind="tool_preview",
                    text=_clip_text(original, max(800, target_tokens * 3)),
                )
            ]
            method_trace.append({"method": "fallback_preview", "atom_count": 1})

        scored = self._score_atoms(query=query, atoms=atoms)
        reranked, rerank_meta = self._rerank_atoms(query=query, atoms=scored)
        method_trace.append({"method": "semantic_rerank", **rerank_meta})
        selected, dropped = self._select_atoms(query=query, atoms=reranked, target_tokens=target_tokens)
        method_trace.append(
            {
                "method": "semantic_mmr",
                "selected_count": len(selected),
                "dropped_count": len(dropped),
                "target_tokens": target_tokens,
            }
        )

        if self._llm_enabled(enable_llm=enable_llm, llm=llm) and len(selected) > 1:
            selected, dropped, llm_meta = self._llm_pack(
                query=query,
                atoms=selected,
                dropped=dropped,
                target_tokens=target_tokens,
                llm=llm,
            )
            method_trace.append(llm_meta)

        source_resolution_status = self._source_resolution_status(tool_name=tool_name, parsed=parsed, atoms=selected)
        source_resolution_plan = self._source_resolution_plan(
            tool_name=tool_name,
            parsed=parsed,
            atoms=selected,
            status=source_resolution_status,
        )
        compacted_payload = {
            "evidence_ledger": [atom.to_prompt_dict() for atom in selected],
            "source_resolution_plan": source_resolution_plan,
        }
        compacted_tokens = self.estimate_tokens(json.dumps(compacted_payload, ensure_ascii=False))
        warnings: List[str] = []
        if source_resolution_status == "required":
            warnings.append("Graph results are source leads and should be resolved through RAG/document evidence before final factual synthesis.")
        if compacted_tokens > target_tokens:
            warnings.append("Compacted evidence still exceeds the target token budget.")
        return ContextCompactionResult(
            tool_name=tool_name,
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            selected_atoms=selected,
            dropped_atoms=dropped,
            source_resolution_status=source_resolution_status,
            source_resolution_plan=source_resolution_plan,
            method_trace=method_trace,
            warnings=warnings,
            full_result_ref=full_result_ref,
        )

    def compact_tool_results_for_prompt(
        self,
        *,
        query: str,
        tool_results: Sequence[dict[str, Any]],
        target_tokens: int,
        llm: Any | None = None,
        enable_llm: bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        results = list(tool_results or [])
        if not results:
            return "[]", {}
        per_tool = max(256, target_tokens // max(1, len(results)))
        compacted: List[ContextCompactionResult] = []
        rendered_items: List[dict[str, Any]] = []
        for index, result in enumerate(results):
            tool_name = str(result.get("tool") or result.get("name") or "")
            raw_output = result.get("output")
            if not isinstance(raw_output, str):
                raw_output = json.dumps(raw_output, ensure_ascii=False)
            compacted_result = self.compact_tool_content(
                query=query,
                tool_name=tool_name,
                content=str(raw_output or ""),
                target_tokens=per_tool,
                full_result_ref=str(result.get("full_result_ref") or ""),
                llm=llm,
                enable_llm=enable_llm,
            )
            compacted.append(compacted_result)
            rendered_items.append(
                {
                    "index": index,
                    "tool": tool_name,
                    "args": make_json_compatible(result.get("args") or {}),
                    **compacted_result.render_for_prompt(),
                }
            )

        payload = {
            "object": "compacted_tool_results",
            "query": query,
            "items": rendered_items,
        }
        text = json.dumps(make_json_compatible(payload), ensure_ascii=False)
        if self.estimate_tokens(text) > target_tokens:
            payload["items"] = self._trim_rendered_items(rendered_items, target_tokens=target_tokens)
            text = json.dumps(make_json_compatible(payload), ensure_ascii=False)
        metadata = self._aggregate_metadata(compacted, rendered_text=text)
        return text, metadata

    def _extract_atoms(self, *, tool_name: str, parsed: Any, original: str) -> List[EvidenceAtom]:
        if isinstance(parsed, dict):
            if tool_name == "search_graph_index" or "graph_id" in parsed or "evidence_status" in parsed:
                return self._extract_graph_atoms(tool_name=tool_name, parsed=parsed)
            if tool_name == "rag_agent_tool" or "answer" in parsed or "citations" in parsed:
                return self._extract_rag_atoms(tool_name=tool_name, parsed=parsed)
            return self._extract_structured_atoms(tool_name=tool_name, parsed=parsed)
        if isinstance(parsed, list):
            return self._extract_list_atoms(tool_name=tool_name, items=parsed)
        return [
            EvidenceAtom(
                atom_id=f"{tool_name or 'tool'}:text:0",
                tool_name=tool_name,
                kind="tool_text",
                text=_clip_text(original, 3000),
            )
        ]

    def _extract_graph_atoms(self, *, tool_name: str, parsed: dict[str, Any]) -> List[EvidenceAtom]:
        atoms: List[EvidenceAtom] = []
        graph_id = str(parsed.get("graph_id") or "")
        evidence_status = str(parsed.get("evidence_status") or "")
        requires_source_read = bool(parsed.get("requires_source_read"))
        summary = str(parsed.get("graph_context_summary") or parsed.get("summary") or "").strip()
        if summary:
            atoms.append(
                EvidenceAtom(
                    atom_id=f"{tool_name}:graph_summary:0",
                    tool_name=tool_name,
                    kind="graph_summary",
                    text=summary,
                    source="graph_context_summary",
                    metadata={
                        "graph_id": graph_id,
                        "evidence_status": evidence_status,
                        "requires_source_read": requires_source_read,
                    },
                )
            )
        for index, item in enumerate(_to_list(parsed.get("results"))):
            if not isinstance(item, dict):
                continue
            atom = self._dict_atom(
                tool_name=tool_name,
                kind="graph_result",
                item=item,
                atom_id=f"{tool_name}:result:{index}",
                parent_metadata={
                    "graph_id": graph_id,
                    "evidence_status": evidence_status,
                    "requires_source_read": requires_source_read,
                },
            )
            if atom is not None:
                atoms.append(atom)
        for index, item in enumerate(_to_list(parsed.get("citations"))):
            if not isinstance(item, dict):
                continue
            atom = self._dict_atom(
                tool_name=tool_name,
                kind="citation",
                item=item,
                atom_id=f"{tool_name}:citation:{index}",
                parent_metadata={"graph_id": graph_id, "evidence_status": evidence_status},
            )
            if atom is not None:
                atoms.append(atom)
        for source_key in ("source_candidates", "source_documents", "sources"):
            for index, item in enumerate(_to_list(parsed.get(source_key))):
                if not isinstance(item, dict):
                    continue
                atom = self._dict_atom(
                    tool_name=tool_name,
                    kind="source_candidate",
                    item=item,
                    atom_id=f"{tool_name}:{source_key}:{index}",
                    parent_metadata={
                        "graph_id": graph_id,
                        "evidence_status": evidence_status,
                        "requires_source_read": True,
                    },
                )
                if atom is not None:
                    atoms.append(atom)
        return atoms

    def _extract_rag_atoms(self, *, tool_name: str, parsed: dict[str, Any]) -> List[EvidenceAtom]:
        atoms: List[EvidenceAtom] = []
        answer = str(parsed.get("answer") or parsed.get("final_answer") or "").strip()
        if answer:
            atoms.append(
                EvidenceAtom(
                    atom_id=f"{tool_name}:answer:0",
                    tool_name=tool_name,
                    kind="rag_answer",
                    text=answer,
                    source="answer",
                    metadata={"source_type": "rag_answer"},
                )
            )
        for key in ("evidence", "chunks", "sources", "citations", "supporting_chunks"):
            for index, item in enumerate(_to_list(parsed.get(key))):
                if not isinstance(item, dict):
                    continue
                atom = self._dict_atom(
                    tool_name=tool_name,
                    kind="rag_evidence" if key != "citations" else "citation",
                    item=item,
                    atom_id=f"{tool_name}:{key}:{index}",
                )
                if atom is not None:
                    atoms.append(atom)
        return atoms or self._extract_structured_atoms(tool_name=tool_name, parsed=parsed)

    def _extract_structured_atoms(self, *, tool_name: str, parsed: dict[str, Any]) -> List[EvidenceAtom]:
        atoms: List[EvidenceAtom] = []
        direct = self._dict_atom(
            tool_name=tool_name,
            kind="tool_summary",
            item=parsed,
            atom_id=f"{tool_name}:summary:0",
        )
        if direct is not None:
            atoms.append(direct)
        for key in ("rows", "records", "items", "results", "data"):
            value = parsed.get(key)
            if not isinstance(value, list):
                continue
            for index, item in enumerate(value[:80]):
                if not isinstance(item, dict):
                    continue
                atom = self._dict_atom(
                    tool_name=tool_name,
                    kind="structured_row",
                    item=item,
                    atom_id=f"{tool_name}:{key}:{index}",
                )
                if atom is not None:
                    atoms.append(atom)
        return atoms

    def _extract_list_atoms(self, *, tool_name: str, items: Sequence[Any]) -> List[EvidenceAtom]:
        atoms: List[EvidenceAtom] = []
        for index, item in enumerate(items[:120]):
            if isinstance(item, dict):
                atom = self._dict_atom(
                    tool_name=tool_name,
                    kind="structured_row",
                    item=item,
                    atom_id=f"{tool_name}:item:{index}",
                )
                if atom is not None:
                    atoms.append(atom)
            else:
                atoms.append(
                    EvidenceAtom(
                        atom_id=f"{tool_name}:item:{index}",
                        tool_name=tool_name,
                        kind="tool_item",
                        text=_clip_text(str(item), 1200),
                    )
                )
        return atoms

    def _dict_atom(
        self,
        *,
        tool_name: str,
        kind: str,
        item: dict[str, Any],
        atom_id: str,
        parent_metadata: dict[str, Any] | None = None,
    ) -> EvidenceAtom | None:
        parts: List[str] = []
        for key, value in item.items():
            key_text = str(key)
            if key_text in _TEXT_KEYS and str(value or "").strip():
                parts.append(f"{key}: {value}")
            elif key_text in _STRUCTURAL_GRAPH_KEYS and value:
                parts.append(f"{key}: {_clip_text(json.dumps(make_json_compatible(value), ensure_ascii=False), 1200)}")
        if not parts:
            simple_items = [
                f"{key}: {value}"
                for key, value in item.items()
                if isinstance(value, (str, int, float, bool)) and str(value or "").strip()
            ]
            parts.extend(simple_items[:16])
        if not parts:
            return None
        title = self._first_value(item, _TITLE_KEYS)
        doc_id = self._first_value(item, _DOC_KEYS)
        citation = self._first_citation(item)
        metadata = dict(parent_metadata or {})
        for key in ("backend", "collection_id", "method", "score", "source_type"):
            if key in item:
                metadata[key] = item.get(key)
        return EvidenceAtom(
            atom_id=atom_id,
            tool_name=tool_name,
            kind=kind,
            text=_clip_text("; ".join(parts), 2400),
            title=title,
            doc_id=doc_id,
            citation_id=citation,
            source=kind,
            metadata=metadata,
        )

    def _first_value(self, item: dict[str, Any], keys: set[str]) -> str:
        for key in keys:
            value = item.get(key)
            if str(value or "").strip():
                return str(value)
        return ""

    def _first_citation(self, item: dict[str, Any]) -> str:
        for key in _CITATION_KEYS:
            value = item.get(key)
            values = _to_list(value)
            for candidate in values:
                if str(candidate or "").strip():
                    return str(candidate)
        return ""

    def _score_atoms(self, *, query: str, atoms: Sequence[EvidenceAtom]) -> List[EvidenceAtom]:
        query_tokens = _tokens(query)
        requested_labels = _label_tokens(query)
        scored: List[EvidenceAtom] = []
        for atom in atoms:
            haystack = " ".join([atom.text, atom.title, atom.doc_id, atom.citation_id])
            atom_tokens = _tokens(haystack)
            overlap = len(query_tokens & atom_tokens)
            union = len(query_tokens | atom_tokens) or 1
            score = overlap / union
            if requested_labels and requested_labels & _label_tokens(haystack):
                score += 2.0
            if atom.kind in {"rag_answer", "tool_summary"}:
                score += 1.2
            elif atom.kind in {"rag_evidence", "structured_row"}:
                score += 0.9
            elif atom.kind == "graph_result":
                score += 0.75
            elif atom.kind == "graph_summary":
                score += 0.5
            elif atom.kind == "citation":
                score += 0.25
            if atom.doc_id:
                score += 0.15
            if atom.citation_id:
                score += 0.15
            scored.append(
                EvidenceAtom(
                    atom_id=atom.atom_id,
                    tool_name=atom.tool_name,
                    kind=atom.kind,
                    text=atom.text,
                    title=atom.title,
                    doc_id=atom.doc_id,
                    citation_id=atom.citation_id,
                    source=atom.source,
                    score=score,
                    metadata=dict(atom.metadata),
                )
            )
        return sorted(scored, key=lambda item: (-item.score, item.atom_id))

    def _rerank_atoms(self, *, query: str, atoms: Sequence[EvidenceAtom]) -> tuple[List[EvidenceAtom], dict[str, Any]]:
        if not atoms:
            return [], {"status": "skipped_empty"}
        settings = self.settings
        if settings is None:
            return list(atoms), {"status": "skipped_no_settings"}
        try:
            reranked, decision = rerank_items(
                settings,
                query=query,
                items=list(atoms),
                id_fn=lambda _index, atom: atom.atom_id,
                text_fn=lambda atom: " ".join([atom.title, atom.doc_id, atom.text]),
            )
            return list(reranked), decision.to_dict()
        except Exception as exc:
            return list(atoms), {"status": "fallback", "error": f"{type(exc).__name__}: {exc}"}

    def _select_atoms(
        self,
        *,
        query: str,
        atoms: Sequence[EvidenceAtom],
        target_tokens: int,
    ) -> tuple[List[EvidenceAtom], List[EvidenceAtom]]:
        del query
        remaining = list(atoms)
        selected: List[EvidenceAtom] = []
        used_doc_counts: dict[str, int] = {}
        used_kind_counts: dict[str, int] = {}
        budget = max(160, target_tokens)
        current_tokens = 0
        while remaining:
            ranked = sorted(
                remaining,
                key=lambda atom: (
                    -(
                        atom.score
                        - (0.18 * used_doc_counts.get(atom.doc_id, 0) if atom.doc_id else 0.0)
                        - (0.08 * used_kind_counts.get(atom.kind, 0))
                    ),
                    atom.atom_id,
                ),
            )
            candidate = ranked[0]
            remaining.remove(candidate)
            candidate_tokens = self.estimate_tokens(json.dumps(candidate.to_prompt_dict(), ensure_ascii=False)) + 12
            if selected and current_tokens + candidate_tokens > budget:
                continue
            selected.append(candidate)
            current_tokens += candidate_tokens
            if candidate.doc_id:
                used_doc_counts[candidate.doc_id] = used_doc_counts.get(candidate.doc_id, 0) + 1
            used_kind_counts[candidate.kind] = used_kind_counts.get(candidate.kind, 0) + 1
            if current_tokens >= budget:
                break
        selected_ids = {atom.atom_id for atom in selected}
        dropped = [atom for atom in atoms if atom.atom_id not in selected_ids]
        return selected, dropped

    def _llm_enabled(self, *, enable_llm: bool | None, llm: Any | None) -> bool:
        if llm is None:
            return False
        if enable_llm is not None:
            return bool(enable_llm)
        return bool(getattr(self.settings, "context_smart_compaction_llm_enabled", False))

    def _llm_pack(
        self,
        *,
        query: str,
        atoms: Sequence[EvidenceAtom],
        dropped: Sequence[EvidenceAtom],
        target_tokens: int,
        llm: Any,
    ) -> tuple[List[EvidenceAtom], List[EvidenceAtom], dict[str, Any]]:
        atom_map = {atom.atom_id: atom for atom in atoms}
        candidates = [atom.to_prompt_dict() for atom in atoms[:40]]
        prompt = (
            "Select the smallest set of evidence items needed to answer the user query. "
            "Preserve direct evidence, citations, conflicts, source-resolution warnings, and structured values. "
            "Do not invent facts. Return JSON only with selected_ids as an array of evidence ids.\n\n"
            f"USER_QUERY:\n{query}\n\n"
            f"TOKEN_BUDGET:{target_tokens}\n\n"
            f"EVIDENCE:\n{json.dumps(candidates, ensure_ascii=False)}"
        )
        try:
            response = llm.invoke(
                [
                    SystemMessage(content="You compact evidence ledgers for answer synthesis."),
                    HumanMessage(content=prompt),
                ]
            )
            text = str(getattr(response, "content", None) or response)
            parsed = extract_json(text)
            selected_ids = parsed.get("selected_ids") if isinstance(parsed, dict) else None
            if not isinstance(selected_ids, list):
                return list(atoms), list(dropped), {"method": "llm_pack_failed", "error": "missing selected_ids"}
            chosen = [atom_map[str(atom_id)] for atom_id in selected_ids if str(atom_id) in atom_map]
            if not chosen:
                return list(atoms), list(dropped), {"method": "llm_pack_failed", "error": "empty selection"}
            chosen_ids = {atom.atom_id for atom in chosen}
            new_dropped = list(dropped) + [atom for atom in atoms if atom.atom_id not in chosen_ids]
            return chosen, new_dropped, {"method": "llm_semantic_pack", "selected_count": len(chosen)}
        except Exception as exc:
            return list(atoms), list(dropped), {"method": "llm_pack_failed", "error": f"{type(exc).__name__}: {exc}"}

    def _source_resolution_status(self, *, tool_name: str, parsed: Any, atoms: Sequence[EvidenceAtom]) -> str:
        if tool_name == "rag_agent_tool" or any(atom.kind in {"rag_answer", "rag_evidence"} for atom in atoms):
            return "resolved"
        if isinstance(parsed, dict):
            evidence_status = str(parsed.get("evidence_status") or "").lower()
            if bool(parsed.get("requires_source_read")) or evidence_status in {"source_candidates_only", "requires_source_read"}:
                return "required"
            if tool_name == "search_graph_index" and not atoms:
                return "insufficient"
        return "not_required"

    def _source_resolution_plan(
        self,
        *,
        tool_name: str,
        parsed: Any,
        atoms: Sequence[EvidenceAtom],
        status: str,
    ) -> dict[str, Any]:
        doc_ids = _dedupe_strings(atom.doc_id for atom in atoms if atom.doc_id)
        citation_ids = _dedupe_strings(atom.citation_id for atom in atoms if atom.citation_id)
        if isinstance(parsed, dict):
            doc_ids = _dedupe_strings([*doc_ids, *self._collect_doc_ids(parsed)])
            citation_ids = _dedupe_strings([*citation_ids, *self._collect_citation_ids(parsed)])
        return {
            "status": status,
            "tool_name": tool_name,
            "preferred_doc_ids": doc_ids[:12],
            "citation_ids": citation_ids[:12],
            "next_step": "resolve_with_rag_or_document_read" if status == "required" else "",
        }

    def _collect_doc_ids(self, payload: Any) -> List[str]:
        found: List[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in _DOC_KEYS and str(value or "").strip():
                    found.append(str(value))
                elif isinstance(value, (dict, list)):
                    found.extend(self._collect_doc_ids(value))
        elif isinstance(payload, list):
            for item in payload:
                found.extend(self._collect_doc_ids(item))
        return found

    def _collect_citation_ids(self, payload: Any) -> List[str]:
        found: List[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key in _CITATION_KEYS:
                    found.extend(str(item) for item in _to_list(value) if str(item or "").strip())
                elif isinstance(value, (dict, list)):
                    found.extend(self._collect_citation_ids(value))
        elif isinstance(payload, list):
            for item in payload:
                found.extend(self._collect_citation_ids(item))
        return found

    def _trim_rendered_items(self, items: Sequence[dict[str, Any]], *, target_tokens: int) -> List[dict[str, Any]]:
        trimmed: List[dict[str, Any]] = []
        current = 0
        for item in items:
            clone = dict(item)
            ledger = clone.get("evidence_ledger")
            if isinstance(ledger, list):
                clone["evidence_ledger"] = ledger[: max(1, min(4, len(ledger)))]
            tokens = self.estimate_tokens(json.dumps(clone, ensure_ascii=False))
            if trimmed and current + tokens > target_tokens:
                continue
            trimmed.append(clone)
            current += tokens
        return trimmed

    def _aggregate_metadata(self, results: Sequence[ContextCompactionResult], *, rendered_text: str) -> dict[str, Any]:
        if not results:
            return {}
        statuses = _dedupe_strings(result.source_resolution_status for result in results if result.source_resolution_status)
        return {
            "original_tokens": sum(result.original_tokens for result in results),
            "compacted_tokens": self.estimate_tokens(rendered_text),
            "method": "+".join(_dedupe_strings(result.method for result in results)),
            "selected_evidence_count": sum(len(result.selected_atoms) for result in results),
            "dropped_evidence_count": sum(len(result.dropped_atoms) for result in results),
            "full_result_ref": next((result.full_result_ref for result in results if result.full_result_ref), ""),
            "source_resolution_status": statuses[0] if len(statuses) == 1 else ",".join(statuses),
        }


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


__all__ = [
    "ContextCompactionResult",
    "ContextCompactionService",
    "EvidenceAtom",
]
