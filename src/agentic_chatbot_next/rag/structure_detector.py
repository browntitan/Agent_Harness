"""Document structure detection.

Pure-function module — no LLM calls, no I/O.
Inspects raw document text and classifies its structure type
so that the ingestion pipeline can choose the right splitter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches lines that start a numbered clause, section, or article.
# Examples matched:
#   "Clause 3:"         "Clause 3.2 Title"
#   "Section 4."        "Section 10.1 – Definitions"
#   "Article 5"         "ARTICLE VI"
#   "1. Introduction"   "10.2.3 Scope"  (numbered heading — capital-word follows)
CLAUSE_PATTERN = re.compile(
    r"^(?:"
    r"(?:Clause|CLAUSE)\s+\d[\d\.]*"
    r"|(?:Section|SECTION)\s+\d[\d\.]*"
    r"|(?:Article|ARTICLE)\s+(?:\d[\d\.]*|[IVXLCDM]+)"
    r"|\d{1,3}(?:\.\d{1,3}){0,3}\.\s+[A-Z]"  # e.g. "3.1 Definitions"
    r")",
    re.MULTILINE,
)

# Matches requirement-language keywords anywhere in a chunk.
REQUIREMENT_PATTERN = re.compile(
    r"\b("
    r"shall|must|is\s+required\s+to|are\s+required\s+to"
    r"|is\s+prohibited\s+from|shall\s+not|must\s+not"
    r"|REQ-\d+|REQUIREMENT-?\d*"
    r")\b",
    re.IGNORECASE,
)

# Matches process-flow and workflow language.
PROCESS_FLOW_PATTERN = re.compile(
    r"\b("
    r"process\s+flows?|workflows?|flowcharts?|approval\s+flows?|decision\s+tree|"
    r"escalation\s+path|handoff|step[-\s]?by[-\s]?step|swimlane"
    r")\b",
    re.IGNORECASE,
)

# Matches enumerated procedure steps.
STEP_PATTERN = re.compile(
    r"^\s*(?:step\s+\d+[:.)-]?|\d{1,2}[.)]\s+[A-Z]|phase\s+\d+[:.)-]?)",
    re.IGNORECASE | re.MULTILINE,
)

# Matches explicit requirement IDs — used as a strong signal.
REQ_ID_PATTERN = re.compile(r"\b(REQ-\d+|REQUIREMENT-\d+|R\d{3,})\b")

# Policy-document signals (titles / headings commonly seen in policy docs).
POLICY_PATTERN = re.compile(
    r"\b(policy|procedure|guideline|compliance|governance|standard)\b",
    re.IGNORECASE,
)

# Contract-document signals.
CONTRACT_PATTERN = re.compile(
    r"\b(termset|term\s+sheet|agreement|contract|indemnif|warranty|warranties|"
    r"governing\s+law|dispute\s+resolution|force\s+majeure)\b",
    re.IGNORECASE,
)

DocStructureType = Literal[
    "general",
    "structured_clauses",
    "requirements_doc",
    "policy_doc",
    "contract",
    "process_flow_doc",
]


@dataclass
class StructureAnalysis:
    doc_structure_type: DocStructureType
    has_clauses: bool
    has_requirements: bool
    has_process_flow: bool
    clause_density: float   # fraction of non-blank lines that are clause headers


def detect_structure(full_text: str) -> StructureAnalysis:
    """Classify a document's structure using heuristics.

    No LLM call — fast and deterministic. Suitable for ingest-time dispatch.

    Thresholds (all adjustable):
      clause_density > 0.03  → has_clauses=True
      requirement matches >= 5 → has_requirements=True
    """
    lines = full_text.splitlines()
    non_blank = [l for l in lines if l.strip()]
    total_lines = max(len(non_blank), 1)

    clause_hits = sum(1 for l in non_blank if CLAUSE_PATTERN.match(l.strip()))
    clause_density = clause_hits / total_lines

    req_matches = len(REQUIREMENT_PATTERN.findall(full_text))
    req_id_matches = len(REQ_ID_PATTERN.findall(full_text))

    has_clauses = clause_density > 0.03 or clause_hits >= 3
    has_requirements = req_matches >= 5 or req_id_matches >= 2
    process_flow_matches = len(PROCESS_FLOW_PATTERN.findall(full_text))
    step_matches = len(STEP_PATTERN.findall(full_text))
    has_process_flow = process_flow_matches >= 2 or step_matches >= 3

    # Classify
    if req_id_matches >= 2 or (has_requirements and not has_clauses):
        doc_type: DocStructureType = "requirements_doc"
    elif has_process_flow and POLICY_PATTERN.search(full_text):
        doc_type = "process_flow_doc"
    elif has_clauses and CONTRACT_PATTERN.search(full_text):
        doc_type = "contract"
    elif has_clauses and POLICY_PATTERN.search(full_text):
        doc_type = "policy_doc"
    elif has_process_flow:
        doc_type = "process_flow_doc"
    elif has_clauses:
        doc_type = "structured_clauses"
    else:
        doc_type = "general"

    return StructureAnalysis(
        doc_structure_type=doc_type,
        has_clauses=has_clauses,
        has_requirements=has_requirements,
        has_process_flow=has_process_flow,
        clause_density=clause_density,
    )
