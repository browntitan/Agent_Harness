# RFP Corpus Test Prompts

Use these prompts after indexing the bundled defense corpus. The source files live under
`defense_rag_test_corpus/`; answer-time retrieval should use the indexed collection, not raw
filesystem reads.

Recommended setup:

```bash
python run.py sync-defense-corpus
python run.py evaluate-defense-corpus --sync-first
```

The main defense corpus should be tested as collection `rfp-corpus`. The requirements pack
should be tested as collection `requirements-extraction-pack`.

## Inventory And Scope Checks

- `What documents are available in the rfp-corpus collection?`
  Expected path: `general -> list_indexed_docs(source_type="kb", collection_id="rfp-corpus")`.

- `List the PDF, DOCX, TXT, and XLSX files in rfp-corpus and group them by program.`
  Expected path: inventory-first metadata answer; no fabricated citations.

- `Which documents in rfp-corpus mention Ember Reach? Return document titles only with short grounded justifications.`
  Expected path: `research_coordinator` or RAG corpus-discovery path with citations.

## Direct Grounded Lookup

- `Which supplier is named in the Asterion subtier statement of work? Cite the source.`
  Expected path: direct `rag_worker` lookup over `asterion_subtier_sow_rev_b.docx`.

- `What endurance distance does the Iron Vale ground test plan require? Cite the source.`
  Expected path: direct `rag_worker` lookup over `iron_vale_ground_test_plan_final.pdf`.

- `Which bracket required reinforcement after Trident Echo shock qualification? Cite the source.`
  Expected path: direct `rag_worker` lookup over `trident_echo_shock_qualification_brief_final.pdf`.

- `What minimum processing-efficiency threshold does the Ember Reach RFP require? Cite the source.`
  Expected path: direct `rag_worker` lookup over `ember_reach_rfp_overview_final.pdf`.

## Multi-Document Reconciliation

- `Explain why Asterion CDR moved and identify the documents that show both the emerging problem and the final approved answer.`
  Expected path: `research_coordinator` or deep RAG campaign across Asterion draft/status/ECP documents.

- `Why is it inaccurate to describe the Iron Vale endurance miss as only a software problem?`
  Expected path: `research_coordinator` grounded synthesis across the ground test plan,
  endurance summary, proposal, and risk workbook.

- `If someone says Blue Mica Wave 2 slipped because the hardware was bad, what is the better evidence-based answer?`
  Expected path: `research_coordinator` comparison across compliance summary, refresh plan,
  after-action notes, and staffing/spares workbook.

- `Differentiate North Coast Systems LLC from Northcoast Signal Labs and name the program each one supports.`
  Expected path: entity disambiguation with citations from the relevant program documents;
  `research_coordinator` should win when the router treats this as broad corpus reconciliation.

- `Differentiate Halcyon Foundry from Halcyon Microdevices and explain which one had manufacturing void issues versus procurement scoring.`
  Expected path: entity disambiguation across Trident Echo and Ember Reach evidence;
  `research_coordinator` or `rag_researcher` are both valid test paths depending on override.

## RAG Researcher Variants

Run these with `metadata.requested_agent="rag_researcher"` when you want to test the manual
RAG researcher loop directly. Expected tool shape: `plan_rag_queries`, chunk/section or
metadata search, evidence grading/pruning, `validate_evidence_plan`,
`build_rag_controller_hints`, then final `rag_agent_tool`.

- `Use rag_researcher. Which Asterion documents should be treated as authoritative for updated dates, rather than the early draft planning note? Cite each source.`

- `Use rag_researcher. Differentiate North Coast Systems LLC from Northcoast Signal Labs and name the program each one supports.`

- `Use rag_researcher. Search rfp-corpus for spreadsheet evidence about Iron Vale risks, profile the relevant workbook evidence if needed, and answer with citations.`

## Requirements Extraction Pack

- `Pull the shall statements from raven_crest_system_performance_spec_rev_a.docx in the requirements-extraction-pack collection.`
  Expected path: `extract_requirement_statements` in strict shall mode.

- `Extract all mandatory requirement statements from all documents in the requirements-extraction-pack collection and export them.`
  Expected path: `extract_requirement_statements` plus `export_requirement_statements`, with a downloadable CSV artifact.

- `In strict shall mode, pull statements from raven_crest_cybersecurity_and_assurance_appendix.md.`
  Expected path: strict shall extraction only.

- `In mandatory mode, pull statements from raven_crest_cybersecurity_and_assurance_appendix.md.`
  Expected path: broader mandatory-language extraction; compare against strict shall output.

## Tool-Trace Variants

Run these with `metadata.requested_agent="general"` when you want top-level tool traces instead
of a direct specialist start.

- `Search rfp-corpus for documents about Asterion cost changes, then read the most relevant document directly.`
  Expected tools: `search_indexed_docs`, then `read_indexed_doc`.

- `Compare the Asterion ECP and monthly status review and explain the cost/date differences with citations.`
  Expected tools: `resolve_indexed_docs`, `compare_indexed_docs`, or `rag_agent_tool`.

- `Extract requirement-like obligations from the Ember Reach RFP overview and return them as a structured list.`
  Expected tools: `document_extract` or `extract_requirement_statements` depending on routing.

- `Build an evidence binder for the claim that Blue Mica Wave 2 slipped because of supplier documentation and training/spares issues.`
  Expected tools: `evidence_binder` plus grounded RAG citations.

## Graph-Aware Variants

Use these only when graph search and a graph index for the corpus are available.

- `What graph indexes are available for rfp-corpus?`
  Expected path: graph inventory through `graph_manager` or `general -> list_graph_indexes`.

- `Use graph-backed evidence to trace supplier relationships around Ember Reach and explain which documents support each relationship.`
  Expected path: `graph_manager` with `search_graph_index`, `explain_source_plan`, and cited text evidence.

- `Find graph relationships that connect Trident Echo manufacturing constraints to qualification outcomes, then resolve them back to source documents.`
  Expected path: graph search followed by text-grounded citations; graph facts alone are insufficient.

## Failure Checks

- `Answer from rfp-corpus: What is the approved Martian launch date?`
  Expected behavior: clear negative-evidence response; no invented source.

- `Extract shall statements from a document that is not in requirements-extraction-pack.`
  Expected behavior: missing-source guidance or inventory prompt, not fabricated extraction.

- `Summarize every hidden classified appendix in rfp-corpus.`
  Expected behavior: explain that only indexed accessible documents can be used.
