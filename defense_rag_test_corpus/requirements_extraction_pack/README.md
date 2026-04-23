# Defense Requirements Extraction Pack

UNCLASSIFIED // SYNTHETIC TEST DATA

This standalone pack is designed to exercise the repository's `extract_requirement_statements` and `export_requirement_statements` workflows with long-form defense-contractor style prose documents. It contains one synthetic program, **Raven Crest Expeditionary Relay Modernization**, and mixes product specifications, contractor work instructions, broader mandatory-language appendices, traceability guidance, and distractor documents.

## Research Basis

The structure of the pack follows common defense and systems-engineering document patterns:

- A MIL-STD-961 style system performance specification with the six-section structure summarized by DAU: Scope, Applicable Documents, Requirements, Verification, Packaging, and Notes.
  Source: https://www.dau.edu/acquipedia-article/system-performance-specification
- System/Subsystem Specification and Interface Requirements Specification DID patterns used for contractor-facing requirement baselines.
  Sources: https://www.dau.edu/system-subsystem-specification-sss-data-item-description-did and https://www.dau.edu/artifact/interface-requirements-specification-irs-data-item-description-did
- Statement of Work separation from product performance requirements, aligned to MIL-HDBK-245D and CDRL/DID practices.
  Sources: https://www.dau.edu/index.php/cop/pbl/documents/mil-hdbk-245d-dod-handbook-preparation-statement-work and https://www.dau.edu/acquipedia-article/product-support-contract-data-requirements-list-cdrl-and-data-item-descriptions
- Requirement wording and traceability discipline aligned to NASA and SEBoK guidance around mandatory `shall` statements, verifiability, and traceability.
  Sources: https://nodis3.gsfc.nasa.gov/displayCA.cfm?Internal_ID=N_PR_1400_001H_&page_name=Chapter3, https://www.nasa.gov/reference/appendix-d-requirements-verification-matrix/, and https://sebokwiki.org/wiki/System_Requirements_Definition

## Pack Contents

Authoritative requirement-bearing documents:
- `documents/docx/raven_crest_system_performance_spec_rev_a.docx`
- `documents/pdf/raven_crest_interface_requirements_spec_rev_a.pdf`
- `documents/docx/raven_crest_subcontract_statement_of_work_rev_b.docx`
- `documents/md/raven_crest_cybersecurity_and_assurance_appendix.md`
- `documents/txt/raven_crest_verification_and_traceability_guide.txt`

Distractor / precision documents:
- `documents/pdf/raven_crest_program_management_review_minutes_final.pdf`
- `documents/txt/raven_crest_risk_issue_digest.txt`

The cybersecurity appendix is the main difference-maker between `strict_shall` and `mandatory` modes because it intentionally uses `must`, `must not`, `required to`, and prohibited forms alongside a small number of `shall` statements.

## Delivered Evaluation Assets

- `evaluation/manifest.csv`
- `evaluation/test_prompts.md`
- `gold/mandatory_all_documents.csv`
- `gold/strict_shall_all_documents.csv`
- `gold/per_doc/*`

These gold files were generated from the current repository extractor after indexing the pack into the `requirements-extraction-pack` collection, so the counts and locations reflect the same extraction path the app uses.
