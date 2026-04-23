# Requirements Extraction Test Prompts

Collection ID for KB tests: `requirements-extraction-pack`

## Upload Tests

1. Upload only `raven_crest_system_performance_spec_rev_a.docx` and ask:
   `Extract all shall statements from this uploaded document.`

2. Upload only `raven_crest_system_performance_spec_rev_a.docx` and ask:
   `Extract all requirement statements from this uploaded document using mandatory language and return the CSV.`

3. Upload the full pack and ask:
   `Extract all mandatory requirement statements from all uploaded documents and export the CSV.`

4. Upload the full pack and ask:
   `Extract all shall statements from raven_crest_interface_requirements_spec_rev_a.pdf.`

5. Upload the full pack and ask:
   `Extract all shall statements from raven_crest_subcontract_statement_of_work_rev_b.docx and include where each statement appears.`

## KB Tests

1. `Pull the shall statements from raven_crest_system_performance_spec_rev_a.docx in the requirements-extraction-pack collection.`

2. `Pull all shall statements from all documents in the requirements-extraction-pack collection and export them.`

3. `Extract all requirement statements from raven_crest_cybersecurity_and_assurance_appendix.md in the requirements-extraction-pack collection using mandatory language.`

4. `Extract all shall statements from raven_crest_program_management_review_minutes_final.pdf in the requirements-extraction-pack collection.`

## Precision Checks

- `In strict shall mode, pull statements from raven_crest_cybersecurity_and_assurance_appendix.md.`
- `In mandatory mode, pull statements from raven_crest_cybersecurity_and_assurance_appendix.md.`
- `Extract all shall statements from raven_crest_risk_issue_digest.txt.`
