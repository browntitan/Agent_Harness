# `new_demo_notebook`

This folder contains the scenario-first Jupyter showcase for the live
`agentic_chatbot_next` backend runtime.

The notebook is still API-driven and self-starting, but the main walkthrough is now organized as
a scenario-first main path instead of a scenario appendix or a single run-all harness.
After setup/startup, the source notebook at `new_demo_notebook/agentic_system_showcase.ipynb`
becomes a sequence of self-contained demos with one scenario per runnable code cell, one query or one skill action per runnable code cell, and one explanatory markdown cell directly above it.
The executed artifact under `.artifacts/executed/` is only a test output snapshot.

## What the notebook demonstrates

- live FastAPI startup and health inspection
- inline observability logs plus structured post-run summaries inside each core demo
- BASIC vs AGENT routing
- general-agent tool and skill use
- delegated grounded RAG via `metadata.requested_agent="general"`
- direct grounded RAG via `rag_worker`
- workbook-aware RAG
- data-analyst sandbox workflows with returned artifacts
- coordinator orchestration with planner, workers, handoffs, finalizer, and verifier
- sync long-form writing showcase with returned artifact and `metadata.long_output`
- repo-authored skill-pack inspection
- runtime skill CRUD through `/v1/skills`
- optional defense-corpus appendix

## Notebook structure

The main notebook path is scenario-first after setup:

1. environment setup and preflight
2. backend startup and runtime scope
3. driver-code walkthrough
4. basic chat demo
5. general-agent tool-and-skill demo
6. delegated grounded RAG demo
7. direct grounded RAG demo
8. workbook-aware RAG demo
9. data-analyst sandbox demo
10. coordinator orchestration demo
11. long-form writing demo
12. repo skill-pack overview
13. create runtime skill draft
14. preview runtime skill resolution
15. activate runtime skill
16. inspect runtime skill
17. update runtime skill version
18. deactivate updated runtime skill
19. roll back runtime skill

Secondary sections come afterward:

- optional defense corpus appendix
- troubleshooting and cleanup

Each chat demo cell is intentionally self-contained. If a demo needs KB ingest or upload-style
workspace prep, that cell performs the prep before the chat turn so users can run only the
scenarios they care about. The runtime skill control cells are split into one action per cell and
are intended to be run top-to-bottom.

Optional behavior is gated in the top configuration cell:

- `RUN_ADVANCED_DEMOS`
- `RUN_DEFENSE_CORPUS_DEMO`
- `SHOW_SERVER_LOG_TAIL`
- `GATEWAY_TIMEOUT_SECONDS`
- `JOB_WAIT_TIMEOUT_SECONDS`

## Observability behavior

The notebook shows logs plus summaries for substantial scenarios:

- live printed progress lines while the turn is running
- tool and artifact notifications inline with the stream
- structured post-run sections for progress, tools, artifacts, metadata, and traces

Each core demo cell shows safe summarized execution progress, not raw chain-of-thought.

For the sync long-form writing showcase, the notebook also surfaces:

- returned artifact metadata
- `metadata.long_output`
- workspace previews of the generated draft and manifest

## Setup

1. Install the main repository requirements.
2. Install notebook-side extras:

```bash
python -m pip install -r new_demo_notebook/requirements.txt
```

3. Make sure the backend provider and database environment is configured.
4. Launch Jupyter:

```bash
jupyter notebook new_demo_notebook/agentic_system_showcase.ipynb
```

The notebook starts and stops the live backend server itself with `python run.py serve-api`
unless you disable `START_LOCAL_SERVER` and provide a `BASE_URL`.
When `START_LOCAL_SERVER=True`, the notebook also attempts to bootstrap missing local Docker dependencies such as `rag-postgres` and the offline analyst sandbox image before it enforces the strict preflight assert.

## Runtime behavior notes

- The notebook waits for `/health/live` when it starts the server, then inspects `/health/ready`
  separately.
- The preflight uses the same `.env` and runtime settings loader as the backend, so
  DB/provider/model checks reflect the real app configuration rather than notebook-specific
  defaults.
- The preflight includes a strict offline sandbox-image probe for `SANDBOX_DOCKER_IMAGE`. The
  default image is `agentic-chatbot-sandbox:py312`, and the notebook can build it during
  bootstrap when Docker is available.
- The default notebook request timeout is intentionally long for slower personal machines:
  `GATEWAY_TIMEOUT_SECONDS=1800` and `JOB_WAIT_TIMEOUT_SECONDS=900`.
- Persistent memory can be disabled end-to-end with `MEMORY_ENABLED=false`; when disabled, memory
  tools, heuristic extraction, and `memory_maintainer` work are unavailable.
- RAG scenarios use collection-scoped ingest. The system searches indexed content, not raw folders
  at query time.
- The long-form scenario is sync-only on purpose. It is sized below the background thresholds so
  the notebook can demonstrate the multi-call writer path without `/v1/jobs/{job_id}` polling.
- That long-form showcase also injects a compact local reference packet from repo docs into the
  prompt. This keeps the demo deterministic and makes it clear that the multi-call writer path is
  being exercised directly rather than relying on a separate live retrieval step inside the
  composer.

## Default local demo data

The notebook uses current runtime docs from `docs/` plus spreadsheet fixtures in
`new_demo_notebook/demo_data/` for:

- collection-backed RAG scenarios
- workbook-aware RAG
- data-analyst demos
- coordinator demos
- the sync long-form writing showcase's local reference packet

## Optional defense benchmark section

The notebook can ingest the defense test corpus from:

- `defense_rag_test_corpus/documents/`

That section is heavier and disabled by default. For repeatable benchmark work, prefer the CLI:

```bash
python run.py sync-defense-corpus
python run.py evaluate-defense-corpus --sync-first
```

## Acceptance and smoke tests

The notebook still participates in the acceptance flow, but the acceptance scenario manifest stays
separate from the hand-authored notebook.

Run the local acceptance flow in this order:

```bash
python -m pip install -r new_demo_notebook/requirements.txt
ollama list
docker info
python run.py doctor --strict
python run.py migrate
python run.py sync-kb --collection-id default
python run.py index-skills
RUN_NEXT_RUNTIME_ACCEPTANCE=1 pytest -m acceptance tests/test_next_acceptance_harness.py
RUN_NEXT_RUNTIME_NOTEBOOK_ACCEPTANCE=1 pytest -m acceptance tests/test_next_acceptance_harness.py -k notebook
```

The scenario manifest and `ScenarioRunner` remain available for broad agent coverage validation,
but the notebook itself is the primary curated interactive surface.

## Notes

- Server logs are written to `new_demo_notebook/.artifacts/server.log`.
- Executed notebook output is written to
  `new_demo_notebook/.artifacts/executed/agentic_system_showcase.executed.ipynb`.
- Runtime traces are read from `data/runtime`.
- Session workspaces live under `data/workspaces/`.
- Memory artifacts, when memory is enabled, live under `data/memory/`.
- Scenario definitions for the separate acceptance harness live in
  `new_demo_notebook/scenarios/scenarios.json`.
- Data-analyst fixtures live under `new_demo_notebook/demo_data/data_analyst/`.
- Fixture documentation lives in `new_demo_notebook/demo_data/data_analyst/README.md`.
