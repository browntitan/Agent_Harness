# Session Workspace

The live workspace is a session-scoped host directory used by the data-analyst
sandbox, upload flows, long-form writing, and user-facing downloadable artifacts.

## Path model

```text
data/workspaces/<filesystem_key(session_id)>/
```

This is intentionally different from the old raw `session_id` directory layout.

## Why it exists

Each Docker sandbox execution is ephemeral, but the workspace bind mount
persists across turns. That lets the runtime:

- copy uploads into `/workspace`
- keep generated files across turns
- let later turns inspect prior outputs
- publish derived files back to the user without leaving the session scope
- keep long-form draft/manifest files visible to the same session

Under the current analyst sandbox model, that workspace contract is unchanged. What changed is
package provisioning: the sandbox image itself must already contain the analyst dependencies, and
`python run.py build-sandbox-image`, `python run.py doctor --strict`, plus notebook preflight now
verify that image contract before analyst demos run.

## Live creation paths

The workspace is created by the live next runtime in three common paths:

1. `RuntimeService.process_turn(...)`, which eagerly opens the canonical
   session workspace for service-handled turns when `WORKSPACE_DIR` is
   configured
2. `POST /v1/ingest/documents` before the first chat turn, so uploaded files
   are already available in the session-scoped workspace
3. `POST /v1/upload`, which opens the same canonical session workspace for
   multipart client uploads

Both API ingest paths use
`SessionWorkspace.for_session(session_id, WORKSPACE_DIR)`, where `session_id`
is the derived runtime session key for the active request scope.

## Runtime bridge

`SessionState.workspace_root` stores the workspace path. `ToolContext`
reconstructs the workspace handle lazily for tool code. Scoped worker jobs
inherit the same `SessionState.workspace_root`; the runtime does not create
per-job workspaces today.

When a tool publishes a workspace file through `return_file`, the runtime stores
a download manifest in session `downloads` metadata and attaches pending
assistant `artifacts` metadata for the gateway to expose.

Long-form writing uses the same workspace contract. Instead of publishing a tool-produced file,
the runtime writes the document draft directly into the session workspace and then registers the
finished draft as a normal download artifact.

## User-facing file return

The session workspace is also the source of truth for downloadable analyst
outputs.

- `return_file` validates that a file already exists in the workspace
- the tool registers a download manifest in session metadata
- the gateway serves that file through `/v1/files/{download_id}`
- chat responses surface the same handoff through assistant `artifacts`

`return_file` does not move the file out of the workspace. The workspace file
remains the durable source of truth for the download.

That remains true whether the file was created by open-ended `execute_code(...)` or by bounded
analyst helpers running inside the offline image configured by `SANDBOX_DOCKER_IMAGE`.

## What belongs here

- uploaded files copied for analyst and sandbox access
- files written by `workspace_write`
- files created by sandbox code
- derived CSV, Excel, and chart outputs created by the data analyst
- long-form writing drafts such as `long_output_<hash>_<slug>.md`
- long-form manifest files such as `long_output_<hash>_manifest.json`
- optional long-form per-section files such as `long_output_<hash>_section_01.md`

## What does not belong here

- session state
- transcripts
- events
- job artifacts
- download metadata itself
- document source bytes when object storage or upload/source stores own them
- managed memory records and their embeddings
- MCP connection profiles or cached tool catalogs
- capability/access-control rows
- graph index metadata, graph runs, graph query cache rows, and GraphRAG project state
- extracted requirement statement records
- coordinator-owned typed handoff artifacts such as `analysis_summary`, `entity_candidates`,
  `keyword_windows`, `doc_focus`, `evidence_request`, and `evidence_response`

Those live under `data/runtime/...`, PostgreSQL-backed stores, configured object storage, or
GraphRAG project directories depending on the subsystem.

Design note:

- workspace files are user-visible or sandbox-visible filesystem artifacts
- typed handoffs are structured runtime metadata passed between workers through the
  coordinator layer
- long-form drafts and manifests are workspace-backed document artifacts owned by the session,
  not coordinator handoff artifacts

They are intentionally separate so campaign coordination does not depend on ad hoc
workspace file conventions.

For long-form writing, the workspace draft remains the source of truth for the downloadable
document. Session metadata only stores the download registration and high-level artifact
reference.
