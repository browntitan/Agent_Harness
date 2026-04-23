# Technical Reference

## Who This Is For

Technical readers who need implementation-grounded explanations of the control panel’s behavior, storage, and safety model.

## What You Can Do Here

- Understand how token protection and local-admin assumptions work.
- Understand how operator admin auth and end-user RBAC fit together in the current shipped system.
- Learn overlay precedence and reload timing.
- See where control-panel data is stored and how document views are assembled.

## What To Read Next

- Use the [Access Operator Guide](access-operator-guide.md) for the operator-facing RBAC workflow.
- Use the [Task Guide](task-guide.md) for operator workflows.
- Use [Troubleshooting](troubleshooting.md) when a technical detail shows up as a user-facing error.
- Return to the [handbook landing page](../README.md) for the audience reading paths.

## Access Model

The current control panel is still designed for trusted local administrators, but the runtime can
also enforce per-user RBAC for end users when `AUTHZ_ENABLED=true`.

- The UI sends `X-Admin-Token` with admin requests.
- If the control panel is disabled, admin endpoints return `404`.
- If the server has no configured admin token, admin endpoints return `503`.
- If the supplied token does not match, admin endpoints return `401`.
- The runtime RBAC model is driven by trusted request email, not by standalone auth handled inside
  the control panel itself.
- Trusted runtime identity can be forwarded through `X-User-Email`,
  `X-OpenWebUI-User-Email`, `metadata.user_email`, or top-level `userEmail`.
- Protected runtime resources are collections, graphs, tools, and skill families.
- Authorization is deny-by-default when enabled.
- The chat-scoped upload collection remains implicitly allowed to the owning session.
- Graph access requires both a graph grant and a grant to the graph's backing collection.
- Tool use is the intersection of user grants and the selected agent's allowed tools.
- Skill access is granted by skill family id, not per-version skill id.

Use the [Access Operator Guide](access-operator-guide.md) for the step-by-step rollout and recovery
workflows.

## Overlay Precedence

The control panel writes overlays instead of overwriting repo-authored defaults.

### Runtime env

- Base settings are loaded from the normal environment and optional `.env`.
- A runtime overlay file is then loaded with override semantics.
- In-memory preview overrides are applied during validation before any live swap.

Default runtime overlay location:

- `data/control_panel/overlays/runtime.env`

### Prompt overlays

- Agent prompt overlays live under `data/control_panel/overlays/prompts/<prompt_file>`.
- Template prompt overlays use the same overlay directory and file-name lookup.
- Prompt overlays take effect on the next turn that reads the prompt.

### Agent overlays

- Agent overlays live under `data/control_panel/overlays/agents/<agent>.md`.
- Saving an overlay only writes the file.
- The registry changes only after `Reload Agents` succeeds.

### Audit log

Default audit log location:

- `data/control_panel/audit/events.jsonl`

## Validate, Apply, And Reload

These three ideas are easy to confuse, but they do different jobs.

| Action | What it does | Makes a live change | Writes overlay files |
| --- | --- | --- | --- |
| `Validate` | Normalizes and checks config changes against the catalog and attempts a preview build | No | No |
| `Apply` | Writes runtime env overrides and swaps in a rebuilt runtime if validation succeeds | Yes | Yes |
| `Reload Agents` | Validates the merged agent registry and swaps the registry on success | Yes | No new write by itself |

### Config apply behavior

- The config catalog blocks unsupported or read-only fields.
- Secrets are masked in effective-value responses.
- If the runtime rebuild fails, the previous overlay is restored.
- Successful config applies append an audit event.

### Prompt behavior

- Prompt save writes an overlay file immediately.
- The next turn uses the overlaid prompt.
- No full runtime reload is required.

### Agent behavior

- `Save Overlay` writes the markdown overlay and validates its frontmatter shape.
- `Reload Agents` validates the merged registry before making it live.
- Failed reloads keep the prior good registry active.

## Current UI Limitations That Matter

- The UI shows raw JSON payloads in several places rather than a curated admin dashboard.
- The current `Config` page does not expose a dedicated draft reset control for every field type.
- The current `Agents` page does not expose a visible overlay delete button.
- The current `Skills` page exposes edit and status toggles, but not the full version rollback flow.
- The current `Skills` page previews with the `general` scope in the shipped UI.

## Data Shown In Collections And Documents

The collection workspace now combines a persistent collection catalog with document, chunk, and health data from the ingest stores.

- Collection rows include collection ID, create and update timestamps, document counts, graph counts, latest ingest time, source-type counts, readiness details, and storage-profile metadata.
- The collection inspector shows the vector backend, relevant table names, embedding provider, embedding model, configured embedding dimension, actual vector dimensions, and mismatch warnings.
- Document lists include title, source type, logical display path, real source path, chunk count, file type, and ingest time.
- Document detail reconstructs extracted content by concatenating stored chunks in chunk order.
- Raw source is shown only when the original file still exists and is a text-like file type.
- Reindex requires the original source path to still exist.
- Folder uploads preserve relative paths through `source_display_path`, so repeated basenames remain distinguishable in the UI.
- The graph workspace collection dropdown reads from the same catalog, so explicitly created empty collections are available before any document ingest.

## Retrieval And Output-Length Notes

Two implementation details matter when operators debug weak answers.

- the live RAG answer path searches indexed chunks in PostgreSQL / pgvector and PostgreSQL full-text search; it is not reading the filesystem directly at answer time
- the runtime starts with a fast hybrid retrieval pass and can escalate into a deeper multi-round search when the evidence is weak or the request is broad
- retrieval includes grading and filtering, not only top-k recall

Output-length controls now use the newer `*_MAX_OUTPUT_TOKENS` envs.

- `CHAT_MAX_OUTPUT_TOKENS` is the normal chat default
- `DEMO_CHAT_MAX_OUTPUT_TOKENS` is the demo/CLI override
- `JUDGE_MAX_OUTPUT_TOKENS` applies to routing and grading
- `AGENT_<NAME>_MAX_OUTPUT_TOKENS` is the optional per-agent chat override
- leaving those blank means the runtime does not force a provider-side output cap
- legacy `OLLAMA_NUM_PREDICT` fields remain compatibility fallbacks

## Admin Endpoints Backing The UI

The current UI is driven by these endpoint groups:

- `/v1/admin/overview`
- `/v1/admin/operations`
- `/v1/admin/config/*`
- `/v1/admin/agents*`
- `/v1/admin/prompts*`
- `/v1/admin/collections*`
- `/v1/admin/graphs*`
- `/v1/admin/access/*`
- `/v1/skills*`

Use the UI first for normal operations. Reach for the API only when the shipped page does not expose the recovery or cleanup action you need.

## Storage Summary

Default control-panel storage paths:

| Purpose | Default path |
| --- | --- |
| Runtime env overlay | `data/control_panel/overlays/runtime.env` |
| Prompt overlays | `data/control_panel/overlays/prompts/` |
| Agent overlays | `data/control_panel/overlays/agents/` |
| Audit events | `data/control_panel/audit/events.jsonl` |
| Built static UI | `control_panel/dist/` |

> Technical note: these paths are configurable with `CONTROL_PANEL_*` environment variables, but the documentation assumes the default layout unless your environment says otherwise.
