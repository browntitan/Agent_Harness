# Section Reference

## Who This Is For

Anyone who wants a quick explanation of a control-panel page before using it or reviewing a change.

## What You Can Do Here

- See the purpose of each shipped section at a glance.
- Understand which sections are informational and which ones can change live behavior.
- Find the safest recovery path available in the current UI.

## What To Read Next

- Use the [Access Operator Guide](access-operator-guide.md) when you need the full RBAC runbook for users, groups, roles, bindings, and permissions.
- Use the [Task Guide](task-guide.md) for step-by-step workflows.
- Use the [Technical Reference](technical-reference.md) for overlay, reload, and storage details.
- Use [Troubleshooting](troubleshooting.md) if a page or action behaves unexpectedly.

## Capability Matrix

| Section | Primary purpose | Common user | Changes live behavior | Undo or recovery guidance |
| --- | --- | --- | --- | --- |
| `Dashboard` | Runtime and summary view | Lead, operator | No | None needed |
| `Architecture` | Explain the live system map, routing paths, and observed traffic | Lead, operator, technical reviewer | No | None needed |
| `Config` | Draft, validate, and apply runtime config overrides | Operator, technical reviewer | Yes | Reapply a known-good value |
| `Agents` | Review agent definitions and save overlays | Operator, technical reviewer | Yes after reload | Save a corrected overlay and reload again |
| `Prompts` | Edit prompt overlays | Operator, technical reviewer | Yes on next turn | `Reset Overlay` |
| `Collections` | Create, ingest, inspect, repair, and delete empty collections | Operator | Yes | Re-ingest or re-upload source content, or recreate an empty namespace |
| `Graphs` | Create and manage graph projects bound to a collection | Operator, technical reviewer | Yes | Rebuild, refresh, or re-save the graph draft |
| `Skills` | Review, preview, edit, activate, and deactivate skill packs | Operator, technical reviewer | Yes | Toggle status back or restore prior body outside the UI |
| `Access` | Manage principals, groups, roles, bindings, permissions, and effective-access preview | Operator, technical reviewer | Yes | Remove the incorrect binding, membership, or permission, or replace it with a corrected role model |
| `Operations` | Review reload history, jobs, and audit activity | Lead, operator | No | None needed |

## Dashboard

### What You Can Do Here

- Review runtime status and the high-level payload returned by the admin overview endpoint.
- Check collection, agent, skill, tool, and job counts.
- Review the last reload summary.

### What You See

- `Runtime`
- `Reload Summary`

### Major Actions

- Read-only inspection only

**Risk: Low risk**

## Architecture

### What You Can Do Here

- See the current runtime topology based on the live agent registry, router settings, and overlay state.
- Trace canonical routing paths such as `BASIC`, default `AGENT`, grounded lookup, data analysis, and coordinator-led work.
- Review privacy-safe recent routing activity and worker handoffs.

### What You See

- `System Overview`
- `Map`
- `Routing Paths`
- `Live Traffic`

### Major Actions

- read-only inspection only
- click nodes to inspect live capabilities and badges
- trace a canonical route back onto the map

**Risk: Low risk**

## Config

### What You Can Do Here

- Review effective config values grouped by area.
- Draft edits to supported runtime-managed settings.
- Validate and apply config changes.

### What You See

- grouped config sections such as `Providers`, `Runtime`, `Routing`, `Features`, `Sandbox`, `Observability`, and `Agent Models`
- `Validate`
- `Apply`
- `Preview`

### Major Actions

- validate proposed changes without applying them
- apply a valid change and trigger a runtime swap

**Risk: Changes live behavior**

> Technical note: `Bootstrap` fields are intentionally read-only in the current control panel.

## Agents

### What You Can Do Here

- Review available agents and the tool catalog.
- Edit common agent fields in the overlay-backed editor.
- Reload the agent registry after saving an overlay.

### What You See

- `Available Agents`
- `Tool Catalog`
- `Agent Editor`
- `Save Overlay`
- `Reload Agents`

### Major Actions

- save an agent overlay
- reload the agent registry

**Risk: Needs careful review**

## Prompts

### What You Can Do Here

- Review prompt files currently known to the system.
- Edit prompt overlays.
- Reset a prompt overlay back to the base prompt.

### What You See

- `Prompt Files`
- `Prompt Editor`
- `Save Overlay`
- `Reset Overlay`

### Major Actions

- save prompt text for the next turn
- remove an overlay and fall back to the repo-backed prompt

**Risk: Changes live behavior**

## Collections

### What You Can Do Here

- Create a collection explicitly and keep it visible before any ingest succeeds.
- Review available collections and readiness state from the catalog and dropdown.
- Sync configured KB sources.
- Ingest server-visible files by absolute path.
- Upload files from the browser or upload a whole folder while preserving relative paths.
- Inspect documents, reindex or delete them, and review technical storage metadata and health in the same workspace.
- Delete an empty collection once its documents and graphs are cleared.

### What You See

- `Collections`
- `Collection Workspace`
- `Available Collections`
- `Create Collection`
- `Delete Empty`
- `Host Paths`
- `Ingest Host Paths`
- `Upload Files`
- `Upload Folder`
- `Sync KB`
- `Documents`
- `Document Viewer`
- `Collection Inspector`
- `Collection Health`
- `Reindex`
- `Delete`

### Major Actions

- create an empty collection namespace
- ingest content into a collection
- inspect stored document data
- inspect table names, embedding settings, and vector dimensions
- repair duplicate or drifted KB rows
- reindex or delete a document

**Risk: Needs careful review**

## Graphs

### What You Can Do Here

- Create graph drafts that bind to one collection at a time.
- Validate, build, and refresh graph projects.
- Confirm that the graph collection dropdown uses the same catalog as the collections workspace, including brand-new empty collections.
- Review graph sources, runs, logs, prompt overrides, and graph-bound skill overlays.

### What You See

- `Graphs`
- `Graph Workspace`
- `Graph Collection`
- `Save Draft`
- `Validate`
- `Build`
- `Refresh`
- `Graph Inspector`
- `Graph Runs`

### Major Actions

- create a graph draft for a selected collection
- validate or build the graph project
- update graph prompts or bound skills

**Risk: Needs careful review**

## Skills

### What You Can Do Here

- Review the current skill list.
- Preview whether a skill matches a given query.
- Edit an existing skill body.
- Activate or deactivate a skill.

### What You See

- `Skills`
- `Preview Query`
- `Preview Match`
- `Skill Editor`
- `Update Skill`
- `Activate`
- `Deactivate`

### Major Actions

- preview a skill match
- update skill body
- change skill availability

**Risk: Changes live behavior**

> Technical note: the current page is strongest for editing existing skills. It does not expose the full backend lifecycle, such as rollback, in the shipped UI.

## Access

### What You Can Do Here

- Create email-backed user principals and placeholder groups.
- Create reusable roles for collections, graphs, tools, and skill families.
- Bind roles to users or groups.
- Add group memberships for future identity-provider sync readiness.
- Preview the exact effective-access snapshot the runtime will resolve for a user email.

### What You See

- `Principals`
- `Roles`
- `Bindings`
- `Memberships`
- `Permissions`
- `Effective Access`
- `Preview Access`

### Major Actions

- save a principal
- save a role
- add or remove a binding
- add or remove a membership
- add or remove a permission
- preview effective access for a user email

**Risk: Changes live behavior**

> Operator note: use the [Access Operator Guide](access-operator-guide.md) for the full rollout
> order, starter role patterns, API backup flows, and access-denial troubleshooting.

## Operations

### What You Can Do Here

- Review the most recent reload summary.
- See recent jobs and their output paths.
- Review audit events.

### What You See

- `Operations`
- `last_reload`
- `jobs`
- `audit_events`

### Major Actions

- read-only inspection only

**Risk: Low risk**
