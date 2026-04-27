# Control Panel Testing Routines

## Who This Is For

Operators, admins, and QA reviewers who need to exercise the control panel end to end before a demo, release, or access-policy change.

Use this page when you want a repeatable manual test run. Use the [Task Guide](task-guide.md) when you only need the normal operating workflow, and use the [Access Operator Guide](access-operator-guide.md) when you need the full RBAC rollout runbook.

## Test Setup And Safety Conventions

### Start from a known environment

1. Start the stack from the repo root:

   ```bash
   docker compose up -d --build
   ```

2. Open the control panel:

   `http://127.0.0.1:18000/control-panel`

3. Unlock it with `CONTROL_PANEL_ADMIN_TOKEN` from `.env`.
4. Open `Dashboard` and confirm `Runtime` reports a healthy status.
5. Open `Operations` and note the latest `last_reload`, active jobs, and recent audit events.
6. Open Open WebUI for live runtime checks:

   `http://127.0.0.1:3001`

### Use disposable QA names

Use names that are easy to search and safe to clean up:

- collection id: `qa-control-panel-smoke`
- graph id: `qa-vendor-risk`
- graph display name: `QA Vendor Risk Graph`
- test user: `qa.user@example.com`
- test group: `qa-control-panel-reviewers`
- role names: `qa-kb-reader`, `qa-graph-analyst`, `qa-skill-tool-user`
- skill family id: `control-panel-qa-readiness`

### Record these values during each run

- admin token environment used
- collection id
- graph id
- source filenames and folder-relative paths
- user email used for effective-access preview
- role ids or role names
- skill id and family id
- tool name or MCP registry name
- `Operations` audit event timestamp after each mutation

### General pass criteria

A routine passes when:

- the expected UI row, status, or result appears without a generic error
- `Operations` shows the expected audit activity after mutations
- live chat can use the resource when grants are present
- live chat cannot use the resource when the negative check intentionally removes or withholds a required grant
- cleanup leaves no QA documents, empty collections, bindings, permissions, or temporary skill changes behind unless the run intentionally preserves them for follow-up testing

## Routine 1: Access And User Permissions

**Risk: Changes live behavior**

Use this routine to confirm that user, group, role, binding, permission, and effective-access previews all line up with runtime behavior.

### Steps

1. Open `Access`.
2. In `Principals`, create a `User` principal with email `qa.user@example.com`.
3. In `Principals`, create a `Group` principal named `qa-control-panel-reviewers`.
4. In `Memberships`, add `qa.user@example.com` as a member of `qa-control-panel-reviewers`.
5. In `Roles`, create `qa-kb-reader` with a description such as `QA KB collection use role`.
6. In `Permissions`, add:
   - `collection` + `use` + `qa-control-panel-smoke`
7. In `Bindings`, bind `qa-kb-reader` to `qa.user@example.com` or to the QA group.
8. In `Effective Access`, preview `qa.user@example.com`.
9. Confirm the effective snapshot includes the expected collection grant.
10. Open `Operations` and confirm audit events were recorded for the principal, membership, role, permission, and binding changes.

### Expected Results

- The user principal appears with normalized lowercase email.
- The group principal appears as a group.
- The membership row links the group to the user.
- The role appears and can be selected by bindings and permissions.
- The binding appears as active.
- The effective-access preview shows the collection selector under the user's resolved grants.

### Live Chat Check

Run this in Open WebUI as, or on behalf of, `qa.user@example.com`:

```text
What knowledge bases do I have access to?
```

Expected result: the answer includes the QA collection once it exists and has an access grant. If `AUTHZ_ENABLED=false`, record that RBAC runtime enforcement is not active and treat the UI preview as the primary check.

### Negative Checks

- Preview `QA.User@Example.com`. It should normalize to the same email identity instead of creating a casing-specific runtime identity.
- Remove the binding and preview again. The collection grant should disappear.
- Grant `graph:use qa-vendor-risk` without `collection:use qa-control-panel-smoke`. Graph runtime access should still fail because graph access requires both graph and backing-collection grants.
- Enter a typo such as `qa.uesr@example.com`. It should preview as a different principal or no useful grants, which confirms the operator must bind the corrected identity.

### Cleanup

Remove temporary bindings, permissions, memberships, and roles from `Access`. The shipped UI does not expose principal deletion, so leave mistakenly created principals unused and document them in the QA notes.

## Routine 2: Knowledge Base Collection Ingest

**Risk: Changes live behavior**

Use this routine to confirm collection creation, document ingest, upload handling, health reporting, reindexing, deletion, and empty-collection cleanup.

### Steps

1. Open `Collections`.
2. Enter `qa-control-panel-smoke` in `Collection ID`.
3. Click `Create Collection`.
4. Confirm `qa-control-panel-smoke` appears in `Available Collections`.
5. In `Add Documents`, stay on `Host Paths`.
6. Enter one or more server-visible absolute paths, such as a small repo markdown or CSV file the app container can read.
7. Click `Ingest Host Paths`.
8. Review the action result for `ingested_count`, `skipped_count`, `failed_count`, `missing_paths`, and file-level outcomes.
9. Switch to `Upload Files`.
10. Upload a small `.txt`, `.md`, `.csv`, `.docx`, `.pdf`, `.xls`, or `.xlsx` file.
11. Switch to `Upload Folder`.
12. Upload a folder that contains duplicate basenames in different folders, such as `alpha/same.txt` and `beta/same.txt`.
13. Switch to `Sync Configured Sources` and click `Sync Configured Sources` only if the selected collection should receive configured KB sources.
14. Review `Documents`.
15. Select each uploaded document and review `Document Viewer`.
16. Open the `Raw` tab for a text-like source and confirm raw source text is visible when the original file still exists.
17. Expand `Collection Inspector`.
18. Confirm table names, embedding provider, embedding model, configured vector dimension, actual vector dimensions, source-type counts, and graph count.
19. Review `Collection Health`.
20. If health reports duplicate or drift issues, click `Repair Collection` and confirm health returns to `Healthy`.
21. Select one document and click `Reindex`.
22. Confirm the document updates or reports a clear source-path error if the original source no longer exists.
23. Delete one test document and confirm it disappears from `Documents`.

### Expected Results

- Explicit collection creation works before any ingest succeeds.
- Host-path ingest reports file-level outcomes instead of a generic failure.
- Browser file upload ingests selected files and creates document rows.
- Folder upload preserves folder-relative paths, so duplicate basenames remain distinguishable.
- `Collection Inspector` shows storage metadata and vector dimension warnings when applicable.
- `Collection Health` reports duplicate or drift issues and shows a successful repair result when repair is available.
- `Reindex` succeeds only when the original source path is still available.

### Live Chat Checks

Run these after at least one document is indexed:

```text
What knowledge bases do I have access to?
```

```text
List documents in qa-control-panel-smoke.
```

```text
Summarize the documents in qa-control-panel-smoke and cite your sources.
```

Expected result: inventory prompts should mention the QA collection and document titles. Grounded prompts should cite the indexed QA files.

### Negative Checks

- Upload an unsupported file type and confirm the action result names the failed file or unsupported suffix.
- Upload two files named `same.txt` through folder upload from different folders and confirm both folder-relative paths remain visible.
- Enter a host path the backend cannot see and confirm `missing_paths` reports it.
- Try `Delete Empty` while documents still exist. The UI should block deletion or the backend should return a conflict.

### Cleanup

Delete QA documents. If no graph references the collection, click `Delete Empty` for `qa-control-panel-smoke`.

## Routine 3: Managed Graph Ingest

**Risk: Changes live behavior**

Use this routine to confirm graph draft creation, validation, build, refresh, prompt tuning, graph-bound skills, and query-ready status.

### Steps

1. Make sure `qa-control-panel-smoke` exists and contains at least one indexed document.
2. Open `Graphs`.
3. Set `Graph Display Name` to `QA Vendor Risk Graph`.
4. Set `Graph ID` to `qa-vendor-risk`.
5. Set `Graph Collection` to `qa-control-panel-smoke`.
6. Keep `Use Entire Collection` selected for the first pass.
7. Confirm the workspace reports the number of indexed documents that will be used.
8. Optionally switch to `Choose Documents` and select a subset when testing manual source selection.
9. In `Graph Prompt Overrides`, enter:

   ```json
   {"extract_graph.txt":"Prioritize vendor names, approval chains, mitigation controls, and exception owners."}
   ```

10. In `Graph Config Overrides`, enter:

   ```json
   {"extract_graph":{"entity_types":["vendor","risk","approval","control"]}}
   ```

11. Click `Save Draft`.
12. Confirm the graph appears in the graph list.
13. Click `Validate`.
14. Review `Graph Inspector` for runtime validation status and warnings.
15. Click `Build`.
16. Confirm `Graph Inspector` reaches `Query Ready`.
17. Click `Refresh` and confirm a refresh run appears in `Graph Runs`.
18. In `Research & Tune`, enter guidance:

   ```text
   Prioritize supplier ownership, approval chains, mitigation controls, exception workflows, and source traceability.
   ```

19. Click `Run Research & Tune`.
20. Review the scratchpad preview and prompt drafts.
21. Click `Apply Selected Prompts` only if the prompt drafts match the goal.
22. In `Bound Graph Skill IDs`, enter a known active skill id if one should be bound.
23. In `Graph Overlay Skill Markdown`, enter a graph-specific overlay such as:

   ```markdown
   # QA Vendor Graph Overlay
   agent_scope: rag

   ## Workflow

   - Prefer approval-chain and control-owner relationships when this graph is selected.
   - Resolve graph findings back to source documents before final synthesis.
   ```

24. Click `Save Skill Overlay`.
25. Reopen `Graph Inspector` and confirm the graph remains query ready and shows bound skills.

### Expected Results

- Brand-new empty collections appear in the graph collection dropdown.
- `Save Draft` creates the graph without building it.
- `Validate` reports runtime readiness and source issues without building.
- `Build` creates a query-ready graph when source documents and backend requirements are satisfied.
- `Refresh` creates a new run using previously recorded graph sources.
- Research tuning previews prompt changes before applying them.
- Graph skill overlay save creates or updates a graph-scoped skill reference.

### Live Chat Check

Run this after the graph is query ready:

```text
Search the vendor-risk graph for approval-chain dependencies and cite evidence.
```

Expected result: the router should prefer graph-backed retrieval when graph search is enabled and the graph is accessible. The answer should include relationship-style findings and source-grounded evidence.

### Negative Checks

- Click `Build` before any source documents are available. The UI or backend should report a source/readiness error.
- Save malformed JSON in prompt or config overrides and confirm validation catches it.
- Grant only `graph:use qa-vendor-risk` without granting `collection:use qa-control-panel-smoke`; runtime graph use should fail.
- Remove the backing collection documents and confirm graph refresh reports a clear source problem.

### Cleanup

Remove temporary graph skill overlays or bound skill ids if they were only for QA. Delete source documents before deleting the backing collection. If a graph keeps the collection from being deleted, record the graph id and clear the graph relationship through the supported admin path for your environment.

## Routine 4: Agents And Skills

**Risk: Changes live behavior**

Use this routine to confirm agent overlays, agent reload, skill preview, skill CRUD, activation, deactivation, and dependency-blocker handling.

### Agent Steps

1. Open `Agents`.
2. Select `general`.
3. Review `Available Agents`, `Tool Catalog`, and the current agent detail payload.
4. Make a small, reversible editor change such as adding a QA note to `metadata` or adjusting a non-production QA overlay field.
5. Click `Save Overlay`.
6. Click `Reload Agents`.
7. Confirm the reload succeeds in the page or in `Operations`.
8. Reopen the agent and confirm overlay state is visible.

### Agent Expected Results

- `Save Overlay` writes an overlay but does not swap the running registry by itself.
- `Reload Agents` validates the merged registry and swaps it only on success.
- Failed reloads keep the last good registry active.

### Skill Steps

1. Open `Skills`.
2. Open `Preview`.
3. Enter:

   ```text
   help me test control-panel graph permissions and readiness
   ```

4. Click `Preview Match`.
5. Confirm existing matching skills appear without changing runtime behavior.
6. Open `Editor`.
7. Click `New Skill`.
8. Paste the `Control Panel QA Readiness Review` skill from this document.
9. Click `Create Skill`.
10. Select the created skill and note its `Skill Status` values, especially `Family`.
11. Click `Deactivate`.
12. Confirm status changes to `archived`.
13. Run `Preview Match` again and confirm the deactivated skill no longer appears.
14. Click `Activate`.
15. Confirm status returns to `active`.
16. Edit the body with a small additional QA instruction and click `Update Skill`.
17. Confirm a new active version is shown and the family id remains stable.

### Skill Expected Results

- Preview is read-only.
- Created skills default to the expected visibility and lifecycle for runtime-authored skills.
- Deactivated skills are hidden from active retrieval.
- Reactivated skills appear in preview again.
- Updates preserve the family relationship through `version_parent`.

### Negative Checks

- Deactivate a skill with active dependents in a safe test environment. The page should show a dependency blocker rather than archiving it.
- Create a skill with malformed metadata and confirm creation or activation reports validation detail.
- Add a permission for the current skill id instead of the family id, then preview effective access. The guide should catch that runtime skill access must use the family id.

### Cleanup

Deactivate temporary QA skills when the test is complete, or keep the QA readiness skill active only if it is intentionally part of the test fixture. Restore any agent overlay fields changed for the run and reload agents again.

## Routine 5: Tools And MCP Connections

**Risk: Changes live behavior**

Use this routine when `MCP_TOOL_PLANE_ENABLED=true` and your environment has a safe Streamable HTTP MCP test server available. If the tool plane is disabled, record the section as not applicable and verify that the control panel communicates the disabled state clearly.

### Steps

1. Open `Tools`.
2. Create a connection with a QA display name such as `QA Readiness Tools`.
3. Enter the test MCP server URL.
4. Choose the correct auth type for the server. Use `none` only for local test servers that do not require secrets.
5. Set allowed agents to include `general`.
6. Click the connection create action.
7. Click `Test Connection`.
8. Confirm the connection health result is successful or reports an actionable error.
9. Click `Refresh Tools`.
10. Confirm discovered tools appear in the tool catalog.
11. Select a harmless read-only tool.
12. Set `enabled=true`, `read_only=true`, `destructive=false`, and `background_safe=true` when those flags match the tool behavior.
13. If deferred discovery is enabled, set a useful `search_hint`.
14. Open `Access`.
15. Add `tool` + `use` permission for the exact registry name, such as `mcp__qa_readiness_tools__status`.
16. Preview effective access for `qa.user@example.com`.
17. Confirm the tool grant appears.
18. Open `Agents` and confirm the selected agent allows the tool through a direct selector or `mcp__*`.

### Expected Results

- The connection is stored with the selected visibility and owner context.
- `Test Connection` returns health status.
- `Refresh Tools` caches remote tool metadata into the runtime tool catalog.
- Tool flags are reflected in the selected tool row.
- Effective access includes the exact tool selector when RBAC is enabled.
- Runtime tool availability is still clipped by both the user's grant and the selected agent's allowed tools.

### Live Chat Check

Run a prompt that should require the QA MCP tool, using the exact capability exposed by your test server:

```text
Use the QA readiness status tool to check whether the control-panel smoke setup is available.
```

Expected result: the agent either calls the allowed read-only tool directly or discovers it through deferred tool discovery, depending on the tool flags and agent setup.

### Negative Checks

- Disable the MCP connection and confirm the tool is no longer callable.
- Remove the `tool:use` permission and confirm runtime access is denied when RBAC is enabled.
- Remove the tool from the agent allow-list and confirm the grant alone does not expose it.
- Mark a tool as destructive in a read-only flow and confirm it is not treated as background safe.

### Cleanup

Disable or remove the QA connection through the supported admin path for your environment. Remove temporary tool permissions and bindings from `Access`.

## Proposed Tool Fixture: `qa_readiness_report`

This tool is a proposed implementation target for future work. It is documented here as a test fixture so operators can validate skill matching, permissions, and expected runtime behavior before the Python tool exists.

### Tool Contract

| Field | Value |
| --- | --- |
| Name | `qa_readiness_report` |
| Group | `utility` |
| Purpose | Inspect whether a control-panel QA setup is usable by a target user. |
| Read only | `true` |
| Background safe | `true` |
| Requires workspace | `false` |
| Destructive | `false` |

### Inputs

| Input | Type | Required | Meaning |
| --- | --- | --- | --- |
| `user_email` | string | yes | Email to preview through effective access. |
| `collection_id` | string | no | Collection to check for readiness and grants. |
| `graph_id` | string | no | Graph to check for query readiness and grants. |
| `skill_family_id` | string | no | Skill family to check for status and grants. |
| `tool_name` | string | no | Tool registry name to check for catalog visibility and grants. |
| `include_samples` | boolean | no | Include sample chat prompts and next-step suggestions. |

### Output Shape

```json
{
  "object": "qa_readiness_report",
  "user_email": "qa.user@example.com",
  "summary": {
    "status": "pass",
    "pass_count": 7,
    "warn_count": 1,
    "fail_count": 0
  },
  "checks": [
    {
      "status": "pass",
      "area": "collection",
      "message": "Collection qa-control-panel-smoke exists and has indexed documents."
    },
    {
      "status": "warn",
      "area": "skill",
      "message": "Skill family control-panel-qa-readiness exists but is archived."
    }
  ],
  "next_actions": [
    "Open Skills and activate control-panel-qa-readiness.",
    "Preview effective access for qa.user@example.com before live chat testing."
  ],
  "sample_prompts": [
    "List documents in qa-control-panel-smoke.",
    "Run a QA readiness report for qa.user@example.com, collection qa-control-panel-smoke, graph qa-vendor-risk, skill family control-panel-qa-readiness, and tool qa_readiness_report."
  ]
}
```

### Implementation Note

When implemented, add the tool to the in-repo utility tool group so it follows the existing `ToolDefinition` path:

- tool implementation: `src/agentic_chatbot_next/tools/qa_readiness.py`
- tool group binding: `src/agentic_chatbot_next/tools/groups/utility.py`
- registry definition: `src/agentic_chatbot_next/tools/registry.py`
- docs update: `docs/TOOLS_AND_TOOL_CALLING.md`
- focused tests: tool policy, access preview handling, and utility group binding

The implementation should read existing stores and services only. It should not create, repair, reindex, delete, activate, deactivate, or reload anything.

## Proposed Skill Fixture: Control Panel QA Readiness Review

Paste this into `Skills` -> `Editor` -> `New Skill` when you want a QA skill fixture.

```markdown
---
skill_id: control-panel-qa-readiness
name: Control Panel QA Readiness Review
kind: retrievable
agent_scope: general
tool_tags: qa_readiness_report, list_indexed_docs, inspect_graph_index, search_skills
task_tags: control-panel, qa, readiness, rbac, graph, skill, tool
version: 1
enabled: true
description: Guide an operator through control-panel readiness checks for collections, graphs, skills, tools, and user grants.
keywords: control panel qa, readiness report, rbac smoke test, graph permission test, skill tool access
when_to_apply: Use when a user asks whether a control-panel test setup is ready for live chat or asks for a QA smoke-test diagnosis.
avoid_when: Avoid using this for normal application answers that do not involve control-panel resources or operator validation.
examples: qa readiness report, graph permission smoke test, control panel test setup
---
# Control Panel QA Readiness Review

## Workflow

Ask for any missing ids before diagnosing readiness: user email, collection id, graph id, skill family id, and tool name.

If `qa_readiness_report` is available, call it first with the supplied ids and `include_samples=true`.

If the report is not available, use inventory tools to approximate the checks:

- use `list_indexed_docs` for collection and document inventory
- use `inspect_graph_index` for graph readiness
- use `search_skills` for skill-family or skill-guidance discovery

Explain results as `pass`, `warn`, and `fail` checks. For each failed check, name the control-panel page where the operator should fix it.

Never claim that a user can query a graph unless both graph access and backing collection access are present.

For skill access, prefer the skill family id over the current version id.

End with one or two live chat prompts the operator can run to confirm behavior.
```

### Skill Preview Query

Use this in `Skills` -> `Preview`:

```text
Preview the best skill for a control-panel graph permission smoke test.
```

Expected result: `Control Panel QA Readiness Review` should rank near the top once active.

## End-To-End Smoke Script

Run this compact sequence when you need one full pass instead of every detail above.

1. Unlock the control panel and confirm `Dashboard` health.
2. Create collection `qa-control-panel-smoke`.
3. Upload two small files, including duplicate basenames through folder upload.
4. Confirm document rows, viewer content, raw text, inspector metadata, and collection health.
5. Create graph `qa-vendor-risk` against `qa-control-panel-smoke`.
6. Save draft, validate, build, and confirm `Query Ready`.
7. Create or activate `Control Panel QA Readiness Review`.
8. Create `qa.user@example.com`, a QA group, roles, permissions, memberships, and bindings for:
   - `collection:use qa-control-panel-smoke`
   - `graph:use qa-vendor-risk`
   - `skill_family:use control-panel-qa-readiness`
   - `tool:use qa_readiness_report` if the proposed tool exists
9. Preview effective access for `qa.user@example.com`.
10. Open Open WebUI and run the live chat prompts below.
11. Open `Operations` and confirm audit events and recent jobs look expected.
12. Clean up temporary documents, graph bindings, permissions, roles, memberships, and inactive QA fixtures.

## Live Chat Prompt Pack

Use these prompts after the matching routine has completed.

### Access and inventory

```text
What knowledge bases do I have access to?
```

```text
List documents in qa-control-panel-smoke.
```

### Grounded KB behavior

```text
Summarize the documents in qa-control-panel-smoke and cite your sources.
```

### Graph behavior

```text
Search the vendor-risk graph for approval-chain dependencies and cite evidence.
```

### Proposed tool behavior

```text
Run a QA readiness report for qa.user@example.com, collection qa-control-panel-smoke, graph qa-vendor-risk, skill family control-panel-qa-readiness, and tool qa_readiness_report.
```

### Skill behavior

```text
Preview the best skill for a control-panel graph permission smoke test.
```

## Final Cleanup Checklist

- Delete QA documents from `qa-control-panel-smoke`.
- Delete the empty QA collection when it has no documents or graph bindings.
- Remove QA graph skill overlays or bound skill ids if they are not permanent fixtures.
- Deactivate temporary QA skills, or explicitly record why they remain active.
- Remove QA permissions, bindings, memberships, and roles.
- Disable QA MCP connections or remove them through the supported admin path.
- Reopen `Operations` and confirm the final audit trail matches the cleanup actions.
