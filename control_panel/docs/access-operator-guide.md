# Access Operator Guide

> Use the repo-root [`README.md`](../../README.md) for full stack bring-up, Docker commands,
> backend restart workflows, Open WebUI setup, and collection seeding. This guide assumes the
> backend and control panel are already running and focuses only on operating per-user access
> control safely.

## Who This Is For

Operators and admins who need to turn on RBAC, assign access to real users, validate the resolved
grants, and troubleshoot denials without reading backend code first.

## What You Can Do Here

- Turn the current RBAC model on safely.
- Understand exactly what the runtime protects today.
- Use the shipped `Access` page in the control panel without guessing what each field means.
- Roll out first-user access, expand to more users, and revoke access cleanly.
- Use the admin API only when the shipped UI does not expose the exact recovery action you need.

## What To Read Next

- Use the [Task Guide](task-guide.md) for the broader control-panel workflow catalog.
- Use the [Section Reference](section-reference.md) if you want a quick explanation of the `Access`
  page before following the runbook here.
- Use the [Technical Reference](technical-reference.md) when you need implementation detail about
  admin auth, overlays, or endpoint groups.
- Use the gateway reference at [`docs/OPENAI_GATEWAY.md`](../../docs/OPENAI_GATEWAY.md) for the full
  API contract.
- Use [Troubleshooting](troubleshooting.md) if the control panel itself is unavailable or the admin
  token fails before you reach the `Access` page.

## What This Access Model Controls Today

When `AUTHZ_ENABLED=true`, the runtime enforces per-user RBAC for protected runtime resources.

Current protected resource types:

- shared KB collections
- named graphs
- tools
- skill families

Current behavior operators must understand:

- authorization is deny-by-default for protected resources
- a user with no grants can still do normal non-protected chat, but cannot use protected KB
  collections, graphs, tools, or skill families
- the chat-scoped upload collection remains implicitly usable by the owning session, even when no
  ACL row exists for that upload collection
- graph access requires both a `graph:use` grant and a `collection:use` grant for the graph's
  backing collection
- tool access is the intersection of what the selected agent allows and what the user is granted
- skill access is bound to the skill family id (`version_parent`), not an individual skill version id
- skill mutation routes require either the control-panel admin token or a `skill_family:manage`
  grant
- the runtime resolves a fresh access snapshot on each request and on worker-job start, so
  revocations take effect on the next request rather than waiting for a long-lived session to expire

## What This Access Model Does Not Control Today

This is important because operators often over-assume what RBAC is doing.

The current model does not:

- provide standalone end-user authentication
- replace the need for a trusted upstream identity source
- implement Azure Entra ID sync yet
- ACL the static repo-authored prompt and skill markdown files on disk
- automatically grant access to a graph because the user can reach its backing collection
- automatically grant tool access because the user can reach a KB or graph
- require an ACL row for the session's own upload collection

## Before You Turn It On

### Required setup

1. Set `AUTHZ_ENABLED=true` in the repo-root `.env`.
2. Restart the API or local stack so the env-backed runtime setting is active.
3. Confirm the control panel is reachable and unlocked with `CONTROL_PANEL_ADMIN_TOKEN`.
4. Confirm your caller is forwarding a trustworthy user email.
5. Confirm the resources you want to protect already exist, especially collections and graphs.

### Trusted identity inputs

The runtime expects a trustworthy user email from the caller. Current supported inputs are:

- `X-User-Email`
- `X-OpenWebUI-User-Email`
- `metadata.user_email`
- top-level `userEmail`

If none of those are present, the runtime cannot map the request to a real user principal. When
RBAC is enabled, that usually means the user ends up with no protected-resource access.

### Recommended readiness checklist

Before you create any grants, gather these exact ids:

- target collection ids from `Collections`
- target graph ids and each graph's backing collection from `Graphs`
- target skill family ids from `Skills`
- target tool names from `Agents` -> `Tool Catalog`

The guide below uses those ids directly. Operators lose the most time when they know the display
name but not the real id.

## Concept Model

### Core objects

| Term | What it means in this system | Where operators use it |
| --- | --- | --- |
| Principal | A subject that can receive access. Today that is usually an email-backed user, or a placeholder group for future IdP sync. | `Principals` |
| Group | A principal of type `group` used to collect users under one shared access profile. | `Principals`, `Memberships` |
| Role | A reusable permission bundle. Roles do not do anything until they are bound to a principal. | `Roles` |
| Binding | A link from a role to a principal. Active bindings drive runtime access. | `Bindings` |
| Membership | A link from a group to one of its members. Memberships let a user inherit a group's role bindings. | `Memberships` |
| Permission | A single rule on a role, such as `collection:use default` or `tool:use search_documents`. | `Permissions` |
| Effective access | The fully resolved per-request access snapshot the runtime will use after principal lookup, group expansion, role expansion, and permission union. | `Effective Access` |

### Resolution flow

The runtime resolves access in this order:

`trusted email -> user principal -> optional group memberships -> active role bindings -> role permissions -> per-request access snapshot`

That final snapshot is what the runtime uses when deciding whether the user can:

- search a KB collection
- query a graph
- call a tool
- retrieve or manage a skill family

### Resource types and actions

| Resource type | Typical selector | Meaning of `use` | Meaning of `manage` |
| --- | --- | --- | --- |
| `collection` | a collection id such as `default` | Allow runtime use of that KB collection | Reserved for future admin-style collection management logic |
| `graph` | a graph id | Allow runtime use of that graph, but still only if the backing collection is also granted | Reserved for future graph management logic |
| `tool` | a tool name from `Tool Catalog` | Allow the tool only when the chosen agent also exposes it | Reserved for future tool-management logic |
| `skill_family` | the family id from `Skill Status` -> `Family` | Allow retrieval and use of that skill family | Allow skill mutation routes for that family |

### Family ids vs skill ids

This is the most common skill-related mistake.

- Skill permissions should target the family id, not the current version id.
- In the UI, use the value shown in `Skill Status` -> `Family`.
- If a skill is versioned, new versions continue to match the same family grant.

## Recommended Operating Conventions

These conventions prevent most first-rollout mistakes.

### Identity conventions

- Enter user emails in lowercase form.
- Treat the normalized email as the canonical identity key.
- Do not create multiple principals for the same real person with different email casing.

### Naming conventions

- Name roles after business intent, not implementation detail.
- Good examples: `finance-kb-reader`, `ops-graph-analyst`, `platform-skill-manager`
- Avoid vague names such as `role1`, `default-access`, or `misc-users`.

### Granting conventions

- Start with exact resource ids before you consider `*`.
- Use groups only when multiple users truly share the same access profile.
- Prefer one focused role per access pattern instead of one huge “everything” role.
- Separate “can use this thing” from “can manage this thing”.
- Always preview effective access before asking a real user to test.

### Validation conventions

- Use the resource pages to find ids and check the high-level `RBAC Access` summaries.
- Use `Effective Access` as the source of truth for a specific user email.
- Validate with a real runtime request after the preview matches your expectations.

## The `Access` Page, Card By Card

This section tells you exactly what each card does in the shipped UI, what field matters, what
success looks like, and what your fastest recovery path is.

| Card | Use it for | Important fields | Main action | Success looks like | Fast recovery |
| --- | --- | --- | --- | --- | --- |
| `Principals` | Create email users and placeholder groups | `Principal Type`, `Provider`, `Email` or `Group Name` | `Save Principal` | The principal appears in the `Principals` list with the expected display value and provider | The shipped UI does not expose principal deletion. Create the corrected principal and stop using the mistaken one. |
| `Roles` | Create reusable permission bundles | `Role Name`, `Description` | `Save Role` | The role appears in `Roles` and can be selected in `Bindings` and `Permissions` | Delete the wrong role if nothing should use it, or move grants off it before deleting. |
| `Bindings` | Attach a role to a user or group | `Role`, `Principal` | `Add Binding` | The binding appears in the list with `Active` status | Use `Remove` to revoke that role from the principal. The shipped UI does not expose a disable toggle. |
| `Memberships` | Put users into groups | `Group`, `Member` | `Add Membership` | The membership appears with a timestamp | Use `Remove` to break the inheritance path. |
| `Permissions` | Add `use` or `manage` rules to roles | `Role`, `Resource Type`, `Action`, `Resource Selector` | `Add Permission` | The permission appears in the list with the correct selector | Use `Remove` to delete the wrong permission. |
| `Effective Access` | Preview the exact resolved access snapshot for an email | `User Email` | `Preview Access` | You see `Preview Email`, `Authz Enabled`, and the `Effective access snapshot` JSON | Change the email and preview again. No runtime mutation happens here. |

### Important UI details

- In `Permissions`, leaving `Resource Selector` at `*` creates a wildcard permission.
- In `Permissions`, `Skill Family` selectors come from the skill family id, not the currently
  selected skill version id.
- In `Bindings`, the subtitle says disabled bindings are ignored at runtime, but the shipped UI
  currently exposes `Remove`, not a disable toggle. Use the API backup flow if you need to keep a
  binding record but mark it disabled.
- In `Principals`, `Provider` currently offers `Email` and `Future Entra`. Only email-backed
  identity is active today.

## First Rollout Workflow

This is the safest order for a first deployment.

### 1. Inventory the resources you plan to protect

Before you create any RBAC records, collect the ids you will need.

Use these pages:

- `Collections` -> note the collection id
- `Graphs` -> note the graph id and its `Collection`
- `Skills` -> note `Skill Status` -> `Family`
- `Agents` -> `Tool Catalog` -> note the exact tool name

Success looks like:

- you know the exact selectors you plan to use
- you are not relying on display labels or memory

### 2. Create user principals

1. Open `Access`.
2. In `Principals`, set `Principal Type` to `User`.
3. Leave `Provider` as `Email`.
4. Enter the user's lowercase email in `Email`.
5. Click `Save Principal`.

Success looks like:

- the user appears in `Principals`
- the email is shown in normalized form

Fast recovery:

- if you mistype the email, create the corrected principal immediately and use that one in future
  bindings
- do not keep building roles and bindings on top of a known-bad principal

### 3. Create placeholder groups only if you actually need shared access

Only create a group when multiple users will share the same access profile.

1. In `Principals`, set `Principal Type` to `Group`.
2. Enter a clear shared label in `Group Name`.
3. Click `Save Principal`.

Good reasons to use a group:

- several finance analysts should inherit the same KB access
- several reviewers should inherit the same graph access
- you want a clean landing zone for future Entra group sync

Bad reasons to use a group:

- a single one-off user
- you have not decided the shared access pattern yet

### 4. Create roles before you create bindings

1. In `Roles`, enter a clear `Role Name`.
2. Add a human-readable `Description`.
3. Click `Save Role`.

Good examples:

- `finance-kb-reader`
- `defense-graph-analyst`
- `platform-skill-manager`

Success looks like:

- the role appears in `Roles`
- you can now choose it in `Bindings` and `Permissions`

### 5. Add permissions to the role

1. In `Permissions`, choose the target `Role`.
2. Choose the `Resource Type`.
3. Choose `Action` as `Use` or `Manage`.
4. Choose the `Resource Selector`.
5. Click `Add Permission`.

Examples:

- `collection` + `use` + `default`
- `graph` + `use` + `defense_rag_test_corpus`
- `tool` + `use` + `search_documents`
- `skill_family` + `manage` + `graph-research-pack`

Success looks like:

- the permission appears in the timeline list
- the selector value is exactly the one you intended

Fast recovery:

- use `Remove` on the incorrect permission
- add the corrected permission immediately after

### 6. Bind roles to users or groups

1. In `Bindings`, choose the `Role`.
2. Choose the target `Principal`.
3. Click `Add Binding`.

Success looks like:

- the binding appears in `Bindings`
- the status reads `Active`

Fast recovery:

- click `Remove` if the wrong role or principal was selected

### 7. Add memberships if you are using groups

If you created a group, connect users to it now.

1. In `Memberships`, choose the `Group`.
2. Choose the `Member`.
3. Click `Add Membership`.

Success looks like:

- the membership appears with a timestamp
- the group can now pass its role bindings down to that user

Fast recovery:

- click `Remove` on the incorrect membership

### 8. Preview effective access before live testing

1. In `Effective Access`, enter the target user's email in `User Email`.
2. Click `Preview Access`.
3. Review:
   - `Preview Email`
   - `Authz Enabled`
   - the `Effective access snapshot`

What you want to see:

- `Authz Enabled` shows `Yes`
- the snapshot includes the expected `principal_id` and `role_ids`
- the snapshot shows the expected resources under `collection`, `graph`, `tool`, and `skill_family`

What usually means something is wrong:

- `Authz Enabled` is `No`
- the snapshot has no `role_ids`
- the resource list is empty even though you thought you granted access

### 9. Validate with a real request

After the preview looks correct, validate with the real runtime path.

Recommended validation flow:

1. Sign in to Open WebUI as the target user, or send a direct API request that includes the same
   user email.
2. Ask a simple resource-awareness question such as `What knowledge base collections do we have access to?`
3. If you granted collection access, confirm the answer includes only the intended collections.
4. If you granted graph access, confirm graph operations succeed only after both the graph and its
   backing collection are granted.
5. If you removed access, confirm the next request reflects the revocation.

## Day-2 Workflows

These are the workflows operators use after the first rollout.

### Add a new user to an existing access profile

Use this when the role model is already correct and you are only onboarding another person.

1. Create the user in `Principals`.
2. If you are using direct user bindings, add the needed `Bindings`.
3. If you are using groups, add the `Membership`.
4. Run `Preview Access` for the user's email.

Prefer this over creating a new role when the access profile is already defined.

### Grant KB-only access

Use this when the user should search one or more shared collections but should not use graphs or
skill-management flows.

Required permission rows:

- `collection` + `use` + `<collection_id>`

Optional, only if intended:

- `skill_family` + `use` + `<family_id>` for skills you want that user to retrieve
- `tool` + `use` + `<tool_name>` if the intended workflow depends on an explicitly gated tool

Do not assume collection access implies graph access.

### Grant graph access correctly

Use this when the user should use a named graph.

Required permission rows:

- `collection` + `use` + `<backing_collection_id>`
- `graph` + `use` + `<graph_id>`

Important:

- granting only the graph is not enough
- granting only the collection is not enough if the user needs the graph

Always confirm the graph's backing collection in `Graphs` -> `Graph Inspector` before you create
the permission rows.

### Grant tool access

Use this when the user needs a specific protected tool.

Required permission row:

- `tool` + `use` + `<tool_name>`

Also confirm:

- the selected agent already exposes that tool in `Agents`

If the role grants the tool but the chosen agent does not allow it, the tool still will not run.

### Grant skill-family access

Use this when the user should retrieve or manage a runtime-authored skill family.

Required permission rows:

- for runtime use only: `skill_family` + `use` + `<family_id>`
- for mutation workflows: `skill_family` + `manage` + `<family_id>`

Optional combined pattern:

- give both `use` and `manage` when a user needs to both retrieve and maintain the same family

Always source `<family_id>` from `Skills` -> `Skill Status` -> `Family`.

### Revoke access safely

Choose the smallest change that solves the problem.

Use `Remove` on a binding when:

- one user or group should lose a role
- other principals should keep the role unchanged

Use `Remove` on a permission when:

- the role itself should lose one capability
- every principal bound to that role should lose that capability

Use `Delete` on a role when:

- the role is obsolete
- you have already removed or intentionally accepted the impact on all bindings

Current shipped-UI limitation:

- the `Bindings` card does not expose a disable toggle
- if you need a disabled binding instead of a removed binding, use the API backup flow

### Answer “who can use this?”

Use two levels of checking:

1. High-level resource summary on the resource page:
   - `Collections` -> `Collection Inspector` -> `RBAC Access`
   - `Graphs` -> `Graph Inspector` -> `RBAC Access`
   - `Skills` -> `Skill Status` -> `RBAC Access`
2. User-specific confirmation:
   - `Access` -> `Effective Access` -> `Preview Access`

Use the resource-page summary for a quick answer and `Effective Access` for the source-of-truth
answer about one specific user email.

## Recommended Starter Access Patterns

These are opinionated starting patterns that work well for first rollout.

### `No Protected Access`

Recommended grants:

- none

What the user can do:

- normal non-protected chat
- use the chat's own upload collection for that session

What the user still cannot do:

- use shared KB collections
- use named graphs
- use protected tools
- retrieve protected skill families

Common mistake to avoid:

- creating a special deny role for users who should simply have no grants

### `KB Reader`

Recommended grants:

- one permission row per allowed collection: `collection` + `use` + `<collection_id>`

What the user can do:

- search only the granted shared KB collections

What the user still cannot do:

- use graphs unless separately granted
- use protected tools unless separately granted
- mutate skills

Common mistake to avoid:

- assuming a collection grant automatically grants graphs or all tools connected to that KB

### `Graph Analyst`

Recommended grants:

- `collection` + `use` + `<backing_collection_id>`
- `graph` + `use` + `<graph_id>`

Optional additions:

- more `collection:use` rows for any other KB collections the analyst should browse
- `tool:use` rows only when a protected tool is also needed in your deployment

What the user can do:

- use the specific named graph
- reach the graph's backing corpus

What the user still cannot do:

- use other graphs
- use unrelated collections
- manage skill families unless separately granted

Common mistake to avoid:

- granting only `graph:use` and forgetting the required `collection:use` grant for the backing collection

### `Skill Author/Manager`

Recommended grants:

- `skill_family` + `use` + `<family_id>`
- `skill_family` + `manage` + `<family_id>`

Optional broader admin pattern:

- use `*` only for tightly controlled platform operators who intentionally manage many families

What the user can do:

- retrieve that skill family at runtime
- use the skill mutation routes for that family

What the user still cannot do:

- manage unrelated skill families unless granted
- use protected collections, graphs, or tools that the skill may reference unless those are also granted

Common mistake to avoid:

- granting permissions to the current `skill_id` instead of the stable `Family` value

## Troubleshooting

Use this section when the role model looks correct on paper but behavior is still wrong.

| Symptom | Most likely cause | What to check | Likely fix |
| --- | --- | --- | --- |
| User can chat but sees no KBs | No `collection:use` grants, or the request is missing trusted email | `Effective Access`, request headers or metadata, and the target collection id | Add the collection grant and confirm `X-User-Email` or equivalent is present |
| Graph queries fail | Missing `graph:use`, missing backing `collection:use`, or both | `Graph Inspector` for backing collection, then `Effective Access` | Add both grants for the same user or inherited role |
| Tool never runs | Missing `tool:use`, or the selected agent does not allow that tool | `Effective Access` plus `Agents` -> `Tool Catalog` and the agent's allowed tools | Grant the tool and confirm the chosen agent can expose it |
| Upload works but writing to a shared collection fails | The session upload collection is implicitly allowed, but the shared target collection is not | The explicit collection id used by the request and `Effective Access` | Add `collection:use` for the shared collection |
| Skill update is denied | No admin token and no `skill_family:manage` grant | Whether the caller is the control panel admin or a runtime user, plus the family id | Use `X-Admin-Token` or grant `skill_family:manage` on the correct family |
| New skill version behavior is confusing | The operator granted a version id instead of the family id, or is expecting version-specific ACL behavior | `Skills` -> `Skill Status` -> `Family`, plus the role's permission selector | Use the family id in the permission row |
| `Preview Access` shows `Authz Enabled` = `No` | RBAC is not enabled in runtime settings or the service was not restarted after setting the env | `.env`, restart status, and the preview payload | Set `AUTHZ_ENABLED=true` and restart the backend |
| A user loses access only after the next request | This is expected | Request timing | No fix needed; per-request refresh is the intended behavior |

## API Backup Appendix

Use the UI first. Use the API only when the shipped page does not expose the exact action you need.

### Admin access endpoints

- `GET /v1/admin/access/principals`
- `POST /v1/admin/access/principals`
- `GET /v1/admin/access/memberships`
- `POST /v1/admin/access/memberships`
- `DELETE /v1/admin/access/memberships/{membership_id}`
- `GET /v1/admin/access/roles`
- `POST /v1/admin/access/roles`
- `DELETE /v1/admin/access/roles/{role_id}`
- `GET /v1/admin/access/bindings`
- `POST /v1/admin/access/bindings`
- `DELETE /v1/admin/access/bindings/{binding_id}`
- `GET /v1/admin/access/permissions`
- `POST /v1/admin/access/permissions`
- `DELETE /v1/admin/access/permissions/{permission_id}`
- `GET /v1/admin/access/effective-access?email=<user_email>`

### Trusted identity inputs for runtime requests

- `X-User-Email`
- `X-OpenWebUI-User-Email`
- `metadata.user_email`
- top-level `userEmail`

### Example effective-access preview request

```bash
curl \
  -H "X-Admin-Token: change-me-local-admin-token" \
  "http://127.0.0.1:18000/v1/admin/access/effective-access?email=alex@example.com"
```

Optional multi-tenant header:

```bash
-H "X-Tenant-ID: your-tenant-id"
```

### Example runtime validation request

Use this only when you want a direct API test instead of Open WebUI:

```bash
curl http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <gateway-token-if-required>" \
  -H "X-User-Email: alex@example.com" \
  -d '{
    "model": "enterprise-agent",
    "messages": [
      {"role": "user", "content": "What knowledge base collections do we have access to?"}
    ]
  }'
```

For the full gateway contract, request headers, and skill-mutation rules, use
[`docs/OPENAI_GATEWAY.md`](../../docs/OPENAI_GATEWAY.md).
