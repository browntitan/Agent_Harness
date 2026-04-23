# Manager Overview

## Who This Is For

Leads, managers, and reviewers who need to understand what the control panel governs, what can change live behavior, and when to ask for a deeper technical review.

## What You Can Do Here

- Understand the role of the control panel in the live agent system.
- Separate low-risk inspection tasks from higher-risk operational changes.
- Set expectations for review habits and change governance.

## What To Read Next

- Read [Getting Started](getting-started.md) for access and first-use guidance.
- Read [Section Reference](section-reference.md) for the current capabilities of each page.
- Read [Technical Reference](technical-reference.md) when you need implementation detail behind a behavior.

## What The Control Panel Controls

The control panel gives local administrators a single place to:

- review runtime health and recent reload activity
- inspect collections, documents, agents, prompts, skills, and jobs
- draft and apply configuration overrides
- save prompt and agent overlays
- ingest, reindex, and remove collection documents

It does not introduce full end-user administration, tenant self-service, or role-based access control. In the current design, it is a trusted local-admin tool protected by a shared admin token.

## What Changes Live Behavior

Use these labels when reviewing or approving actions.

| Label | Meaning | Typical examples |
| --- | --- | --- |
| `Low risk` | Informational or reversible with little operational impact. | Viewing the dashboard, reviewing collections, reading operations history |
| `Changes live behavior` | Affects future runtime behavior or visible outputs. | Applying a config change, reloading agents, editing prompts |
| `Needs careful review` | Can remove data, break runtime behavior, or affect production results. | Deleting documents, changing provider/model settings, editing agent tool access |

## Informational vs Mutating Actions

| Action type | What it does | Review expectation |
| --- | --- | --- |
| Informational | Reads current state without changing files or runtime behavior. | Safe for routine review |
| Drafting | Prepares a change without making it live yet. | Good checkpoint for review |
| Applying or reloading | Makes a live change visible to future requests. | Should be deliberate and documented |
| Destructive cleanup | Removes data or overrides. | Should be approved and recoverable first |

In the shipped UI, the following pages are primarily informational:

- `Dashboard`
- `Operations`
- most of `Collections` when used for viewing

The following pages can directly change live behavior:

- `Config`
- `Agents`
- `Prompts`
- `Collections`
- `Skills`

## Governance Habits To Recommend

- Ask operators to validate before they apply any config change.
- Treat provider changes, model changes, and tool/agent permission edits as review-worthy changes.
- Expect operators to check `Operations` after a change so they can confirm the reload summary and audit trail.
- Require masked screenshots or screen shares whenever someone demonstrates the control panel to others.
- Keep a habit of documenting why a change was made, not just what button was pressed.

## Where To Request Technical Review

Request technical review when a change:

- affects provider selection, model routing, or judge behavior
- changes agent tools, worker access, or step limits
- edits prompts that influence answer quality or safety
- removes documents from an actively used collection
- fails validation or produces a reload error

> Technical note: the current control panel writes overlays instead of changing repo-authored defaults. That makes review easier, but it still means live behavior can change immediately after `Apply` or `Reload Agents`.

## Manager Checklist

Before approving a change, confirm:

- the operator can explain the goal in plain language
- the affected section and action are understood
- the expected live effect is clear
- the rollback or recovery path is known
- the operator plans to verify the result in `Dashboard` or `Operations`
