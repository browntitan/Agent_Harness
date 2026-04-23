# Demo KB Packs

`data/kb/` is demo seed content, not the primary operational source of truth.

## Operational model

- operational corpus: PostgreSQL
- demo seed corpus: `data/kb/`
- workspace files: `data/workspaces/`
- runtime session/job state: `data/runtime/`

The next-runtime cutover did not change the KB storage model. It changed how live AGENT
turns execute around that data.

## When to use `data/kb`

Use the bundled demo KB when you want:

- local demos
- regression scenarios
- starter content for UI testing

Do not assume those files are already indexed in PostgreSQL until you explicitly sync them.

## Common commands

```bash
python run.py sync-kb --collection-id default
python run.py reindex-document <path>
```

`init-kb` still exists as a deprecated compatibility command, but the live operator path is
`sync-kb`.

Automatic startup seeding is still supported:

```env
SEED_DEMO_KB_ON_STARTUP=true
```

## Key distinction

The demo KB is still content.

The current runtime durability features such as transcripts, jobs, and notifications live
under `data/runtime/*`, not under `data/kb/`.

The defense benchmark corpus follows a different operator path. It is intended to be indexed into
its own persistent collection with:

```bash
python run.py sync-defense-corpus
python run.py evaluate-defense-corpus --sync-first
```

Like the rest of the live RAG stack, that benchmark is collection-first and index-first. The
runtime does not query raw corpus folders directly at answer time.
