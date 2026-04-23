# Agentic Chatbot v3

Docker-first Agentic RAG chatbot stack.

v3 is intentionally different from the old local-dev setup: Open WebUI is the chat surface, the control panel is served by the FastAPI app, and runtime state lives in Docker volumes.

## Stack

- FastAPI agent runtime on `http://127.0.0.1:18000`
- Control panel on `http://127.0.0.1:18000/control-panel`
- Open WebUI on `http://127.0.0.1:3001`
- Optional Langfuse on `http://127.0.0.1:3000`
- PostgreSQL plus pgvector
- Native macOS Ollama for chat, judge, embeddings, and Microsoft GraphRAG model calls
- Microsoft GraphRAG projects stored in the `graphrag_projects` Docker volume

## First Run

```bash
cd /Users/shivbalodi/Desktop/Rag_Research/agentic_chatbot_v3
cp .env.example .env
docker compose up -d --build
```

The first run can take a while because Docker builds the app and Open WebUI images.

For production-style Compose runs, rebuild after changing `src`, `data/agents`,
`data/prompts`, `data/router`, `requirements.txt`, `Dockerfile`, or Open WebUI
pipe/bootstrap files. `app-bootstrap` runs a runtime registry smoke check before
the API starts so stale images fail early instead of surfacing as a chat-time
Open WebUI `503`.

## Core Startup

Start the backend, control panel, Open WebUI, and Enterprise Agent registration:

```bash
cd /Users/shivbalodi/Desktop/Rag_Research/agentic_chatbot_v3
docker compose up -d --no-build
```

If the Enterprise Agent model does not appear in Open WebUI, rerun just the Open WebUI bootstrap:

```bash
docker compose up -d --force-recreate --no-deps --no-build openwebui-bootstrap
```

Langfuse is opt-in:

```bash
docker compose --profile observability up -d --no-build
```

## Hot Reload Development

Use the dev compose overlay when you are editing Python API/agent code or the control panel and do not want to rebuild the app image after every save:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build app control-panel-dev
```

After the first build, restart the dev stack without rebuilding:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build app control-panel-dev
```

Dev URLs:

- API with backend reload: `http://127.0.0.1:18000`
- Vite control panel with HMR: `http://127.0.0.1:4174`
- Static control panel served by FastAPI: `http://127.0.0.1:18000/control-panel`

In this mode, `./src` and `./run.py` are mounted into the app container and uvicorn watches `/app/src`. Control panel source is served by Vite with `/v1` and `/health` proxied to the app container. Rebuild the app image only when Docker-level inputs change, such as `requirements.txt`, `Dockerfile`, system packages, or production static assets.

## Health Checks

```bash
docker compose ps
curl http://127.0.0.1:18000/health/live
curl http://127.0.0.1:18000/health/ready
```

Expected URLs:

- API: `http://127.0.0.1:18000`
- Control panel: `http://127.0.0.1:18000/control-panel`
- Open WebUI: `http://127.0.0.1:3001`
- Langfuse: `http://127.0.0.1:3000`

## Bootstrap Flow

Compose starts the stack in this order:

1. `rag-postgres`
2. `app-bootstrap`, which runs migrations, skill indexing, KB sync, and `runtime-smoke --registry-only --json`
3. `app`, which serves the API and control panel after `/health/ready` is healthy
4. `openwebui` and `openwebui-bootstrap`, which install the Enterprise Agent pipe/model

## Default Credentials

Change these in `.env` before any non-local use:

- `CONTROL_PANEL_ADMIN_TOKEN`
- `GATEWAY_SHARED_BEARER_TOKEN`
- `OPENWEBUI_ADMIN_EMAIL`
- `OPENWEBUI_ADMIN_PASSWORD`
- `OPENWEBUI_SECRET_KEY`
- `LANGFUSE_INIT_USER_EMAIL`
- `LANGFUSE_INIT_USER_PASSWORD`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`

## Runtime Data

Source-controlled seed/config data lives in:

- `data/agents`
- `data/prompts`
- `data/skills`
- `data/skill_packs`
- `data/router`
- `data/kb`
- `data/demo`
- `docs`

Mutable runtime state is intentionally not committed. Docker owns it through named volumes:

- `rag_postgres_data`
- `ollama_data`
- `openwebui_data`
- `app_runtime`
- `app_uploads`
- `app_workspaces`
- `app_cache`
- `app_control_panel_data`
- `graphrag_projects`
- Langfuse Postgres, ClickHouse, and MinIO volumes

## Useful Commands

```bash
docker compose logs -f app
docker compose logs -f app-bootstrap
docker compose logs -f openwebui-bootstrap
docker compose logs -f langfuse-web
```

Reset containers but keep volumes:

```bash
docker compose down --remove-orphans
```

Full fresh reset:

```bash
docker compose down --remove-orphans -v
```

Rerun only app bootstrap:

```bash
docker compose up -d --force-recreate app-bootstrap
```

Rerun only Open WebUI bootstrap:

```bash
docker compose up -d --force-recreate openwebui-bootstrap
```

## Smoke Test

After `app` and `openwebui-bootstrap` are healthy:

1. Open `http://127.0.0.1:3001`.
2. Sign in with the Open WebUI admin credentials from `.env`.
3. Select the `Enterprise Agent` model.
4. Ask: `What knowledge base collections do we have access to?`
5. Open Langfuse and confirm traces are landing for model-backed turns.

## Development Notes

The Python package is still named `agentic_chatbot_next` under `src/agentic_chatbot_next` in this migration so v3 starts from a stable runtime. Rename the package only in a later, isolated refactor.

The old custom chat client and browser SDK example were intentionally not copied into v3. Open WebUI is the supported chat UI.
