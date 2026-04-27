# Local Docker Stack

v3 is Docker-first. Use this stack instead of running the API, Postgres, Ollama, or chat UI directly on the host.

## Start

```bash
cd /Users/shivbalodi/Desktop/Rag_Research/agentic_chatbot_v3
cp .env.example .env
docker compose up -d --build
```

Default URLs:

- API: `http://127.0.0.1:18000`
- Control panel: `http://127.0.0.1:18000/control-panel`
- Open WebUI: `http://127.0.0.1:3001`
- Langfuse: `http://127.0.0.1:3000` when started with the `observability` profile

## Core Restart

Use this for normal local work after images already exist:

```bash
cd /Users/shivbalodi/Desktop/Rag_Research/agentic_chatbot_v3
docker compose up -d --no-build
```

Use `--no-build` only when the app image already matches the source-controlled
runtime files. Rebuild after changing `src`, `data/agents`, `data/prompts`,
`data/router`, dependency/image files, or Open WebUI pipe/bootstrap files:

```bash
docker compose up -d --build app app-bootstrap openwebui-bootstrap openwebui
```

If Open WebUI loads but the `Enterprise Agent` model is missing, refresh the registration:

```bash
docker compose up -d --force-recreate --no-deps --no-build openwebui-bootstrap
```

Start Langfuse only when you need observability:

```bash
docker compose --profile observability up -d --no-build
```

## Dev Overlay

Use the dev overlay when you want Python hot reload and do not want to rebuild
the app image for every `src/` edit.

Start or switch into dev mode:

```bash
cd /Users/shivbalodi/Desktop/Rag_Research/agentic_chatbot_v3
docker compose down --remove-orphans
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build app app-bootstrap openwebui openwebui-bootstrap control-panel-dev
```

After that first build, you can bring the dev stack back without rebuilding:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build app openwebui control-panel-dev
```

In this mode:

- `./src` is mounted into `/app/src`
- `./run.py` is mounted into `/app/run.py`
- the API runs through uvicorn with `--reload`
- the control panel is served by Vite on `http://127.0.0.1:4174`

Use this workflow:

- `src/` changes: save and let uvicorn reload automatically
- `data/agents`, `data/prompts`, or `data/router` changes: restart `app`
- schema/bootstrap changes: rerun `app-bootstrap`
- dependency or image changes such as `requirements.txt` or `Dockerfile`: rebuild

Helpful commands:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart app
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --force-recreate --no-deps app-bootstrap
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build app app-bootstrap
```

Confirm that dev mode is active:

```bash
docker inspect agentic-chatbot-v3-app-1 --format '{{json .Config.Cmd}}'
docker inspect agentic-chatbot-v3-app-1 --format '{{range .Mounts}}{{println .Destination " <- " .Source}}{{end}}'
```

The running `app` container should show a uvicorn `--reload` command and a bind
mount for `/app/src`.

## Services

- `rag-postgres`: PostgreSQL with pgvector
- native macOS Ollama: model runtime reached from containers through `host.docker.internal`
- `ollama` / `ollama-bootstrap`: optional Dockerized model runtime behind the `docker-ollama` profile
- `app-bootstrap`: runs migrations, skill indexing, KB sync, and a runtime registry smoke check
- `app`: FastAPI gateway and control-panel static host, marked healthy only after `/health/ready` passes
- `openwebui`: chat UI
- `openwebui-bootstrap`: installs the Enterprise Agent pipe/model
- `langfuse-*`: optional local Langfuse observability stack behind the `observability` profile

## Verify

```bash
docker compose config --quiet
docker compose ps
curl http://127.0.0.1:18000/health/live
curl http://127.0.0.1:18000/health/ready
docker compose logs -n 100 app-bootstrap
docker compose logs -n 100 openwebui-bootstrap
```

## Smoke Tests

- In Open WebUI, select `Enterprise Agent` and ask `What knowledge base collections do we have access to?`
- Upload a small file in OpenWebUI and ask the agent to summarize it. OpenWebUI is only
  a byte transport in this stack; the file is first ingested into the agent document
  repository, and all retrieval/citations come from that repository.
- In the control panel, confirm collections, graphs, skills, and operations load.
- In Langfuse, confirm a model-backed turn appears after keys from `.env` are initialized.

## Reset

Stop containers while keeping state:

```bash
docker compose down --remove-orphans
```

Full fresh reset:

```bash
docker compose down --remove-orphans -v
```

The v3 runtime intentionally stores mutable state in Docker volumes, including uploaded files, workspaces, runtime traces, control-panel overlays, GraphRAG projects, Open WebUI data, and Langfuse stores.
