# Control Panel

The v3 control panel is built into the app image and served by FastAPI at:

`http://127.0.0.1:18000/control-panel`

It is not the chat UI. Chat happens in Open WebUI at:

`http://127.0.0.1:3001`

## Start

Use the repo-root Docker stack:

```bash
cd /Users/shivbalodi/Desktop/Rag_Research/agentic_chatbot_v3
docker compose up -d --build
```

The panel uses `CONTROL_PANEL_ADMIN_TOKEN` from `.env`.

## What It Manages

- effective runtime configuration
- agents and prompts
- collections and KB health
- managed GraphRAG indexes
- skill packs
- access control
- operational health and bootstrap status

## Local Panel Development

For panel-only development, run the Vite dev server from this directory and point it at the Dockerized API:

```bash
npm install
APP_API_HOST=127.0.0.1 npm run dev -- --host 0.0.0.0 --port 4174
```

The production v3 path remains the Dockerized app-hosted panel.
