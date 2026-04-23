# Getting Started

The v3 control panel is served by the Dockerized FastAPI app:

`http://127.0.0.1:18000/control-panel`

Open WebUI is the chat UI:

`http://127.0.0.1:3001`

Start the full stack from the repo root:

```bash
docker compose up -d --build
```

Use `CONTROL_PANEL_ADMIN_TOKEN` from `.env` to unlock the panel.
