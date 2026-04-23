# Troubleshooting

## Panel Does Not Load

Check the Dockerized app:

```bash
docker compose ps app
docker compose logs -n 100 app
curl http://127.0.0.1:18000/health/live
```

The control panel is served at `http://127.0.0.1:18000/control-panel`.

## Login Fails

Confirm `CONTROL_PANEL_ADMIN_TOKEN` in `.env`, then restart the app:

```bash
docker compose restart app
```

## Collections Or Graphs Look Empty

Check app bootstrap:

```bash
docker compose logs -n 200 app-bootstrap
docker compose up -d --force-recreate app-bootstrap
```

The bootstrap service runs migrations, skill indexing, and KB sync.

## Open WebUI Model Is Missing

Rerun the Open WebUI bootstrap service:

```bash
docker compose up -d --force-recreate openwebui-bootstrap
docker compose logs -n 100 openwebui-bootstrap
```
