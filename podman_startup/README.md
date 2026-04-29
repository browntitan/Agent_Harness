# Rootless Podman Startup

This folder contains a rootless Podman Quadlet startup kit for the core Agentic Chatbot v3 stack on RHEL 9.7:

- PostgreSQL with pgvector
- GPU-backed Ollama
- Ollama model bootstrap
- FastAPI app bootstrap
- FastAPI app and control panel
- Open WebUI
- Open WebUI Enterprise Agent bootstrap

Langfuse, SeaweedFS, and dev-only services are intentionally not included.

## 1. Install Host Packages

Run this on the EC2 host:

```bash
sudo dnf install -y container-tools git curl jq
```

Install NVIDIA Container Toolkit if it is not already available through your base image or company repo:

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo dnf install -y nvidia-container-toolkit
sudo systemctl enable --now nvidia-cdi-refresh.path
sudo systemctl restart nvidia-cdi-refresh.service
```

Verify the driver and Podman GPU access:

```bash
nvidia-smi
nvidia-ctk cdi list

podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  docker.io/nvidia/cuda:12.9.0-base-ubuntu22.04 \
  nvidia-smi
```

Enable rootless services to run after logout/reboot:

```bash
sudo loginctl enable-linger "$USER"
```

If `systemctl --user` cannot connect to the user bus in a non-interactive shell, reconnect over SSH or run:

```bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
```

## 2. Prepare `.env`

From the repo root:

```bash
cp .env.example .env
```

If `.env` already exists, do not overwrite it. Add the Podman-specific values from:

```bash
podman_startup/env.podman.example
```

At minimum, set:

```env
OLLAMA_BASE_URL=http://ollama:11434
GRAPHRAG_BASE_URL=http://ollama:11434/v1
PG_DSN=postgresql://raguser:ragpass@rag-postgres:5432/ragdb
LANGFUSE_HOST=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

For browser use on EC2, replace `EC2_HOST` with the public DNS name or IP address:

```env
PUBLIC_AGENT_API_BASE_URL=http://EC2_HOST:18000
CONNECTOR_ALLOWED_ORIGINS=http://EC2_HOST:3001,http://EC2_HOST:18000,http://localhost:3001,http://localhost:18000
```

Rotate these before using the instance with other people:

```env
OPENWEBUI_ADMIN_PASSWORD=
OPENWEBUI_SECRET_KEY=
CONTROL_PANEL_ADMIN_TOKEN=
GATEWAY_SHARED_BEARER_TOKEN=
DOWNLOAD_URL_SECRET=
CONNECTOR_SECRET_API_KEY=
```

## 3. Build Images

The Open WebUI build needs the ONNX Runtime GPU artifact that `onnxruntime-node` normally downloads from GitHub during `npm install`. The build script downloads that artifact on the host first, verifies it with `tar`, then injects it into the Podman image build so the in-container `npm install` does not make that GitHub artifact request.

The default artifact is:

```text
podman_startup/vendor/onnxruntime-linux-x64-gpu-1.20.1.tgz
```

The build script passes these proxy defaults unless overridden by your shell or `.env`:

```env
HTTP_PROXY=http://contractorproxyeast.northgrum.com:80
HTTPS_PROXY=http://contractorproxyeast.northgrum.com:80
NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0,rag-postgres,ollama,app,openwebui,.northgrum.com,169.254.169.254,169.254.170.2
ONNX_RUNTIME_VERSION=1.20.1
ONNX_RUNTIME_ARTIFACT=onnxruntime-linux-x64-gpu-1.20.1.tgz
ONNX_RUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
```

From the repo root:

```bash
podman_startup/scripts/build-images.sh
```

This builds:

- `localhost/agentic-chatbot-v3-app:latest`
- `localhost/agentic-chatbot-v3-openwebui:latest`

To force a clean rebuild after a failed build attempt:

```bash
podman_startup/scripts/stop.sh || true
systemctl --user reset-failed || true

podman rm -f rag-postgres ollama ollama-bootstrap app-bootstrap app openwebui openwebui-bootstrap 2>/dev/null || true
podman rmi -f localhost/agentic-chatbot-v3-app:latest localhost/agentic-chatbot-v3-openwebui:latest 2>/dev/null || true
podman builder prune -af
podman image prune -af
```

To force the ONNX Runtime artifact to download again:

```bash
rm -f podman_startup/vendor/onnxruntime-linux-x64-gpu-1.20.1.tgz
```

## 4. Install Rootless Quadlets

From the repo root:

```bash
podman_startup/scripts/install-rootless.sh
```

The installer:

- renders Quadlet files into `~/.config/containers/systemd/`
- writes generated env files into `~/.config/agentic-chatbot-v3/env/`
- runs `systemctl --user daemon-reload`

The generated env files are outside the repo so secrets are not committed.

## 5. Start The Stack

```bash
podman_startup/scripts/start.sh
```

The startup order is:

1. Podman network
2. PostgreSQL and Ollama
3. Ollama model bootstrap
4. App bootstrap
5. FastAPI app
6. Open WebUI
7. Open WebUI Enterprise Agent bootstrap

The default models pulled into Ollama are:

```text
gpt-oss:20b
nomic-embed-text:latest
```

## 6. Verify

```bash
podman_startup/scripts/status.sh

curl http://127.0.0.1:18000/health/live
curl http://127.0.0.1:18000/health/ready

podman exec ollama nvidia-smi
podman exec ollama ollama list
```

Open the app:

```text
http://EC2_HOST:3001
http://EC2_HOST:18000/control-panel
```

Use `OPENWEBUI_ADMIN_EMAIL` and `OPENWEBUI_ADMIN_PASSWORD` from `.env` for Open WebUI.

Use `CONTROL_PANEL_ADMIN_TOKEN` from `.env` for the control panel.

## Logs, Stop, Restart

Follow all service logs:

```bash
podman_startup/scripts/logs.sh
```

Follow one service:

```bash
podman_startup/scripts/logs.sh app
podman_startup/scripts/logs.sh ollama-bootstrap
```

Stop everything:

```bash
podman_startup/scripts/stop.sh
```

Restart:

```bash
podman_startup/scripts/start.sh
```

## Troubleshooting

If a Quadlet service is missing after install:

```bash
systemctl --user daemon-reload
systemctl --user list-unit-files | grep -E 'rag-postgres|ollama|app|openwebui'
```

If the GPU is not visible inside Ollama:

```bash
nvidia-ctk cdi list
podman exec ollama nvidia-smi
```

If Open WebUI starts but the Enterprise Agent model is missing:

```bash
systemctl --user restart openwebui-bootstrap.service
podman_startup/scripts/logs.sh openwebui-bootstrap
```

If app bootstrap fails after changing `.env`:

```bash
podman_startup/scripts/install-rootless.sh
systemctl --user restart app-bootstrap.service
podman_startup/scripts/logs.sh app-bootstrap
```
