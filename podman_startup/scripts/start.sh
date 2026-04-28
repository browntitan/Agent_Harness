#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_command podman
require_command systemctl
require_command curl

APP_API_PORT="$(env_value APP_API_PORT 18000)"
OPENWEBUI_PORT="$(env_value OPENWEBUI_PORT 3001)"

echo "Starting network and persistent services"
systemctl --user start "${INFRA_SERVICES[@]}"
systemctl --user start rag-postgres.service
systemctl --user start ollama.service

wait_for_container_health rag-postgres 180
wait_for_container_health ollama 300

echo "Running Ollama model bootstrap"
systemctl --user restart ollama-bootstrap.service

echo "Running application bootstrap"
systemctl --user restart app-bootstrap.service

echo "Starting API"
systemctl --user start app.service
wait_for_http "http://127.0.0.1:${APP_API_PORT}/health/ready" 240

echo "Starting Open WebUI"
systemctl --user start openwebui.service
wait_for_http "http://127.0.0.1:${OPENWEBUI_PORT}/" 300

echo "Running Open WebUI bootstrap"
systemctl --user restart openwebui-bootstrap.service

echo "Startup complete"
echo "Open WebUI:     http://127.0.0.1:${OPENWEBUI_PORT}"
echo "Control panel:  http://127.0.0.1:${APP_API_PORT}/control-panel"
