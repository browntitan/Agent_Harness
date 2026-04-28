#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_command systemctl

systemctl --user stop openwebui-bootstrap.service openwebui.service app.service app-bootstrap.service ollama-bootstrap.service ollama.service rag-postgres.service || true
echo "Stopped Agentic Chatbot v3 Podman services"
