#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_repo_root
require_command podman

OPENWEBUI_VERSION="$(env_value OPENWEBUI_VERSION v0.8.6)"

echo "Building application image from $REPO_ROOT"
podman build \
  -t localhost/agentic-chatbot-v3-app:latest \
  -f "$REPO_ROOT/Dockerfile" \
  "$REPO_ROOT"

echo "Building Open WebUI image with OPENWEBUI_VERSION=$OPENWEBUI_VERSION"
podman build \
  -t localhost/agentic-chatbot-v3-openwebui:latest \
  -f "$REPO_ROOT/deployment/openwebui/image/Dockerfile" \
  --build-arg "OPENWEBUI_VERSION=$OPENWEBUI_VERSION" \
  "$REPO_ROOT"

echo "Images built:"
podman images localhost/agentic-chatbot-v3-app localhost/agentic-chatbot-v3-openwebui
