#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_command systemctl
require_command podman

systemctl --user --no-pager --full status "${ALL_SERVICES[@]}" || true
echo
podman ps --all --filter 'name=rag-postgres|ollama|app|openwebui'
