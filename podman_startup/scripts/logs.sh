#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_command journalctl

if [[ "${1:-}" == "" || "${1:-}" == "all" ]]; then
  journalctl --user -f \
    -u rag-postgres.service \
    -u ollama.service \
    -u ollama-bootstrap.service \
    -u app-bootstrap.service \
    -u app.service \
    -u openwebui.service \
    -u openwebui-bootstrap.service
else
  service="$1"
  [[ "$service" == *.service ]] || service="${service}.service"
  journalctl --user -f -u "$service"
fi
