#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PODMAN_STARTUP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PODMAN_STARTUP_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
QUADLET_DIR="${QUADLET_DIR:-$XDG_CONFIG_HOME/containers/systemd}"
ENV_DIR="${ENV_DIR:-$XDG_CONFIG_HOME/agentic-chatbot-v3/env}"

INFRA_SERVICES=(
  agentic-network.service
  rag_postgres_data-volume.service
  ollama_data-volume.service
  openwebui_data-volume.service
  app_runtime-volume.service
  app_uploads-volume.service
  app_workspaces-volume.service
  app_cache-volume.service
  app_control_panel_data-volume.service
  graphrag_projects-volume.service
)

LONG_RUNNING_SERVICES=(
  rag-postgres.service
  ollama.service
  app.service
  openwebui.service
)

BOOTSTRAP_SERVICES=(
  ollama-bootstrap.service
  app-bootstrap.service
  openwebui-bootstrap.service
)

ALL_SERVICES=(
  "${INFRA_SERVICES[@]}"
  rag-postgres.service
  ollama.service
  ollama-bootstrap.service
  app-bootstrap.service
  app.service
  openwebui.service
  openwebui-bootstrap.service
)

require_repo_root() {
  [[ -f "$REPO_ROOT/Dockerfile" ]] || die "could not find repo Dockerfile at $REPO_ROOT"
  [[ -f "$REPO_ROOT/deployment/openwebui/image/Dockerfile" ]] || die "could not find Open WebUI Dockerfile"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required but was not found in PATH"
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

strip_quotes() {
  local value="$1"
  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi
  printf '%s' "$value"
}

env_value() {
  local key="$1"
  local default="${2:-}"
  local line value
  if [[ -f "$ENV_FILE" ]]; then
    line="$(grep -E "^[[:space:]]*$key=" "$ENV_FILE" | tail -n 1 || true)"
    if [[ -n "$line" ]]; then
      value="${line#*=}"
      value="$(trim "$value")"
      strip_quotes "$value"
      return 0
    fi
  fi
  printf '%s' "$default"
}

write_env_line() {
  local file="$1"
  local key="$2"
  local value="$3"
  printf '%s=%s\n' "$key" "$value" >> "$file"
}

copy_normalized_env_without() {
  local source="$1"
  local dest="$2"
  shift 2
  local excluded=" $* "
  local line key value

  : > "$dest"
  [[ -f "$source" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$(trim "$line")" ]] && continue
    [[ "$(trim "$line")" == \#* ]] && continue
    [[ "$line" == *"="* ]] || continue
    key="$(trim "${line%%=*}")"
    case "$key" in
      ''|*[!A-Za-z0-9_]*)
        continue
        ;;
    esac
    [[ "$excluded" == *" $key "* ]] && continue
    value="${line#*=}"
    value="$(trim "$value")"
    value="$(strip_quotes "$value")"
    write_env_line "$dest" "$key" "$value"
  done < "$source"
}

wait_for_container_health() {
  local container="$1"
  local timeout="${2:-180}"
  local deadline status
  deadline=$((SECONDS + timeout))

  while (( SECONDS < deadline )); do
    status="$(podman inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container" 2>/dev/null || true)"
    if [[ "$status" == "healthy" || "$status" == "running" ]]; then
      echo "$container is $status"
      return 0
    fi
    sleep 2
  done

  podman ps --all --filter "name=$container" || true
  die "$container did not become healthy within ${timeout}s"
}

wait_for_http() {
  local url="$1"
  local timeout="${2:-180}"
  local deadline
  deadline=$((SECONDS + timeout))

  while (( SECONDS < deadline )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$url is reachable"
      return 0
    fi
    sleep 2
  done

  die "$url did not become reachable within ${timeout}s"
}
