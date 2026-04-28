#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_repo_root
require_command podman
require_command systemctl

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  die "run this as the application user, not root; these are rootless user Quadlets"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  die "missing $ENV_FILE; copy .env.example to .env and edit it first"
fi

mkdir -p "$QUADLET_DIR" "$ENV_DIR"

RAG_DB_NAME="$(env_value RAG_DB_NAME ragdb)"
RAG_DB_USER="$(env_value RAG_DB_USER raguser)"
RAG_DB_PASSWORD="$(env_value RAG_DB_PASSWORD ragpass)"
RAG_DB_PORT="$(env_value RAG_DB_PORT 55432)"
OLLAMA_PORT="$(env_value OLLAMA_PORT 11434)"
APP_API_PORT="$(env_value APP_API_PORT 18000)"
OPENWEBUI_PORT="$(env_value OPENWEBUI_PORT 3001)"
DEFAULT_COLLECTION_ID="$(env_value DEFAULT_COLLECTION_ID default)"
GATEWAY_MODEL_ID="$(env_value GATEWAY_MODEL_ID enterprise-agent)"
GATEWAY_SHARED_BEARER_TOKEN="$(env_value GATEWAY_SHARED_BEARER_TOKEN change-me-openwebui-shared-token)"
OPENWEBUI_ADMIN_EMAIL="$(env_value OPENWEBUI_ADMIN_EMAIL admin@example.com)"
OPENWEBUI_ADMIN_PASSWORD="$(env_value OPENWEBUI_ADMIN_PASSWORD change-me-openwebui-admin)"
OPENWEBUI_ADMIN_NAME="$(env_value OPENWEBUI_ADMIN_NAME "Open WebUI Admin")"
OPENWEBUI_SECRET_KEY="$(env_value OPENWEBUI_SECRET_KEY change-me-openwebui-secret)"
PUBLIC_AGENT_API_BASE_URL="$(env_value PUBLIC_AGENT_API_BASE_URL "http://localhost:${APP_API_PORT}")"
OPENWEBUI_AGENT_REQUEST_TIMEOUT_SECONDS="$(env_value OPENWEBUI_AGENT_REQUEST_TIMEOUT_SECONDS 600)"
OPENWEBUI_COLLECTION_PREFIX="$(env_value OPENWEBUI_COLLECTION_PREFIX owui-chat-)"
OPENWEBUI_KB_COLLECTION_ID="$(env_value OPENWEBUI_KB_COLLECTION_ID "$DEFAULT_COLLECTION_ID")"
OPENWEBUI_THIN_MODE="$(env_value OPENWEBUI_THIN_MODE true)"
OPENWEBUI_ENABLE_HELPERS="$(env_value OPENWEBUI_ENABLE_HELPERS false)"
OPENWEBUI_ALLOW_FILE_BYTE_TRANSPORT="$(env_value OPENWEBUI_ALLOW_FILE_BYTE_TRANSPORT true)"
OPENWEBUI_BOOTSTRAP_TIMEOUT_SECONDS="$(env_value OPENWEBUI_BOOTSTRAP_TIMEOUT_SECONDS 180)"
OLLAMA_BOOTSTRAP_MODELS="$(env_value OLLAMA_BOOTSTRAP_MODELS "gpt-oss:20b,nomic-embed-text:latest")"

APP_ENV="$ENV_DIR/app.env"
POSTGRES_ENV="$ENV_DIR/postgres.env"
OLLAMA_ENV="$ENV_DIR/ollama.env"
OLLAMA_BOOTSTRAP_ENV="$ENV_DIR/ollama-bootstrap.env"
OPENWEBUI_ENV="$ENV_DIR/openwebui.env"
OPENWEBUI_BOOTSTRAP_ENV="$ENV_DIR/openwebui-bootstrap.env"

copy_normalized_env_without "$ENV_FILE" "$APP_ENV" \
  DATABASE_BACKEND VECTOR_STORE_BACKEND SKILLS_BACKEND PROMPTS_BACKEND \
  PG_DSN OLLAMA_BASE_URL GRAPHRAG_BASE_URL LANGFUSE_HOST LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY LANGFUSE_DEBUG
write_env_line "$APP_ENV" DATABASE_BACKEND postgres
write_env_line "$APP_ENV" VECTOR_STORE_BACKEND pgvector
write_env_line "$APP_ENV" SKILLS_BACKEND local
write_env_line "$APP_ENV" PROMPTS_BACKEND local
write_env_line "$APP_ENV" PG_DSN "postgresql://${RAG_DB_USER}:${RAG_DB_PASSWORD}@rag-postgres:5432/${RAG_DB_NAME}"
write_env_line "$APP_ENV" OLLAMA_BASE_URL "http://ollama:11434"
write_env_line "$APP_ENV" GRAPHRAG_BASE_URL "http://ollama:11434/v1"
write_env_line "$APP_ENV" LANGFUSE_HOST ""
write_env_line "$APP_ENV" LANGFUSE_PUBLIC_KEY ""
write_env_line "$APP_ENV" LANGFUSE_SECRET_KEY ""
write_env_line "$APP_ENV" LANGFUSE_DEBUG false

: > "$POSTGRES_ENV"
write_env_line "$POSTGRES_ENV" POSTGRES_DB "$RAG_DB_NAME"
write_env_line "$POSTGRES_ENV" POSTGRES_USER "$RAG_DB_USER"
write_env_line "$POSTGRES_ENV" POSTGRES_PASSWORD "$RAG_DB_PASSWORD"

: > "$OLLAMA_ENV"
write_env_line "$OLLAMA_ENV" OLLAMA_HOST "0.0.0.0:11434"
for key in OLLAMA_FLASH_ATTENTION OLLAMA_KV_CACHE_TYPE OLLAMA_KEEP_ALIVE OLLAMA_NUM_PARALLEL OLLAMA_MAX_LOADED_MODELS OLLAMA_CONTEXT_LENGTH; do
  value="$(env_value "$key" "")"
  [[ -n "$value" ]] && write_env_line "$OLLAMA_ENV" "$key" "$value"
done

: > "$OLLAMA_BOOTSTRAP_ENV"
write_env_line "$OLLAMA_BOOTSTRAP_ENV" OLLAMA_HOST "http://ollama:11434"
write_env_line "$OLLAMA_BOOTSTRAP_ENV" OLLAMA_BOOTSTRAP_MODELS "$OLLAMA_BOOTSTRAP_MODELS"

: > "$OPENWEBUI_ENV"
write_env_line "$OPENWEBUI_ENV" WEBUI_SECRET_KEY "$OPENWEBUI_SECRET_KEY"
write_env_line "$OPENWEBUI_ENV" ADMIN_EMAIL "$OPENWEBUI_ADMIN_EMAIL"
write_env_line "$OPENWEBUI_ENV" OPENWEBUI_ADMIN_EMAIL "$OPENWEBUI_ADMIN_EMAIL"
write_env_line "$OPENWEBUI_ENV" OPENWEBUI_ADMIN_PASSWORD "$OPENWEBUI_ADMIN_PASSWORD"
write_env_line "$OPENWEBUI_ENV" OPENWEBUI_ADMIN_NAME "$OPENWEBUI_ADMIN_NAME"
write_env_line "$OPENWEBUI_ENV" ENABLE_FORWARD_USER_INFO_HEADERS true
write_env_line "$OPENWEBUI_ENV" FORWARD_USER_INFO_HEADER_USER_ID X-User-ID
write_env_line "$OPENWEBUI_ENV" FORWARD_SESSION_INFO_HEADER_CHAT_ID X-Conversation-ID
write_env_line "$OPENWEBUI_ENV" FORWARD_SESSION_INFO_HEADER_MESSAGE_ID X-Request-ID
write_env_line "$OPENWEBUI_ENV" AGENT_BASE_URL "http://app:8000/v1"
write_env_line "$OPENWEBUI_ENV" AGENT_API_KEY "$GATEWAY_SHARED_BEARER_TOKEN"
write_env_line "$OPENWEBUI_ENV" AGENT_MODEL_ID "$GATEWAY_MODEL_ID"
write_env_line "$OPENWEBUI_ENV" KB_COLLECTION_ID "$OPENWEBUI_KB_COLLECTION_ID"
write_env_line "$OPENWEBUI_ENV" PUBLIC_AGENT_API_BASE_URL "$PUBLIC_AGENT_API_BASE_URL"
write_env_line "$OPENWEBUI_ENV" REQUEST_TIMEOUT_SECONDS "$OPENWEBUI_AGENT_REQUEST_TIMEOUT_SECONDS"
write_env_line "$OPENWEBUI_ENV" COLLECTION_PREFIX "$OPENWEBUI_COLLECTION_PREFIX"
write_env_line "$OPENWEBUI_ENV" OPENWEBUI_THIN_MODE "$OPENWEBUI_THIN_MODE"
write_env_line "$OPENWEBUI_ENV" OPENWEBUI_ENABLE_HELPERS "$OPENWEBUI_ENABLE_HELPERS"
write_env_line "$OPENWEBUI_ENV" OPENWEBUI_ALLOW_FILE_BYTE_TRANSPORT "$OPENWEBUI_ALLOW_FILE_BYTE_TRANSPORT"
write_env_line "$OPENWEBUI_ENV" ENABLE_WEB_SEARCH false
write_env_line "$OPENWEBUI_ENV" ENABLE_RAG_WEB_SEARCH false
write_env_line "$OPENWEBUI_ENV" ENABLE_CODE_INTERPRETER false
write_env_line "$OPENWEBUI_ENV" ENABLE_IMAGE_GENERATION false
write_env_line "$OPENWEBUI_ENV" ENABLE_SEARCH_QUERY_GENERATION false
write_env_line "$OPENWEBUI_ENV" ENABLE_RETRIEVAL_QUERY_GENERATION false
write_env_line "$OPENWEBUI_ENV" BYPASS_EMBEDDING_AND_RETRIEVAL true
write_env_line "$OPENWEBUI_ENV" RAG_FULL_CONTEXT false

: > "$OPENWEBUI_BOOTSTRAP_ENV"
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_BASE_URL "http://openwebui:8080"
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_ADMIN_EMAIL "$OPENWEBUI_ADMIN_EMAIL"
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_ADMIN_PASSWORD "$OPENWEBUI_ADMIN_PASSWORD"
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_ADMIN_NAME "$OPENWEBUI_ADMIN_NAME"
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_PIPE_FILE "/config/enterprise_agent_pipe.py"
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_PIPE_ID enterprise_agent_pipe
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_PIPE_NAME "Enterprise Agent Pipe"
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_PIPE_SUBMODEL_ID owui_enterprise_agent
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_PIPE_MODEL_NAME "Enterprise Agent"
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_THIN_MODE "$OPENWEBUI_THIN_MODE"
write_env_line "$OPENWEBUI_BOOTSTRAP_ENV" OPENWEBUI_BOOTSTRAP_TIMEOUT_SECONDS "$OPENWEBUI_BOOTSTRAP_TIMEOUT_SECONDS"

chmod 600 "$ENV_DIR"/*.env

repo_escaped="$(printf '%s' "$REPO_ROOT" | sed 's/[\/&]/\\&/g')"
env_escaped="$(printf '%s' "$ENV_DIR" | sed 's/[\/&]/\\&/g')"

for template in "$PODMAN_STARTUP_DIR"/quadlet/*; do
  name="$(basename "$template")"
  output_name="${name%.in}"
  sed \
    -e "s|__REPO_ROOT__|$repo_escaped|g" \
    -e "s|__ENV_DIR__|$env_escaped|g" \
    -e "s|__RAG_DB_USER__|$RAG_DB_USER|g" \
    -e "s|__RAG_DB_NAME__|$RAG_DB_NAME|g" \
    -e "s|__RAG_DB_PORT__|$RAG_DB_PORT|g" \
    -e "s|__OLLAMA_PORT__|$OLLAMA_PORT|g" \
    -e "s|__APP_API_PORT__|$APP_API_PORT|g" \
    -e "s|__OPENWEBUI_PORT__|$OPENWEBUI_PORT|g" \
    "$template" > "$QUADLET_DIR/$output_name"
done

systemctl --user daemon-reload

echo "Installed rootless Quadlets to $QUADLET_DIR"
echo "Generated env files in $ENV_DIR"
echo "Run: $PODMAN_STARTUP_DIR/scripts/start.sh"
