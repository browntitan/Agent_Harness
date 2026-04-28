#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_repo_root
require_command podman

OPENWEBUI_VERSION="$(env_value OPENWEBUI_VERSION v0.8.6)"
DEFAULT_PROXY="http://contractorproxyeast.northgrum.com:80"
DEFAULT_NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0,rag-postgres,ollama,app,openwebui,.northgrum.com,169.254.169.254,169.254.170.2"
BUILD_HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-$(env_value HTTP_PROXY "$DEFAULT_PROXY")}}"
BUILD_HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-$(env_value HTTPS_PROXY "$DEFAULT_PROXY")}}"
BUILD_NO_PROXY="${NO_PROXY:-${no_proxy:-$(env_value NO_PROXY "$DEFAULT_NO_PROXY")}}"
BUILD_CA_CERT="$(env_value PODMAN_BUILD_CA_CERT "podman_startup/certs/NG-Certificate-Chain.cer")"
BUILD_CA_CERT_NORMALIZED=""

if [[ "$BUILD_CA_CERT" != /* ]]; then
  BUILD_CA_CERT="$REPO_ROOT/$BUILD_CA_CERT"
fi

cleanup() {
  [[ -n "$BUILD_CA_CERT_NORMALIZED" && -f "$BUILD_CA_CERT_NORMALIZED" ]] && rm -f "$BUILD_CA_CERT_NORMALIZED"
}
trap cleanup EXIT

CUSTOM_CA_CERT_B64=""
if [[ -f "$BUILD_CA_CERT" ]]; then
  echo "Using custom build CA certificate: $BUILD_CA_CERT"
  if head -n 1 "$BUILD_CA_CERT" | grep -q "BEGIN CERTIFICATE"; then
    BUILD_CA_CERT_NORMALIZED="$BUILD_CA_CERT"
  else
    require_command openssl
    BUILD_CA_CERT_NORMALIZED="$(mktemp)"
    openssl x509 -inform DER -in "$BUILD_CA_CERT" -out "$BUILD_CA_CERT_NORMALIZED"
  fi
  CUSTOM_CA_CERT_B64="$(base64 < "$BUILD_CA_CERT_NORMALIZED" | tr -d '\n')"
else
  echo "No custom build CA certificate found at $BUILD_CA_CERT"
  echo "If your network intercepts TLS, place NG-Certificate-Chain.cer there or set PODMAN_BUILD_CA_CERT."
fi

BUILD_ARGS=(
  --build-arg "HTTP_PROXY=$BUILD_HTTP_PROXY"
  --build-arg "HTTPS_PROXY=$BUILD_HTTPS_PROXY"
  --build-arg "NO_PROXY=$BUILD_NO_PROXY"
  --build-arg "CUSTOM_CA_CERT_B64=$CUSTOM_CA_CERT_B64"
)

echo "Building application image from $REPO_ROOT"
podman build \
  "${BUILD_ARGS[@]}" \
  -t localhost/agentic-chatbot-v3-app:latest \
  -f "$REPO_ROOT/Dockerfile" \
  "$REPO_ROOT"

echo "Building Open WebUI image with OPENWEBUI_VERSION=$OPENWEBUI_VERSION"
podman build \
  "${BUILD_ARGS[@]}" \
  -t localhost/agentic-chatbot-v3-openwebui:latest \
  -f "$REPO_ROOT/deployment/openwebui/image/Dockerfile" \
  --build-arg "OPENWEBUI_VERSION=$OPENWEBUI_VERSION" \
  "$REPO_ROOT"

echo "Images built:"
podman images localhost/agentic-chatbot-v3-app localhost/agentic-chatbot-v3-openwebui
