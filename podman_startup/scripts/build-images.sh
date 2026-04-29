#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_repo_root
require_command podman
require_command curl
require_command tar

OPENWEBUI_VERSION="$(env_value OPENWEBUI_VERSION v0.8.6)"
DEFAULT_PROXY="http://contractorproxyeast.northgrum.com:80"
DEFAULT_NO_PROXY="localhost,127.0.0.1,::1,0.0.0.0,rag-postgres,ollama,app,openwebui,.northgrum.com,169.254.169.254,169.254.170.2"
BUILD_HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-$(env_value HTTP_PROXY "$DEFAULT_PROXY")}}"
BUILD_HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-$(env_value HTTPS_PROXY "$DEFAULT_PROXY")}}"
BUILD_NO_PROXY="${NO_PROXY:-${no_proxy:-$(env_value NO_PROXY "$DEFAULT_NO_PROXY")}}"
export HTTP_PROXY="$BUILD_HTTP_PROXY"
export HTTPS_PROXY="$BUILD_HTTPS_PROXY"
export NO_PROXY="$BUILD_NO_PROXY"
export http_proxy="$BUILD_HTTP_PROXY"
export https_proxy="$BUILD_HTTPS_PROXY"
export no_proxy="$BUILD_NO_PROXY"
ONNX_RUNTIME_VERSION="$(env_value ONNX_RUNTIME_VERSION 1.20.1)"
ONNX_RUNTIME_ARTIFACT="$(env_value ONNX_RUNTIME_ARTIFACT "onnxruntime-linux-x64-gpu-${ONNX_RUNTIME_VERSION}.tgz")"
ONNX_RUNTIME_URL="$(env_value ONNX_RUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/${ONNX_RUNTIME_ARTIFACT}")"
VENDOR_DIR="$REPO_ROOT/podman_startup/vendor"
ONNX_RUNTIME_TARBALL="$VENDOR_DIR/$ONNX_RUNTIME_ARTIFACT"

BUILD_ARGS=(
  --build-arg "HTTP_PROXY=$BUILD_HTTP_PROXY"
  --build-arg "HTTPS_PROXY=$BUILD_HTTPS_PROXY"
  --build-arg "NO_PROXY=$BUILD_NO_PROXY"
)

ensure_onnx_runtime_tarball() {
  mkdir -p "$VENDOR_DIR"

  if [[ -f "$ONNX_RUNTIME_TARBALL" ]]; then
    echo "Using existing ONNX Runtime GPU artifact: $ONNX_RUNTIME_TARBALL"
  else
    local tmp_file
    local curl_args
    echo "Downloading ONNX Runtime GPU artifact:"
    echo "  $ONNX_RUNTIME_URL"
    tmp_file="$(mktemp "$VENDOR_DIR/${ONNX_RUNTIME_ARTIFACT}.tmp.XXXXXX")"
    curl_args=(
      -fL
      --retry 5
      --retry-delay 5
      --connect-timeout 30
      --output "$tmp_file"
    )
    if [[ -n "$BUILD_HTTPS_PROXY" ]]; then
      curl_args+=(--proxy "$BUILD_HTTPS_PROXY")
    fi
    curl "${curl_args[@]}" "$ONNX_RUNTIME_URL"
    mv "$tmp_file" "$ONNX_RUNTIME_TARBALL"
  fi

  echo "Verifying ONNX Runtime GPU artifact"
  tar -tzf "$ONNX_RUNTIME_TARBALL" >/dev/null
}

ensure_onnx_runtime_tarball

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
  -f "$REPO_ROOT/podman_startup/Containerfile.openwebui" \
  --build-arg "OPENWEBUI_VERSION=$OPENWEBUI_VERSION" \
  --build-arg "ONNX_RUNTIME_VERSION=$ONNX_RUNTIME_VERSION" \
  --build-arg "ONNX_RUNTIME_ARTIFACT=$ONNX_RUNTIME_ARTIFACT" \
  "$REPO_ROOT"

echo "Images built:"
podman images localhost/agentic-chatbot-v3-app localhost/agentic-chatbot-v3-openwebui
