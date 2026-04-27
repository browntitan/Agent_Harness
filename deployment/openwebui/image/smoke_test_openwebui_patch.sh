#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PATCH_FILE="${REPO_ROOT}/deployment/openwebui/image/patches/openwebui-status-upserts.patch"
DOCKERFILE="${REPO_ROOT}/deployment/openwebui/image/Dockerfile"
OPENWEBUI_VERSION="${OPENWEBUI_VERSION:-v0.8.6}"
IMAGE_TAG="${IMAGE_TAG:-agentic-rag-openwebui-smoke:${OPENWEBUI_VERSION}}"
WORKDIR="$(mktemp -d)"

cleanup() {
  rm -rf "${WORKDIR}"
}
trap cleanup EXIT

echo "Cloning OpenWebUI ${OPENWEBUI_VERSION} into ${WORKDIR}"
git clone --depth 1 --branch "${OPENWEBUI_VERSION}" https://github.com/open-webui/open-webui.git "${WORKDIR}/open-webui"

echo "Checking patch applicability"
git -C "${WORKDIR}/open-webui" apply --recount --check "${PATCH_FILE}"

echo "Checking patched tool-call trace UI files"
git -C "${WORKDIR}/open-webui" apply --recount "${PATCH_FILE}"
test -f "${WORKDIR}/open-webui/src/lib/components/chat/Messages/ResponseMessage/StatusHistory/ToolCallStack.svelte"
grep -R "agentic_tool_call" \
  "${WORKDIR}/open-webui/src/lib/components/chat/Chat.svelte" \
  "${WORKDIR}/open-webui/src/lib/components/chat/Messages/ResponseMessage/StatusHistory" >/dev/null

if [[ "${SKIP_DOCKER_BUILD:-0}" == "1" ]]; then
  echo "Patch applies cleanly. Docker build skipped because SKIP_DOCKER_BUILD=1."
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for the smoke build" >&2
  exit 1
fi

echo "Building ${IMAGE_TAG}"
docker build \
  --build-arg "OPENWEBUI_VERSION=${OPENWEBUI_VERSION}" \
  -t "${IMAGE_TAG}" \
  -f "${DOCKERFILE}" \
  "${REPO_ROOT}"

echo "Smoke check completed successfully."
