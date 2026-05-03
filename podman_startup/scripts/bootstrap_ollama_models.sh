#!/usr/bin/env sh
set -eu

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
OLLAMA_BOOTSTRAP_MODELS="${OLLAMA_BOOTSTRAP_MODELS:-nemotron-cascade-2:30b,nomic-embed-text:latest}"
OLLAMA_BOOTSTRAP_TIMEOUT_SECONDS="${OLLAMA_BOOTSTRAP_TIMEOUT_SECONDS:-300}"
export OLLAMA_HOST

echo "Waiting for Ollama at ${OLLAMA_HOST}"
deadline=$(( $(date +%s) + OLLAMA_BOOTSTRAP_TIMEOUT_SECONDS ))
while ! ollama list >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "Timed out waiting for Ollama at ${OLLAMA_HOST}" >&2
    exit 1
  fi
  sleep 2
done

echo "Bootstrapping Ollama models: ${OLLAMA_BOOTSTRAP_MODELS}"
printf '%s' "$OLLAMA_BOOTSTRAP_MODELS" | tr ',' '\n' | while IFS= read -r model; do
  model="$(printf '%s' "$model" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -n "$model" ] || continue

  if ollama list | awk 'NR>1 {print $1}' | grep -Fx "$model" >/dev/null 2>&1; then
    echo "Model ${model} already present"
  else
    echo "Pulling ${model}"
    ollama pull "$model"
  fi
done

echo "Ollama bootstrap complete"
