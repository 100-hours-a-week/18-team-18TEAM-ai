#!/usr/bin/env bash
set -euo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${HOOK_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOOK_DIR}/env.sh"
else
  echo "[start-ai-container] env.sh not found in ${HOOK_DIR}" >&2
  exit 11
fi

: "${RELEASE_ID:?RELEASE_ID is required}"
: "${APP_STAGE:?APP_STAGE is required}"
: "${IMAGE_URI:?IMAGE_URI is required}"

export RELEASE_ID
export APP_STAGE
export IMAGE_URI
export HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-${HEALTH_URL:-http://127.0.0.1:8000/ai/health}}"
export ENV_FILE="${ENV_FILE:-/home/ubuntu/.env-ai}"
export CONTAINER_NAME="${CONTAINER_NAME:-bizkit-ai}"
export HOST_PORT="${HOST_PORT:-8000}"
export CONTAINER_PORT="${CONTAINER_PORT:-8000}"
export LEGACY_SERVICES="${LEGACY_SERVICES:-bizkit-ai.service,bizkit-ai-worker.service}"

chmod +x "${HOOK_DIR}/deploy-ai-container.sh"
exec "${HOOK_DIR}/deploy-ai-container.sh"
