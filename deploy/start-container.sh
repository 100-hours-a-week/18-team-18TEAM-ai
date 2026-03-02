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
export API_CONTAINER_NAME="${API_CONTAINER_NAME:-bizkit-ai}"
export WORKER_CONTAINER_NAME="${WORKER_CONTAINER_NAME:-bizkit-ai-worker}"
export HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-${HEALTH_URL:-http://127.0.0.1:8000/ai/health}}"
export ENV_FILE="${ENV_FILE:-/home/ubuntu/.env-ai}"

chmod +x "${HOOK_DIR}/deploy-ai-container.sh"
exec "${HOOK_DIR}/deploy-ai-container.sh"
