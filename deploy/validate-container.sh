#!/usr/bin/env bash
set -euo pipefail

UBUNTU_USER="${UBUNTU_USER:-ubuntu}"
UBUNTU_HOME="/home/${UBUNTU_USER}"
UBUNTU_UID="$(id -u "${UBUNTU_USER}")"
USER_RUNTIME_DIR="/run/user/${UBUNTU_UID}"

API_CONTAINER_NAME="${API_CONTAINER_NAME:-bizkit-ai}"
WORKER_CONTAINER_NAME="${WORKER_CONTAINER_NAME:-bizkit-ai-worker}"
API_UNIT_NAME="container-${API_CONTAINER_NAME}.service"
WORKER_UNIT_NAME="container-${WORKER_CONTAINER_NAME}.service"
URL="${URL:-http://127.0.0.1:8000/ai/health}"
ENV_FILE="${UBUNTU_HOME}/deploy/env.sh"
LEGACY_SERVICES="${LEGACY_SERVICES:-bizkit-ai.service,bizkit-ai-worker.service}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_FILE}" || true
fi

URL="${URL:-${HEALTH_CHECK_URL:-${URL}}}"

for service in ${LEGACY_SERVICES//,/ }; do
  if systemctl is-active --quiet "${service}"; then
    echo "[validate-ai-container] legacy service still active: ${service}" >&2
    exit 4
  fi
done

TRIES="${TRIES:-10}"
SLEEP_SEC="${SLEEP_SEC:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2}"

for cmd in podman runuser systemctl curl; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[validate-ai-container] missing command: ${cmd}" >&2
    exit 2
  fi
done

if [[ ! -d "${USER_RUNTIME_DIR}" ]]; then
  echo "[validate-ai-container] user runtime dir not found: ${USER_RUNTIME_DIR}" >&2
  exit 3
fi

as_ubuntu() {
  runuser -u "${UBUNTU_USER}" -- env \
    HOME="${UBUNTU_HOME}" \
    XDG_RUNTIME_DIR="${USER_RUNTIME_DIR}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=${USER_RUNTIME_DIR}/bus" \
    "$@"
}

echo "[validate-ai-container] checking api container=${API_CONTAINER_NAME}, worker=${WORKER_CONTAINER_NAME}"
for i in $(seq 1 "${TRIES}"); do
  if ! as_ubuntu systemctl --user is-active --quiet "${API_UNIT_NAME}"; then
    echo "[validate-ai-container] api unit not active: ${API_UNIT_NAME} (try=${i})"
    sleep "${SLEEP_SEC}"
    continue
  fi

  if ! as_ubuntu systemctl --user is-active --quiet "${WORKER_UNIT_NAME}"; then
    echo "[validate-ai-container] worker unit not active: ${WORKER_UNIT_NAME} (try=${i})"
    sleep "${SLEEP_SEC}"
    continue
  fi

  if ! as_ubuntu podman ps --format "{{.Names}}" | grep -Fxq "${API_CONTAINER_NAME}"; then
    echo "[validate-ai-container] api container not running (try=${i})"
    sleep "${SLEEP_SEC}"
    continue
  fi

  if ! as_ubuntu podman ps --format "{{.Names}}" | grep -Fxq "${WORKER_CONTAINER_NAME}"; then
    echo "[validate-ai-container] worker container not running (try=${i})"
    sleep "${SLEEP_SEC}"
    continue
  fi

  HTTP_CODE="$(curl -fsSL -o /dev/null -w "%{http_code}" --max-time "${TIMEOUT_SEC}" "${URL}" 2>/dev/null || echo "0")"
  echo "[validate-ai-container] HTTP_CODE=${HTTP_CODE} (try=${i}/${TRIES})"
  if [[ "${HTTP_CODE}" =~ ^2[0-9]{2}$ ]]; then
    echo "[validate-ai-container] AI validated OK (HTTP=${HTTP_CODE})"
    exit 0
  fi
  sleep "${SLEEP_SEC}"
done

echo "[validate-ai-container] FAILED"
as_ubuntu systemctl --user status "${API_UNIT_NAME}" "${WORKER_UNIT_NAME}" --no-pager || true
as_ubuntu podman ps -a || true
as_ubuntu podman logs --tail 200 "${API_CONTAINER_NAME}" || true
as_ubuntu podman logs --tail 200 "${WORKER_CONTAINER_NAME}" || true
exit 1
