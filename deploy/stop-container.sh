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
LEGACY_SERVICES="${LEGACY_SERVICES:-bizkit-ai.service,bizkit-ai-worker.service}"

echo "[stop-ai-container] stopping api=${API_CONTAINER_NAME}, worker=${WORKER_CONTAINER_NAME}"
for service in ${LEGACY_SERVICES//,/ }; do
  systemctl stop "${service}" || true
done

for cmd in runuser systemctl; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[stop-ai-container] missing command: ${cmd}. skip stop." >&2
    exit 0
  fi
done

if [[ ! -d "${USER_RUNTIME_DIR}" ]]; then
  echo "[stop-ai-container] user runtime dir not found: ${USER_RUNTIME_DIR}. skip stop." >&2
  exit 0
fi

as_ubuntu() {
  runuser -u "${UBUNTU_USER}" -- env \
    HOME="${UBUNTU_HOME}" \
    XDG_RUNTIME_DIR="${USER_RUNTIME_DIR}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=${USER_RUNTIME_DIR}/bus" \
    "$@"
}

as_ubuntu systemctl --user disable --now "${API_UNIT_NAME}" >/dev/null 2>&1 || true
as_ubuntu systemctl --user disable --now "${WORKER_UNIT_NAME}" >/dev/null 2>&1 || true

if as_ubuntu podman ps --format "{{.Names}}" 2>/dev/null | grep -Fxq "${API_CONTAINER_NAME}"; then
  as_ubuntu podman rm -f "${API_CONTAINER_NAME}" || true
  echo "[stop-ai-container] container removed: ${API_CONTAINER_NAME}"
else
  echo "[stop-ai-container] container not found: ${API_CONTAINER_NAME}"
fi

if as_ubuntu podman ps --format "{{.Names}}" 2>/dev/null | grep -Fxq "${WORKER_CONTAINER_NAME}"; then
  as_ubuntu podman rm -f "${WORKER_CONTAINER_NAME}" || true
  echo "[stop-ai-container] container removed: ${WORKER_CONTAINER_NAME}"
else
  echo "[stop-ai-container] container not found: ${WORKER_CONTAINER_NAME}"
fi
