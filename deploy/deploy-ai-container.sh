#!/usr/bin/env bash
set -euo pipefail

APP_HOME="/home/ubuntu"
STATE_DIR="${APP_HOME}/artifact/ai-container"
CURRENT_IMAGE_FILE="${STATE_DIR}/.current_image"

UBUNTU_USER="${UBUNTU_USER:-ubuntu}"
UBUNTU_HOME="/home/${UBUNTU_USER}"
UBUNTU_UID="$(id -u "${UBUNTU_USER}")"
USER_RUNTIME_DIR="/run/user/${UBUNTU_UID}"

CONTAINER_NAME="${CONTAINER_NAME:-bizkit-ai}"
UNIT_NAME="container-${CONTAINER_NAME}.service"
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://127.0.0.1:8000/ai/health}"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
ENV_FILE="${ENV_FILE:-/home/ubuntu/.env-ai}"
LEGACY_SERVICES="${LEGACY_SERVICES:-bizkit-ai.service,bizkit-ai-worker.service}"

: "${RELEASE_ID:?RELEASE_ID is required}"
: "${APP_STAGE:?APP_STAGE is required}"
: "${IMAGE_URI:?IMAGE_URI is required}"

for cmd in podman aws curl runuser systemctl; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[deploy-ai-container] missing command: ${cmd}" >&2
    exit 2
  fi
done

if [[ ! -d "${USER_RUNTIME_DIR}" ]]; then
  echo "[deploy-ai-container] user runtime dir not found: ${USER_RUNTIME_DIR}" >&2
  echo "[deploy-ai-container] run: loginctl enable-linger ${UBUNTU_USER}" >&2
  exit 3
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[deploy-ai-container] env file not found: ${ENV_FILE}" >&2
  exit 4
fi

as_ubuntu() {
  runuser -u "${UBUNTU_USER}" -- env \
    HOME="${UBUNTU_HOME}" \
    XDG_RUNTIME_DIR="${USER_RUNTIME_DIR}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=${USER_RUNTIME_DIR}/bus" \
    "$@"
}

install -d -m 0755 -o "${UBUNTU_USER}" -g "${UBUNTU_USER}" "${STATE_DIR}"

PREV_IMAGE_URI=""
if [[ -f "${CURRENT_IMAGE_FILE}" ]]; then
  PREV_IMAGE_URI="$(tr -d '\r' < "${CURRENT_IMAGE_FILE}" | xargs || true)"
fi

REGISTRY="${IMAGE_URI%%/*}"
if [[ ! "${REGISTRY}" =~ \.ecr\.[a-z0-9-]+\.amazonaws\.com$ ]]; then
  echo "[deploy-ai-container] invalid ECR registry in IMAGE_URI=${IMAGE_URI}" >&2
  exit 4
fi
AWS_REGION="$(echo "${REGISTRY}" | sed -E 's#^.*\.ecr\.([a-z0-9-]+)\.amazonaws\.com$#\1#')"

echo "[deploy-ai-container] release=${RELEASE_ID} stage=${APP_STAGE}"
echo "[deploy-ai-container] image=${IMAGE_URI}"
echo "[deploy-ai-container] previous_image=${PREV_IMAGE_URI:-none}"
echo "[deploy-ai-container] container_name=${CONTAINER_NAME}"
echo "[deploy-ai-container] health_check_url=${HEALTH_CHECK_URL}"

echo "[deploy-ai-container] stopping legacy services ${LEGACY_SERVICES} (if exist)"
for service in ${LEGACY_SERVICES//,/ }; do
  systemctl stop "${service}" || true
done

as_ubuntu bash -lc "aws ecr get-login-password --region '${AWS_REGION}' | podman login --username AWS --password-stdin '${REGISTRY}'"
as_ubuntu podman pull "${IMAGE_URI}"
as_ubuntu systemctl --user disable --now "${UNIT_NAME}" >/dev/null 2>&1 || true
as_ubuntu podman rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

as_ubuntu podman create \
  --name "${CONTAINER_NAME}" \
  --replace \
  --env-file "${ENV_FILE}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  "${IMAGE_URI}"

as_ubuntu bash -lc "mkdir -p '${UBUNTU_HOME}/.config/systemd/user' && cd '${UBUNTU_HOME}/.config/systemd/user' && podman generate systemd --new --name '${CONTAINER_NAME}' --files --restart-policy always"
as_ubuntu systemctl --user daemon-reload
as_ubuntu systemctl --user enable --now "${UNIT_NAME}"

ok=0
for i in $(seq 1 30); do
  if curl -fsSL --max-time 2 "${HEALTH_CHECK_URL}" >/dev/null; then
    ok=1
    break
  fi
  sleep 1
done

if [[ "${ok}" -eq 1 ]]; then
  echo "${IMAGE_URI}" > "${CURRENT_IMAGE_FILE}"
  chown "${UBUNTU_USER}:${UBUNTU_USER}" "${CURRENT_IMAGE_FILE}" || true
  echo "[deploy-ai-container] SUCCESS release=${RELEASE_ID}"
  exit 0
fi

echo "[deploy-ai-container] FAILED healthcheck. rollback start." >&2
as_ubuntu systemctl --user status "${UNIT_NAME}" --no-pager || true
as_ubuntu podman ps --all || true
as_ubuntu podman inspect "${CONTAINER_NAME}" --format 'pre-rollback ID={{.ID}} State={{.State.Status}} ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}} Error={{.State.Error}}' || true
as_ubuntu podman logs --tail 200 "${CONTAINER_NAME}" || true

as_ubuntu systemctl --user disable --now "${UNIT_NAME}" >/dev/null 2>&1 || true
as_ubuntu podman rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

if [[ -n "${PREV_IMAGE_URI}" ]]; then
  echo "[deploy-ai-container] rollback image=${PREV_IMAGE_URI}" >&2
  as_ubuntu podman pull "${PREV_IMAGE_URI}"
  as_ubuntu podman create \
    --name "${CONTAINER_NAME}" \
    --replace \
    --env-file "${ENV_FILE}" \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    "${PREV_IMAGE_URI}"
  as_ubuntu bash -lc "mkdir -p '${UBUNTU_HOME}/.config/systemd/user' && cd '${UBUNTU_HOME}/.config/systemd/user' && podman generate systemd --new --name '${CONTAINER_NAME}' --files --restart-policy always"
  as_ubuntu systemctl --user daemon-reload
  as_ubuntu systemctl --user enable --now "${UNIT_NAME}"

  for i in $(seq 1 20); do
    if curl -fsSL --max-time 2 "${HEALTH_CHECK_URL}" >/dev/null; then
      echo "[deploy-ai-container] ROLLBACK OK" >&2
      exit 1
    fi
    sleep 1
  done
  as_ubuntu podman inspect "${CONTAINER_NAME}" --format 'rollback ID={{.ID}} State={{.State.Status}} ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} Error={{.State.Error}}' || true
  as_ubuntu podman logs --tail 200 "${CONTAINER_NAME}" || true
fi

echo "[deploy-ai-container] ROLLBACK FAILED" >&2
exit 1
