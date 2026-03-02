#!/usr/bin/env bash
set -euo pipefail

APP_HOME="/home/ubuntu"
STATE_DIR="${APP_HOME}/artifact/ai-container"
CURRENT_IMAGE_FILE="${STATE_DIR}/.current_image"

UBUNTU_USER="${UBUNTU_USER:-ubuntu}"
UBUNTU_HOME="/home/${UBUNTU_USER}"
UBUNTU_UID="$(id -u "${UBUNTU_USER}")"
USER_RUNTIME_DIR="/run/user/${UBUNTU_UID}"

API_CONTAINER_NAME="${API_CONTAINER_NAME:-bizkit-ai}"
WORKER_CONTAINER_NAME="${WORKER_CONTAINER_NAME:-bizkit-ai-worker}"
API_UNIT_NAME="container-${API_CONTAINER_NAME}.service"
WORKER_UNIT_NAME="container-${WORKER_CONTAINER_NAME}.service"
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://127.0.0.1:8000/ai/health}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-120}"
HEALTH_CHECK_SLEEP="${HEALTH_CHECK_SLEEP:-2}"
ENV_FILE="${ENV_FILE:-/home/ubuntu/.env-ai}"
LEGACY_SERVICES="${LEGACY_SERVICES:-bizkit-ai.service,bizkit-ai-worker.service}"
PODMAN_NETWORK="${PODMAN_NETWORK:-host}"

: "${RELEASE_ID:?RELEASE_ID is required}"
: "${APP_STAGE:?APP_STAGE is required}"
: "${IMAGE_URI:?IMAGE_URI is required}"

for cmd in podman aws curl runuser systemctl; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[deploy-ai-container] missing command: ${cmd}" >&2
    exit 2
  fi
done

if [[ ! -x /usr/bin/loginctl ]]; then
  echo "[deploy-ai-container] missing command: loginctl" >&2
  exit 2
fi

ensure_user_runtime_dir() {
  if [[ ! -d /run/user ]]; then
    mkdir -p /run/user
  fi
  chmod 755 /run/user || true

  if [[ ! -d "${USER_RUNTIME_DIR}" ]]; then
    echo "[deploy-ai-container] user runtime dir not found: ${USER_RUNTIME_DIR}"
    echo "[deploy-ai-container] enabling linger and creating runtime dir for ${UBUNTU_USER}"
    loginctl enable-linger "${UBUNTU_USER}" || true
    mkdir -p "${USER_RUNTIME_DIR}"
  fi

  chmod 700 "${USER_RUNTIME_DIR}" || true
  chown "${UBUNTU_UID}:${UBUNTU_UID}" "${USER_RUNTIME_DIR}" || true
  mkdir -p "${USER_RUNTIME_DIR}/libpod/tmp"
  chmod 700 "${USER_RUNTIME_DIR}/libpod/tmp" || true
}

ensure_user_runtime_dir

if [[ ! -d "${USER_RUNTIME_DIR}" ]]; then
  echo "[deploy-ai-container] user runtime dir still missing: ${USER_RUNTIME_DIR}" >&2
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

create_container_unit() {
  local container_name="$1"
  local unit_name="$2"
  local unit_image="$3"
  shift 3
  local command_args=("$@")

  as_ubuntu podman rm -f "${container_name}" >/dev/null 2>&1 || true
  as_ubuntu systemctl --user disable --now "${unit_name}" >/dev/null 2>&1 || true

  as_ubuntu podman create \
    --name "${container_name}" \
    --replace \
    --env-file "${ENV_FILE}" \
    --network "${PODMAN_NETWORK}" \
    "${unit_image}" \
    "${command_args[@]}"

  as_ubuntu bash -lc "mkdir -p '${UBUNTU_HOME}/.config/systemd/user' && cd '${UBUNTU_HOME}/.config/systemd/user' && podman generate systemd --new --name '${container_name}' --files --restart-policy always"
  as_ubuntu systemctl --user daemon-reload
  as_ubuntu systemctl --user enable --now "${unit_name}"
}

stop_container_units() {
  as_ubuntu systemctl --user disable --now "${API_UNIT_NAME}" >/dev/null 2>&1 || true
  as_ubuntu systemctl --user disable --now "${WORKER_UNIT_NAME}" >/dev/null 2>&1 || true
  as_ubuntu podman rm -f "${API_CONTAINER_NAME}" >/dev/null 2>&1 || true
  as_ubuntu podman rm -f "${WORKER_CONTAINER_NAME}" >/dev/null 2>&1 || true
}

stop_legacy_services() {
  for service in ${LEGACY_SERVICES//,/ }; do
    systemctl stop "${service}" || true
  done
}

show_debug_info() {
  local container_name="$1"
  as_ubuntu podman ps -a || true
  if as_ubuntu podman ps --format "{{.Names}}" | grep -Fxq "${container_name}"; then
    as_ubuntu podman inspect "${container_name}" --format 'container ID={{.ID}} State={{.State.Status}} ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}} Error={{.State.Error}}' || true
    as_ubuntu podman logs --tail 200 "${container_name}" || true
  fi
}

check_health() {
  local max="${1}"
  local current=1
  while [[ "${current}" -le "${max}" ]]; do
    local code
    code="$(curl -sS -o /tmp/healthcheck_body.txt -w "%{http_code}" --max-time 3 "${HEALTH_CHECK_URL}" || true)"
    local status=$?
    if [[ "${status}" -eq 0 ]] && [[ "${code}" =~ ^[23][0-9]{2}$ ]]; then
      echo "[deploy-ai-container] healthcheck ok (${code}) at attempt=${current}" 
      rm -f /tmp/healthcheck_body.txt
      return 0
    fi
    if [[ -f /tmp/healthcheck_body.txt ]]; then
      local body
      body="$(cat /tmp/healthcheck_body.txt | tr '\n' ' ' | tr -s ' ')"
      rm -f /tmp/healthcheck_body.txt
    else
      body=""
    fi
    echo "[deploy-ai-container] healthcheck fail attempt=${current}/${max} code=${code} status=${status} body=${body}" >&2
    current=$((current + 1))
    sleep "${HEALTH_CHECK_SLEEP}"
  done
  rm -f /tmp/healthcheck_body.txt
  return 1
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
echo "[deploy-ai-container] api_container=${API_CONTAINER_NAME} worker_container=${WORKER_CONTAINER_NAME}"
echo "[deploy-ai-container] health_check_url=${HEALTH_CHECK_URL}"

stop_legacy_services

as_ubuntu bash -lc "aws ecr get-login-password --region '${AWS_REGION}' | podman login --username AWS --password-stdin '${REGISTRY}'"
as_ubuntu podman pull "${IMAGE_URI}"
stop_container_units

echo "[deploy-ai-container] start API container"
create_container_unit "${API_CONTAINER_NAME}" "${API_UNIT_NAME}" "${IMAGE_URI}"

echo "[deploy-ai-container] start worker container"
create_container_unit "${WORKER_CONTAINER_NAME}" "${WORKER_UNIT_NAME}" "${IMAGE_URI}" "python" "run_worker.py"

ok=0
if check_health "${HEALTH_CHECK_RETRIES}"; then
  ok=1
fi

if [[ "${ok}" -eq 1 ]]; then
  echo "${IMAGE_URI}" > "${CURRENT_IMAGE_FILE}"
  chown "${UBUNTU_USER}:${UBUNTU_USER}" "${CURRENT_IMAGE_FILE}" || true
  echo "[deploy-ai-container] SUCCESS release=${RELEASE_ID}"
  exit 0
fi

echo "[deploy-ai-container] FAILED healthcheck. rollback start." >&2
show_debug_info "${API_CONTAINER_NAME}"
show_debug_info "${WORKER_CONTAINER_NAME}"
stop_container_units

if [[ -n "${PREV_IMAGE_URI}" ]]; then
  echo "[deploy-ai-container] rollback image=${PREV_IMAGE_URI}" >&2
  as_ubuntu podman pull "${PREV_IMAGE_URI}"
  create_container_unit "${API_CONTAINER_NAME}" "${API_UNIT_NAME}" "${PREV_IMAGE_URI}"
  create_container_unit "${WORKER_CONTAINER_NAME}" "${WORKER_UNIT_NAME}" "${PREV_IMAGE_URI}" "python" "run_worker.py"

  for i in $(seq 1 60); do
    if check_health 1; then
      echo "[deploy-ai-container] ROLLBACK OK" >&2
      exit 1
    fi
  done
fi

show_debug_info "${API_CONTAINER_NAME}"
show_debug_info "${WORKER_CONTAINER_NAME}"
echo "[deploy-ai-container] ROLLBACK FAILED" >&2
exit 1
