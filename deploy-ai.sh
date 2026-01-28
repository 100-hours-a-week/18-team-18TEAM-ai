#!/usr/bin/env bash
set -euo pipefail

BASE="/home/ubuntu"

SERVICE_UNIT="bizkit-ai.service"
ENV_FILE="${BASE}/.env-ai"

ARTIFACT_DIR="${BASE}/artifact/ai"
BACKUP_DIR="${BASE}/backup/ai"
CURRENT_FILE="${ARTIFACT_DIR}/.current_version"

RELEASE_ID="${1:-}"
RELEASE_DIR="${ARTIFACT_DIR}/${RELEASE_ID}"

# 공유 venv (원하면 다른 경로로)
VENV_DIR="${BASE}/caro-ai-venv"

# 실행 파라미터 (기본값, .env-ai에서 덮어쓰는 방식 추천)
APP_MODULE="${APP_MODULE:-app.main:app}"
APP_PORT="${APP_PORT:-8000}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${APP_PORT}/ai/health}"

if [[ -z "${RELEASE_ID}" ]]; then
  echo "[deploy-ai] missing RELEASE_ID. usage: deploy-ai.sh <version>" >&2
  exit 2
fi

if [[ ! -d "${RELEASE_DIR}" ]]; then
  echo "[deploy-ai] release dir not found: ${RELEASE_DIR}" >&2
  exit 3
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[deploy-ai] env not found: ${ENV_FILE}" >&2
  exit 4
fi

mkdir -p "${BACKUP_DIR}"

# --- 이전 버전 읽기 ---
PREV_ID=""
if [[ -f "${CURRENT_FILE}" ]]; then
  PREV_ID="$(cat "${CURRENT_FILE}" | tr -d '\r' | xargs || true)"
fi

# --- venv 준비 + requirements 설치(공유 venv 기준) ---
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

REQ_FILE="${RELEASE_DIR}/requirements.txt"
if [[ -f "${REQ_FILE}" ]]; then
  "${VENV_DIR}/bin/pip" install -U pip >/dev/null
  "${VENV_DIR}/bin/pip" install -r "${REQ_FILE}" --no-cache-dir
fi

# --- systemd override 갱신: "java -jar" -> "uvicorn" ---
OVERRIDE_DIR="/etc/systemd/system/${SERVICE_UNIT}.d"
sudo mkdir -p "${OVERRIDE_DIR}"

sudo tee "${OVERRIDE_DIR}/override.conf" >/dev/null <<EOF
[Service]
EnvironmentFile=-${ENV_FILE}
WorkingDirectory=${RELEASE_DIR}
ExecStart=
ExecStart=${VENV_DIR}/bin/uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${APP_PORT}
EOF

sudo systemctl daemon-reload
sudo systemctl restart "${SERVICE_UNIT}"

# --- health check (up to 30s) ---
ok=0
for i in $(seq 1 30); do
  if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null; then
    ok=1
    break
  fi
  sleep 1
done

if [[ "${ok}" -eq 1 ]]; then
  echo "${RELEASE_ID}" | sudo tee "${CURRENT_FILE}" >/dev/null
  echo "[deploy-ai] SUCCESS version=${RELEASE_ID}"
  exit 0
fi

# --- rollback to prev (BE와 동일 컨셉) ---
echo "[deploy-ai] FAILED healthcheck. try rollback to prev=${PREV_ID}" >&2

if [[ -n "${PREV_ID}" && -d "${ARTIFACT_DIR}/${PREV_ID}" ]]; then
  PREV_DIR="${ARTIFACT_DIR}/${PREV_ID}"

  sudo tee "${OVERRIDE_DIR}/override.conf" >/dev/null <<EOF
[Service]
EnvironmentFile=-${ENV_FILE}
WorkingDirectory=${PREV_DIR}
ExecStart=
ExecStart=${VENV_DIR}/bin/uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${APP_PORT}
EOF

  sudo systemctl daemon-reload
  sudo systemctl restart "${SERVICE_UNIT}"

  curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null && \
    echo "[deploy-ai] ROLLBACK OK -> ${PREV_ID}" >&2 || \
    echo "[deploy-ai] ROLLBACK ALSO FAILED" >&2
fi

exit 1
