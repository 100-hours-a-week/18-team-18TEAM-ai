#!/usr/bin/env bash
set -euo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REV_DIR="$(cd "${HOOK_DIR}/../../.." && pwd)"

echo "[start.sh] HOOK_DIR=${HOOK_DIR}"
echo "[start.sh] REV_DIR=${REV_DIR}"

# env 주입 파일 로드 (bundle/codedeploy/ai/env.sh)
if [[ -f "${REV_DIR}/codedeploy/ai/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REV_DIR}/codedeploy/ai/env.sh"
fi

: "${RELEASE_ID:?RELEASE_ID is required}"
: "${APP_URL:?APP_URL is required}"

ARTIFACT_DIR="/home/ubuntu/artifact/ai"
RELEASE_DIR="${ARTIFACT_DIR}/${RELEASE_ID}"
PKG_TMP="/tmp/bizkit-ai-${RELEASE_ID}.tar.gz"

sudo mkdir -p "${RELEASE_DIR}"

echo "[start.sh] download app -> ${PKG_TMP}"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "${APP_URL}" -o "${PKG_TMP}"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "${PKG_TMP}" "${APP_URL}"
else
  echo "[start.sh] neither curl nor wget is installed" >&2
  exit 10
fi

echo "[start.sh] extract -> ${RELEASE_DIR}"
sudo tar -xzf "${PKG_TMP}" -C "${RELEASE_DIR}"
sudo rm -f "${PKG_TMP}"

# (권장) ubuntu 서비스가 읽을 수 있게
sudo chown -R ubuntu:ubuntu "${RELEASE_DIR}" || true

chmod +x "${REV_DIR}/deploy-ai.sh"
exec "${REV_DIR}/deploy-ai.sh" "${RELEASE_ID}"
