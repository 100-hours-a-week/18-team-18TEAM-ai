#!/usr/bin/env bash
set -euo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# hooks -> ai -> codedeploy -> (bundle root)
REV_DIR="$(cd "${HOOK_DIR}/../../.." && pwd)"
AI_DIR="$(cd "${HOOK_DIR}/.." && pwd)"  # .../codedeploy/ai

echo "[start-ai] HOOK_DIR=${HOOK_DIR}"
echo "[start-ai] AI_DIR=${AI_DIR}"
echo "[start-ai] REV_DIR=${REV_DIR}"

# env 주입 파일 로드 (우선순위: hooks/env.sh -> ai/env.sh -> REV_DIR/codedeploy/ai/env.sh)
if [[ -f "${HOOK_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOOK_DIR}/env.sh"
elif [[ -f "${AI_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${AI_DIR}/env.sh"
elif [[ -f "${REV_DIR}/codedeploy/ai/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REV_DIR}/codedeploy/ai/env.sh"
else
  echo "[start-ai] env.sh not found. tried:" >&2
  echo "  - ${HOOK_DIR}/env.sh" >&2
  echo "  - ${AI_DIR}/env.sh" >&2
  echo "  - ${REV_DIR}/codedeploy/ai/env.sh" >&2
  echo "[start-ai] list AI_DIR:" >&2
  ls -al "${AI_DIR}" >&2 || true
  exit 11
fi

: "${RELEASE_ID:?RELEASE_ID is required}"
: "${ARCHIVE_URL:?ARCHIVE_URL is required}"

ARTIFACT_DIR="/home/ubuntu/artifact/ai"
RELEASE_DIR="${ARTIFACT_DIR}/${RELEASE_ID}"
PKG_TMP="/tmp/bizkit-ai-${RELEASE_ID}.tar.gz"

sudo mkdir -p "${RELEASE_DIR}"

echo "[start-ai] download -> ${PKG_TMP}"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "${ARCHIVE_URL}" -o "${PKG_TMP}"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "${PKG_TMP}" "${ARCHIVE_URL}"
else
  echo "[start-ai] neither curl nor wget is installed" >&2
  exit 10
fi

echo "[start-ai] extract -> ${RELEASE_DIR}"
sudo tar -xzf "${PKG_TMP}" -C "${RELEASE_DIR}"
sudo rm -f "${PKG_TMP}"

sudo chown -R ubuntu:ubuntu "${RELEASE_DIR}" || true

if [[ ! -f "${REV_DIR}/deploy-ai.sh" ]]; then
  echo "[start-ai] deploy-ai.sh not found at ${REV_DIR}/deploy-ai.sh" >&2
  ls -al "${REV_DIR}" >&2 || true
  exit 12
fi

chmod +x "${REV_DIR}/deploy-ai.sh"
exec "${REV_DIR}/deploy-ai.sh" "${RELEASE_ID}"

