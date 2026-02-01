#!/usr/bin/env bash
set -euo pipefail

# 기본 헬스 URL (원하면 .env-ai에서 HEALTH_URL로 덮어쓰기)
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/ai/health}"

curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null
systemctl is-active --quiet bizkit-ai.service
echo "[codedeploy][ai] validate OK"

