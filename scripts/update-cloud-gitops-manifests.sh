#!/usr/bin/env bash

set -euo pipefail

APP_STAGE="${1:-${APP_STAGE:-}}"
IMAGE_URI="${2:-${IMAGE_URI:-}}"
CLOUD_REPO="${CLOUD_REPO:-100-hours-a-week/18-team-18TEAM-cloud}"
CLOUD_REPO_BRANCH="${CLOUD_REPO_BRANCH:-main}"
CLOUD_REPO_TOKEN="${CLOUD_REPO_TOKEN:-}"
CLOUD_REPO_URL="${CLOUD_REPO_URL:-}"
BOT_NAME="${GITOPS_BOT_NAME:-github-actions[bot]}"
BOT_EMAIL="${GITOPS_BOT_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}"

if [[ -z "${APP_STAGE}" ]]; then
  echo "APP_STAGE is required" >&2
  exit 1
fi

if [[ "${APP_STAGE}" != "dev" && "${APP_STAGE}" != "prod" ]]; then
  echo "APP_STAGE must be one of: dev, prod" >&2
  exit 1
fi

if [[ -z "${IMAGE_URI}" ]]; then
  echo "IMAGE_URI is required" >&2
  exit 1
fi

if [[ -z "${CLOUD_REPO_URL}" ]]; then
  if [[ -z "${CLOUD_REPO_TOKEN}" ]]; then
    echo "CLOUD_REPO_TOKEN is required when CLOUD_REPO_URL is not set" >&2
    exit 1
  fi
  CLOUD_REPO_URL="https://x-access-token:${CLOUD_REPO_TOKEN}@github.com/${CLOUD_REPO}.git"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

git clone --depth=1 --branch "${CLOUD_REPO_BRANCH}" "${CLOUD_REPO_URL}" "${tmp_dir}/cloud"
cd "${tmp_dir}/cloud"

git config user.name "${BOT_NAME}"
git config user.email "${BOT_EMAIL}"

export APP_STAGE IMAGE_URI
python3 <<'PY'
import os
import re
from pathlib import Path

stage = os.environ["APP_STAGE"]
image = os.environ["IMAGE_URI"]
targets = {
    "dev": [
        "k8s/apps/ai-api/overlays/dev/resources.yaml",
        "k8s/apps/ai-worker/overlays/dev/resources.yaml",
    ],
    "prod": [
        "k8s/apps/ai-api/base/deployment.yaml",
        "k8s/apps/ai-worker/base/deployment.yaml",
    ],
}

for relpath in targets[stage]:
    path = Path(relpath)
    original = path.read_text()
    updated, count = re.subn(
        r"^(\s*image:\s*).+$",
        lambda match: f"{match.group(1)}{image}",
        original,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
      raise SystemExit(f"expected exactly one image field in {relpath}, got {count}")
    path.write_text(updated)
PY

if git diff --quiet --exit-code; then
  echo "Cloud repo already references ${IMAGE_URI}; nothing to update."
  exit 0
fi

git add \
  k8s/apps/ai-api/base/deployment.yaml \
  k8s/apps/ai-api/overlays/dev/resources.yaml \
  k8s/apps/ai-worker/base/deployment.yaml \
  k8s/apps/ai-worker/overlays/dev/resources.yaml

image_tag="${IMAGE_URI##*:}"
source_sha="${GITHUB_SHA:-unknown}"
short_sha="${source_sha:0:7}"
commit_message="chore(gitops): update AI ${APP_STAGE} image to ${image_tag}"
if [[ "${short_sha}" != "unknown" ]]; then
  commit_message="${commit_message} (${short_sha})"
fi

git commit -m "${commit_message}"

for attempt in 1 2 3; do
  if git push origin "HEAD:${CLOUD_REPO_BRANCH}"; then
    echo "Updated cloud repo ${CLOUD_REPO_BRANCH} with ${IMAGE_URI}"
    exit 0
  fi

  if [[ "${attempt}" -eq 3 ]]; then
    echo "failed to push updated cloud repo after ${attempt} attempts" >&2
    exit 1
  fi

  git fetch origin "${CLOUD_REPO_BRANCH}"
  git rebase "origin/${CLOUD_REPO_BRANCH}"
done
