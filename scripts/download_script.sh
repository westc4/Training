#!/usr/bin/env bash
set -euo pipefail

export ACCOUNT_ID=""   # or export it outside
export AWS_ACCESS_KEY_ID=""        # or export it outside
export AWS_SECRET_ACCESS_KEY=""  # or export it outside

BUCKET="music-train"

TARGET_PREFIX="data"              # no leading slash
#TARGET_PREFIX="data/experiments"              # no leading slash
BASE_DIR="/root/workspace"                       # absolute local base

# ---- checks (won't print secrets) ----
echo "[check] ACCOUNT_ID=${ACCOUNT_ID}"
echo "[check] AWS_ACCESS_KEY_ID len=${#AWS_ACCESS_KEY_ID}"
echo "[check] AWS_SECRET_ACCESS_KEY len=${#AWS_SECRET_ACCESS_KEY}"



# ---- download ----
mkdir -p "${BASE_DIR}/${TARGET_PREFIX}"

s5cmd --endpoint-url "https://${ACCOUNT_ID}.r2.cloudflarestorage.com" \
  cp "s3://${BUCKET}/${TARGET_PREFIX}/*" "${BASE_DIR}/${TARGET_PREFIX}/"

# ---- upload ----
s5cmd --endpoint-url "https://${ACCOUNT_ID}.r2.cloudflarestorage.com" \
  sync "${BASE_DIR}/${TARGET_PREFIX}/" "s3://${BUCKET}/${TARGET_PREFIX}/"