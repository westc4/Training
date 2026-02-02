#!/usr/bin/env bash
set -euo pipefail

export ACCOUNT_ID="6c8acbda921889bbaf880ae4bdbf99f0"   # or export it outside
export AWS_ACCESS_KEY_ID="38633d51c6288fe36b6e26faee0b7350"        # or export it outside
export AWS_SECRET_ACCESS_KEY="206ba17204b3d6f04610e189e05f54848ef0e50013181c5b132cd9e1174878ed"  # or export it outside

BUCKET="music-train"

TARGET_PREFIX="data"              # no leading slash
#TARGET_PREFIX="data/experiments"              # no leading slash
BASE_DIR="/root/workspace"                       # absolute local base

# ---- checks (won't print secrets) ----
echo "[check] ACCOUNT_ID=${ACCOUNT_ID}"
echo "[check] AWS_ACCESS_KEY_ID len=${#AWS_ACCESS_KEY_ID}"
echo "[check] AWS_SECRET_ACCESS_KEY len=${#AWS_SECRET_ACCESS_KEY}"



# mkdir -p "${BASE_DIR}/${TARGET_PREFIX}"

# s5cmd --endpoint-url "https://${ACCOUNT_ID}.r2.cloudflarestorage.com" \
#   cp "s3://${BUCKET}/${TARGET_PREFIX}/*" "${BASE_DIR}/${TARGET_PREFIX}/"

# ---- upload ----
s5cmd --endpoint-url "https://${ACCOUNT_ID}.r2.cloudflarestorage.com" \
  sync "${BASE_DIR}/${TARGET_PREFIX}/" "s3://${BUCKET}/${TARGET_PREFIX}/"
