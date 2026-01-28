#!/usr/bin/env bash
set -euo pipefail

# Make the gcloud installer non-interactive
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

cd /

# Install Google Cloud SDK if missing
GCLOUD_BIN="$HOME/google-cloud-sdk/bin/gcloud"
if [[ -x "$GCLOUD_BIN" ]]; then
  echo "[startup] gcloud already installed at $GCLOUD_BIN; skipping install."
else
  echo "[startup] Installing Google Cloud SDK (non-interactive)..."
  curl -sSL https://sdk.cloud.google.com | bash
fi

# Add gcloud to PATH for this session
if [[ -f "$HOME/google-cloud-sdk/path.bash.inc" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/google-cloud-sdk/path.bash.inc"
elif [[ -d "$HOME/google-cloud-sdk/bin" ]]; then
  export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

# Workspace setup
mkdir -p /root/workspace/data
cd /root/workspace

# Clone repo if missing
if [[ ! -d "/root/workspace/Training/.git" ]]; then
  git clone https://github.com/westc4/Training
else
  echo "[startup] Repo already exists: /root/workspace/Training"
fi

# Configure git identity only if env vars exist
if [[ -n "${GITMAIL:-}" ]]; then
  git config --global user.email "$GITMAIL"
fi
if [[ -n "${GITUSER:-}" ]]; then
  git config --global user.name "$GITUSER"
fi

echo "[startup] Done."


gcloud storage ls gs://music_train


gsutil -m \
  -o "GSUtil:parallel_process_count=128" \
  -o "GSUtil:parallel_thread_count=16" \
  cp -r "gs://music_train/data/fma_large" "/root/workspace/data"

gcsfuse music_train /root/

git clone https://github.com/westc4/Training

gcloud auth login

top -o %CPU