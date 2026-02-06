#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

cd /

echo "[startup] Updating apt + installing utilities..."
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl unzip git tar \
  nload iftop bmon jq ffmpeg
apt-get update && apt-get install -y dstat
apt-get update && apt-get install -y iotop




# -----------------------------
# Install AWS CLI v2 (idempotent)
# -----------------------------
AWS_BIN="/usr/local/bin/aws"
if [[ -x "$AWS_BIN" ]]; then
  echo "[startup] AWS CLI already installed at $AWS_BIN; skipping install."
else
  echo "[startup] Installing AWS CLI v2..."
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64) AWS_ZIP_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" ;;
    aarch64|arm64) AWS_ZIP_URL="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" ;;
    *)
      echo "[startup] ERROR: Unsupported architecture: $ARCH"
      exit 1
      ;;
  esac

  curl -fSL -o /tmp/awscliv2.zip "$AWS_ZIP_URL"
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install || true
  rm -rf /tmp/aws /tmp/awscliv2.zip
fi

# Ensure aws is on PATH (and provide /usr/bin/aws as convenience)
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
ln -sf /usr/local/bin/aws /usr/bin/aws || true

# -----------------------------
# Install s5cmd (idempotent)
# -----------------------------
S5CMD_BIN="/usr/local/bin/s5cmd"
if [[ -x "$S5CMD_BIN" ]]; then
  echo "[startup] s5cmd already installed at $S5CMD_BIN; skipping install."
else
  echo "[startup] Installing s5cmd..."
  # pin a known-good version
  S5_VER="2.3.0"
  S5_URL="https://github.com/peak/s5cmd/releases/download/v${S5_VER}/s5cmd_${S5_VER}_Linux-64bit.tar.gz"

  curl -fSL -o /tmp/s5cmd.tar.gz "$S5_URL"
  tar -xzf /tmp/s5cmd.tar.gz -C /tmp
  install -m 0755 /tmp/s5cmd /usr/local/bin/s5cmd
  rm -f /tmp/s5cmd /tmp/s5cmd.tar.gz
fi

# -----------------------------
# Workspace setup
# -----------------------------
mkdir -p /root/workspace/data
cd /root/workspace

REPO_DIR="/root/workspace/Training"
REPO_URL="https://github.com/westc4/Training"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[startup] Repo already exists: $REPO_DIR"
else
  if [[ -d "$REPO_DIR" ]]; then
    echo "[startup] WARNING: $REPO_DIR exists but is not a git repo; leaving as-is."
    echo "[startup] If you want to re-clone, delete it and restart the container."
  else
    echo "[startup] Cloning repo -> $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
  fi
fi


# -----------------------------
# Install Libraries
# -----------------------------

WORKDIR="/root/workspace"
VENV_DIR="${WORKDIR}/.venv"

# Candidate python binaries (first match wins)
PY_CANDIDATES=(
  "/usr/bin/python"
  "/usr/bin/python3.10"
)

echo "[info] Target workdir: ${WORKDIR}"
mkdir -p "${WORKDIR}"

# Pick a python that exists and is executable
PY_BIN=""
for c in "${PY_CANDIDATES[@]}"; do
  if [[ -x "$c" ]]; then
    PY_BIN="$c"
    break
  fi
done

if [[ -z "${PY_BIN}" ]]; then
  echo "[error] No usable python found. Tried:"
  printf '  - %s\n' "${PY_CANDIDATES[@]}"
  exit 1
fi

echo "[info] Using python: ${PY_BIN}"
"${PY_BIN}" --version

# Ensure venv module is available
if ! "${PY_BIN}" -c "import venv" >/dev/null 2>&1; then
  echo "[error] Selected python cannot import 'venv' module."
  echo "        On Debian/Ubuntu, you may need: apt-get update && apt-get install -y python3-venv"
  exit 1
fi

# Create venv if missing
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[info] Creating venv at: ${VENV_DIR}"
  "${PY_BIN}" -m venv "${VENV_DIR}"
else
  echo "[info] Venv already exists at: ${VENV_DIR}"
fi

# Activate venv
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[info] Venv python: $(command -v python)"
python --version

# Upgrade pip tooling
echo "[info] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel aiohttp aiofiles tqdm

# Install packages
echo "[info] Installing packages: httpx pandas pyarrow tqdm tenacity essentia essentia-tensorflow ffmpeg"
python -m pip install --upgrade httpx pandas pyarrow tqdm tenacity essentia essentia-tensorflow ffmpeg

echo "[done] Installed into venv: ${VENV_DIR}"
echo "To use it now: source ${VENV_DIR}/bin/activate"


# export final variables (RunPod secrets)
export ACCOUNT_ID="{{ RUNPOD_SECRET_ACCOUNT_ID }}"
export AWS_ACCESS_KEY_ID="{{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}"
export AWS_SECRET_ACCESS_KEY="{{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}"
export AWS_DEFAULT_REGION="auto"
export GITMAIL="cliftonw101@gmail.com"
export GITUSER="westc4"
export HF_TOKEN="{{ RUNPOD_SECRET_HF_TOKEN }}"
export GH_PAT="{{ RUNPOD_SECRET_GH_PAT }}"

# Configure git identity only if env vars exist
if [[ -n "${GITMAIL:-}" ]]; then
  git config --global user.email "$GITMAIL"
fi
if [[ -n "${GITUSER:-}" ]]; then
  git config --global user.name "$GITUSER"
fi

echo "[startup] Installed tools:"
echo " - aws:   $(aws --version 2>&1 || true)"
echo " - s5cmd: $(s5cmd version 2>&1 || true)"
echo " - nload: $(command -v nload || true)"
echo " - iftop: $(command -v iftop || true)"
echo " - bmon:  $(command -v bmon || true)"
echo " - jq:    $(command -v jq || true)"

echo "[startup] Env sanity (lengths only):"
echo " - ACCOUNT_ID: ${ACCOUNT_ID:-<unset>}"
echo " - AWS_ACCESS_KEY_ID len: ${#AWS_ACCESS_KEY_ID}"
echo " - AWS_SECRET_ACCESS_KEY len: ${#AWS_SECRET_ACCESS_KEY}"

echo "[startup] Workspace:"
echo " - /root/workspace: $(ls -1 /root/workspace | tr '\n' ' ' || true)"
echo " - data dir: $(test -d /root/workspace/data && echo OK || echo MISSING)"
echo " - repo dir: $(test -d "$REPO_DIR" && echo OK || echo MISSING)"

echo "[startup] Done."

# NOTE:
# Don't auto-run nload/iftop/bmon here (they block).
# Run manually when needed:
#   nload
#   bmon
#   iftop

# For disk I/O monitoring, use:
# iostat -dmx 1 | awk '
# /^[a-z]/ { r += $6; w += $7; u += $NF; n++ }
# NR>1 && /Device/ { if (n>0) { printf "readMB/s=%.2f writeMB/s=%.2f avg%%util=%.1f devices=%d\n", r, w, u/n, n } r=w=u=n=0 }
# '

# To check disk usage by folder:
# du -h --max-depth=0 /root/workspace | sort -h

# For GPU monitoring, use:
# watch -n 1 nvidia-smi
# nvidia-smi topo -m