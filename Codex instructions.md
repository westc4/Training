

# CODEX TASK: Reproduce FMA-small-mini → AudioCraft (Dora) training in 2 RunPod Jupyter notebooks

## Goal
Create **two Jupyter notebooks** that fully reproduce the end-to-end workflow we just executed:

1) **Notebook A (Setup + Data):**
   - Sets up system + Python dependencies on a RunPod machine
   - Clones the correct AudioCraft repo
   - Downloads / prepares a **small FMA-based mini dataset**
   - Creates 10s mono 32k segments
   - Produces train/valid split manifests
   - Creates `egs/train` and `egs/valid` folders (symlinks)
   - Generates AudioCraft-native `data.jsonl` (includes `duration` + `sample_rate`)
   - Creates Hydra config `config/dset/audio/fma_small_mini.yaml`

2) **Notebook B (Train):**
   - Validates environment + dataset artifacts exist
   - Runs `python -m dora run solver=compression/debug ...` successfully
   - Writes logs to `/workspace/experiments/audiocraft/xps/<xp_id>`
   - Includes sanity checks (counts, sample durations, first lines of data.jsonl, etc.)

## Constraints / Assumptions
- Environment: RunPod Jupyter, working directory `/workspace`
- GPU: large GPU (>= 96GB VRAM assumed ok), CUDA available
- Python: 3.11
- Keep everything **idempotent** (safe to re-run)
- Prefer symlinks over copies to save disk
- Do NOT require manual steps outside notebook cells
- Use `%%bash` for shell commands; Python cells for Python logic
- If xFormers is incompatible, do not fail the run—silence or uninstall is acceptable

## Deliverables
- `notebooks/01_fma_small_mini_setup.ipynb`
- `notebooks/02_audiocraft_train_compression_debug.ipynb`

Both notebooks should be readable and “single-click runnable” top-to-bottom.

---

# Notebook A: 01_fma_small_mini_setup.ipynb

## Sections & Required Cells

### 0. Parameters (Python cell)
Define:
- `BASE_DIR=/workspace`
- `DATA_DIR=/workspace/data/fma_small_mini`
- `AUDIOCRAFT_REPO_DIR=/workspace/audiocraft`
- `EXPERIMENTS_DIR=/workspace/experiments/audiocraft`
- `SEGMENT_SECONDS=10`
- `TARGET_SR=32000`
- `CHANNELS=1`
- `TRAIN_RATIO=0.9`
- `RANDOM_SEED=42`
- `NUM_SAMPLES_TOTAL` (small fixed value; e.g., 100 tracks or 300 segments)

### 1. System dependencies (%%bash)
Install:
- `pkg-config`
- `ffmpeg`
- `libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswresample-dev libswscale-dev`

Also verify:
- `ffmpeg -version`
- `ffprobe -version`

### 2. Python deps (%%bash)
- Upgrade build tools:
  - `python -m pip install -U pip setuptools wheel`
- Install dora:
  - `python -m pip install -U dora-search`
- Install torch CUDA (try cu121 first):
  - `python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Install transformers (pin if needed):
  - `python -m pip install -U transformers`

Add a quick verification Python cell:
- `import torch; print(torch.__version__, torch.cuda.is_available())`

### 3. Clone AudioCraft repo (%%bash)
- Ensure a clean clone:
  - remove existing `/workspace/audiocraft` if present (guarded)
  - `git clone https://github.com/facebookresearch/audiocraft.git /workspace/audiocraft`
- Verify repo contains config directory:
  - `ls -la /workspace/audiocraft/config`

### 4. Install AudioCraft (%%bash)
- `cd /workspace/audiocraft`
- `python -m pip install -r requirements.txt`
- `python -m pip install -e .`
- Verify import (Python cell):
  - `import audiocraft; print("audiocraft OK")`

### 5. Optional: handle xformers mismatch (%%bash)
If xformers emits warnings / incompatible:
- Either uninstall:
  - `python -m pip uninstall -y xformers || true`
OR set env var to reduce noise (acceptable).

### 6. Download FMA small subset (Python + %%bash)
Implement a minimal “mini” dataset strategy:
- Download a small archive (or use FMA official zip) to a raw folder:
  - `/workspace/data/fma_raw/`
- Extract
- Select a limited subset of audio files (by count)
- Convert to wav mono 32k (ffmpeg) into:
  - `/workspace/data/fma_small_mini/wav_32k_mono/`

**Must include checks:**
- Print file counts
- Print a few filenames

### 7. Segment audio into 10s wav (Python cell)
Create:
- `/workspace/data/fma_small_mini/segments_10s/`
Segment each wav into `SEGMENT_SECONDS` chunks.
Naming convention must be stable and unique (e.g., `<source_id>_<segment_idx>.wav`).

**Must include checks:**
- Count segments created
- Validate 5 random segments are ~10s using ffprobe

### 8. Create train/valid manifests (Python cell)
Create:
- `/workspace/data/fma_small_mini/manifests/train.jsonl`
- `/workspace/data/fma_small_mini/manifests/valid.jsonl`

Each line:
```json
{"path": "/workspace/data/fma_small_mini/segments_10s/<file>.wav"}
```

Split should be deterministic using `RANDOM_SEED` and `TRAIN_RATIO`.

### 9. Create `egs/train` and `egs/valid` using symlinks (Python cell)
Create:
- `/workspace/data/fma_small_mini/egs/train/`
- `/workspace/data/fma_small_mini/egs/valid/`

For each path in manifests:
- Create symlink in corresponding egs folder using basename

**Must include checks:**
- Count wavs in egs/train and egs/valid
- Ensure all symlink targets exist

### 10. Generate AudioCraft-native data.jsonl (Python cell)
In each split folder, create:
- `/workspace/data/fma_small_mini/egs/train/data.jsonl`
- `/workspace/data/fma_small_mini/egs/valid/data.jsonl`

Each line MUST include:
```json
{
  "path": "/workspace/data/fma_small_mini/egs/train/<file>.wav",
  "duration": <float_seconds>,
  "sample_rate": <int>,
  "channels": 1
}
```

Compute `duration` and `sample_rate` via `ffprobe` (parallel ok).
**Must include checks:**
- Print first line of each data.jsonl parsed
- Verify `duration` ~ 10s for random samples

### 11. Create Hydra dataset config for Dora (%%bash)
Create:
- `/workspace/audiocraft/config/dset/audio/fma_small_mini.yaml`

Content (exact):
```yaml
# @package __global__

datasource:
  max_sample_rate: 32000
  max_channels: 1
  train: /workspace/data/fma_small_mini/egs/train
  valid: /workspace/data/fma_small_mini/egs/valid
  evaluate: /workspace/data/fma_small_mini/egs/valid
  generate: /workspace/data/fma_small_mini/egs/valid
```

Verify file exists and print it.

### 12. Final “ready-to-train” checklist (Python cell)
Print:
- torch version + cuda available
- audiocraft import ok
- train/valid counts
- sample jsonl line(s)
- config file path exists
- experiments dir exists or can be created

---

# Notebook B: 02_audiocraft_train_compression_debug.ipynb

## Sections & Required Cells

### 0. Parameters (Python cell)
- `AUDIOCRAFT_REPO_DIR=/workspace/audiocraft`
- `EXPERIMENTS_DIR=/workspace/experiments/audiocraft`
- `DSET=audio/fma_small_mini`
- `SOLVER=compression/debug`
- `SEGMENT_SECONDS=10`
- `BATCH_SIZE=8`
- `NUM_WORKERS=4`
- `UPDATES_PER_EPOCH=50`
- `VALID_NUM_SAMPLES=30`
- `GENERATE_EVERY=1`
- `EVALUATE_EVERY=1`

### 1. Sanity checks (Python cell)
Validate:
- `/workspace/audiocraft/config/dset/audio/fma_small_mini.yaml` exists
- `/workspace/data/fma_small_mini/egs/train/data.jsonl` exists
- `/workspace/data/fma_small_mini/egs/valid/data.jsonl` exists
- `import audiocraft` works
- `torch.cuda.is_available()` is True

If any missing, raise a clear error with instructions: “run Notebook A”.

### 2. Run training via Dora (%%bash)
Use `%%bash` (NOT Python cell) and set env vars:

- `AUDIOCRAFT_TEAM=default`
- `AUDIOCRAFT_DORA_DIR=/workspace/experiments/audiocraft`
- `USER=root`

Command:
```bash
python -m dora run solver=compression/debug \
  dset=audio/fma_small_mini \
  dataset.segment_duration=10 \
  dataset.batch_size=8 \
  dataset.num_workers=4 \
  optim.updates_per_epoch=50 \
  dataset.valid.num_samples=30 \
  generate.every=1 \
  evaluate.every=1
```

### 3. Capture XP id + show logs path (Python cell)
Parse output or locate latest XP directory under:
- `/workspace/experiments/audiocraft/xps/`

Print:
- XP id
- log directory
- list a few key log files

### 4. Post-run validation (Python cell)
Check:
- XP dir exists
- At least one checkpoint or artifact is created (if debug run produces)
- Basic metrics/log lines exist (read tail of a log file)

### 5. Optional: quick re-run cell (%%bash)
Provide a single cell to re-run with slightly different batch size or workers.

---

# Success Criteria (must all pass)
- Notebook A finishes without manual intervention and ends with “ready-to-train” summary.
- Notebook B starts Dora successfully and creates an XP directory under `/workspace/experiments/audiocraft/xps/`.
- No `MissingConfigException` for dset.
- No `AudioMeta.__init__ missing duration/sample_rate`.
- Any xformers warnings do not stop training.

---

# Notes / Guardrails
- Keep notebooks robust to re-runs (skip downloads if already present, verify hashes/counts).
- Prefer symlinks for egs split.
- Avoid huge downloads: keep dataset small.
- Do not assume conda; use system python + pip.
- Keep all paths absolute and consistent with `/workspace`.

END TASK