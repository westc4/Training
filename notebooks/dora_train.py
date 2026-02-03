from pathlib import Path
import os

# Single switch for all paths (defaults to your new location)
BASE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/root/workspace"))

AUDIOCRAFT_REPO_DIR = BASE_DIR / "audiocraft"
EXPERIMENTS_DIR     = BASE_DIR / "experiments" / "audiocraft"

#DSET = "audio/all_data"
DSET = "audio/all_data"
SOLVER = "compression/debug"
#SOLVER = "compression/encodec_musicgen_32khz"

#SEGMENT_SECONDS = 10
SEGMENT_SECONDS = 60
BATCH_SIZE = 8  # Lower to 4 if training on 60s increase to 32 for 10s
#BATCH_SIZE = 64  # Increased from 8 to improve GPU utilization

# Auto-pick workers: reduce to 4-6 to avoid CPU contention
_cpu_count = os.cpu_count() or 32
# NUM_WORKERS = min(16, max(8, _cpu_count // 4))   # 8–16 is usually the sweet spot
# NUM_WORKERS = 16 # this worked well on v84 cpu
NUM_WORKERS = 16 # this worked well on v84 cpu

# === AUTO-SCALE UPDATES PER EPOCH ===
# Option 1: Use full dataset (set to None for auto-detection)
# Option 2: Set target hours to use a subset (e.g., 100, 500, 3791 for full dataset)
TARGET_HOURS = 10  # None = use full dataset, or set a number like 100, 500, etc.

# Detect world size for DDP (Distributed Data Parallel)
def _get_world_size():
    """Get DDP world size from environment or torch, fallback to 1."""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except (ImportError, RuntimeError):
        return 1

WORLD_SIZE = _get_world_size()

# Calculate updates per epoch based on dataset size
def _get_dataset_size(jsonl_path):
    """Count lines in JSONL file to get dataset size. Uses wc -l for speed, falls back to Python."""
    try:
        # Fast path: use wc -l
        import subprocess
        result = subprocess.run(
            ['wc', '-l', str(jsonl_path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            count = int(result.stdout.split()[0])
            return count
    except (subprocess.SubprocessError, ValueError, IndexError):
        pass
    
    # Fallback: Python line counting
    try:
        with open(jsonl_path, 'r') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Warning: {jsonl_path} not found, using default updates_per_epoch=100")
        return None

TRAIN_JSONL_FOR_COUNT = BASE_DIR / "data" / "all_data" / "egs" / "train" / "data.jsonl"
dataset_size = _get_dataset_size(TRAIN_JSONL_FOR_COUNT)

# Note: Assumes JSONL lines represent pre-segmented clips of SEGMENT_SECONDS duration
GLOBAL_BATCH_SIZE = BATCH_SIZE * WORLD_SIZE  # Effective batch size across all GPUs

if TARGET_HOURS is not None and dataset_size:
    # Use target hours to determine subset
    target_samples = (TARGET_HOURS * 3600) // SEGMENT_SECONDS
    UPDATES_PER_EPOCH = min(target_samples, dataset_size) // GLOBAL_BATCH_SIZE
    print(f"Using TARGET_HOURS={TARGET_HOURS}h → {UPDATES_PER_EPOCH:,} updates/epoch")
elif dataset_size:
    # Use full dataset - one epoch = one full pass through dataset under DDP
    UPDATES_PER_EPOCH = dataset_size // GLOBAL_BATCH_SIZE
    total_hours = (dataset_size * SEGMENT_SECONDS) / 3600
    print(f"\n=== DDP Configuration ===")
    print(f"WORLD_SIZE (GPUs): {WORLD_SIZE}")
    print(f"Per-GPU batch size: {BATCH_SIZE}")
    print(f"Global batch size: {GLOBAL_BATCH_SIZE}")
    print(f"\n=== Dataset Coverage ===")
    print(f"Dataset: {dataset_size:,} samples ({total_hours:.1f}h of audio)")
    print(f"Updates per epoch: {UPDATES_PER_EPOCH:,}")
    effective_samples = UPDATES_PER_EPOCH * GLOBAL_BATCH_SIZE
    coverage_ratio = effective_samples / dataset_size
    print(f"Samples per epoch: {effective_samples:,} ({coverage_ratio:.2%} dataset coverage)")
    if coverage_ratio < 0.95:
        print(f"⚠️  Warning: Epoch covers only {coverage_ratio:.2%} of dataset (expected ~100%)")
else:
    # Fallback
    UPDATES_PER_EPOCH = 100
    print(f"Using default UPDATES_PER_EPOCH={UPDATES_PER_EPOCH}")

VALID_NUM_SAMPLES = 10
GENERATE_EVERY = "null"  # Disable automatic generation during training
EVALUATE_EVERY = "null"  # Disable automatic evaluation during training
AUTOCAST = True
# NUM_THREADS = 16  # Set number of threads for PyTorch operations
NUM_THREADS = 2  # Set number of threads for PyTorch operations
#MP_START_METHOD = "fork" # Use 'fork' to reduce overhead on Linux systems
#MP_START_METHOD = "fork" # Alternative method if issues arise with 'fork'
MP_START_METHOD = "fork" # Alternative method if issues arise with 'fork'
DATASET_BATCH_SIZE = 1 # Batch size for multi gpu setup
DATASET_NUM_SAMPLES = 2000
CHECKPOINT_SAVE = 10000  # Save checkpoint every N optimizer steps (NOT per epoch)

CONFIG_PATH = AUDIOCRAFT_REPO_DIR / "config" / "dset" / "audio" / "all_data.yaml"
TRAIN_JSONL  = BASE_DIR / "data" / "all_data" / "egs" / "train" / "data.jsonl"
VALID_JSONL  = BASE_DIR / "data" / "all_data" / "egs" / "valid" / "data.jsonl"

print(NUM_WORKERS)
print(CONFIG_PATH)
print(TRAIN_JSONL)
print(VALID_JSONL)
print(AUDIOCRAFT_REPO_DIR)

import os
import shlex
import signal
import subprocess
import atexit

# ---- Keep a handle to the currently running process (across cell reruns) ----
try:
    _DORA_PROC
except NameError:
    _DORA_PROC = None

def _kill_proc_tree(proc: subprocess.Popen, timeout=10):
    """Terminate a process group (best for dora/ddp) safely."""
    if proc is None:
        return
    if proc.poll() is not None:
        return  # already exited

    try:
        # Kill the whole process group
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

def _on_exit():
    global _DORA_PROC
    _kill_proc_tree(_DORA_PROC)
    _DORA_PROC = None

atexit.register(_on_exit)

# ---- Pass Python variables to environment ----
env = os.environ.copy()
env["AUDIOCRAFT_TEAM"] = "default"
env["AUDIOCRAFT_DORA_DIR"] = str(EXPERIMENTS_DIR)
env["USER"] = env.get("USER", "root")
env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

# ---- If a previous run is still alive, kill it before starting a new one ----
_kill_proc_tree(_DORA_PROC)
print(f"Using config: dset={DSET}, solver={SOLVER}")
print(f"Training params: segment_duration={SEGMENT_SECONDS}, batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}")
print(f"Optimizer: updates_per_epoch={UPDATES_PER_EPOCH}")
print(f"Validation: num_samples={VALID_NUM_SAMPLES}")
print(f"Evaluate every: {EVALUATE_EVERY}")
print(f"Autocast: {AUTOCAST}")
print(f"Generate every: {GENERATE_EVERY}")
print(f"Num threads: {NUM_THREADS}")
print(f"dataset.generate.batch_size: {DATASET_BATCH_SIZE}")
print(f"MP start method: {MP_START_METHOD}")
print(f"Checkpoint save every: {CHECKPOINT_SAVE}")

# ---- Build the command as a list (avoid shell=True) ----
cmd = [
    "python", "-m", "dora", "run", "-d",
    f"solver={SOLVER}",
    f"dset={DSET}",
    f"dataset.segment_duration={SEGMENT_SECONDS}",
    f"dataset.batch_size={BATCH_SIZE}",
    f"dataset.generate.batch_size={DATASET_BATCH_SIZE}",
    f"dataset.num_workers={NUM_WORKERS}",
    f"optim.updates_per_epoch={UPDATES_PER_EPOCH}",
    f"dataset.valid.num_samples={VALID_NUM_SAMPLES}",
    f"generate.every={GENERATE_EVERY}",
    f"evaluate.every={EVALUATE_EVERY}",
    f"autocast={str(AUTOCAST).lower()}",
    f"num_threads={NUM_THREADS}",
    f"mp_start_method={MP_START_METHOD}",
    f"checkpoint.save_every={CHECKPOINT_SAVE}",
    
]
#f"dataset.num_samples={DATASET_NUM_SAMPLES}",
print("Launching:", " ".join(shlex.quote(x) for x in cmd))
print("Dora dir:", env["AUDIOCRAFT_DORA_DIR"])

# ---- Start in the repo directory + new process group so we can kill all workers ----
_DORA_PROC = subprocess.Popen(
    cmd,
    cwd=str(AUDIOCRAFT_REPO_DIR),
    env=env,
    start_new_session=True,   # <-- creates a new process group/session
    stdout=None,
    stderr=None,
)

# Optional: wait for completion (Ctrl-C will stop this cell; atexit will still clean up on kernel exit)
try:
    rc = _DORA_PROC.wait()
    if rc != 0:
        raise RuntimeError(f"Dora exited with code {rc}")
finally:
    _kill_proc_tree(_DORA_PROC)
    _DORA_PROC = None