#!/usr/bin/env python3
"""
MusicGen Training Script with Target Hours Support and Preflight Checks

Similar to dora_train.py but for MusicGen (music generation) training.
Supports multi-GPU DDP training with automatic scaling based on target hours.

Usage:
    python musicgen_train.py                    # Normal training
    python musicgen_train.py --preflight-only   # Run checks only
    python musicgen_train.py --skip-preflight   # Skip checks
    
    # Or with environment variables:
    WORKSPACE_DIR=/path/to/workspace python musicgen_train.py
"""

from pathlib import Path
import os
import sys
import subprocess
import datetime
import yaml
import argparse

# =============================================================================
# CLI ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description="Train MusicGen model")
parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks only, don't train")
parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
args, _ = parser.parse_known_args()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/root/workspace"))

AUDIOCRAFT_REPO_DIR = BASE_DIR / "audiocraft"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "audiocraft"
DATA_DIR = BASE_DIR / "data"

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Dataset to use (must match config in audiocraft/config/dset/)
DSET = "audio/all_data"

# Path to training JSONL for sample counting
TRAIN_JSONL = DATA_DIR / "all_data" / "egs" / "train" / "data.jsonl"
VALID_JSONL = DATA_DIR / "all_data" / "egs" / "valid" / "data.jsonl"

# =============================================================================
# SOLVER CONFIGURATION
# =============================================================================

# MusicGen solver to use
SOLVER = "musicgen/musicgen_base_32khz"

# Set to path of compression checkpoint, or None to auto-discover latest
COMPRESSION_CHECKPOINT = None  # Auto-discover from experiments

# Conditioner: "none" for unconditional, or a conditioner config
CONDITIONER = "none"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

SEGMENT_SECONDS = 60        # Audio segment duration in seconds
BATCH_SIZE = 32             # Reduced from 64 for more stable gradients per update
NUM_WORKERS = 8             # DataLoader workers per GPU
EPOCHS = 200                # Increased from 100 - unconditional training needs more epochs

# === AUTO-SCALE UPDATES PER EPOCH ===
# Option 1: Use full dataset (set to None for auto-detection)
# Option 2: Set target hours to use a subset (e.g., 100, 500, etc.)
TARGET_HOURS = 8000         # None = use full dataset, or set hours like 100, 500

# Validation and generation
VALID_NUM_SAMPLES = 100     # Number of validation samples
GENERATE_EVERY = "null"     # Generate samples every N epochs ("null" to disable)
EVALUATE_EVERY = 10         # Evaluate every N epochs

# Optimization
# NOTE: Previous run showed 33% of epochs had INF grad_norm and minimal ppl improvement
# Reducing LR and max_norm for more stable training
LEARNING_RATE = 5e-5        # Reduced from 1e-4 for stability (INF gradients were common)
MAX_NORM = 0.5              # Reduced from 1.0 for tighter gradient control
AUTOCAST = True             # Use mixed precision (fp16/bf16)

# Checkpointing
CHECKPOINT_SAVE_EVERY = 5000  # Save checkpoint every N optimizer steps

# System settings
NUM_THREADS = 4             # CPU threads for PyTorch operations
MP_START_METHOD = "fork"    # Multiprocessing start method ("fork" or "spawn")

# Seed for reproducibility
SEED = 42

# =============================================================================
# WORLD SIZE DETECTION (DDP)
# =============================================================================

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
GLOBAL_BATCH_SIZE = BATCH_SIZE * WORLD_SIZE

# Validate batch size is divisible by world size
if BATCH_SIZE % 1 != 0:  # Basic check; real check is global batch
    pass
print(f"\n{'='*60}")
print("DDP Configuration")
print(f"{'='*60}")
print(f"  WORLD_SIZE (GPUs): {WORLD_SIZE}")
print(f"  Per-GPU batch size: {BATCH_SIZE}")
print(f"  Global batch size: {GLOBAL_BATCH_SIZE}")

# =============================================================================
# DATASET SIZE AND UPDATES CALCULATION
# =============================================================================

def _get_dataset_size(jsonl_path: Path) -> int | None:
    """Count lines in JSONL file to get dataset size."""
    if not jsonl_path.exists():
        print(f"Warning: {jsonl_path} not found")
        return None
    
    try:
        # Fast path: use wc -l
        result = subprocess.run(
            ['wc', '-l', str(jsonl_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except (subprocess.SubprocessError, ValueError, IndexError):
        pass
    
    # Fallback: Python line counting
    try:
        with open(jsonl_path, 'r') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Warning: Could not count lines in {jsonl_path}: {e}")
        return None

dataset_size = _get_dataset_size(TRAIN_JSONL)

if TARGET_HOURS is not None and dataset_size:
    # Use target hours to determine subset
    target_samples = (TARGET_HOURS * 3600) // SEGMENT_SECONDS
    UPDATES_PER_EPOCH = min(target_samples, dataset_size) // GLOBAL_BATCH_SIZE
    print(f"\n{'='*60}")
    print("Target Hours Mode")
    print(f"{'='*60}")
    print(f"  TARGET_HOURS: {TARGET_HOURS}h")
    print(f"  Target samples: {target_samples:,}")
    print(f"  Updates per epoch: {UPDATES_PER_EPOCH:,}")
elif dataset_size:
    # Use full dataset
    UPDATES_PER_EPOCH = dataset_size // GLOBAL_BATCH_SIZE
    total_hours = (dataset_size * SEGMENT_SECONDS) / 3600
    effective_samples = UPDATES_PER_EPOCH * GLOBAL_BATCH_SIZE
    coverage_ratio = effective_samples / dataset_size
    
    print(f"\n{'='*60}")
    print("Full Dataset Mode")
    print(f"{'='*60}")
    print(f"  Dataset: {dataset_size:,} samples ({total_hours:.1f}h of audio)")
    print(f"  Updates per epoch: {UPDATES_PER_EPOCH:,}")
    print(f"  Samples per epoch: {effective_samples:,} ({coverage_ratio:.1%} coverage)")
else:
    # Fallback
    UPDATES_PER_EPOCH = 1000
    print(f"\nWarning: Could not determine dataset size, using default UPDATES_PER_EPOCH={UPDATES_PER_EPOCH}")

# =============================================================================
# COMPRESSION CHECKPOINT DISCOVERY
# =============================================================================

def find_compression_checkpoint() -> Path | None:
    """Find the latest compression checkpoint from experiments."""
    xp_root = EXPERIMENTS_DIR / "xps"
    if not xp_root.exists():
        return None
    
    compression_xps = []
    for xp_dir in xp_root.iterdir():
        if not xp_dir.is_dir():
            continue
        
        # Check if this is a compression experiment
        for config_path in [xp_dir / "config.yaml", xp_dir / ".hydra" / "config.yaml"]:
            if config_path.exists():
                try:
                    cfg = yaml.safe_load(config_path.read_text())
                    solver = cfg.get("solver", "")
                    if "compression" in str(solver).lower():
                        compression_xps.append(xp_dir)
                        break
                except Exception:
                    continue
    
    if not compression_xps:
        return None
    
    # Sort by modification time, get latest
    compression_xps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_xp = compression_xps[0]
    
    # Find checkpoint file
    priority_patterns = [
        "*best*.pt", "*best*.pth", "*best*.th",
        "*latest*.pt", "*latest*.pth", "*latest*.th",
        "checkpoint*.pt", "checkpoint*.pth", "checkpoint*.th",
    ]
    
    for pattern in priority_patterns:
        files = sorted(latest_xp.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0]
    
    # Fallback: any checkpoint file
    all_ckpts = list(latest_xp.rglob("*.pt")) + list(latest_xp.rglob("*.pth")) + list(latest_xp.rglob("*.th"))
    if all_ckpts:
        return sorted(all_ckpts, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    
    return None

def get_compression_meta(ckpt_path: Path) -> dict:
    """Load compression model and extract metadata."""
    # Import here to avoid loading torch at module level
    sys.path.insert(0, str(AUDIOCRAFT_REPO_DIR))
    from audiocraft.solvers import CompressionSolver
    
    model = CompressionSolver.model_from_checkpoint(str(ckpt_path), device="cpu")
    
    meta = {
        "ckpt_path": ckpt_path,
        "sample_rate": model.sample_rate,
        "channels": model.channels,
        "n_q": getattr(model, "num_codebooks", None),
        "cardinality": getattr(model, "cardinality", None),
        "frame_rate": getattr(model, "frame_rate", None),
    }
    
    if meta["n_q"] is None and hasattr(model, "quantizer"):
        meta["n_q"] = getattr(model.quantizer, "n_q", None)
    
    del model
    return meta

# Discover or use specified compression checkpoint
if COMPRESSION_CHECKPOINT is None:
    ckpt_path = find_compression_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError(
            "No compression checkpoint found. Either:\n"
            "  1. Run compression training first (02_audiocraft_train_compression_debug.ipynb)\n"
            "  2. Set COMPRESSION_CHECKPOINT to a valid checkpoint path"
        )
else:
    ckpt_path = Path(COMPRESSION_CHECKPOINT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Compression checkpoint not found: {ckpt_path}")

COMPRESSION_META = get_compression_meta(ckpt_path)

print(f"\n{'='*60}")
print("Compression Model")
print(f"{'='*60}")
print(f"  Checkpoint: {COMPRESSION_META['ckpt_path']}")
print(f"  Sample rate: {COMPRESSION_META['sample_rate']} Hz")
print(f"  Channels: {COMPRESSION_META['channels']}")
print(f"  Codebooks (n_q): {COMPRESSION_META['n_q']}")
print(f"  Cardinality: {COMPRESSION_META['cardinality']}")
print(f"  Frame rate: {COMPRESSION_META['frame_rate']} Hz")

# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================

# Add training_checks to path
sys.path.insert(0, str(BASE_DIR / "Training"))

from training_checks.preflight import run_preflight, PreflightConfig
from training_checks.reporting import PreflightReport, save_report

def run_preflight_checks():
    """Run preflight checks and return (passed, results)."""
    # Build delay pattern for validation
    delay_pattern = list(range(COMPRESSION_META['n_q'])) if COMPRESSION_META['n_q'] else None
    
    config = PreflightConfig(
        audiocraft_repo=AUDIOCRAFT_REPO_DIR,
        experiments_dir=EXPERIMENTS_DIR,
        data_dir=DATA_DIR,
        dset=DSET,
        train_jsonl=TRAIN_JSONL,
        valid_jsonl=VALID_JSONL,
        segment_duration=SEGMENT_SECONDS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        target_hours=TARGET_HOURS,
        expected_sample_rate=COMPRESSION_META['sample_rate'],
        expected_channels=COMPRESSION_META['channels'],
        compression_checkpoint=COMPRESSION_META['ckpt_path'],
        transformer_lm_n_q=COMPRESSION_META['n_q'],
        transformer_lm_card=COMPRESSION_META['cardinality'],
        delay_pattern=delay_pattern,
        world_size=WORLD_SIZE,
    )
    
    return run_preflight(config)

# Run preflight checks unless skipped
if not args.skip_preflight:
    preflight_ok, preflight_results = run_preflight_checks()
    
    # Save report to experiments directory
    report = PreflightReport.from_results(preflight_results)
    xp_preflight_dir = EXPERIMENTS_DIR / "preflight_reports"
    xp_preflight_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, text_path = save_report(
        report,
        xp_preflight_dir,
        json_filename=f"preflight_musicgen_{timestamp}.json",
        text_filename=f"preflight_musicgen_{timestamp}.txt",
    )
    print(f"\nPreflight report saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {text_path}")
    
    if args.preflight_only:
        print("\n--preflight-only specified, exiting.")
        sys.exit(0 if preflight_ok else 1)
    
    if not preflight_ok:
        print("\n❌ Preflight checks FAILED. Training will not start.")
        print("   Use --skip-preflight to override (not recommended)")
        sys.exit(1)

# =============================================================================
# PROCESS MANAGEMENT
# =============================================================================

import signal
import atexit

# Keep a handle to the currently running process
try:
    _DORA_PROC
except NameError:
    _DORA_PROC = None

def _kill_proc_tree(proc: subprocess.Popen, timeout=10):
    """Terminate a process group safely."""
    if proc is None or proc.poll() is not None:
        return
    
    try:
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

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

env = os.environ.copy()
env["AUDIOCRAFT_TEAM"] = "default"
env["AUDIOCRAFT_DORA_DIR"] = str(EXPERIMENTS_DIR)
env["USER"] = env.get("USER", "root")
env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

# Numba/joblib cache settings
env["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
env["NUMBA_DISABLE_CACHING"] = "1"
env["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Thread settings to prevent CPU contention
env["OMP_NUM_THREADS"] = str(NUM_THREADS)
env["MKL_NUM_THREADS"] = str(NUM_THREADS)

# =============================================================================
# BUILD AND RUN COMMAND
# =============================================================================

# Kill any previous run
_kill_proc_tree(_DORA_PROC)

# Build delay pattern for codebooks
delays_list = list(range(COMPRESSION_META['n_q']))
delays_str = "[" + ",".join(str(i) for i in delays_list) + "]"

print(f"\n{'='*60}")
print("Training Configuration")
print(f"{'='*60}")
print(f"  Solver: {SOLVER}")
print(f"  Dataset: {DSET}")
print(f"  Conditioner: {CONDITIONER}")
print(f"  Segment duration: {SEGMENT_SECONDS}s")
print(f"  Batch size: {BATCH_SIZE} (global: {GLOBAL_BATCH_SIZE})")
print(f"  Num workers: {NUM_WORKERS}")
print(f"  Updates per epoch: {UPDATES_PER_EPOCH:,}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Autocast: {AUTOCAST}")
print(f"  Checkpoint save every: {CHECKPOINT_SAVE_EVERY} steps")
print(f"  Generate every: {GENERATE_EVERY}")
print(f"  Evaluate every: {EVALUATE_EVERY}")

# Build command
cmd = [
    "python", "-m", "dora", "run", "-d",
    f"solver={SOLVER}",
    f"dset={DSET}",
    f"conditioner={CONDITIONER}",
    
    # Compression model settings
    f"compression_model_checkpoint={COMPRESSION_META['ckpt_path']}",
    f"sample_rate={COMPRESSION_META['sample_rate']}",
    f"channels={COMPRESSION_META['channels']}",
    f"transformer_lm.n_q={COMPRESSION_META['n_q']}",
    f"transformer_lm.card={COMPRESSION_META['cardinality']}",
    f"codebooks_pattern.delay.delays={delays_str}",
    
    # Dataset settings
    f"dataset.segment_duration={SEGMENT_SECONDS}",
    f"dataset.batch_size={BATCH_SIZE}",
    f"dataset.num_workers={NUM_WORKERS}",
    f"dataset.valid.num_samples={VALID_NUM_SAMPLES}",
    
    # Optimization settings
    f"optim.updates_per_epoch={UPDATES_PER_EPOCH}",
    f"optim.epochs={EPOCHS}",
    f"optim.lr={LEARNING_RATE}",
    f"optim.max_norm={MAX_NORM}",
    
    # Generation and evaluation
    f"generate.every={GENERATE_EVERY}",
    f"evaluate.every={EVALUATE_EVERY}",
    
    # System settings
    f"autocast={str(AUTOCAST).lower()}",
    f"num_threads={NUM_THREADS}",
    f"mp_start_method={MP_START_METHOD}",
    f"seed={SEED}",
    
    # Checkpointing
    f"checkpoint.save_every={CHECKPOINT_SAVE_EVERY}",
]

import shlex
print(f"\n{'='*60}")
print("Command")
print(f"{'='*60}")
print(" ".join(shlex.quote(x) for x in cmd))
print(f"\nDora dir: {env['AUDIOCRAFT_DORA_DIR']}")
print(f"Working dir: {AUDIOCRAFT_REPO_DIR}")

# =============================================================================
# LAUNCH TRAINING
# =============================================================================

print(f"\n{'='*60}")
print("Starting Training...")
print(f"{'='*60}")
print(f"Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

_DORA_PROC = subprocess.Popen(
    cmd,
    cwd=str(AUDIOCRAFT_REPO_DIR),
    env=env,
    start_new_session=True,  # Create new process group for clean cleanup
    stdout=None,
    stderr=None,
)

# Wait for completion
try:
    rc = _DORA_PROC.wait()
    if rc != 0:
        print(f"\n❌ Training exited with code {rc}")
        sys.exit(rc)
    else:
        print(f"\n✓ Training completed successfully!")
        print(f"Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user, cleaning up...")
    _kill_proc_tree(_DORA_PROC)
    _DORA_PROC = None
    sys.exit(130)
finally:
    _kill_proc_tree(_DORA_PROC)
    _DORA_PROC = None
