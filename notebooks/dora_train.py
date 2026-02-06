#!/usr/bin/env python3
"""
Compression Model Training Script with Preflight Checks

Trains AudioCraft compression models (EnCodec) with:
- Preflight validation of datasets, configs, and environment
- Target hours mode for subset training
- Multi-GPU DDP support
- Optional adversarial loss tuning via environment variables

Usage:
    python dora_train.py                    # Normal training
    python dora_train.py --preflight-only   # Run checks only
    python dora_train.py --skip-preflight   # Skip checks
    python dora_train.py --codec-check-every 5 --codec-check-samples 10 # Run codec quality check every 5 epochs with 10 samples

Adversarial Training Toggles (env vars):
    # Reduce adversarial weights (losses.adv=2.0, losses.feat=2.0):
    REDUCE_ADV=1 python -u dora_train.py

    # Update discriminator every 2 steps:
    ADV_EVERY=2 python -u dora_train.py

    # Scale adversarial losses to 50% (takes priority over REDUCE_ADV):
    ADV_SCALE=0.5 python -u dora_train.py

    # Scale + change update frequency:
    ADV_SCALE=0.5 ADV_EVERY=2 python -u dora_train.py
"""

from pathlib import Path
import os
import sys
import argparse

# =============================================================================
# CLI ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(description="Train AudioCraft compression model")
parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks only, don't train")
parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
args, _ = parser.parse_known_args()

# =============================================================================
# PATH CONFIGURATION  
# =============================================================================

# Single switch for all paths (defaults to your new location)
BASE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/root/workspace"))

AUDIOCRAFT_REPO_DIR = BASE_DIR / "audiocraft"
EXPERIMENTS_DIR     = BASE_DIR / "experiments" / "audiocraft"

#DSET = "audio/all_data"
DSET = "audio/all_data"
#SOLVER = "compression/debug"
SOLVER = "compression/encodec_musicgen_32khz"

#SEGMENT_SECONDS = 10
SEGMENT_SECONDS = 30  # Changed from 60: only 45% of clips are >=60s, but 89% are >=30s
BATCH_SIZE = 24  # Must be divisible by WORLD_SIZE (num GPUs). For 4 GPUs: 4, 8, 12, etc.
#BATCH_SIZE = 64  # Increased from 8 to improve GPU utilization

# =============================================================================
# TRAINING HYPERPARAMETERS (CRITICAL FOR CONVERGENCE)
# =============================================================================
# Previous runs showed SI-SNR collapse (-2.2 dB) due to:
#   1. No gradient clipping (max_norm=0.0)
#   2. Too few epochs (24 vs needed 100-200)
#   3. Learning rate potentially too high for adversarial training

LEARNING_RATE = 0.0001      # Reduced from 0.0003 for stability
MAX_NORM = 1.0              # CRITICAL: Enable gradient clipping (was 0.0!)
EPOCHS = 200                # Full training (was stopping at ~24)

# Auto-pick workers: reduce to 4-6 to avoid CPU contention
_cpu_count = os.cpu_count() or 32
# NUM_WORKERS = min(16, max(8, _cpu_count // 4))   # 8–16 is usually the sweet spot
# NUM_WORKERS = 16 # this worked well on v84 cpu
NUM_WORKERS = 16 # this worked well on v84 cpu

# === AUTO-SCALE UPDATES PER EPOCH ===
# Option 1: Use full dataset (set to None for auto-detection)
# Option 2: Set target hours to use a subset (e.g., 100, 500, 3791 for full dataset)
TARGET_HOURS = 500  # None = use full dataset, or set a number like 100, 500, etc.

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
EVALUATE_EVERY = 10       # Evaluate every 10 epochs to monitor SI-SNR
AUTOCAST = True

# =============================================================================
# CODEC QUALITY CHECK CONFIGURATION
# =============================================================================
CODEC_CHECK_EVERY = 1     # Run codec quality check every N epochs (0=disabled)
CODEC_CHECK_SAMPLES = 5    # Number of samples to test per check
# NUM_THREADS = 16  # Set number of threads for PyTorch operations
NUM_THREADS = 8  # Set number of threads for PyTorch operations
#MP_START_METHOD = "fork" # Use 'fork' to reduce overhead on Linux systems
#MP_START_METHOD = "fork" # Alternative method if issues arise with 'fork'
MP_START_METHOD = "fork" # Alternative method if issues arise with 'fork'
DATASET_BATCH_SIZE = 4 # Batch size for multi gpu setup
DATASET_NUM_SAMPLES = 2000
CHECKPOINT_SAVE = 10000  # Save checkpoint every N optimizer steps (NOT per epoch)

CONFIG_PATH = AUDIOCRAFT_REPO_DIR / "config" / "dset" / "audio" / "all_data.yaml"
TRAIN_JSONL  = BASE_DIR / "data" / "all_data" / "egs" / "train" / "data.jsonl"
VALID_JSONL  = BASE_DIR / "data" / "all_data" / "egs" / "valid" / "data.jsonl"

CODEC_CHECK_OUTPUT_DIR = BASE_DIR / "Training" / "outputs" / "codec_quality_check"

if CODEC_CHECK_EVERY > 0:
    print(f"\n=== Codec Quality Check ===")
    print(f"Running every {CODEC_CHECK_EVERY} epochs with {CODEC_CHECK_SAMPLES} samples")
    print(f"Output: {CODEC_CHECK_OUTPUT_DIR}")

# =============================================================================
# ADVERSARIAL TRAINING OVERRIDES
# =============================================================================
# Set these directly OR use env vars (env vars take priority if set)
#
# Options:
#   REDUCE_ADV = True   → losses.adv=2.0, losses.feat=2.0 (half of default)
#   ADV_SCALE  = 0.5    → losses.adv=2.0, losses.feat=2.0 (50% of default)
#   ADV_EVERY  = 2      → adversarial.every=2 (update discriminator every 2 steps)
#
# Priority: ADV_SCALE > REDUCE_ADV (if both set, ADV_SCALE wins)
# Set to None to use defaults (losses.adv=4.0, losses.feat=4.0, adversarial.every=1)

REDUCE_ADV = True      # Set to True to halve adversarial weights (losses.adv=2.0, losses.feat=2.0)
ADV_SCALE  = None      # Set to float (e.g., 0.5) to scale adversarial weights. Takes priority over REDUCE_ADV
ADV_EVERY  = 2      # Set to int >= 1 to change discriminator update frequency

# Defaults (from compression/encodec_musicgen_32khz.yaml)
BASE_ADV_LOSS = 4.0
BASE_FEAT_LOSS = 4.0

def get_adversarial_overrides():
    """
    Build adversarial overrides from script variables OR env vars.
    Env vars take priority over script variables if set.
    
    Priority: ADV_SCALE > REDUCE_ADV (if both set, ADV_SCALE wins)
    """
    overrides = {}
    summary_parts = []
    
    # Read from script variables first, then check env vars (env wins if set)
    reduce_adv_val = REDUCE_ADV
    adv_scale_val = ADV_SCALE
    adv_every_val = ADV_EVERY
    
    # Env vars override script variables
    env_reduce = os.environ.get("REDUCE_ADV", "").strip()
    env_scale = os.environ.get("ADV_SCALE", "").strip()
    env_every = os.environ.get("ADV_EVERY", "").strip()
    
    if env_reduce == "1":
        reduce_adv_val = True
    if env_scale:
        try:
            adv_scale_val = float(env_scale)
        except ValueError:
            print(f"Warning: env ADV_SCALE='{env_scale}' is not a valid float, ignoring")
    if env_every:
        try:
            adv_every_val = int(env_every)
        except ValueError:
            print(f"Warning: env ADV_EVERY='{env_every}' is not a valid int, ignoring")
    
    # Determine losses.adv and losses.feat
    adv_loss = None
    feat_loss = None
    scale_source = None
    
    # ADV_SCALE takes priority over REDUCE_ADV
    if adv_scale_val is not None:
        if adv_scale_val <= 0:
            print(f"Warning: ADV_SCALE={adv_scale_val} must be positive, ignoring")
        else:
            adv_loss = BASE_ADV_LOSS * adv_scale_val
            feat_loss = BASE_FEAT_LOSS * adv_scale_val
            scale_source = f"ADV_SCALE={adv_scale_val}"
    
    # REDUCE_ADV only applies if ADV_SCALE wasn't used
    if scale_source is None and reduce_adv_val:
        adv_loss = 2.0
        feat_loss = 2.0
        scale_source = "REDUCE_ADV=True"
    
    # Apply loss overrides
    if adv_loss is not None:
        overrides["losses.adv"] = adv_loss
        overrides["losses.feat"] = feat_loss
        summary_parts.append(f"{scale_source} → losses.adv={adv_loss}, losses.feat={feat_loss}")
    
    # ADV_EVERY override
    if adv_every_val is not None:
        if adv_every_val < 1:
            print(f"Warning: ADV_EVERY={adv_every_val} must be >= 1, ignoring")
        else:
            overrides["adversarial.every"] = adv_every_val
            summary_parts.append(f"ADV_EVERY={adv_every_val} → adversarial.every={adv_every_val}")
    
    return overrides, summary_parts

# Get adversarial overrides
ADV_OVERRIDES, ADV_SUMMARY = get_adversarial_overrides()

# Print adversarial config summary (always print to make it clear what's happening)
print(f"\n=== Adversarial Training Config ===")
print(f"Script vars: REDUCE_ADV={REDUCE_ADV}, ADV_SCALE={ADV_SCALE}, ADV_EVERY={ADV_EVERY}")
print(f"Env vars:    REDUCE_ADV={os.environ.get('REDUCE_ADV', '(not set)')}, "
      f"ADV_SCALE={os.environ.get('ADV_SCALE', '(not set)')}, "
      f"ADV_EVERY={os.environ.get('ADV_EVERY', '(not set)')}")
if ADV_OVERRIDES:
    print(f"✅ OVERRIDES ACTIVE:")
    for part in ADV_SUMMARY:
        print(f"   {part}")
else:
    print(f"   (using defaults: losses.adv=4.0, losses.feat=4.0, adversarial.every=1)")

print(NUM_WORKERS)
print(CONFIG_PATH)
print(TRAIN_JSONL)
print(VALID_JSONL)
print(AUDIOCRAFT_REPO_DIR)

# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================

# Add training_checks to path
sys.path.insert(0, str(BASE_DIR / "Training"))

from training_checks.preflight import run_preflight, PreflightConfig
from training_checks.reporting import PreflightReport, save_report

def run_preflight_checks():
    """Run preflight checks and return (passed, results)."""
    config = PreflightConfig(
        audiocraft_repo=AUDIOCRAFT_REPO_DIR,
        experiments_dir=EXPERIMENTS_DIR,
        data_dir=BASE_DIR / "data",
        dset=DSET,
        train_jsonl=TRAIN_JSONL,
        valid_jsonl=VALID_JSONL,
        segment_duration=SEGMENT_SECONDS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        target_hours=TARGET_HOURS,
        expected_sample_rate=32000,
        expected_channels=1,
        world_size=WORLD_SIZE,
        skip_compression_check=True,  # Skip for compression training (no checkpoint exists yet)
    )
    
    return run_preflight(config)

# Run preflight checks unless skipped
if not args.skip_preflight:
    preflight_ok, preflight_results = run_preflight_checks()
    
    # Save report to experiments directory
    report = PreflightReport.from_results(preflight_results)
    xp_preflight_dir = EXPERIMENTS_DIR / "preflight_reports"
    xp_preflight_dir.mkdir(parents=True, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, text_path = save_report(
        report,
        xp_preflight_dir,
        json_filename=f"preflight_{timestamp}.json",
        text_filename=f"preflight_{timestamp}.txt",
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
print(f"Optimizer: updates_per_epoch={UPDATES_PER_EPOCH}, epochs={EPOCHS}, lr={LEARNING_RATE}")
print(f"Gradient clipping: max_norm={MAX_NORM}  ← CRITICAL for stability")
print(f"Validation: num_samples={VALID_NUM_SAMPLES}")
print(f"Evaluate every: {EVALUATE_EVERY} epochs  ← Monitor SI-SNR here!")
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
    f"optim.epochs={EPOCHS}",
    f"optim.lr={LEARNING_RATE}",
    f"optim.max_norm={MAX_NORM}",  # CRITICAL: Gradient clipping to prevent divergence
    f"dataset.valid.num_samples={VALID_NUM_SAMPLES}",
    f"generate.every={GENERATE_EVERY}",
    f"evaluate.every={EVALUATE_EVERY}",
    f"autocast={str(AUTOCAST).lower()}",
    f"num_threads={NUM_THREADS}",
    f"mp_start_method={MP_START_METHOD}",
    f"checkpoint.save_every={CHECKPOINT_SAVE}",
]

# ---- Inject adversarial overrides ----
if ADV_OVERRIDES:
    print(f"\n🎯 Injecting adversarial overrides into command:")
    for key, value in ADV_OVERRIDES.items():
        override_arg = f"{key}={value}"
        cmd.append(override_arg)
        print(f"   + {override_arg}")

#f"dataset.num_samples={DATASET_NUM_SAMPLES}",
print("\nLaunching:", " ".join(shlex.quote(x) for x in cmd))
print("Dora dir:", env["AUDIOCRAFT_DORA_DIR"])

# =============================================================================
# CODEC QUALITY CHECK INTEGRATION
# =============================================================================

import re
import threading
import queue

def run_codec_quality_check(checkpoint_path: Path, epoch: int, num_samples: int, output_dir: Path):
    """Run codec quality check in a subprocess."""
    print(f"\n{'='*60}")
    print(f"CODEC QUALITY CHECK - Epoch {epoch}")
    print(f"{'='*60}")
    
    check_script = BASE_DIR / "Training" / "training_checks" / "codec_quality_check.py"
    if not check_script.exists():
        print(f"Warning: {check_script} not found, skipping codec check")
        return
    
    check_cmd = [
        "python", str(check_script),
        "--checkpoint", str(checkpoint_path),
        "--num-samples", str(num_samples),
        "--output-dir", str(output_dir / f"epoch_{epoch:04d}"),
    ]
    
    try:
        result = subprocess.run(
            check_cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("Codec check timed out (5 min)")
    except Exception as e:
        print(f"Codec check failed: {e}")
    
    print(f"{'='*60}\n")


def find_latest_checkpoint():
    """Find the latest compression checkpoint."""
    xps_dir = EXPERIMENTS_DIR / "xps"
    if not xps_dir.exists():
        return None
    
    compression_checkpoints = []
    for xp_dir in xps_dir.iterdir():
        if not xp_dir.is_dir():
            continue
        
        config_path = xp_dir / ".hydra" / "config.yaml"
        checkpoint_path = xp_dir / "checkpoint.th"
        
        if config_path.exists() and checkpoint_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg.get("solver") == "compression":
                    compression_checkpoints.append(checkpoint_path)
            except:
                pass
    
    if not compression_checkpoints:
        return None
    
    # Return most recently modified
    compression_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return compression_checkpoints[0]


def monitor_training_output(proc, output_queue):
    """Thread to read process stdout and detect epoch completions."""
    epoch_pattern = re.compile(r"Train Summary \| Epoch (\d+)")
    
    for line in iter(proc.stdout.readline, ''):
        if not line:
            break
        # Pass through to console
        print(line, end='', flush=True)
        # Check for epoch completion
        match = epoch_pattern.search(line)
        if match:
            epoch = int(match.group(1))
            output_queue.put(('epoch', epoch))
    
    output_queue.put(('done', None))


# ---- Start in the repo directory + new process group so we can kill all workers ----
if CODEC_CHECK_EVERY > 0:
    # Use PIPE to monitor output for epoch completions
    _DORA_PROC = subprocess.Popen(
        cmd,
        cwd=str(AUDIOCRAFT_REPO_DIR),
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )
    
    # Start output monitor thread
    epoch_queue = queue.Queue()
    monitor_thread = threading.Thread(
        target=monitor_training_output,
        args=(_DORA_PROC, epoch_queue),
        daemon=True
    )
    monitor_thread.start()
    
    # Main loop: check for epoch completions and run codec check
    last_checked_epoch = 0
    try:
        while True:
            try:
                msg_type, value = epoch_queue.get(timeout=1)
                
                if msg_type == 'done':
                    break
                
                if msg_type == 'epoch':
                    epoch = value
                    # Check if we should run codec check
                    if epoch > 0 and epoch % CODEC_CHECK_EVERY == 0 and epoch > last_checked_epoch:
                        last_checked_epoch = epoch
                        checkpoint = find_latest_checkpoint()
                        if checkpoint:
                            run_codec_quality_check(
                                checkpoint_path=checkpoint,
                                epoch=epoch,
                                num_samples=CODEC_CHECK_SAMPLES,
                                output_dir=CODEC_CHECK_OUTPUT_DIR,
                            )
                        else:
                            print(f"No checkpoint found at epoch {epoch}, skipping codec check")
            
            except queue.Empty:
                # Check if process is still running
                if _DORA_PROC.poll() is not None:
                    break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        rc = _DORA_PROC.poll()
        if rc is None:
            _kill_proc_tree(_DORA_PROC)
            rc = -1
        _DORA_PROC = None
        if rc != 0:
            print(f"Dora exited with code {rc}")

else:
    # Original behavior: no codec check, just run training
    _DORA_PROC = subprocess.Popen(
        cmd,
        cwd=str(AUDIOCRAFT_REPO_DIR),
        env=env,
        start_new_session=True,
        stdout=None,
        stderr=None,
    )
    
    try:
        rc = _DORA_PROC.wait()
        if rc != 0:
            raise RuntimeError(f"Dora exited with code {rc}")
    finally:
        _kill_proc_tree(_DORA_PROC)
        _DORA_PROC = None