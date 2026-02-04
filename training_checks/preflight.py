"""
Preflight Checks for AudioCraft Training

Centralized validation that runs before training starts.
Catches configuration errors, data issues, and compatibility problems early.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import json
import random
import sys
import os

# Type alias for check results
CheckResult = tuple[bool, str, dict]  # (passed, message, details)


@dataclass
class PreflightConfig:
    """Configuration for preflight checks."""
    
    # Paths
    audiocraft_repo: Path = Path("/root/workspace/audiocraft")
    experiments_dir: Path = Path("/root/workspace/experiments/audiocraft")
    data_dir: Path = Path("/root/workspace/data")
    
    # Dataset configuration
    dset: str = "audio/all_data"
    train_jsonl: Optional[Path] = None
    valid_jsonl: Optional[Path] = None
    
    # Training parameters
    segment_duration: float = 30.0
    batch_size: int = 8
    num_workers: int = 8
    target_hours: Optional[float] = None
    
    # Expected audio properties
    expected_sample_rate: int = 32000
    expected_channels: int = 1
    
    # Compression checkpoint (None = auto-discover)
    compression_checkpoint: Optional[Path] = None
    
    # MusicGen/LM configuration
    transformer_lm_n_q: Optional[int] = None
    transformer_lm_card: Optional[int] = None
    delay_pattern: Optional[list[int]] = None
    
    # Check thresholds
    manifest_sample_size: int = 200
    missing_file_fatal_pct: float = 2.0  # Fatal if > 2% files missing
    short_clip_warn_pct: float = 10.0    # Warn if > 10% clips shorter than segment
    short_clip_fatal_pct: float = 30.0   # Fatal if > 30% clips shorter
    allow_short_clips: bool = False
    
    # Roundtrip test
    roundtrip_samples: int = 4
    roundtrip_min_snr_db: float = 5.0    # Minimum acceptable SI-SNR
    
    # Skip flags
    skip_compression_check: bool = False  # Skip compression compatibility for compression training
    
    # DDP
    world_size: Optional[int] = None     # None = auto-detect
    
    def __post_init__(self):
        """Resolve paths based on dset if not explicitly set."""
        if self.train_jsonl is None:
            # Derive from dset: audio/all_data -> data/all_data/egs/train/data.jsonl
            dset_name = self.dset.replace("audio/", "")
            self.train_jsonl = self.data_dir / dset_name / "egs" / "train" / "data.jsonl"
        
        if self.valid_jsonl is None:
            dset_name = self.dset.replace("audio/", "")
            self.valid_jsonl = self.data_dir / dset_name / "egs" / "valid" / "data.jsonl"


@dataclass
class PreflightResults:
    """Aggregated results from all preflight checks."""
    
    passed: bool = True
    fatal_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    
    # Detailed check results
    checks: dict[str, dict] = field(default_factory=dict)
    
    # Resolved configuration
    resolved_config: dict[str, Any] = field(default_factory=dict)
    
    def add_fatal(self, check_name: str, message: str, details: dict = None):
        self.passed = False
        self.fatal_errors.append(f"[{check_name}] {message}")
        self.checks[check_name] = {
            "status": "FATAL",
            "message": message,
            "details": details or {},
        }
    
    def add_warning(self, check_name: str, message: str, details: dict = None):
        self.warnings.append(f"[{check_name}] {message}")
        self.checks[check_name] = {
            "status": "WARNING",
            "message": message,
            "details": details or {},
        }
    
    def add_ok(self, check_name: str, message: str, details: dict = None):
        self.info.append(f"[{check_name}] {message}")
        self.checks[check_name] = {
            "status": "OK",
            "message": message,
            "details": details or {},
        }


# =============================================================================
# TASK 1A: Manifest Existence and Schema Validation
# =============================================================================

def check_manifest_exists(config: PreflightConfig, results: PreflightResults) -> bool:
    """Check that train and valid manifests exist and are readable."""
    
    for split, jsonl_path in [("train", config.train_jsonl), ("valid", config.valid_jsonl)]:
        check_name = f"manifest_{split}_exists"
        
        if not jsonl_path.exists():
            results.add_fatal(check_name, f"{split} manifest not found: {jsonl_path}")
            return False
        
        # Try to read first line
        try:
            with open(jsonl_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    results.add_fatal(check_name, f"{split} manifest is empty: {jsonl_path}")
                    return False
                
                # Validate JSON
                try:
                    json.loads(first_line)
                except json.JSONDecodeError as e:
                    results.add_fatal(check_name, f"{split} manifest has invalid JSON: {e}")
                    return False
            
            results.add_ok(check_name, f"{split} manifest exists and is readable")
        
        except Exception as e:
            results.add_fatal(check_name, f"Cannot read {split} manifest: {e}")
            return False
    
    return True


def check_manifest_schema(config: PreflightConfig, results: PreflightResults) -> bool:
    """Validate manifest schema and sample files for existence."""
    
    required_fields = {"path", "duration", "sample_rate"}
    optional_fields = {"channels", "n_frames"}
    
    all_ok = True
    
    for split, jsonl_path in [("train", config.train_jsonl), ("valid", config.valid_jsonl)]:
        check_name = f"manifest_{split}_schema"
        
        # Count total lines and sample
        total_lines = 0
        samples = []
        
        try:
            with open(jsonl_path, 'r') as f:
                lines = f.readlines()
                total_lines = len(lines)
                
                if total_lines == 0:
                    results.add_fatal(check_name, f"{split} manifest is empty")
                    all_ok = False
                    continue
                
                # Random sample
                sample_indices = random.sample(
                    range(total_lines),
                    min(config.manifest_sample_size, total_lines)
                )
                samples = [json.loads(lines[i]) for i in sample_indices]
        
        except Exception as e:
            results.add_fatal(check_name, f"Failed to read {split} manifest: {e}")
            all_ok = False
            continue
        
        # Validate schema
        schema_errors = []
        missing_files = []
        sr_mismatches = []
        channel_mismatches = []
        durations = []
        
        for i, entry in enumerate(samples):
            # Check required fields
            missing_fields = required_fields - set(entry.keys())
            if missing_fields:
                schema_errors.append(f"Entry {i}: missing fields {missing_fields}")
                continue
            
            # Check file exists
            audio_path = Path(entry["path"])
            if not audio_path.exists():
                missing_files.append(str(audio_path))
            
            # Check duration
            duration = entry.get("duration", 0)
            if duration <= 0 or duration != duration:  # NaN check
                schema_errors.append(f"Entry {i}: invalid duration {duration}")
            else:
                durations.append(duration)
            
            # Check sample rate
            sr = entry.get("sample_rate", 0)
            if sr != config.expected_sample_rate:
                sr_mismatches.append((str(audio_path), sr))
            
            # Check channels (if present)
            ch = entry.get("channels")
            if ch is not None and ch != config.expected_channels:
                channel_mismatches.append((str(audio_path), ch))
        
        # Evaluate results
        missing_pct = (len(missing_files) / len(samples)) * 100 if samples else 0
        sr_mismatch_pct = (len(sr_mismatches) / len(samples)) * 100 if samples else 0
        
        details = {
            "total_entries": total_lines,
            "sampled_entries": len(samples),
            "schema_errors": len(schema_errors),
            "missing_files": len(missing_files),
            "missing_file_pct": round(missing_pct, 2),
            "sr_mismatches": len(sr_mismatches),
            "sr_mismatch_pct": round(sr_mismatch_pct, 2),
            "channel_mismatches": len(channel_mismatches),
            "duration_stats": {},
        }
        
        if durations:
            import statistics
            sorted_dur = sorted(durations)
            p95_idx = int(len(sorted_dur) * 0.95)
            details["duration_stats"] = {
                "min": round(min(durations), 2),
                "max": round(max(durations), 2),
                "median": round(statistics.median(durations), 2),
                "p95": round(sorted_dur[p95_idx] if p95_idx < len(sorted_dur) else sorted_dur[-1], 2),
                "mean": round(statistics.mean(durations), 2),
            }
        
        # Determine status
        if schema_errors:
            results.add_fatal(check_name, f"Schema errors in {split}: {len(schema_errors)} issues", details)
            all_ok = False
        elif missing_pct > config.missing_file_fatal_pct:
            results.add_fatal(
                check_name,
                f"{split}: {missing_pct:.1f}% files missing (threshold: {config.missing_file_fatal_pct}%)",
                details
            )
            all_ok = False
        elif sr_mismatch_pct > 0:
            results.add_fatal(
                check_name,
                f"{split}: {sr_mismatch_pct:.1f}% sample rate mismatches (expected {config.expected_sample_rate}Hz)",
                details
            )
            all_ok = False
        elif missing_pct > 0:
            results.add_warning(
                check_name,
                f"{split}: {missing_pct:.1f}% files missing (below fatal threshold)",
                details
            )
        else:
            results.add_ok(check_name, f"{split} schema valid ({total_lines:,} entries)", details)
    
    return all_ok


# =============================================================================
# TASK 1B: Segment Duration Feasibility
# =============================================================================

def check_segment_duration(config: PreflightConfig, results: PreflightResults) -> bool:
    """Check if clips are long enough for segment_duration."""
    
    check_name = "segment_duration_feasibility"
    
    all_durations = []
    
    for split, jsonl_path in [("train", config.train_jsonl), ("valid", config.valid_jsonl)]:
        try:
            with open(jsonl_path, 'r') as f:
                lines = f.readlines()
                sample_indices = random.sample(
                    range(len(lines)),
                    min(config.manifest_sample_size, len(lines))
                )
                for i in sample_indices:
                    entry = json.loads(lines[i])
                    dur = entry.get("duration", 0)
                    if dur > 0:
                        all_durations.append(dur)
        except Exception:
            pass
    
    if not all_durations:
        results.add_warning(check_name, "No duration data available")
        return True
    
    short_clips = [d for d in all_durations if d < config.segment_duration]
    short_pct = (len(short_clips) / len(all_durations)) * 100
    
    details = {
        "segment_duration": config.segment_duration,
        "sampled_clips": len(all_durations),
        "short_clips": len(short_clips),
        "short_clip_pct": round(short_pct, 2),
        "min_duration": round(min(all_durations), 2),
        "max_duration": round(max(all_durations), 2),
    }
    
    if short_pct > config.short_clip_fatal_pct and not config.allow_short_clips:
        results.add_fatal(
            check_name,
            f"{short_pct:.1f}% clips shorter than {config.segment_duration}s "
            f"(fatal threshold: {config.short_clip_fatal_pct}%)",
            details
        )
        return False
    elif short_pct > config.short_clip_warn_pct:
        results.add_warning(
            check_name,
            f"{short_pct:.1f}% clips shorter than {config.segment_duration}s "
            f"(may cause repetitive content)",
            details
        )
    else:
        results.add_ok(
            check_name,
            f"Segment duration OK: {short_pct:.1f}% clips < {config.segment_duration}s",
            details
        )
    
    return True


# =============================================================================
# TASK 2A: Compression Checkpoint Compatibility
# =============================================================================

def check_compression_compatibility(config: PreflightConfig, results: PreflightResults) -> bool:
    """Validate compression checkpoint matches MusicGen configuration."""
    
    check_name = "compression_lm_compatibility"
    
    # Find compression checkpoint if not specified
    ckpt_path = config.compression_checkpoint
    if ckpt_path is None:
        ckpt_path = _find_latest_compression_checkpoint(config.experiments_dir)
        if ckpt_path is None:
            results.add_fatal(check_name, "No compression checkpoint found")
            return False
    
    if not ckpt_path.exists():
        results.add_fatal(check_name, f"Compression checkpoint not found: {ckpt_path}")
        return False
    
    # Load compression model to get metadata
    try:
        sys.path.insert(0, str(config.audiocraft_repo))
        from audiocraft.solvers import CompressionSolver
        
        model = CompressionSolver.model_from_checkpoint(str(ckpt_path), device="cpu")
        
        comp_meta = {
            "sample_rate": model.sample_rate,
            "channels": model.channels,
            "n_q": getattr(model, "num_codebooks", None),
            "cardinality": getattr(model, "cardinality", None),
            "frame_rate": getattr(model, "frame_rate", None),
        }
        
        if comp_meta["n_q"] is None and hasattr(model, "quantizer"):
            comp_meta["n_q"] = getattr(model.quantizer, "n_q", None)
        
        del model
        
    except Exception as e:
        results.add_fatal(check_name, f"Failed to load compression checkpoint: {e}")
        return False
    
    # Store resolved values
    results.resolved_config["compression"] = {
        "checkpoint": str(ckpt_path),
        **{k: _to_native(v) for k, v in comp_meta.items()}
    }
    
    # Check against MusicGen config if provided
    mismatches = []
    
    if config.transformer_lm_n_q is not None:
        if config.transformer_lm_n_q != comp_meta["n_q"]:
            mismatches.append(
                f"n_q mismatch: LM config has {config.transformer_lm_n_q}, "
                f"compression has {comp_meta['n_q']}"
            )
    
    if config.transformer_lm_card is not None:
        if config.transformer_lm_card != comp_meta["cardinality"]:
            mismatches.append(
                f"cardinality mismatch: LM config has {config.transformer_lm_card}, "
                f"compression has {comp_meta['cardinality']}"
            )
    
    if config.delay_pattern is not None:
        if len(config.delay_pattern) != comp_meta["n_q"]:
            mismatches.append(
                f"delay pattern length mismatch: {len(config.delay_pattern)} delays, "
                f"but n_q={comp_meta['n_q']}"
            )
    
    if config.expected_sample_rate != comp_meta["sample_rate"]:
        mismatches.append(
            f"sample_rate mismatch: expected {config.expected_sample_rate}, "
            f"compression has {comp_meta['sample_rate']}"
        )
    
    details = {
        "checkpoint": str(ckpt_path),
        "compression_meta": {k: _to_native(v) for k, v in comp_meta.items()},
        "mismatches": mismatches,
    }
    
    if mismatches:
        results.add_fatal(check_name, f"Compression-LM mismatch: {'; '.join(mismatches)}", details)
        return False
    
    results.add_ok(
        check_name,
        f"Compression-LM compatibility OK: n_q={comp_meta['n_q']}, "
        f"card={comp_meta['cardinality']}, sr={comp_meta['sample_rate']}Hz",
        details
    )
    return True


def _find_latest_compression_checkpoint(experiments_dir: Path) -> Optional[Path]:
    """Find the most recent compression checkpoint."""
    import yaml
    
    xp_root = experiments_dir / "xps"
    if not xp_root.exists():
        return None
    
    compression_xps = []
    for xp_dir in xp_root.iterdir():
        if not xp_dir.is_dir():
            continue
        
        for config_path in [xp_dir / "config.yaml", xp_dir / ".hydra" / "config.yaml"]:
            if config_path.exists():
                try:
                    cfg = yaml.safe_load(config_path.read_text())
                    solver = str(cfg.get("solver", ""))
                    if "compression" in solver.lower():
                        compression_xps.append(xp_dir)
                        break
                except Exception:
                    continue
    
    if not compression_xps:
        return None
    
    compression_xps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_xp = compression_xps[0]
    
    # Find checkpoint
    for pattern in ["*best*.th", "*latest*.th", "checkpoint*.th", "*.th", "*.pt", "*.pth"]:
        files = sorted(latest_xp.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0]
    
    return None


# =============================================================================
# TASK 2B: Audio Roundtrip Sanity Test
# =============================================================================

def check_compression_roundtrip(config: PreflightConfig, results: PreflightResults) -> bool:
    """Quick encode-decode test to verify compression checkpoint works."""
    
    check_name = "compression_roundtrip"
    
    # Get some audio paths from validation set
    audio_paths = []
    try:
        with open(config.valid_jsonl, 'r') as f:
            lines = f.readlines()
            sample_indices = random.sample(
                range(len(lines)),
                min(config.roundtrip_samples, len(lines))
            )
            for i in sample_indices:
                entry = json.loads(lines[i])
                path = Path(entry.get("path", ""))
                if path.exists():
                    audio_paths.append(path)
    except Exception as e:
        results.add_warning(check_name, f"Could not sample audio paths: {e}")
        return True
    
    if not audio_paths:
        results.add_warning(check_name, "No audio files found for roundtrip test")
        return True
    
    try:
        import torch
        import torchaudio
        sys.path.insert(0, str(config.audiocraft_repo))
        from audiocraft.solvers import CompressionSolver
        
        ckpt_path = config.compression_checkpoint
        if ckpt_path is None:
            ckpt_path = _find_latest_compression_checkpoint(config.experiments_dir)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CompressionSolver.model_from_checkpoint(str(ckpt_path), device=device)
        model.eval()
        
        roundtrip_results = []
        has_nan = False
        has_zero_energy = False
        
        for audio_path in audio_paths[:config.roundtrip_samples]:
            try:
                # Load audio
                wav, sr = torchaudio.load(audio_path)
                if sr != model.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
                
                # Trim to manageable length
                max_samples = int(5 * model.sample_rate)  # 5 seconds max
                wav = wav[:, :max_samples]
                wav = wav.unsqueeze(0).to(device)
                
                # Encode-decode
                with torch.no_grad():
                    codes = model.encode(wav)
                    reconstructed = model.decode(codes)
                
                # Check for NaN
                if not torch.isfinite(reconstructed).all():
                    has_nan = True
                    roundtrip_results.append({
                        "path": str(audio_path),
                        "status": "NaN in output",
                    })
                    continue
                
                # Compute metrics
                orig = wav.squeeze().cpu()
                recon = reconstructed.squeeze().cpu()
                
                # Match lengths
                min_len = min(orig.shape[-1], recon.shape[-1])
                orig = orig[..., :min_len]
                recon = recon[..., :min_len]
                
                # Energy check
                orig_energy = torch.mean(orig ** 2).item()
                recon_energy = torch.mean(recon ** 2).item()
                
                if recon_energy < 1e-10:
                    has_zero_energy = True
                
                # SI-SNR (simplified)
                noise = recon - orig
                si_snr = 10 * torch.log10(
                    torch.sum(orig ** 2) / (torch.sum(noise ** 2) + 1e-10)
                ).item()
                
                roundtrip_results.append({
                    "path": str(audio_path),
                    "status": "OK",
                    "si_snr_db": round(si_snr, 2),
                    "orig_rms": round(orig_energy ** 0.5, 4),
                    "recon_rms": round(recon_energy ** 0.5, 4),
                })
                
            except Exception as e:
                roundtrip_results.append({
                    "path": str(audio_path),
                    "status": f"Error: {e}",
                })
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        details = {
            "samples_tested": len(roundtrip_results),
            "results": roundtrip_results,
        }
        
        if has_nan:
            results.add_fatal(check_name, "Compression roundtrip produced NaN values", details)
            return False
        
        if has_zero_energy:
            results.add_fatal(check_name, "Compression roundtrip produced zero-energy output", details)
            return False
        
        # Check SI-SNR threshold
        snrs = [r.get("si_snr_db", 0) for r in roundtrip_results if r.get("status") == "OK"]
        if snrs:
            avg_snr = sum(snrs) / len(snrs)
            details["avg_si_snr_db"] = round(avg_snr, 2)
            
            if avg_snr < config.roundtrip_min_snr_db:
                results.add_warning(
                    check_name,
                    f"Low roundtrip SI-SNR: {avg_snr:.1f}dB (threshold: {config.roundtrip_min_snr_db}dB)",
                    details
                )
            else:
                results.add_ok(
                    check_name,
                    f"Compression roundtrip OK: avg SI-SNR={avg_snr:.1f}dB",
                    details
                )
        else:
            results.add_warning(check_name, "No successful roundtrip tests", details)
        
        return True
        
    except Exception as e:
        results.add_warning(check_name, f"Roundtrip test failed: {e}")
        return True


# =============================================================================
# TASK 3A: Training Configuration and Epoch Math
# =============================================================================

def check_training_config(config: PreflightConfig, results: PreflightResults) -> bool:
    """Validate training configuration and compute epoch math."""
    
    check_name = "training_config"
    
    # Detect world size
    world_size = config.world_size
    if world_size is None:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            try:
                import torch
                world_size = max(1, torch.cuda.device_count())
            except (ImportError, RuntimeError):
                world_size = 1
    
    # Validate batch size divisibility
    global_batch_size = config.batch_size * world_size
    
    # Count dataset size
    dataset_size = None
    try:
        with open(config.train_jsonl, 'r') as f:
            dataset_size = sum(1 for _ in f)
    except Exception:
        pass
    
    # Compute updates per epoch
    if config.target_hours is not None and dataset_size:
        target_samples = int((config.target_hours * 3600) / config.segment_duration)
        updates_per_epoch = max(1, min(target_samples, dataset_size) // global_batch_size)
        epoch_mode = "target_hours"
        epoch_meaning = f"An epoch is {updates_per_epoch:,} updates (subset of dataset based on TARGET_HOURS={config.target_hours}h)"
    elif dataset_size:
        updates_per_epoch = max(1, dataset_size // global_batch_size)
        epoch_mode = "full_dataset"
        total_hours = (dataset_size * config.segment_duration) / 3600
        epoch_meaning = f"An epoch is ~1 full pass over {dataset_size:,} samples ({total_hours:.1f}h of audio)"
    else:
        updates_per_epoch = 1000
        epoch_mode = "fallback"
        epoch_meaning = "Epoch size unknown (fallback to 1000 updates)"
    
    details = {
        "world_size": world_size,
        "per_gpu_batch_size": config.batch_size,
        "global_batch_size": global_batch_size,
        "dataset_size": dataset_size,
        "segment_duration": config.segment_duration,
        "target_hours": config.target_hours,
        "epoch_mode": epoch_mode,
        "updates_per_epoch": updates_per_epoch,
        "epoch_meaning": epoch_meaning,
    }
    
    if config.target_hours is not None:
        details["target_samples"] = int((config.target_hours * 3600) / config.segment_duration)
    
    # Store resolved config
    results.resolved_config["training"] = details
    
    # Check for issues
    if updates_per_epoch <= 0:
        results.add_fatal(check_name, "updates_per_epoch <= 0", details)
        return False
    
    if global_batch_size > (dataset_size or float('inf')):
        results.add_warning(
            check_name,
            f"Global batch size ({global_batch_size}) > dataset size ({dataset_size})",
            details
        )
    
    results.add_ok(check_name, epoch_meaning, details)
    return True


# =============================================================================
# TASK 3B: Epoch Meaning Banner
# =============================================================================

def get_epoch_meaning_banner(results: PreflightResults) -> str:
    """Generate a clear banner explaining what 'epoch' means for this run."""
    
    training = results.resolved_config.get("training", {})
    
    lines = [
        "=" * 60,
        "EPOCH DEFINITION FOR THIS RUN",
        "=" * 60,
    ]
    
    epoch_mode = training.get("epoch_mode", "unknown")
    
    if epoch_mode == "target_hours":
        target_hours = training.get("target_hours", "?")
        updates = training.get("updates_per_epoch", "?")
        lines.append(f"Mode: TARGET_HOURS ({target_hours}h)")
        lines.append(f"→ One epoch = {updates:,} updates (NOT a full dataset pass)")
        lines.append(f"→ This is a SUBSET of your data")
    elif epoch_mode == "full_dataset":
        dataset_size = training.get("dataset_size", "?")
        updates = training.get("updates_per_epoch", "?")
        lines.append(f"Mode: FULL DATASET")
        lines.append(f"→ One epoch ≈ 1 full pass over {dataset_size:,} samples")
        lines.append(f"→ {updates:,} updates per epoch")
    else:
        lines.append(f"Mode: FALLBACK (dataset size unknown)")
        lines.append(f"→ Using default 1000 updates per epoch")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# =============================================================================
# Main Preflight Runner
# =============================================================================

def run_preflight(config: PreflightConfig) -> tuple[bool, PreflightResults]:
    """
    Run all preflight checks.
    
    Returns:
        (ok, results): ok is True if no fatal errors, results contains all details.
    """
    
    results = PreflightResults()
    
    print("\n" + "=" * 60)
    print("PREFLIGHT CHECKS")
    print("=" * 60)
    
    # Task 1A: Manifest existence
    print("\n[1/6] Checking manifests...")
    if not check_manifest_exists(config, results):
        return False, results
    
    # Task 1A: Manifest schema
    print("[2/6] Validating manifest schema...")
    check_manifest_schema(config, results)
    
    # Task 1B: Segment duration feasibility
    print("[3/6] Checking segment duration feasibility...")
    check_segment_duration(config, results)
    
    # Task 2A: Compression compatibility (skip for compression training)
    if config.skip_compression_check:
        print("[4/6] Skipping compression-LM compatibility (compression training mode)")
        results.add_ok("compression_lm_compatibility", "Skipped (compression training mode)")
    else:
        print("[4/6] Checking compression-LM compatibility...")
        check_compression_compatibility(config, results)
    
    # Task 2B: Compression roundtrip (skip for compression training)
    if config.skip_compression_check:
        print("[5/6] Skipping compression roundtrip test (compression training mode)")
        results.add_ok("compression_roundtrip", "Skipped (compression training mode)")
    else:
        print("[5/6] Running compression roundtrip test...")
        check_compression_roundtrip(config, results)
    
    # Task 3A/3B: Training config
    print("[6/6] Validating training configuration...")
    check_training_config(config, results)
    
    # Print results
    print("\n" + "=" * 60)
    print("PREFLIGHT RESULTS")
    print("=" * 60)
    
    if results.fatal_errors:
        print("\n❌ FATAL ERRORS:")
        for err in results.fatal_errors:
            print(f"   • {err}")
    
    if results.warnings:
        print("\n⚠️  WARNINGS:")
        for warn in results.warnings:
            print(f"   • {warn}")
    
    if results.info:
        print("\n✓ PASSED:")
        for info in results.info:
            print(f"   • {info}")
    
    # Print epoch meaning banner
    banner = get_epoch_meaning_banner(results)
    print("\n" + banner)
    
    if results.passed:
        print("\n✓ All preflight checks passed!")
    else:
        print("\n❌ Preflight checks FAILED - training will not start")
    
    return results.passed, results


def _to_native(x):
    """Convert numpy/torch types to Python native types."""
    if hasattr(x, 'item'):
        return x.item()
    return x


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run preflight checks")
    parser.add_argument("--dset", default="audio/all_data", help="Dataset name")
    parser.add_argument("--segment-duration", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-hours", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--channels", type=int, default=1)
    
    args = parser.parse_args()
    
    config = PreflightConfig(
        dset=args.dset,
        segment_duration=args.segment_duration,
        batch_size=args.batch_size,
        target_hours=args.target_hours,
        expected_sample_rate=args.sample_rate,
        expected_channels=args.channels,
    )
    
    ok, results = run_preflight(config)
    sys.exit(0 if ok else 1)
