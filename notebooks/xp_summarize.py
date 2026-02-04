#!/usr/bin/env python3
"""
XP Summarizer — Parse Dora XP logs and generate health summary

Usage:
    python xp_summarize.py --xp <xp_id>
    python xp_summarize.py --xp f86fb7e9
    python xp_summarize.py --xp f86fb7e9 --epochs 10
    python xp_summarize.py --list  # List all experiments
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import argparse
import json
import yaml
import datetime
import re
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/root/workspace")) if "os" in dir() else Path("/root/workspace")
import os
BASE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/root/workspace"))
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "audiocraft"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class XPSummary:
    """Summary of a Dora experiment."""
    
    xp_id: str
    xp_path: Path
    solver: str = ""
    status: str = "unknown"
    
    # Timestamps
    created: Optional[str] = None
    last_modified: Optional[str] = None
    
    # Config summary
    config: dict = field(default_factory=dict)
    
    # Training metrics (last N epochs)
    epochs: list[dict] = field(default_factory=list)
    latest_metrics: dict = field(default_factory=dict)
    
    # Health indicators
    anomalies: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    # Canary generations
    canaries: list[dict] = field(default_factory=list)
    
    # Checkpoints
    checkpoints: list[str] = field(default_factory=list)
    best_checkpoint: Optional[str] = None
    
    # Preflight report
    preflight_passed: Optional[bool] = None


# =============================================================================
# XP DISCOVERY
# =============================================================================

def list_experiments(experiments_dir: Path = EXPERIMENTS_DIR) -> list[tuple[str, str, str]]:
    """List all experiments with their solver and modification time."""
    xp_root = experiments_dir / "xps"
    if not xp_root.exists():
        return []
    
    results = []
    for xp_dir in xp_root.iterdir():
        if not xp_dir.is_dir():
            continue
        
        xp_id = xp_dir.name
        solver = "unknown"
        mtime = datetime.datetime.fromtimestamp(xp_dir.stat().st_mtime)
        
        # Read solver from config
        for config_path in [xp_dir / "config.yaml", xp_dir / ".hydra" / "config.yaml"]:
            if config_path.exists():
                try:
                    cfg = yaml.safe_load(config_path.read_text())
                    solver = str(cfg.get("solver", "unknown"))
                    break
                except Exception:
                    pass
        
        results.append((xp_id, solver, mtime.strftime("%Y-%m-%d %H:%M")))
    
    # Sort by modification time (newest first)
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def find_xp_by_id(xp_id: str, experiments_dir: Path = EXPERIMENTS_DIR) -> Optional[Path]:
    """Find experiment directory by ID (supports partial match)."""
    xp_root = experiments_dir / "xps"
    if not xp_root.exists():
        return None
    
    # Exact match first
    exact = xp_root / xp_id
    if exact.exists():
        return exact
    
    # Partial match
    matches = [d for d in xp_root.iterdir() if d.is_dir() and xp_id in d.name]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"Ambiguous XP ID '{xp_id}', matches: {[m.name for m in matches]}")
        return None
    
    return None


# =============================================================================
# CONFIG PARSING
# =============================================================================

def parse_config(xp_dir: Path) -> dict:
    """Parse experiment config."""
    config = {}
    
    for config_path in [xp_dir / ".hydra" / "config.yaml", xp_dir / "config.yaml"]:
        if config_path.exists():
            try:
                full_cfg = yaml.safe_load(config_path.read_text())
                
                # Extract key fields
                config["solver"] = full_cfg.get("solver", "unknown")
                config["dset"] = full_cfg.get("dset", "unknown")
                
                # Dataset config
                if "dataset" in full_cfg:
                    ds = full_cfg["dataset"]
                    config["segment_duration"] = ds.get("segment_duration")
                    config["batch_size"] = ds.get("batch_size")
                    config["num_workers"] = ds.get("num_workers")
                
                # Optimization config
                if "optim" in full_cfg:
                    opt = full_cfg["optim"]
                    config["epochs"] = opt.get("epochs")
                    config["updates_per_epoch"] = opt.get("updates_per_epoch")
                    config["lr"] = opt.get("lr")
                
                # Model config
                if "transformer_lm" in full_cfg:
                    lm = full_cfg["transformer_lm"]
                    config["n_q"] = lm.get("n_q")
                    config["card"] = lm.get("card")
                
                config["sample_rate"] = full_cfg.get("sample_rate")
                config["channels"] = full_cfg.get("channels")
                
                break
            except Exception as e:
                config["parse_error"] = str(e)
    
    return config


# =============================================================================
# LOG PARSING
# =============================================================================

def parse_logs(xp_dir: Path, max_epochs: int = 20) -> tuple[list[dict], dict]:
    """Parse training logs for metrics."""
    epochs = []
    latest = {}
    
    # Common log patterns
    patterns = {
        "epoch": re.compile(r"epoch[:\s]+(\d+)", re.IGNORECASE),
        "train_loss": re.compile(r"train.*?loss[=:\s]+([\d.]+)", re.IGNORECASE),
        "valid_loss": re.compile(r"valid.*?loss[=:\s]+([\d.]+)", re.IGNORECASE),
        "train_ce": re.compile(r"train.*?ce[=:\s]+([\d.]+)", re.IGNORECASE),
        "valid_ce": re.compile(r"valid.*?ce[=:\s]+([\d.]+)", re.IGNORECASE),
        "train_ppl": re.compile(r"train.*?ppl[=:\s]+([\d.]+)", re.IGNORECASE),
        "valid_ppl": re.compile(r"valid.*?ppl[=:\s]+([\d.]+)", re.IGNORECASE),
        "grad_norm": re.compile(r"grad_norm[=:\s]+([\d.]+|inf)", re.IGNORECASE),
        "it_per_sec": re.compile(r"([\d.]+)\s*it/s", re.IGNORECASE),
        "mel": re.compile(r"\bmel[=:\s]+([\d.]+)", re.IGNORECASE),
        "sisnr": re.compile(r"sisnr[=:\s]+([\d.-]+)", re.IGNORECASE),
    }
    
    # Find log files
    log_files = list(xp_dir.glob("*.log")) + list(xp_dir.glob("train*.txt"))
    
    epoch_data = {}
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', errors='ignore') as f:
                for line in f:
                    metrics = {}
                    
                    for name, pattern in patterns.items():
                        match = pattern.search(line)
                        if match:
                            value = match.group(1)
                            if value.lower() == "inf":
                                metrics[name] = float('inf')
                            else:
                                try:
                                    metrics[name] = float(value)
                                except ValueError:
                                    pass
                    
                    if "epoch" in metrics and len(metrics) > 1:
                        epoch = int(metrics["epoch"])
                        if epoch not in epoch_data:
                            epoch_data[epoch] = {}
                        epoch_data[epoch].update(metrics)
        except Exception:
            pass
    
    # Sort epochs and get last N
    sorted_epochs = sorted(epoch_data.keys())
    for epoch in sorted_epochs[-max_epochs:]:
        epochs.append({"epoch": epoch, **epoch_data[epoch]})
    
    # Get latest metrics
    if sorted_epochs:
        latest = epoch_data[sorted_epochs[-1]]
    
    return epochs, latest


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def detect_anomalies(epochs: list[dict], config: dict) -> tuple[list[str], list[str]]:
    """Detect anomalies in training metrics."""
    anomalies = []
    warnings = []
    
    if not epochs:
        warnings.append("No epoch data found in logs")
        return anomalies, warnings
    
    # Count inf grad norms
    inf_count = sum(1 for e in epochs if e.get("grad_norm") == float('inf'))
    if inf_count > 0:
        pct = (inf_count / len(epochs)) * 100
        if pct > 50:
            anomalies.append(f"High INF grad_norm rate: {pct:.1f}% of epochs")
        elif pct > 10:
            warnings.append(f"Elevated INF grad_norm rate: {pct:.1f}% of epochs")
    
    # Check for loss plateau
    if len(epochs) >= 5:
        losses = [e.get("valid_loss") or e.get("valid_ce") for e in epochs[-5:]]
        losses = [l for l in losses if l is not None and l != float('inf')]
        if len(losses) >= 3:
            # Check if variance is very low (plateau)
            mean_loss = sum(losses) / len(losses)
            variance = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
            if variance < 0.0001 and mean_loss > 0:
                warnings.append(f"Loss plateau detected (variance={variance:.6f})")
    
    # Check for exploding metrics
    for epoch in epochs:
        for key, value in epoch.items():
            if isinstance(value, float) and value > 1000 and "ratio" in key.lower():
                anomalies.append(f"Exploding {key}: {value:.2f} at epoch {epoch.get('epoch', '?')}")
                break
    
    # Check throughput degradation
    if len(epochs) >= 10:
        early_it = [e.get("it_per_sec") for e in epochs[:5] if e.get("it_per_sec")]
        late_it = [e.get("it_per_sec") for e in epochs[-5:] if e.get("it_per_sec")]
        if early_it and late_it:
            early_avg = sum(early_it) / len(early_it)
            late_avg = sum(late_it) / len(late_it)
            if early_avg > 0 and late_avg < early_avg * 0.7:
                warnings.append(f"Throughput drop: {early_avg:.2f} → {late_avg:.2f} it/s ({(1 - late_avg/early_avg)*100:.0f}% drop)")
    
    return anomalies, warnings


# =============================================================================
# CANARY PARSING
# =============================================================================

def parse_canaries(xp_dir: Path) -> list[dict]:
    """Parse canary generation results."""
    canary_dir = xp_dir / "canaries"
    if not canary_dir.exists():
        return []
    
    canaries = []
    for json_file in sorted(canary_dir.glob("canary_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                canaries.append({
                    "epoch": data.get("epoch"),
                    "rms": data.get("heuristics", {}).get("rms_loudness"),
                    "spectral_centroid": data.get("heuristics", {}).get("spectral_centroid_mean"),
                    "silence_ratio": data.get("heuristics", {}).get("silence_ratio"),
                    "audio_file": Path(data.get("audio_path", "")).name,
                })
        except Exception:
            pass
    
    return canaries


# =============================================================================
# CHECKPOINT DISCOVERY
# =============================================================================

def find_checkpoints(xp_dir: Path) -> tuple[list[str], Optional[str]]:
    """Find checkpoints and identify best one."""
    checkpoints = []
    best = None
    
    for pattern in ["*.th", "*.pt", "*.pth"]:
        for ckpt in xp_dir.glob(pattern):
            checkpoints.append(ckpt.name)
            if "best" in ckpt.name.lower():
                best = ckpt.name
    
    # Sort by modification time
    checkpoints.sort()
    
    return checkpoints, best


# =============================================================================
# PREFLIGHT PARSING
# =============================================================================

def parse_preflight(xp_dir: Path) -> Optional[bool]:
    """Check if preflight passed."""
    preflight_json = xp_dir / "preflight.json"
    if preflight_json.exists():
        try:
            with open(preflight_json, 'r') as f:
                data = json.load(f)
                return data.get("passed")
        except Exception:
            pass
    return None


# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def generate_summary(xp_id: str, max_epochs: int = 20) -> Optional[XPSummary]:
    """Generate summary for an experiment."""
    xp_path = find_xp_by_id(xp_id)
    if xp_path is None:
        print(f"Experiment '{xp_id}' not found")
        return None
    
    summary = XPSummary(
        xp_id=xp_path.name,
        xp_path=xp_path,
    )
    
    # Timestamps
    stat = xp_path.stat()
    summary.created = datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
    summary.last_modified = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
    
    # Parse config
    summary.config = parse_config(xp_path)
    summary.solver = summary.config.get("solver", "unknown")
    
    # Parse logs
    summary.epochs, summary.latest_metrics = parse_logs(xp_path, max_epochs)
    
    # Detect anomalies
    summary.anomalies, summary.warnings = detect_anomalies(summary.epochs, summary.config)
    
    # Parse canaries
    summary.canaries = parse_canaries(xp_path)
    
    # Find checkpoints
    summary.checkpoints, summary.best_checkpoint = find_checkpoints(xp_path)
    
    # Parse preflight
    summary.preflight_passed = parse_preflight(xp_path)
    
    # Determine status
    if summary.anomalies:
        summary.status = "⚠️ anomalies"
    elif summary.warnings:
        summary.status = "⚡ warnings"
    elif summary.epochs:
        summary.status = "✓ running"
    else:
        summary.status = "? unknown"
    
    return summary


# =============================================================================
# DISPLAY
# =============================================================================

def print_summary(summary: XPSummary):
    """Print formatted summary."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT SUMMARY: {summary.xp_id}")
    print("=" * 70)
    
    print(f"\nStatus: {summary.status}")
    print(f"Solver: {summary.solver}")
    print(f"Path: {summary.xp_path}")
    print(f"Last modified: {summary.last_modified}")
    
    # Config
    if summary.config:
        print("\n" + "-" * 40)
        print("CONFIGURATION")
        print("-" * 40)
        for key, value in summary.config.items():
            if value is not None and key != "solver":
                print(f"  {key}: {value}")
    
    # Anomalies and warnings
    if summary.anomalies:
        print("\n" + "-" * 40)
        print("❌ ANOMALIES")
        print("-" * 40)
        for a in summary.anomalies:
            print(f"  • {a}")
    
    if summary.warnings:
        print("\n" + "-" * 40)
        print("⚠️ WARNINGS")
        print("-" * 40)
        for w in summary.warnings:
            print(f"  • {w}")
    
    # Latest metrics
    if summary.latest_metrics:
        print("\n" + "-" * 40)
        print("LATEST METRICS")
        print("-" * 40)
        for key, value in summary.latest_metrics.items():
            if key != "epoch":
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}" if value < 100 else f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    
    # Recent epochs
    if summary.epochs:
        print("\n" + "-" * 40)
        print(f"RECENT EPOCHS (last {len(summary.epochs)})")
        print("-" * 40)
        
        # Header
        cols = ["epoch"]
        for key in summary.epochs[0].keys():
            if key != "epoch" and key in ["valid_loss", "valid_ce", "valid_ppl", "train_loss", "grad_norm", "it_per_sec"]:
                cols.append(key)
        
        header = " | ".join(f"{c:>12}" for c in cols[:6])
        print(f"  {header}")
        print("  " + "-" * len(header))
        
        for epoch_data in summary.epochs[-10:]:  # Last 10
            values = []
            for col in cols[:6]:
                v = epoch_data.get(col, "")
                if isinstance(v, float):
                    if v == float('inf'):
                        values.append("INF")
                    elif v < 0.01:
                        values.append(f"{v:.6f}")
                    elif v < 100:
                        values.append(f"{v:.4f}")
                    else:
                        values.append(f"{v:.1f}")
                else:
                    values.append(str(v))
            print("  " + " | ".join(f"{v:>12}" for v in values))
    
    # Canaries
    if summary.canaries:
        print("\n" + "-" * 40)
        print(f"CANARY GENERATIONS ({len(summary.canaries)})")
        print("-" * 40)
        for c in summary.canaries[-5:]:
            print(f"  Epoch {c.get('epoch', '?'):4d}: RMS={c.get('rms', 0):.4f}, "
                  f"SpectralCentroid={c.get('spectral_centroid', 0):.0f}Hz, "
                  f"Silence={c.get('silence_ratio', 0)*100:.1f}%")
    
    # Checkpoints
    if summary.checkpoints:
        print("\n" + "-" * 40)
        print(f"CHECKPOINTS ({len(summary.checkpoints)})")
        print("-" * 40)
        for ckpt in summary.checkpoints[-5:]:
            marker = " ⭐ BEST" if ckpt == summary.best_checkpoint else ""
            print(f"  {ckpt}{marker}")
    
    # Preflight
    if summary.preflight_passed is not None:
        print("\n" + "-" * 40)
        print("PREFLIGHT")
        print("-" * 40)
        status = "✓ PASSED" if summary.preflight_passed else "❌ FAILED"
        print(f"  {status}")
    
    print("\n" + "=" * 70)


def print_experiment_list(experiments: list[tuple[str, str, str]]):
    """Print list of experiments."""
    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 70)
    print(f"\n{'XP ID':<12} {'Solver':<40} {'Modified':<16}")
    print("-" * 70)
    
    for xp_id, solver, mtime in experiments:
        # Truncate solver if too long
        solver_short = solver[:38] + ".." if len(solver) > 40 else solver
        print(f"{xp_id:<12} {solver_short:<40} {mtime:<16}")
    
    print(f"\nTotal: {len(experiments)} experiments")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="XP Summarizer - Parse Dora experiment logs and generate health summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python xp_summarize.py --list
    python xp_summarize.py --xp f86fb7e9
    python xp_summarize.py --xp f86fb7e9 --epochs 20
    python xp_summarize.py --xp f86fb7e9 --json
        """
    )
    
    parser.add_argument("--xp", type=str, help="Experiment ID (supports partial match)")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to show (default: 20)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--experiments-dir", type=str, help="Override experiments directory")
    
    args = parser.parse_args()
    
    if args.experiments_dir:
        global EXPERIMENTS_DIR
        EXPERIMENTS_DIR = Path(args.experiments_dir)
    
    if args.list:
        experiments = list_experiments()
        if not experiments:
            print("No experiments found")
            sys.exit(1)
        print_experiment_list(experiments)
        sys.exit(0)
    
    if args.xp:
        summary = generate_summary(args.xp, max_epochs=args.epochs)
        if summary is None:
            sys.exit(1)
        
        if args.json:
            output = {
                "xp_id": summary.xp_id,
                "xp_path": str(summary.xp_path),
                "solver": summary.solver,
                "status": summary.status,
                "config": summary.config,
                "latest_metrics": summary.latest_metrics,
                "epochs": summary.epochs,
                "anomalies": summary.anomalies,
                "warnings": summary.warnings,
                "canaries": summary.canaries,
                "checkpoints": summary.checkpoints,
                "best_checkpoint": summary.best_checkpoint,
                "preflight_passed": summary.preflight_passed,
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            print_summary(summary)
        
        sys.exit(0)
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
