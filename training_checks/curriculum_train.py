#!/usr/bin/env python3
"""
Task 4: Curriculum Training Script

Implements curriculum learning by progressively increasing sequence length.
Tracks CE/PPL progression to determine when to advance to next stage.

Curriculum stages:
1. short_10s  - Fast initial learning, 50 epochs
2. medium_20s - Structure learning, 50 epochs  
3. full_30s+  - Long-form coherence, remaining epochs

Usage:
    python curriculum_train.py --stage short_10s
    python curriculum_train.py --stage medium_20s --continue-from /path/to/checkpoint
    python curriculum_train.py --auto  # Automatically progress through stages
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/root/workspace")
AUDIOCRAFT_DIR = BASE_DIR / "audiocraft"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "audiocraft"

CURRICULUM_STAGES = {
    "short_10s": {
        "segment_duration": 10,
        "batch_size": 128,
        "epochs": 50,
        "updates_per_epoch": 2000,
        "lr": 1e-4,
        "target_ppl_drop": 0.15,  # Expect 15% PPL reduction
        "description": "Short sequences for fast initial learning",
    },
    "medium_20s": {
        "segment_duration": 20,
        "batch_size": 64,
        "epochs": 50,
        "updates_per_epoch": 3000,
        "lr": 5e-5,
        "target_ppl_drop": 0.10,  # Expect 10% PPL reduction
        "description": "Medium sequences for structure learning",
    },
    "full_30s": {
        "segment_duration": 30,
        "batch_size": 32,
        "epochs": 100,
        "updates_per_epoch": 5000,
        "lr": 2e-5,
        "target_ppl_drop": 0.05,
        "description": "Full sequences for long-form coherence",
    },
}


def get_latest_checkpoint(experiments_dir: Path) -> tuple[Path, dict] | None:
    """Find the latest MusicGen checkpoint."""
    xps_dir = experiments_dir / "xps"
    if not xps_dir.exists():
        return None
    
    musicgen_xps = []
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
                if cfg.get("solver") == "musicgen":
                    musicgen_xps.append((xp_dir, checkpoint_path, cfg))
            except:
                pass
    
    if not musicgen_xps:
        return None
    
    # Sort by modification time
    musicgen_xps.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    xp_dir, ckpt_path, cfg = musicgen_xps[0]
    
    return ckpt_path, cfg


def analyze_training_progress(experiments_dir: Path) -> dict | None:
    """Analyze training progress from history.json."""
    xps_dir = experiments_dir / "xps"
    
    # Find latest musicgen experiment with history
    musicgen_xps = []
    for xp_dir in xps_dir.iterdir():
        if not xp_dir.is_dir():
            continue
        
        history_path = xp_dir / "history.json"
        config_path = xp_dir / ".hydra" / "config.yaml"
        
        if history_path.exists() and config_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg.get("solver") == "musicgen":
                    musicgen_xps.append((xp_dir, history_path, cfg))
            except:
                pass
    
    if not musicgen_xps:
        return None
    
    musicgen_xps.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    xp_dir, history_path, cfg = musicgen_xps[0]
    
    with open(history_path) as f:
        history = json.load(f)
    
    if not history:
        return None
    
    # Extract metrics
    first_epoch = history[0].get("train", {})
    last_epoch = history[-1].get("train", {})
    
    ppl_start = first_epoch.get("ppl", None)
    ppl_end = last_epoch.get("ppl", None)
    ce_start = first_epoch.get("ce", None)
    ce_end = last_epoch.get("ce", None)
    
    progress = {
        "xp_dir": str(xp_dir),
        "epochs_trained": len(history),
        "segment_duration": cfg.get("dataset", {}).get("segment_duration"),
        "ppl_start": ppl_start,
        "ppl_end": ppl_end,
        "ce_start": ce_start,
        "ce_end": ce_end,
    }
    
    if ppl_start and ppl_end:
        progress["ppl_reduction"] = (ppl_start - ppl_end) / ppl_start
    
    if ce_start and ce_end:
        progress["ce_reduction"] = (ce_start - ce_end) / ce_start
    
    return progress


def determine_next_stage(progress: dict) -> str | None:
    """Determine which curriculum stage to run next."""
    
    current_segment = progress.get("segment_duration", 0)
    ppl_reduction = progress.get("ppl_reduction", 0)
    
    # Determine current stage based on segment duration
    if current_segment is None or current_segment <= 10:
        current_stage = "short_10s"
        next_stage = "medium_20s"
    elif current_segment <= 20:
        current_stage = "medium_20s"
        next_stage = "full_30s"
    elif current_segment <= 30:
        current_stage = "full_30s"
        next_stage = None  # Already at final stage
    else:
        # Current training uses longer segments than curriculum
        # Recommend starting fresh with curriculum
        print(f"  Current segment_duration ({current_segment}s) exceeds curriculum max (30s)")
        print(f"  Consider starting fresh with short_10s stage")
        return "short_10s"
    
    # Check if ready to advance
    target_drop = CURRICULUM_STAGES[current_stage]["target_ppl_drop"]
    
    if ppl_reduction >= target_drop:
        print(f"✓ PPL reduction {ppl_reduction:.1%} meets target {target_drop:.1%}")
        if next_stage:
            print(f"  Ready to advance to {next_stage}")
            return next_stage
        else:
            print(f"  At final stage (full_30s), continue training")
            return "full_30s"
    else:
        print(f"⚠️  PPL reduction {ppl_reduction:.1%} below target {target_drop:.1%}")
        print(f"   Continue training at {current_stage}")
        return current_stage


def build_training_command(
    stage: str,
    continue_from: Path = None,
    dset: str = "audio/all_data",
    solver: str = "musicgen/musicgen_base_32khz",
) -> list[str]:
    """Build dora training command for a curriculum stage."""
    
    stage_config = CURRICULUM_STAGES[stage]
    
    cmd = [
        "python", "-m", "dora", "run", "-d",
        f"solver={solver}",
        f"dset={dset}",
        f"dataset.segment_duration={stage_config['segment_duration']}",
        f"dataset.batch_size={stage_config['batch_size']}",
        f"optim.epochs={stage_config['epochs']}",
        f"optim.updates_per_epoch={stage_config['updates_per_epoch']}",
        f"optim.lr={stage_config['lr']}",
        "optim.max_norm=1.0",
        "autocast=true",
    ]
    
    if continue_from:
        cmd.append(f"continue_from={continue_from}")
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Curriculum training for MusicGen")
    parser.add_argument("--stage", choices=list(CURRICULUM_STAGES.keys()),
                        help="Curriculum stage to run")
    parser.add_argument("--continue-from", type=Path,
                        help="Checkpoint to continue from")
    parser.add_argument("--auto", action="store_true",
                        help="Automatically determine stage from training progress")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command without running")
    parser.add_argument("--dset", default="audio/all_data",
                        help="Dataset to use")
    args = parser.parse_args()
    
    print("=" * 70)
    print("CURRICULUM TRAINING")
    print("=" * 70)
    
    # Print stage info
    print("\nCurriculum stages:")
    for name, cfg in CURRICULUM_STAGES.items():
        print(f"  {name}: {cfg['description']}")
        print(f"    segment_duration={cfg['segment_duration']}s, "
              f"batch_size={cfg['batch_size']}, "
              f"epochs={cfg['epochs']}")
    
    # Determine stage
    stage = args.stage
    continue_from = args.continue_from
    
    if args.auto:
        print("\n[Auto mode] Analyzing training progress...")
        progress = analyze_training_progress(EXPERIMENTS_DIR)
        
        if progress:
            print(f"  Current experiment: {progress['xp_dir']}")
            print(f"  Epochs trained: {progress['epochs_trained']}")
            print(f"  Segment duration: {progress['segment_duration']}s")
            print(f"  PPL: {progress.get('ppl_start', 'N/A')} → {progress.get('ppl_end', 'N/A')}")
            if progress.get('ppl_reduction'):
                print(f"  PPL reduction: {progress['ppl_reduction']:.1%}")
            
            stage = determine_next_stage(progress)
            
            # Get checkpoint for continuation
            ckpt_result = get_latest_checkpoint(EXPERIMENTS_DIR)
            if ckpt_result:
                continue_from = ckpt_result[0]
        else:
            print("  No training history found, starting from short_10s")
            stage = "short_10s"
    
    if not stage:
        print("\n❌ No stage specified. Use --stage or --auto")
        sys.exit(1)
    
    print(f"\n[Selected stage: {stage}]")
    print(f"  {CURRICULUM_STAGES[stage]['description']}")
    
    if continue_from:
        print(f"  Continuing from: {continue_from}")
    
    # Build command
    cmd = build_training_command(
        stage=stage,
        continue_from=continue_from,
        dset=args.dset,
    )
    
    print(f"\nCommand:")
    print(f"  {' '.join(cmd)}")
    
    if args.dry_run:
        print("\n[Dry run - not executing]")
        return
    
    # Run training
    print(f"\nStarting training...")
    print("-" * 70)
    
    env = {
        **dict(__import__('os').environ),
        "AUDIOCRAFT_TEAM": "default",
        "AUDIOCRAFT_DORA_DIR": str(EXPERIMENTS_DIR),
    }
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(AUDIOCRAFT_DIR),
            env=env,
        )
        
        if result.returncode != 0:
            print(f"\n❌ Training exited with code {result.returncode}")
            sys.exit(result.returncode)
        
        print(f"\n✓ Curriculum stage {stage} completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
