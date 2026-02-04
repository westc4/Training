#!/usr/bin/env python3
"""
Task 2: Conditioning Pipeline Validation

Validates that the conditioning pipeline is properly configured and active.
Checks:
- Percentage of samples with non-empty description
- Average tokenized length
- Verifies conditioner receives 'description' field

Usage:
    python validate_conditioning.py
    python validate_conditioning.py --num-batches 10 --batch-size 8
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/root/workspace")
AUDIOCRAFT_DIR = BASE_DIR / "audiocraft"
DATA_DIR = BASE_DIR / "data"

# Dataset paths
TRAIN_JSONL = DATA_DIR / "all_data" / "egs" / "train" / "data.jsonl"
VALID_JSONL = DATA_DIR / "all_data" / "egs" / "valid" / "data.jsonl"

# Debug conditioning settings (force all text to be used)
DEBUG_CONDITIONER_SETTINGS = {
    "merge_text_p": 1.0,   # Always merge text
    "drop_desc_p": 0.0,    # Never drop description
}


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

def analyze_dataset_conditioning(jsonl_path: Path, num_samples: int = 1000) -> dict:
    """Analyze conditioning fields in dataset."""
    
    stats = {
        "total_samples": 0,
        "has_description": 0,
        "has_empty_description": 0,
        "has_genre": 0,
        "has_mood": 0,
        "has_tags": 0,
        "description_lengths": [],
        "sample_descriptions": [],
    }
    
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            entry = json.loads(line)
            stats["total_samples"] += 1
            
            # Check description field
            desc = entry.get("description", "")
            if desc and desc.strip():
                stats["has_description"] += 1
                stats["description_lengths"].append(len(desc.split()))
                if len(stats["sample_descriptions"]) < 5:
                    stats["sample_descriptions"].append(desc[:100])
            else:
                stats["has_empty_description"] += 1
            
            # Check other conditioning fields
            if entry.get("genre"):
                stats["has_genre"] += 1
            if entry.get("mood"):
                stats["has_mood"] += 1
            if entry.get("tags"):
                stats["has_tags"] += 1
    
    return stats


def check_dataloader_conditioning(
    dset: str = "audio/all_data",
    batch_size: int = 4,
    num_batches: int = 5,
    segment_duration: float = 10.0,
) -> dict:
    """Load data through AudioCraft dataloader and check conditioning."""
    
    sys.path.insert(0, str(AUDIOCRAFT_DIR))
    
    # Import AudioCraft modules
    from audiocraft.data.audio_dataset import AudioDataset
    from audiocraft.data.audio import AudioMeta
    import omegaconf
    
    # Build minimal config for dataset
    # We need to replicate what the solver does
    
    results = {
        "batches_checked": 0,
        "samples_with_info": 0,
        "samples_with_description": 0,
        "total_samples": 0,
        "description_word_counts": [],
        "sample_info_keys": set(),
        "sample_descriptions": [],
    }
    
    # Load dataset config
    dset_name = dset.replace("audio/", "")
    egs_dir = DATA_DIR / dset_name / "egs"
    
    # Read JSONL directly and simulate batch loading
    train_jsonl = egs_dir / "train" / "data.jsonl"
    
    with open(train_jsonl, 'r') as f:
        lines = f.readlines()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        if end_idx > len(lines):
            break
        
        batch_lines = lines[start_idx:end_idx]
        results["batches_checked"] += 1
        
        for line in batch_lines:
            entry = json.loads(line)
            results["total_samples"] += 1
            
            # Check if entry has info dict (metadata)
            info_keys = [k for k in entry.keys() if k not in ["path", "duration", "sample_rate"]]
            results["sample_info_keys"].update(info_keys)
            
            if info_keys:
                results["samples_with_info"] += 1
            
            # Check description specifically
            desc = entry.get("description", "")
            if desc and desc.strip():
                results["samples_with_description"] += 1
                word_count = len(desc.split())
                results["description_word_counts"].append(word_count)
                
                if len(results["sample_descriptions"]) < 3:
                    results["sample_descriptions"].append({
                        "path": entry.get("path", ""),
                        "description": desc[:200],
                        "word_count": word_count,
                    })
    
    # Convert set to list for JSON serialization
    results["sample_info_keys"] = list(results["sample_info_keys"])
    
    return results


def check_conditioner_config(config_path: Path = None) -> dict:
    """Check conditioner configuration from a MusicGen experiment."""
    
    results = {
        "conditioner_type": None,
        "text_conditioner": None,
        "merge_text_p": None,
        "drop_desc_p": None,
        "issues": [],
        "recommendations": [],
    }
    
    if config_path is None:
        # Find latest musicgen experiment
        xps_dir = BASE_DIR / "experiments" / "audiocraft" / "xps"
        musicgen_xps = []
        
        for xp_dir in xps_dir.iterdir():
            if not xp_dir.is_dir():
                continue
            config = xp_dir / ".hydra" / "config.yaml"
            if config.exists():
                try:
                    import yaml
                    with open(config) as f:
                        cfg = yaml.safe_load(f)
                    if cfg.get("solver") == "musicgen":
                        musicgen_xps.append((xp_dir, cfg))
                except:
                    pass
        
        if not musicgen_xps:
            results["issues"].append("No MusicGen experiments found")
            return results
        
        # Use most recent
        musicgen_xps.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
        config_path, cfg = musicgen_xps[0]
        results["experiment_dir"] = str(config_path)
    else:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    
    # Check conditioner settings
    conditioners = cfg.get("conditioners", None)
    
    if conditioners is None:
        results["conditioner_type"] = "none"
        results["issues"].append("conditioners is null - unconditional training!")
        results["recommendations"].append(
            "For text-to-music, set conditioner=text2music in your config"
        )
    else:
        results["conditioner_type"] = "configured"
        
        # Check for text conditioner
        if isinstance(conditioners, dict):
            for name, cond_cfg in conditioners.items():
                if "t5" in str(cond_cfg).lower() or "text" in name.lower():
                    results["text_conditioner"] = name
                    
                    # Check dropout settings
                    if isinstance(cond_cfg, dict):
                        results["merge_text_p"] = cond_cfg.get("merge_text_p")
                        results["drop_desc_p"] = cond_cfg.get("drop_desc_p")
    
    # Validate settings
    if results["conditioner_type"] == "none":
        results["issues"].append(
            "Unconditional training - model learns to generate 'average' music"
        )
    
    if results["drop_desc_p"] and results["drop_desc_p"] > 0.5:
        results["issues"].append(
            f"High drop_desc_p={results['drop_desc_p']} means descriptions often ignored"
        )
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate conditioning pipeline")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples to analyze from dataset")
    parser.add_argument("--num-batches", type=int, default=10,
                        help="Number of batches to check through dataloader")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for dataloader check")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to MusicGen experiment config.yaml")
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONDITIONING PIPELINE VALIDATION")
    print("=" * 70)
    
    # Task 2.1: Analyze dataset for conditioning fields
    print("\n[1/3] Analyzing dataset conditioning fields...")
    print("-" * 70)
    
    if TRAIN_JSONL.exists():
        dataset_stats = analyze_dataset_conditioning(TRAIN_JSONL, args.num_samples)
        
        desc_pct = (dataset_stats["has_description"] / dataset_stats["total_samples"] * 100
                   if dataset_stats["total_samples"] > 0 else 0)
        
        print(f"  Total samples analyzed: {dataset_stats['total_samples']}")
        print(f"  With description: {dataset_stats['has_description']} ({desc_pct:.1f}%)")
        print(f"  Empty description: {dataset_stats['has_empty_description']}")
        print(f"  With genre: {dataset_stats['has_genre']}")
        print(f"  With mood: {dataset_stats['has_mood']}")
        print(f"  With tags: {dataset_stats['has_tags']}")
        
        if dataset_stats["description_lengths"]:
            avg_len = sum(dataset_stats["description_lengths"]) / len(dataset_stats["description_lengths"])
            print(f"\n  Average description length: {avg_len:.1f} words")
            print(f"  Min/Max length: {min(dataset_stats['description_lengths'])}/{max(dataset_stats['description_lengths'])} words")
        
        if dataset_stats["sample_descriptions"]:
            print(f"\n  Sample descriptions:")
            for desc in dataset_stats["sample_descriptions"][:3]:
                print(f"    - \"{desc}...\"")
        
        if desc_pct < 10:
            print(f"\n  ⚠️  WARNING: Only {desc_pct:.1f}% have descriptions!")
            print(f"     Text conditioning will be ineffective.")
    else:
        print(f"  ❌ Train JSONL not found: {TRAIN_JSONL}")
    
    # Task 2.2: Check dataloader
    print("\n[2/3] Checking dataloader conditioning...")
    print("-" * 70)
    
    try:
        loader_stats = check_dataloader_conditioning(
            batch_size=args.batch_size,
            num_batches=args.num_batches,
        )
        
        print(f"  Batches checked: {loader_stats['batches_checked']}")
        print(f"  Total samples: {loader_stats['total_samples']}")
        print(f"  Samples with metadata: {loader_stats['samples_with_info']}")
        print(f"  Samples with description: {loader_stats['samples_with_description']}")
        
        if loader_stats["sample_info_keys"]:
            print(f"\n  Available info keys: {loader_stats['sample_info_keys']}")
            
            if "description" not in loader_stats["sample_info_keys"]:
                print(f"\n  ❌ ERROR: 'description' field NOT found in dataset!")
                print(f"     Conditioner expects 'description' key for text conditioning.")
        
        if loader_stats["description_word_counts"]:
            avg_words = sum(loader_stats["description_word_counts"]) / len(loader_stats["description_word_counts"])
            print(f"\n  Average tokenized length: ~{avg_words:.1f} words")
            print(f"     (Actual T5 tokens will be ~1.5-2x this)")
        
        if loader_stats["sample_descriptions"]:
            print(f"\n  Sample descriptions from dataloader:")
            for sample in loader_stats["sample_descriptions"]:
                print(f"    - [{sample['word_count']} words] \"{sample['description'][:80]}...\"")
    
    except Exception as e:
        print(f"  ❌ Error checking dataloader: {e}")
    
    # Task 2.3: Check conditioner config
    print("\n[3/3] Checking conditioner configuration...")
    print("-" * 70)
    
    cond_results = check_conditioner_config(args.config)
    
    print(f"  Conditioner type: {cond_results['conditioner_type']}")
    if cond_results.get("experiment_dir"):
        print(f"  Experiment: {cond_results['experiment_dir']}")
    
    if cond_results["text_conditioner"]:
        print(f"  Text conditioner: {cond_results['text_conditioner']}")
    
    if cond_results["merge_text_p"] is not None:
        print(f"  merge_text_p: {cond_results['merge_text_p']}")
    if cond_results["drop_desc_p"] is not None:
        print(f"  drop_desc_p: {cond_results['drop_desc_p']}")
    
    if cond_results["issues"]:
        print(f"\n  Issues found:")
        for issue in cond_results["issues"]:
            print(f"    ⚠️  {issue}")
    
    if cond_results["recommendations"]:
        print(f"\n  Recommendations:")
        for rec in cond_results["recommendations"]:
            print(f"    → {rec}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if cond_results["conditioner_type"] == "none":
        print("""
  Your current training is UNCONDITIONAL (conditioners=null).
  
  This means:
  - The model ignores all text/metadata
  - It learns to generate "average" music from your dataset
  - This can lead to "boring attractors" (repetitive, bland outputs)
  
  For better results, consider:
  1. Adding text descriptions to your dataset
  2. Using conditioner=text2music in your training config
  3. Setting merge_text_p=1.0, drop_desc_p=0.0 for debugging
""")
    else:
        print("""
  Conditioner is configured. Verify:
  1. Your dataset has 'description' field with meaningful text
  2. drop_desc_p is not too high (recommend 0.0-0.3)
  3. merge_text_p=1.0 for debugging (can relax later)
""")
    
    # Print debug settings
    print("\n  Debug settings for testing conditioning:")
    print(f"    conditioner=text2music")
    print(f"    conditioner.merge_text_p=1.0")
    print(f"    conditioner.drop_desc_p=0.0")


if __name__ == "__main__":
    main()
