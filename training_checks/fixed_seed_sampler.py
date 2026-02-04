#!/usr/bin/env python3
"""
Task 5: Fixed-Seed Sample Generation Harness

Generates audio samples with fixed seeds and parameters for reproducible
A/B comparison across training epochs.

Features:
- Fixed seeds for reproducibility
- Fixed decode params (temperature, top-k, top-p)
- Saves outputs to samples/epoch_XX/
- Generates HTML index for easy comparison

Usage:
    python fixed_seed_sampler.py --epoch 10
    python fixed_seed_sampler.py --checkpoint /path/to/checkpoint.th
    python fixed_seed_sampler.py --generate-index  # Create HTML comparison page
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/root/workspace")
AUDIOCRAFT_DIR = BASE_DIR / "audiocraft"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "audiocraft"
OUTPUT_BASE_DIR = BASE_DIR / "Training" / "outputs" / "samples"

# Fixed generation parameters for reproducibility
FIXED_PARAMS = {
    "seeds": [42, 123, 456, 789, 2024],  # Fixed seeds
    "temperature": 0.8,
    "top_k": 250,
    "top_p": 0.0,
    "duration": 10.0,  # seconds
    "use_sampling": True,
}

# Prompts for conditional generation (ignored if unconditional)
FIXED_PROMPTS = [
    "A warm ambient synth pad with gentle reverb",
    "Upbeat electronic dance music with driving bass",
    "Calm acoustic guitar melody",
    "Dramatic orchestral strings",
    "Funky disco groove with brass",
]


# =============================================================================
# CHECKPOINT DISCOVERY
# =============================================================================

def find_checkpoints_by_epoch(experiments_dir: Path) -> dict[int, Path]:
    """Find all MusicGen checkpoints organized by epoch."""
    
    checkpoints = {}
    xps_dir = experiments_dir / "xps"
    
    if not xps_dir.exists():
        return checkpoints
    
    for xp_dir in xps_dir.iterdir():
        if not xp_dir.is_dir():
            continue
        
        config_path = xp_dir / ".hydra" / "config.yaml"
        history_path = xp_dir / "history.json"
        checkpoint_path = xp_dir / "checkpoint.th"
        
        if not all(p.exists() for p in [config_path, history_path, checkpoint_path]):
            continue
        
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            
            if cfg.get("solver") != "musicgen":
                continue
            
            with open(history_path) as f:
                history = json.load(f)
            
            epoch = len(history)
            if epoch > 0:
                checkpoints[epoch] = checkpoint_path
        except:
            pass
    
    return checkpoints


def find_latest_musicgen_checkpoint(experiments_dir: Path) -> tuple[Path, int] | None:
    """Find the latest MusicGen checkpoint and its epoch."""
    
    checkpoints = find_checkpoints_by_epoch(experiments_dir)
    if not checkpoints:
        return None
    
    max_epoch = max(checkpoints.keys())
    return checkpoints[max_epoch], max_epoch


def find_compression_checkpoint(experiments_dir: Path) -> Path | None:
    """Find the latest compression checkpoint."""
    
    xps_dir = experiments_dir / "xps"
    if not xps_dir.exists():
        return None
    
    compression_xps = []
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
                    compression_xps.append((xp_dir, checkpoint_path))
            except:
                pass
    
    if not compression_xps:
        return None
    
    compression_xps.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    return compression_xps[0][1]


# =============================================================================
# GENERATION
# =============================================================================

def generate_samples(
    musicgen_checkpoint: Path,
    compression_checkpoint: Path,
    output_dir: Path,
    epoch: int,
    device: str = "cuda",
) -> list[dict]:
    """Generate samples with fixed seeds and params."""
    
    import torchaudio
    import omegaconf
    
    sys.path.insert(0, str(AUDIOCRAFT_DIR))
    from audiocraft.solvers import CompressionSolver
    from audiocraft.utils import checkpoint as ckpt_utils
    from audiocraft.models.builders import get_lm_model
    from audiocraft.modules.conditioners import ConditioningAttributes
    
    # Create output directory
    epoch_dir = output_dir / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Load compression model
    print(f"Loading compression model from {compression_checkpoint.name}...")
    compression_model = CompressionSolver.model_from_checkpoint(
        str(compression_checkpoint), device=device
    )
    sample_rate = int(compression_model.sample_rate)
    frame_rate = compression_model.frame_rate
    
    # Load LM model
    musicgen_xp = musicgen_checkpoint.parent
    hydra_config = musicgen_xp / ".hydra" / "config.yaml"
    
    if not hydra_config.exists():
        raise FileNotFoundError(f"Config not found: {hydra_config}")
    
    cfg = omegaconf.OmegaConf.load(hydra_config)
    cfg.device = device
    
    print(f"Loading LM model from {musicgen_checkpoint.name}...")
    lm_model = get_lm_model(cfg)
    checkpoint_data = ckpt_utils.load_checkpoint(musicgen_checkpoint, is_sharded=False)
    lm_model.load_state_dict(checkpoint_data["model"])
    lm_model.to(device)
    lm_model.eval()
    
    # Check if model is conditional
    is_conditional = cfg.get("conditioners") is not None
    
    # Calculate generation length
    max_gen_len = int(FIXED_PARAMS["duration"] * frame_rate)
    
    print(f"\nGenerating samples...")
    print(f"  Duration: {FIXED_PARAMS['duration']}s ({max_gen_len} tokens)")
    print(f"  Temperature: {FIXED_PARAMS['temperature']}")
    print(f"  Top-k: {FIXED_PARAMS['top_k']}")
    print(f"  Conditional: {is_conditional}")
    
    generated_files = []
    
    for i, seed in enumerate(FIXED_PARAMS["seeds"]):
        # Set seed
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        
        # Prepare conditions
        if is_conditional and i < len(FIXED_PROMPTS):
            prompt = FIXED_PROMPTS[i]
            conds = [ConditioningAttributes(text={"description": prompt})]
        else:
            prompt = None
            conds = [ConditioningAttributes()]
        
        print(f"  Sample {i+1}/{len(FIXED_PARAMS['seeds'])} (seed={seed})...")
        
        with torch.no_grad():
            tokens = lm_model.generate(
                prompt=None,
                conditions=conds,
                num_samples=1,
                max_gen_len=max_gen_len,
                use_sampling=FIXED_PARAMS["use_sampling"],
                temp=FIXED_PARAMS["temperature"],
                top_k=FIXED_PARAMS["top_k"],
                top_p=FIXED_PARAMS["top_p"],
            )
            
            audio = compression_model.decode(tokens)
        
        audio = audio.detach().cpu().squeeze(0)
        
        # Save audio
        filename = f"sample_seed{seed:04d}.wav"
        filepath = epoch_dir / filename
        torchaudio.save(str(filepath), audio, sample_rate)
        
        # Compute basic stats
        rms = torch.sqrt(torch.mean(audio ** 2)).item()
        peak = torch.max(torch.abs(audio)).item()
        
        generated_files.append({
            "filename": filename,
            "filepath": str(filepath),
            "seed": seed,
            "prompt": prompt,
            "duration": FIXED_PARAMS["duration"],
            "rms": round(rms, 4),
            "peak": round(peak, 4),
        })
        
        print(f"    Saved: {filename} (RMS={rms:.4f}, peak={peak:.4f})")
    
    # Save metadata
    metadata = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "musicgen_checkpoint": str(musicgen_checkpoint),
        "compression_checkpoint": str(compression_checkpoint),
        "params": FIXED_PARAMS,
        "is_conditional": is_conditional,
        "files": generated_files,
    }
    
    metadata_path = epoch_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Generated {len(generated_files)} samples to {epoch_dir}")
    
    return generated_files


# =============================================================================
# HTML INDEX GENERATION
# =============================================================================

def generate_html_index(output_dir: Path):
    """Generate HTML index for A/B comparison across epochs."""
    
    # Find all epoch directories
    epoch_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    
    if not epoch_dirs:
        print("No epoch directories found!")
        return
    
    # Load metadata from each epoch
    epochs_data = []
    for epoch_dir in epoch_dirs:
        metadata_path = epoch_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
                data["dir_name"] = epoch_dir.name
                epochs_data.append(data)
    
    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>MusicGen Training Samples - A/B Comparison</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .epoch-section {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .sample-card {
            background: #fafafa;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #eee;
        }
        .sample-card h4 { margin: 0 0 10px 0; }
        .sample-card audio { width: 100%; margin: 10px 0; }
        .sample-card .stats {
            font-size: 12px;
            color: #666;
        }
        .sample-card .prompt {
            font-style: italic;
            color: #888;
            font-size: 13px;
            margin-top: 8px;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .comparison-table th, .comparison-table td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }
        .comparison-table th { background: #f0f0f0; }
        .comparison-table audio { width: 150px; }
        .nav {
            position: sticky;
            top: 0;
            background: white;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .nav a {
            margin-right: 15px;
            text-decoration: none;
            color: #0066cc;
        }
        .params {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <h1>🎵 MusicGen Training Samples</h1>
    <p>A/B comparison across training epochs with fixed seeds</p>
    
    <div class="nav">
        <strong>Jump to:</strong>
"""
    
    # Add navigation links
    for data in epochs_data:
        epoch = data["epoch"]
        html += f'        <a href="#epoch-{epoch}">Epoch {epoch}</a>\n'
    
    html += """        <a href="#comparison">Side-by-Side</a>
    </div>
    
    <div class="params">
        <strong>Fixed Generation Parameters:</strong><br>
"""
    
    if epochs_data:
        params = epochs_data[0].get("params", FIXED_PARAMS)
        html += f"        Seeds: {params.get('seeds', FIXED_PARAMS['seeds'])}<br>\n"
        html += f"        Temperature: {params.get('temperature', FIXED_PARAMS['temperature'])}<br>\n"
        html += f"        Top-k: {params.get('top_k', FIXED_PARAMS['top_k'])}<br>\n"
        html += f"        Duration: {params.get('duration', FIXED_PARAMS['duration'])}s\n"
    
    html += """    </div>
"""
    
    # Add epoch sections
    for data in epochs_data:
        epoch = data["epoch"]
        dir_name = data["dir_name"]
        timestamp = data.get("timestamp", "Unknown")
        files = data.get("files", [])
        is_conditional = data.get("is_conditional", False)
        
        html += f"""
    <div class="epoch-section" id="epoch-{epoch}">
        <h2>Epoch {epoch}</h2>
        <p>Generated: {timestamp} | Conditional: {is_conditional}</p>
        <div class="sample-grid">
"""
        
        for file_info in files:
            seed = file_info["seed"]
            filename = file_info["filename"]
            rms = file_info.get("rms", "N/A")
            peak = file_info.get("peak", "N/A")
            prompt = file_info.get("prompt", "")
            
            html += f"""            <div class="sample-card">
                <h4>Seed {seed}</h4>
                <audio controls preload="metadata">
                    <source src="{dir_name}/{filename}" type="audio/wav">
                </audio>
                <div class="stats">RMS: {rms} | Peak: {peak}</div>
"""
            if prompt:
                html += f'                <div class="prompt">"{prompt}"</div>\n'
            
            html += """            </div>
"""
        
        html += """        </div>
    </div>
"""
    
    # Add side-by-side comparison table
    if len(epochs_data) > 1:
        html += """
    <div class="epoch-section" id="comparison">
        <h2>Side-by-Side Comparison</h2>
        <p>Compare the same seed across epochs</p>
        <table class="comparison-table">
            <tr>
                <th>Seed</th>
"""
        
        for data in epochs_data:
            html += f'                <th>Epoch {data["epoch"]}</th>\n'
        
        html += """            </tr>
"""
        
        # Get all unique seeds
        all_seeds = set()
        for data in epochs_data:
            for f in data.get("files", []):
                all_seeds.add(f["seed"])
        
        for seed in sorted(all_seeds):
            html += f"""            <tr>
                <td><strong>Seed {seed}</strong></td>
"""
            for data in epochs_data:
                dir_name = data["dir_name"]
                file_info = next((f for f in data.get("files", []) if f["seed"] == seed), None)
                
                if file_info:
                    filename = file_info["filename"]
                    html += f"""                <td>
                    <audio controls preload="metadata">
                        <source src="{dir_name}/{filename}" type="audio/wav">
                    </audio>
                </td>
"""
                else:
                    html += "                <td>-</td>\n"
            
            html += "            </tr>\n"
        
        html += """        </table>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    # Save HTML
    index_path = output_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"✓ Generated HTML index: {index_path}")
    print(f"  Open in browser to compare samples across epochs")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fixed-seed sample generation")
    parser.add_argument("--epoch", type=int, help="Epoch number (for labeling)")
    parser.add_argument("--checkpoint", type=Path, help="MusicGen checkpoint to use")
    parser.add_argument("--compression-checkpoint", type=Path,
                        help="Compression checkpoint to use")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_BASE_DIR,
                        help="Output directory for samples")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--generate-index", action="store_true",
                        help="Generate HTML index without generating new samples")
    parser.add_argument("--list-epochs", action="store_true",
                        help="List available checkpoints by epoch")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FIXED-SEED SAMPLE GENERATION")
    print("=" * 70)
    
    # List epochs mode
    if args.list_epochs:
        print("\nAvailable MusicGen checkpoints:")
        checkpoints = find_checkpoints_by_epoch(EXPERIMENTS_DIR)
        for epoch in sorted(checkpoints.keys()):
            print(f"  Epoch {epoch}: {checkpoints[epoch]}")
        return
    
    # Generate index mode
    if args.generate_index:
        print(f"\nGenerating HTML index for {args.output_dir}...")
        generate_html_index(args.output_dir)
        return
    
    # Generation mode
    # Find checkpoints
    musicgen_ckpt = args.checkpoint
    epoch = args.epoch
    
    if musicgen_ckpt is None:
        result = find_latest_musicgen_checkpoint(EXPERIMENTS_DIR)
        if result is None:
            print("❌ No MusicGen checkpoint found!")
            sys.exit(1)
        musicgen_ckpt, auto_epoch = result
        if epoch is None:
            epoch = auto_epoch
        print(f"Using latest checkpoint: {musicgen_ckpt}")
    
    if epoch is None:
        epoch = 0
        print("Warning: No epoch specified, using 0")
    
    compression_ckpt = args.compression_checkpoint
    if compression_ckpt is None:
        compression_ckpt = find_compression_checkpoint(EXPERIMENTS_DIR)
        if compression_ckpt is None:
            print("❌ No compression checkpoint found!")
            sys.exit(1)
        print(f"Using compression checkpoint: {compression_ckpt}")
    
    print(f"\nGeneration settings:")
    print(f"  Epoch: {epoch}")
    print(f"  MusicGen: {musicgen_ckpt}")
    print(f"  Compression: {compression_ckpt}")
    print(f"  Output: {args.output_dir}")
    
    # Generate samples
    generate_samples(
        musicgen_checkpoint=musicgen_ckpt,
        compression_checkpoint=compression_ckpt,
        output_dir=args.output_dir,
        epoch=epoch,
        device=args.device,
    )
    
    # Update HTML index
    print("\nUpdating HTML index...")
    generate_html_index(args.output_dir)


if __name__ == "__main__":
    main()
