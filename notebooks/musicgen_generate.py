#!/usr/bin/env python3
"""
MusicGen Audio Generation Script

Generates audio samples from a trained MusicGen model.
Automatically discovers the latest MusicGen and compression checkpoints.

Usage:
    python musicgen_generate.py
    
    # With custom settings via environment:
    NUM_SAMPLES=5 DURATION=10 python musicgen_generate.py
"""

from pathlib import Path
import os
import sys
import datetime
import json
import yaml

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/root/workspace"))

AUDIOCRAFT_REPO_DIR = BASE_DIR / "audiocraft"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "audiocraft"
OUTPUT_DIR = BASE_DIR / "Training" / "outputs" / "generated"

# Add audiocraft to path
sys.path.insert(0, str(AUDIOCRAFT_REPO_DIR))

# =============================================================================
# GENERATION SETTINGS
# =============================================================================

# Number of samples to generate
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 4))

# Duration of each sample in seconds
DURATION_SECONDS = float(os.environ.get("DURATION", 8.0))

# Sampling parameters
USE_SAMPLING = True         # Use sampling (True) or greedy decoding (False)
TEMPERATURE = 0.5           # Sampling temperature (higher = more random)
TOP_K = 250                 # Top-k sampling (0 to disable)
TOP_P = 0.0                 # Nucleus sampling threshold (0 to disable, 0.9 typical)

# Seed settings
BASE_SEED = 2026
FORCE_UNIQUE_SEEDS = True   # Generate each sample with different seed

# Text prompt for conditional generation (ignored if model is unconditional)
PROMPT = "A warm synth pad"

# Device
DEVICE = os.environ.get("DEVICE", "cuda")  # "cuda" or "cpu"

# =============================================================================
# CHECKPOINT DISCOVERY
# =============================================================================

def read_solver_from_config(xp_dir: Path) -> tuple:
    """Read solver name from experiment config."""
    for candidate in [xp_dir / "config.yaml", xp_dir / ".hydra" / "config.yaml"]:
        if candidate.exists():
            try:
                cfg = yaml.safe_load(candidate.read_text())
                solver = cfg.get("solver", "")
                return str(solver), cfg
            except Exception:
                continue
    return "", None


def find_experiments(filter_keyword: str = None) -> list:
    """Find experiment directories, optionally filtered by solver keyword."""
    xp_root = EXPERIMENTS_DIR / "xps"
    if not xp_root.exists():
        return []
    
    results = []
    for xp_dir in xp_root.iterdir():
        if not xp_dir.is_dir():
            continue
        
        solver_str, cfg = read_solver_from_config(xp_dir)
        if filter_keyword and filter_keyword not in solver_str.lower():
            continue
        
        results.append((xp_dir, solver_str, cfg))
    
    # Sort by modification time (newest last)
    results.sort(key=lambda t: t[0].stat().st_mtime)
    return results


def find_checkpoint(xp_dir: Path) -> Path | None:
    """Find the best/latest checkpoint in an experiment directory."""
    priority_patterns = [
        "*best*.pt", "*best*.pth", "*best*.th",
        "*latest*.pt", "*latest*.pth", "*latest*.th",
        "checkpoint*.pt", "checkpoint*.pth", "checkpoint*.th",
    ]
    
    for pattern in priority_patterns:
        files = sorted(xp_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0]
    
    # Fallback: any checkpoint
    all_ckpts = list(xp_dir.rglob("*.pt")) + list(xp_dir.rglob("*.pth")) + list(xp_dir.rglob("*.th"))
    if all_ckpts:
        return sorted(all_ckpts, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    
    return None


def find_compression_checkpoint() -> tuple:
    """Find the latest compression checkpoint and its metadata."""
    compression_xps = find_experiments("compression")
    if not compression_xps:
        raise FileNotFoundError(
            "No compression checkpoint found. Train a compression model first."
        )
    
    xp_dir, solver_str, cfg = compression_xps[-1]  # Latest
    ckpt = find_checkpoint(xp_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {xp_dir}")
    
    return ckpt, xp_dir, cfg


def find_musicgen_checkpoint() -> tuple:
    """Find the latest MusicGen checkpoint."""
    # Try musicgen first, then audiogen
    for keyword in ["musicgen", "audiogen"]:
        xps = find_experiments(keyword)
        if xps:
            xp_dir, solver_str, cfg = xps[-1]  # Latest
            ckpt = find_checkpoint(xp_dir)
            if ckpt is not None:
                return ckpt, xp_dir, cfg
    
    raise FileNotFoundError(
        "No MusicGen/AudioGen checkpoint found. Train a model first using musicgen_train.py"
    )


# =============================================================================
# MAIN GENERATION
# =============================================================================

def main():
    import torch
    import torchaudio
    import omegaconf
    from audiocraft.solvers import CompressionSolver
    from audiocraft.utils import checkpoint as ckpt_utils
    from audiocraft.models.builders import get_lm_model
    from audiocraft.modules.conditioners import ConditioningAttributes
    
    print(f"\n{'='*60}")
    print("MusicGen Audio Generation")
    print(f"{'='*60}")
    print(f"Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Find checkpoints
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("Discovering Checkpoints")
    print(f"{'='*60}")
    
    # Compression checkpoint
    compression_ckpt, compression_xp, _ = find_compression_checkpoint()
    print(f"Compression XP: {compression_xp.name}")
    print(f"Compression checkpoint: {compression_ckpt.name}")
    
    # MusicGen checkpoint
    musicgen_ckpt, musicgen_xp, musicgen_cfg = find_musicgen_checkpoint()
    print(f"MusicGen XP: {musicgen_xp.name}")
    print(f"MusicGen checkpoint: {musicgen_ckpt.name}")
    
    # ==========================================================================
    # Load models
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("Loading Models")
    print(f"{'='*60}")
    
    device = DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu"
    print(f"Device: {device}")
    
    # Load compression model
    print(f"Loading compression model...")
    compression_model = CompressionSolver.model_from_checkpoint(
        str(compression_ckpt), device=device
    )
    sample_rate = int(compression_model.sample_rate)
    frame_rate = compression_model.frame_rate
    n_q = getattr(compression_model, "num_codebooks", None)
    if n_q is None and hasattr(compression_model, "quantizer"):
        n_q = getattr(compression_model.quantizer, "n_q", None)
    
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Frame rate: {frame_rate} Hz")
    print(f"  Codebooks: {n_q}")
    
    # Load LM model config
    hydra_config = musicgen_xp / ".hydra" / "config.yaml"
    if not hydra_config.exists():
        # Try alternative location
        hydra_config = musicgen_xp / "config.yaml"
    
    if not hydra_config.exists():
        raise FileNotFoundError(f"Config not found in {musicgen_xp}")
    
    cfg = omegaconf.OmegaConf.load(hydra_config)
    cfg.device = device
    
    # Build and load LM model
    print(f"Building LM model...")
    lm_model = get_lm_model(cfg)
    
    print(f"Loading LM weights from {musicgen_ckpt.name}...")
    checkpoint_data = ckpt_utils.load_checkpoint(musicgen_ckpt, is_sharded=False)
    lm_model.load_state_dict(checkpoint_data["model"])
    lm_model.to(device)
    lm_model.eval()
    
    # Calculate generation length
    max_gen_len = int(DURATION_SECONDS * frame_rate)
    print(f"✓ Models loaded: generating {max_gen_len} tokens ({DURATION_SECONDS}s)")
    
    # ==========================================================================
    # Generate audio
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("Generating Audio")
    print(f"{'='*60}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Duration: {DURATION_SECONDS}s each")
    print(f"Sampling: {'yes' if USE_SAMPLING else 'no (greedy)'}")
    if USE_SAMPLING:
        print(f"  Temperature: {TEMPERATURE}")
        print(f"  Top-k: {TOP_K}")
        print(f"  Top-p: {TOP_P}")
    print(f"Prompt: {PROMPT[:50]}..." if len(PROMPT) > 50 else f"Prompt: {PROMPT}")
    
    # Prepare conditions (for conditional generation)
    conds = [ConditioningAttributes(text={"description": PROMPT})]
    
    with torch.no_grad():
        if FORCE_UNIQUE_SEEDS and NUM_SAMPLES > 1:
            # Generate one-by-one with different seeds
            print(f"\nGenerating with unique seeds...")
            all_tokens = []
            
            for i in range(NUM_SAMPLES):
                seed = BASE_SEED + i
                torch.manual_seed(seed)
                if device.startswith("cuda"):
                    torch.cuda.manual_seed_all(seed)
                
                print(f"  Sample {i+1}/{NUM_SAMPLES} (seed={seed})...", end=" ", flush=True)
                
                tokens = lm_model.generate(
                    prompt=None,
                    conditions=conds,
                    num_samples=1,
                    max_gen_len=max_gen_len,
                    use_sampling=USE_SAMPLING,
                    temp=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                )
                all_tokens.append(tokens)
                print("done")
            
            tokens = torch.cat(all_tokens, dim=0)
        else:
            # Batched generation
            print(f"\nGenerating batch (seed={BASE_SEED})...")
            torch.manual_seed(BASE_SEED)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(BASE_SEED)
            
            tokens = lm_model.generate(
                prompt=None,
                conditions=conds,
                num_samples=NUM_SAMPLES,
                max_gen_len=max_gen_len,
                use_sampling=USE_SAMPLING,
                temp=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
            )
        
        # Decode tokens to audio
        print(f"Decoding {tokens.shape[0]} samples...")
        audio = compression_model.decode(tokens)
    
    audio = audio.detach().cpu()
    
    # ==========================================================================
    # Save audio files
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("Saving Audio Files")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    for i in range(audio.shape[0]):
        wav = audio[i]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        
        # Generate filename with timestamp
        filename = f"generated_{timestamp}_{i:03d}.wav"
        out_path = OUTPUT_DIR / filename
        
        torchaudio.save(str(out_path), wav, sample_rate)
        
        # Compute stats
        duration = wav.shape[-1] / sample_rate
        rms = torch.sqrt(torch.mean(wav ** 2)).item()
        peak = torch.max(torch.abs(wav)).item()
        
        saved_files.append({
            "path": str(out_path),
            "filename": filename,
            "duration_sec": round(duration, 2),
            "rms": round(rms, 4),
            "peak": round(peak, 4),
        })
        
        print(f"  ✓ {filename}: {duration:.2f}s, RMS={rms:.4f}, peak={peak:.4f}")
    
    # ==========================================================================
    # Save generation metadata
    # ==========================================================================
    
    # Convert numpy types to Python native types for JSON serialization
    def to_native(x):
        if hasattr(x, 'item'):
            return x.item()
        return x
    
    metadata = {
        "timestamp": timestamp,
        "compression_checkpoint": str(compression_ckpt),
        "musicgen_checkpoint": str(musicgen_ckpt),
        "musicgen_xp": musicgen_xp.name,
        "sample_rate": to_native(sample_rate),
        "frame_rate": to_native(frame_rate),
        "num_codebooks": to_native(n_q),
        "generation_settings": {
            "num_samples": NUM_SAMPLES,
            "duration_seconds": DURATION_SECONDS,
            "use_sampling": USE_SAMPLING,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "base_seed": BASE_SEED,
            "force_unique_seeds": FORCE_UNIQUE_SEEDS,
            "prompt": PROMPT,
        },
        "generated_files": saved_files,
    }
    
    metadata_path = OUTPUT_DIR / f"metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved: {metadata_path.name}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print(f"\n{'='*60}")
    print("Generation Complete!")
    print(f"{'='*60}")
    print(f"Generated {len(saved_files)} audio files")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    
    # Token diversity check
    if tokens.shape[0] >= 2:
        identical = (tokens[0] == tokens[1]).all().item()
        print(f"\nDiversity check: samples 0 and 1 {'identical ⚠️' if identical else 'different ✓'}")
    
    return saved_files


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Generation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
