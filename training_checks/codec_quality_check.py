#!/usr/bin/env python3
"""
Task 1: Codec Reconstruction Quality Verification

Samples N clips from validation set, runs encode→decode with a compression
checkpoint, saves original and reconstructed wav pairs, and logs metrics.

NOTE: AudioCraft's SI-SNR is SIGN-INVERTED compared to standard convention!
      - Standard: higher SI-SNR = better (positive values good)
      - AudioCraft: lower (more negative) SI-SNR = better reconstruction
      - This script reports BOTH conventions for clarity.

Usage:
    python codec_quality_check.py
    python codec_quality_check.py --checkpoint /path/to/checkpoint.th
    python codec_quality_check.py --num-samples 20 --output-dir ./codec_test
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torchaudio

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/root/workspace")
AUDIOCRAFT_DIR = BASE_DIR / "audiocraft"
EXPERIMENTS_DIR = BASE_DIR / "experiments" / "audiocraft"
DATA_DIR = BASE_DIR / "data"

# Default checkpoint (the one from your training)
DEFAULT_CHECKPOINT = EXPERIMENTS_DIR / "xps" / "528474a5" / "checkpoint.th"

# Default validation JSONL
DEFAULT_VALID_JSONL = DATA_DIR / "all_data" / "egs" / "valid" / "data.jsonl"

# Default output directory
DEFAULT_OUTPUT_DIR = BASE_DIR / "Training" / "outputs" / "codec_quality_check"

# Number of samples to test
DEFAULT_NUM_SAMPLES = 10

# Segment duration for testing (seconds)
SEGMENT_DURATION = 10.0

# =============================================================================
# METRICS
# =============================================================================

def compute_l1_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute L1 loss between original and reconstructed."""
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    return torch.mean(torch.abs(original[..., :min_len] - reconstructed[..., :min_len])).item()


def compute_mse_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute MSE loss between original and reconstructed."""
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    return torch.mean((original[..., :min_len] - reconstructed[..., :min_len]) ** 2).item()


def compute_sisnr(original: torch.Tensor, reconstructed: torch.Tensor, eps: float = 1e-8) -> tuple[float, float]:
    """
    Compute SI-SNR between original and reconstructed.
    
    Returns:
        (standard_sisnr, audiocraft_sisnr):
            - standard_sisnr: Standard convention (higher = better)
            - audiocraft_sisnr: AudioCraft convention (sign-inverted, lower = better)
    
    NOTE: AudioCraft uses SIGN-INVERTED SI-SNR!
          In their loss functions, they minimize -SI-SNR, so lower values = better.
          This is opposite to standard convention where higher SI-SNR = better.
    """
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    orig = original[..., :min_len].flatten()
    recon = reconstructed[..., :min_len].flatten()
    
    # Zero-mean normalization
    orig = orig - orig.mean()
    recon = recon - recon.mean()
    
    # Compute SI-SNR (standard convention: higher = better)
    dot = torch.dot(orig, recon)
    s_target = dot * orig / (torch.dot(orig, orig) + eps)
    e_noise = recon - s_target
    
    si_snr_standard = 10 * torch.log10(
        torch.dot(s_target, s_target) / (torch.dot(e_noise, e_noise) + eps) + eps
    ).item()
    
    # AudioCraft convention: sign-inverted (lower = better)
    si_snr_audiocraft = -si_snr_standard
    
    return si_snr_standard, si_snr_audiocraft


def compute_mel_spectrogram_loss(
    original: torch.Tensor, 
    reconstructed: torch.Tensor,
    sample_rate: int = 32000,
    n_fft: int = 1024,
    n_mels: int = 80,
    hop_length: int = 256
) -> float:
    """Compute mel spectrogram L1 loss."""
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    orig = original[..., :min_len]
    recon = reconstructed[..., :min_len]
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
    ).to(orig.device)
    
    mel_orig = mel_transform(orig)
    mel_recon = mel_transform(recon)
    
    # Log mel spectrogram
    mel_orig = torch.log(mel_orig + 1e-5)
    mel_recon = torch.log(mel_recon + 1e-5)
    
    return torch.mean(torch.abs(mel_orig - mel_recon)).item()


def compute_multi_scale_spectral_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    scales: list[int] = [512, 1024, 2048]
) -> float:
    """Compute multi-scale spectral loss."""
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    orig = original[..., :min_len]
    recon = reconstructed[..., :min_len]
    
    total_loss = 0.0
    for n_fft in scales:
        hop_length = n_fft // 4
        
        # Pad if necessary
        if orig.shape[-1] < n_fft:
            continue
            
        spec_orig = torch.stft(
            orig.squeeze(), n_fft=n_fft, hop_length=hop_length,
            return_complex=True, window=torch.hann_window(n_fft).to(orig.device)
        )
        spec_recon = torch.stft(
            recon.squeeze(), n_fft=n_fft, hop_length=hop_length,
            return_complex=True, window=torch.hann_window(n_fft).to(recon.device)
        )
        
        # Magnitude spectrogram loss
        mag_orig = torch.abs(spec_orig)
        mag_recon = torch.abs(spec_recon)
        
        total_loss += torch.mean(torch.abs(mag_orig - mag_recon)).item()
    
    return total_loss / len(scales) if scales else 0.0


# =============================================================================
# MAIN
# =============================================================================

def load_audio_samples(jsonl_path: Path, num_samples: int, segment_duration: float, sample_rate: int) -> list[dict]:
    """Load audio samples from validation JSONL."""
    import random
    
    samples = []
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    # Shuffle and sample
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(lines)
    
    for line in lines:
        if len(samples) >= num_samples:
            break
            
        entry = json.loads(line)
        audio_path = Path(entry.get("path", ""))
        duration = entry.get("duration", 0)
        
        if duration < segment_duration:
            continue
            
        if not audio_path.exists():
            continue
        
        samples.append({
            "path": audio_path,
            "duration": duration,
            "entry": entry
        })
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Codec reconstruction quality check")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to compression checkpoint")
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_VALID_JSONL,
                        help="Path to validation JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for wav files and metrics")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Add audiocraft to path
    sys.path.insert(0, str(AUDIOCRAFT_DIR))
    from audiocraft.solvers import CompressionSolver
    
    print("=" * 70)
    print("CODEC RECONSTRUCTION QUALITY CHECK")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation JSONL: {args.valid_jsonl}")
    print(f"Num samples: {args.num_samples}")
    print(f"Device: {args.device}")
    
    # Check checkpoint exists
    if not args.checkpoint.exists():
        print(f"\n❌ ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load compression model
    print(f"\nLoading compression model...")
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = CompressionSolver.model_from_checkpoint(str(args.checkpoint), device=device)
    model.eval()
    
    sample_rate = int(model.sample_rate)
    n_q = getattr(model, "num_codebooks", None)
    if n_q is None and hasattr(model, "quantizer"):
        n_q = getattr(model.quantizer, "n_q", None)
    
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Codebooks: {n_q}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    print(f"\nLoading samples from {args.valid_jsonl}...")
    samples = load_audio_samples(args.valid_jsonl, args.num_samples, SEGMENT_DURATION, sample_rate)
    print(f"  Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("❌ No valid samples found!")
        sys.exit(1)
    
    # Process samples
    print(f"\nProcessing samples...")
    print("-" * 70)
    
    all_metrics = []
    segment_samples = int(SEGMENT_DURATION * sample_rate)
    
    for i, sample in enumerate(samples):
        audio_path = sample["path"]
        
        # Load audio
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Resample if necessary
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Take segment
        if waveform.shape[-1] > segment_samples:
            waveform = waveform[..., :segment_samples]
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]
        
        # Encode and decode
        with torch.no_grad():
            codes, scale = model.encode(waveform)  # encode returns (codes, scale)
            reconstructed = model.decode(codes, scale)
        
        # Move to CPU for metrics
        orig_cpu = waveform.squeeze(0).cpu()
        recon_cpu = reconstructed.squeeze(0).cpu()
        
        # Compute metrics
        l1 = compute_l1_loss(orig_cpu, recon_cpu)
        mse = compute_mse_loss(orig_cpu, recon_cpu)
        sisnr_std, sisnr_ac = compute_sisnr(orig_cpu, recon_cpu)
        mel = compute_mel_spectrogram_loss(orig_cpu, recon_cpu, sample_rate)
        msspec = compute_multi_scale_spectral_loss(orig_cpu, recon_cpu)
        
        metrics = {
            "sample_idx": i,
            "source_path": str(audio_path),
            "l1": l1,
            "mse": mse,
            "sisnr_standard": sisnr_std,  # Higher = better (standard convention)
            "sisnr_audiocraft": sisnr_ac,  # Lower = better (AudioCraft convention)
            "mel": mel,
            "msspec": msspec,
        }
        all_metrics.append(metrics)
        
        # Save wav files
        orig_path = output_dir / f"sample_{i:03d}_orig.wav"
        recon_path = output_dir / f"sample_{i:03d}_recon.wav"
        
        torchaudio.save(str(orig_path), orig_cpu, sample_rate)
        torchaudio.save(str(recon_path), recon_cpu, sample_rate)
        
        print(f"Sample {i+1}/{len(samples)}: "
              f"L1={l1:.4f}, MEL={mel:.4f}, MSSPEC={msspec:.4f}, "
              f"SI-SNR(std)={sisnr_std:+.2f}dB, SI-SNR(AC)={sisnr_ac:+.2f}dB")
    
    # Compute averages
    print("-" * 70)
    print("\nAGGREGATE METRICS:")
    print("-" * 70)
    
    avg_metrics = {
        "l1": sum(m["l1"] for m in all_metrics) / len(all_metrics),
        "mse": sum(m["mse"] for m in all_metrics) / len(all_metrics),
        "sisnr_standard": sum(m["sisnr_standard"] for m in all_metrics) / len(all_metrics),
        "sisnr_audiocraft": sum(m["sisnr_audiocraft"] for m in all_metrics) / len(all_metrics),
        "mel": sum(m["mel"] for m in all_metrics) / len(all_metrics),
        "msspec": sum(m["msspec"] for m in all_metrics) / len(all_metrics),
    }
    
    print(f"  L1 Loss:        {avg_metrics['l1']:.4f}")
    print(f"  MSE Loss:       {avg_metrics['mse']:.6f}")
    print(f"  Mel Loss:       {avg_metrics['mel']:.4f}")
    print(f"  MS-Spec Loss:   {avg_metrics['msspec']:.4f}")
    print()
    print(f"  SI-SNR (standard):   {avg_metrics['sisnr_standard']:+.2f} dB  ← Higher = better")
    print(f"  SI-SNR (AudioCraft): {avg_metrics['sisnr_audiocraft']:+.2f} dB  ← Lower = better (sign-inverted)")
    
    # Quality assessment
    print()
    print("=" * 70)
    print("QUALITY ASSESSMENT:")
    print("=" * 70)
    
    sisnr = avg_metrics['sisnr_standard']
    if sisnr < 0:
        print(f"  ❌ CRITICAL: Negative SI-SNR ({sisnr:.2f} dB) indicates codec NOT converged!")
        print(f"     Reconstruction is WORSE than original signal.")
        print(f"     Training needs to continue significantly longer.")
    elif sisnr < 5:
        print(f"  ⚠️  WARNING: Low SI-SNR ({sisnr:.2f} dB) - codec needs more training")
        print(f"     Target for music: SI-SNR > 10 dB")
    elif sisnr < 10:
        print(f"  ⚠️  ACCEPTABLE: SI-SNR ({sisnr:.2f} dB) - codec partially converged")
        print(f"     May be usable but quality improvements possible with more training")
    else:
        print(f"  ✓ GOOD: SI-SNR ({sisnr:.2f} dB) - codec quality acceptable")
    
    # Save metrics to JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "checkpoint": str(args.checkpoint),
            "num_samples": len(all_metrics),
            "segment_duration": SEGMENT_DURATION,
            "sample_rate": sample_rate,
            "num_codebooks": n_q,
            "average_metrics": avg_metrics,
            "per_sample_metrics": all_metrics,
            "note": "AudioCraft uses SIGN-INVERTED SI-SNR! sisnr_audiocraft is what training logs show.",
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Audio pairs: sample_XXX_orig.wav / sample_XXX_recon.wav")
    print(f"\nListen to the wav pairs to subjectively assess quality!")


if __name__ == "__main__":
    main()
