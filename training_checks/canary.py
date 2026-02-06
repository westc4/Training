"""
Canary Generation for Training Evaluation

Task 5A: Deterministic "canary generation" at eval intervals.
Generates audio with fixed seed/prompt and computes audio heuristics.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import json
import datetime
import sys


@dataclass
class CanaryConfig:
    """Configuration for canary generation."""
    
    seed: int = 42
    duration_seconds: float = 8.0
    prompt: str = "A warm synth pad with a gentle melody"
    
    # Sampling parameters (deterministic by default)
    use_sampling: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0


@dataclass
class CanaryResult:
    """Result from a canary generation."""
    
    epoch: int
    iteration: int
    timestamp: str
    audio_path: str
    
    # Audio heuristics
    rms_loudness: float = 0.0
    peak_amplitude: float = 0.0
    spectral_centroid_mean: float = 0.0
    zero_crossing_rate: float = 0.0
    silence_ratio: float = 0.0
    
    # Generation config snapshot
    config: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "audio_path": self.audio_path,
            "heuristics": {
                "rms_loudness": round(self.rms_loudness, 6),
                "peak_amplitude": round(self.peak_amplitude, 6),
                "spectral_centroid_mean": round(self.spectral_centroid_mean, 2),
                "zero_crossing_rate": round(self.zero_crossing_rate, 6),
                "silence_ratio": round(self.silence_ratio, 4),
            },
            "config": self.config,
        }


class CanaryGenerator:
    """
    Generate deterministic "canary" samples at evaluation intervals.
    
    Produces consistent audio samples and tracks heuristics over training.
    """
    
    def __init__(
        self,
        xp_dir: Path,
        config: Optional[CanaryConfig] = None,
        audiocraft_repo: Optional[Path] = None,
    ):
        self.xp_dir = xp_dir
        self.config = config or CanaryConfig()
        self.audiocraft_repo = audiocraft_repo or Path("/root/workspace/audiocraft")
        
        self.canary_dir = xp_dir / "canaries"
        self.canary_dir.mkdir(parents=True, exist_ok=True)
        
        self._results: list[CanaryResult] = []
    
    def generate(
        self,
        epoch: int,
        iteration: int,
        lm_model: Any,
        compression_model: Any,
        device: str = "cuda",
    ) -> CanaryResult:
        """
        Generate a canary sample and compute heuristics.
        
        Args:
            epoch: Current training epoch
            iteration: Current iteration
            lm_model: The language model
            compression_model: The compression model for decoding
            device: Device to run on
            
        Returns:
            CanaryResult with audio path and heuristics
        """
        import torch
        import torchaudio
        
        sys.path.insert(0, str(self.audiocraft_repo))
        from audiocraft.modules.conditioners import ConditioningAttributes
        
        timestamp = datetime.datetime.now().isoformat()
        
        # Set deterministic seed
        torch.manual_seed(self.config.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.config.seed)
        
        # Prepare conditions
        conds = [ConditioningAttributes(text={"description": self.config.prompt})]
        
        # Calculate generation length
        frame_rate = compression_model.frame_rate
        max_gen_len = int(self.config.duration_seconds * frame_rate)
        
        # Generate tokens
        lm_model.eval()
        with torch.no_grad():
            tokens = lm_model.generate(
                prompt=None,
                conditions=conds,
                num_samples=1,
                max_gen_len=max_gen_len,
                use_sampling=self.config.use_sampling,
                temp=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            
            # Decode to audio
            audio = compression_model.decode(tokens)
        
        audio = audio.squeeze(0).cpu()  # (C, T)
        sample_rate = int(compression_model.sample_rate)
        
        # Save audio
        audio_filename = f"canary_epoch{epoch:04d}_iter{iteration:08d}.wav"
        audio_path = self.canary_dir / audio_filename
        
        # Ensure mono for saving
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(str(audio_path), audio, sample_rate)
        
        # Compute heuristics
        heuristics = self._compute_heuristics(audio, sample_rate)
        
        result = CanaryResult(
            epoch=epoch,
            iteration=iteration,
            timestamp=timestamp,
            audio_path=str(audio_path),
            rms_loudness=heuristics["rms_loudness"],
            peak_amplitude=heuristics["peak_amplitude"],
            spectral_centroid_mean=heuristics["spectral_centroid_mean"],
            zero_crossing_rate=heuristics["zero_crossing_rate"],
            silence_ratio=heuristics["silence_ratio"],
            config={
                "seed": self.config.seed,
                "duration_seconds": self.config.duration_seconds,
                "prompt": self.config.prompt,
                "use_sampling": self.config.use_sampling,
                "temperature": self.config.temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
            },
        )
        
        self._results.append(result)
        
        # Save result JSON
        json_path = self.canary_dir / f"canary_epoch{epoch:04d}_iter{iteration:08d}.json"
        with open(json_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return result
    
    def _compute_heuristics(self, audio: "torch.Tensor", sample_rate: int) -> dict:
        """Compute audio heuristics."""
        import torch
        
        # Ensure 1D for analysis
        if audio.dim() > 1:
            audio_1d = audio.mean(dim=0)  # Mix to mono
        else:
            audio_1d = audio
        
        # RMS loudness
        rms = torch.sqrt(torch.mean(audio_1d ** 2)).item()
        
        # Peak amplitude
        peak = torch.max(torch.abs(audio_1d)).item()
        
        # Zero crossing rate
        signs = torch.sign(audio_1d)
        sign_changes = torch.sum(torch.abs(signs[1:] - signs[:-1]) > 0).item()
        zcr = sign_changes / (len(audio_1d) - 1) if len(audio_1d) > 1 else 0.0
        
        # Silence ratio (samples below threshold)
        silence_threshold = 0.01
        silence_samples = torch.sum(torch.abs(audio_1d) < silence_threshold).item()
        silence_ratio = silence_samples / len(audio_1d) if len(audio_1d) > 0 else 0.0
        
        # Spectral centroid (simplified)
        spectral_centroid = self._compute_spectral_centroid(audio_1d, sample_rate)
        
        return {
            "rms_loudness": rms,
            "peak_amplitude": peak,
            "zero_crossing_rate": zcr,
            "silence_ratio": silence_ratio,
            "spectral_centroid_mean": spectral_centroid,
        }
    
    def _compute_spectral_centroid(self, audio: "torch.Tensor", sample_rate: int) -> float:
        """Compute mean spectral centroid."""
        import torch
        
        # Use STFT
        n_fft = 2048
        hop_length = 512
        
        try:
            # Pad if too short
            if len(audio) < n_fft:
                audio = torch.nn.functional.pad(audio, (0, n_fft - len(audio)))
            
            # Compute STFT
            spec = torch.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
                window=torch.hann_window(n_fft),
            )
            
            # Magnitude spectrum
            mag = torch.abs(spec)  # (freq_bins, time_frames)
            
            # Frequency bins
            freqs = torch.linspace(0, sample_rate / 2, mag.shape[0])
            
            # Spectral centroid per frame
            centroids = []
            for t in range(mag.shape[1]):
                frame_mag = mag[:, t]
                total_mag = torch.sum(frame_mag)
                if total_mag > 0:
                    centroid = torch.sum(freqs * frame_mag) / total_mag
                    centroids.append(centroid.item())
            
            return sum(centroids) / len(centroids) if centroids else 0.0
            
        except Exception:
            return 0.0
    
    def get_results(self) -> list[dict]:
        """Get all canary results as dicts."""
        return [r.to_dict() for r in self._results]
    
    def get_heuristics_trend(self) -> dict:
        """Get heuristics trending over epochs."""
        if not self._results:
            return {}
        
        return {
            "epochs": [r.epoch for r in self._results],
            "rms_loudness": [r.rms_loudness for r in self._results],
            "spectral_centroid_mean": [r.spectral_centroid_mean for r in self._results],
            "zero_crossing_rate": [r.zero_crossing_rate for r in self._results],
            "silence_ratio": [r.silence_ratio for r in self._results],
        }
    
    def save_summary(self):
        """Save summary of all canary generations."""
        summary = {
            "total_canaries": len(self._results),
            "results": self.get_results(),
            "trends": self.get_heuristics_trend(),
        }
        
        summary_path = self.canary_dir / "canary_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
