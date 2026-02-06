"""
Early Stopping for AudioCraft Training

Task 5B: Implement clear stop criteria for compression and MusicGen training.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json
import datetime


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    
    enabled: bool = True
    
    # Primary metric to monitor
    metric: str = "valid_ce"  # valid_ce, valid_ppl, valid_mel, valid_loss
    
    # Stopping criteria
    patience: int = 10  # Number of evaluations without improvement
    min_delta: float = 0.001  # Minimum change to qualify as improvement
    mode: str = "min"  # "min" for loss-like metrics, "max" for accuracy-like
    
    # Additional criteria
    max_epochs: Optional[int] = None  # Hard stop at this many epochs
    max_time_hours: Optional[float] = None  # Hard stop after this many hours
    
    # Ratio explosion detection (for compression)
    ratio_explosion_threshold: float = 100.0  # Flag if any ratio > this
    
    # For MusicGen: canary heuristic degradation
    canary_degradation_patience: int = 5  # Stop if canary quality degrades for N evals


@dataclass
class EarlyStoppingState:
    """Internal state for early stopping."""
    
    best_value: Optional[float] = None
    best_epoch: int = 0
    best_checkpoint_path: Optional[str] = None
    epochs_without_improvement: int = 0
    
    # History
    history: list[dict] = field(default_factory=list)
    
    # Timestamps
    start_time: Optional[str] = None
    
    # Canary tracking
    canary_degradation_count: int = 0
    last_canary_metrics: Optional[dict] = None


class EarlyStopping:
    """
    Early stopping handler for training.
    
    Monitors metrics and determines when to stop training.
    """
    
    def __init__(
        self,
        config: EarlyStoppingConfig,
        xp_dir: Optional[Path] = None,
    ):
        self.config = config
        self.xp_dir = xp_dir
        self.state = EarlyStoppingState()
        self.state.start_time = datetime.datetime.now().isoformat()
        
        # Callbacks
        self._on_new_best: Optional[callable] = None
        self._on_stop: Optional[callable] = None
    
    def set_on_new_best(self, callback: callable):
        """Set callback for when a new best is achieved."""
        self._on_new_best = callback
    
    def set_on_stop(self, callback: callable):
        """Set callback for when stopping is triggered."""
        self._on_stop = callback
    
    def check(
        self,
        epoch: int,
        metrics: dict,
        checkpoint_path: Optional[str] = None,
        canary_metrics: Optional[dict] = None,
    ) -> tuple[bool, str]:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch
            metrics: Dict of metric name -> value
            checkpoint_path: Path to current checkpoint (for best tracking)
            canary_metrics: Optional canary generation heuristics
            
        Returns:
            (should_stop, reason): Tuple of stop flag and reason string
        """
        if not self.config.enabled:
            return False, ""
        
        # Record history
        self.state.history.append({
            "epoch": epoch,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics,
            "canary_metrics": canary_metrics,
        })
        
        # Check max epochs
        if self.config.max_epochs is not None and epoch >= self.config.max_epochs:
            return True, f"Reached max epochs ({self.config.max_epochs})"
        
        # Check max time
        if self.config.max_time_hours is not None:
            start = datetime.datetime.fromisoformat(self.state.start_time)
            elapsed_hours = (datetime.datetime.now() - start).total_seconds() / 3600
            if elapsed_hours >= self.config.max_time_hours:
                return True, f"Reached max time ({self.config.max_time_hours:.1f}h)"
        
        # Check ratio explosion (compression)
        for key, value in metrics.items():
            if "ratio" in key.lower() and isinstance(value, (int, float)):
                if value > self.config.ratio_explosion_threshold:
                    return True, f"Ratio explosion: {key}={value:.2f} > {self.config.ratio_explosion_threshold}"
        
        # Check primary metric
        metric_value = metrics.get(self.config.metric)
        if metric_value is not None:
            is_better = self._is_better(metric_value, self.state.best_value)
            
            if is_better:
                self.state.best_value = metric_value
                self.state.best_epoch = epoch
                self.state.epochs_without_improvement = 0
                
                if checkpoint_path:
                    self.state.best_checkpoint_path = checkpoint_path
                    self._save_best_checkpoint(checkpoint_path, epoch, metrics)
                
                if self._on_new_best:
                    self._on_new_best(epoch, metric_value, checkpoint_path)
            else:
                self.state.epochs_without_improvement += 1
                
                if self.state.epochs_without_improvement >= self.config.patience:
                    return True, (
                        f"No improvement in {self.config.metric} for {self.config.patience} evaluations "
                        f"(best: {self.state.best_value:.6f} at epoch {self.state.best_epoch})"
                    )
        
        # Check canary degradation
        if canary_metrics is not None:
            if self._check_canary_degradation(canary_metrics):
                self.state.canary_degradation_count += 1
                if self.state.canary_degradation_count >= self.config.canary_degradation_patience:
                    return True, f"Canary quality degraded for {self.config.canary_degradation_patience} consecutive evaluations"
            else:
                self.state.canary_degradation_count = 0
            
            self.state.last_canary_metrics = canary_metrics
        
        return False, ""
    
    def _is_better(self, current: float, best: Optional[float]) -> bool:
        """Check if current value is better than best."""
        if best is None:
            return True
        
        if self.config.mode == "min":
            return current < best - self.config.min_delta
        else:
            return current > best + self.config.min_delta
    
    def _check_canary_degradation(self, canary_metrics: dict) -> bool:
        """Check if canary quality has degraded."""
        if self.state.last_canary_metrics is None:
            return False
        
        # Consider degradation if:
        # - RMS drops significantly (audio getting quieter/silent)
        # - Spectral centroid becomes very low or very high (abnormal)
        # - Silence ratio increases
        
        last = self.state.last_canary_metrics
        
        # Silence ratio increasing
        if canary_metrics.get("silence_ratio", 0) > last.get("silence_ratio", 0) + 0.1:
            return True
        
        # RMS dropping significantly
        if canary_metrics.get("rms_loudness", 0) < last.get("rms_loudness", 0) * 0.5:
            return True
        
        return False
    
    def _save_best_checkpoint(self, checkpoint_path: str, epoch: int, metrics: dict):
        """Save a copy of the best checkpoint."""
        if self.xp_dir is None:
            return
        
        import shutil
        
        src = Path(checkpoint_path)
        if not src.exists():
            return
        
        dst = self.xp_dir / "best_checkpoint.th"
        try:
            shutil.copy2(src, dst)
            
            # Save metadata
            meta_path = self.xp_dir / "best_checkpoint_meta.json"
            with open(meta_path, 'w') as f:
                json.dump({
                    "epoch": epoch,
                    "metrics": metrics,
                    "original_path": str(src),
                    "timestamp": datetime.datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save best checkpoint: {e}")
    
    def get_status(self) -> dict:
        """Get current early stopping status."""
        return {
            "enabled": self.config.enabled,
            "metric": self.config.metric,
            "best_value": self.state.best_value,
            "best_epoch": self.state.best_epoch,
            "best_checkpoint_path": self.state.best_checkpoint_path,
            "epochs_without_improvement": self.state.epochs_without_improvement,
            "patience": self.config.patience,
            "history_length": len(self.state.history),
        }
    
    def save_state(self, path: Optional[Path] = None):
        """Save early stopping state to file."""
        if path is None and self.xp_dir:
            path = self.xp_dir / "early_stopping_state.json"
        
        if path:
            with open(path, 'w') as f:
                json.dump({
                    "config": {
                        "enabled": self.config.enabled,
                        "metric": self.config.metric,
                        "patience": self.config.patience,
                        "min_delta": self.config.min_delta,
                        "mode": self.config.mode,
                    },
                    "state": {
                        "best_value": self.state.best_value,
                        "best_epoch": self.state.best_epoch,
                        "best_checkpoint_path": self.state.best_checkpoint_path,
                        "epochs_without_improvement": self.state.epochs_without_improvement,
                        "start_time": self.state.start_time,
                    },
                    "history": self.state.history,
                }, f, indent=2)
    
    @classmethod
    def load_state(cls, path: Path) -> "EarlyStopping":
        """Load early stopping state from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = EarlyStoppingConfig(
            enabled=data["config"]["enabled"],
            metric=data["config"]["metric"],
            patience=data["config"]["patience"],
            min_delta=data["config"]["min_delta"],
            mode=data["config"]["mode"],
        )
        
        es = cls(config, xp_dir=path.parent)
        es.state.best_value = data["state"]["best_value"]
        es.state.best_epoch = data["state"]["best_epoch"]
        es.state.best_checkpoint_path = data["state"]["best_checkpoint_path"]
        es.state.epochs_without_improvement = data["state"]["epochs_without_improvement"]
        es.state.start_time = data["state"]["start_time"]
        es.state.history = data.get("history", [])
        
        return es


# Convenience functions for different training types

def create_compression_early_stopping(
    xp_dir: Path,
    patience: int = 10,
    metric: str = "valid_mel",
) -> EarlyStopping:
    """Create early stopping configured for compression training."""
    config = EarlyStoppingConfig(
        enabled=True,
        metric=metric,
        patience=patience,
        min_delta=0.001,
        mode="min",
        ratio_explosion_threshold=100.0,
    )
    return EarlyStopping(config, xp_dir)


def create_musicgen_early_stopping(
    xp_dir: Path,
    patience: int = 10,
    metric: str = "valid_ce",
) -> EarlyStopping:
    """Create early stopping configured for MusicGen training."""
    config = EarlyStoppingConfig(
        enabled=True,
        metric=metric,
        patience=patience,
        min_delta=0.001,
        mode="min",
        canary_degradation_patience=5,
    )
    return EarlyStopping(config, xp_dir)
