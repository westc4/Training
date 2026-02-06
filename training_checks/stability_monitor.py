#!/usr/bin/env python3
"""
Task 3: Stability Guards for INF/Overflow Events

This module provides utilities for monitoring and handling gradient instability
during training. Can be imported into training scripts or used standalone.

Features:
- Tracks grad overflow events from AMP scaler
- Counts NaN/INF losses
- Implements adaptive LR reduction on instability
- Supports fail-fast on persistent NaNs
- Logs stability metrics per epoch

Usage in training script:
    from stability_monitor import StabilityMonitor
    
    monitor = StabilityMonitor(
        max_nan_ratio=0.1,      # Fail if >10% NaN losses
        lr_reduction_factor=0.5, # Halve LR on instability
        grad_clip_tightening=0.5 # Reduce max_norm by 50% on overflow
    )
    
    for batch in dataloader:
        loss = model(batch)
        
        if monitor.check_loss(loss):  # Returns False if NaN/INF
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            monitor.check_grad_norm(grad_norm)
            
            scaler.step(optimizer)
            scale_before = scaler.get_scale()
            scaler.update()
            scale_after = scaler.get_scale()
            
            monitor.check_scaler_update(scale_before, scale_after)
        
        if monitor.should_reduce_lr():
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
        
        if monitor.should_fail_fast():
            raise RuntimeError("Training unstable - too many NaN losses")
    
    monitor.log_epoch_summary(epoch)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import math

import torch

logger = logging.getLogger(__name__)


@dataclass
class StabilityConfig:
    """Configuration for stability monitoring."""
    
    # Thresholds for failure
    max_nan_ratio_per_epoch: float = 0.1      # Fail if >10% steps have NaN loss
    max_inf_grad_ratio_per_epoch: float = 0.3  # Warn if >30% steps have INF grad
    
    # Adaptive adjustments
    enable_adaptive_lr: bool = True
    lr_reduction_factor: float = 0.5          # Reduce LR by this factor
    lr_reduction_threshold: int = 5           # Reduce after N consecutive overflows
    min_lr: float = 1e-7                      # Don't reduce below this
    
    enable_adaptive_grad_clip: bool = True
    grad_clip_tightening_factor: float = 0.8  # Tighten clip by this factor
    min_grad_clip: float = 0.1                # Don't go below this
    
    # Logging
    log_every_n_steps: int = 100
    save_metrics_path: Optional[Path] = None


@dataclass
class EpochMetrics:
    """Metrics collected during an epoch."""
    
    total_steps: int = 0
    nan_loss_count: int = 0
    inf_loss_count: int = 0
    inf_grad_count: int = 0
    scaler_overflow_count: int = 0
    grad_norms: list = field(default_factory=list)
    loss_values: list = field(default_factory=list)
    lr_reductions: int = 0
    grad_clip_tightenings: int = 0
    
    def nan_ratio(self) -> float:
        return self.nan_loss_count / max(1, self.total_steps)
    
    def inf_grad_ratio(self) -> float:
        return self.inf_grad_count / max(1, self.total_steps)
    
    def overflow_ratio(self) -> float:
        return self.scaler_overflow_count / max(1, self.total_steps)
    
    def avg_grad_norm(self) -> float:
        valid_norms = [n for n in self.grad_norms if math.isfinite(n)]
        return sum(valid_norms) / max(1, len(valid_norms))
    
    def avg_loss(self) -> float:
        valid_losses = [l for l in self.loss_values if math.isfinite(l)]
        return sum(valid_losses) / max(1, len(valid_losses))


class StabilityMonitor:
    """
    Monitors training stability and provides adaptive responses to instability.
    
    Tracks:
    - NaN/INF losses
    - Gradient norm explosions (INF)
    - AMP scaler overflow events
    
    Responds by:
    - Logging warnings
    - Reducing learning rate
    - Tightening gradient clipping
    - Failing fast if too unstable
    """
    
    def __init__(self, config: StabilityConfig = None, **kwargs):
        self.config = config or StabilityConfig(**kwargs)
        self.epoch_metrics = EpochMetrics()
        self.all_epoch_metrics: list[dict] = []
        self.consecutive_overflows = 0
        self.current_lr_factor = 1.0
        self.current_grad_clip_factor = 1.0
        self._step_in_epoch = 0
    
    def reset_epoch(self):
        """Call at the start of each epoch."""
        # Save previous epoch metrics
        if self.epoch_metrics.total_steps > 0:
            self.all_epoch_metrics.append(self._metrics_to_dict(self.epoch_metrics))
        
        self.epoch_metrics = EpochMetrics()
        self._step_in_epoch = 0
    
    def check_loss(self, loss: torch.Tensor) -> bool:
        """
        Check if loss is valid.
        
        Returns:
            True if loss is valid, False if NaN/INF (skip this step)
        """
        self.epoch_metrics.total_steps += 1
        self._step_in_epoch += 1
        
        loss_val = loss.item() if torch.is_tensor(loss) else loss
        
        if math.isnan(loss_val):
            self.epoch_metrics.nan_loss_count += 1
            self._log_instability(f"NaN loss at step {self._step_in_epoch}")
            return False
        
        if math.isinf(loss_val):
            self.epoch_metrics.inf_loss_count += 1
            self._log_instability(f"INF loss at step {self._step_in_epoch}: {loss_val}")
            return False
        
        self.epoch_metrics.loss_values.append(loss_val)
        return True
    
    def check_grad_norm(self, grad_norm: float) -> bool:
        """
        Check if gradient norm is valid.
        
        Returns:
            True if valid, False if INF
        """
        if math.isinf(grad_norm):
            self.epoch_metrics.inf_grad_count += 1
            self.consecutive_overflows += 1
            self._log_instability(f"INF grad_norm at step {self._step_in_epoch}")
            return False
        
        self.epoch_metrics.grad_norms.append(grad_norm)
        self.consecutive_overflows = 0
        return True
    
    def check_scaler_update(self, scale_before: float, scale_after: float) -> bool:
        """
        Check if AMP scaler detected overflow.
        
        Returns:
            True if no overflow, False if overflow occurred
        """
        if scale_after < scale_before:
            self.epoch_metrics.scaler_overflow_count += 1
            self.consecutive_overflows += 1
            
            if self._step_in_epoch % self.config.log_every_n_steps == 0:
                self._log_instability(
                    f"AMP scaler overflow at step {self._step_in_epoch}: "
                    f"scale {scale_before:.0f} → {scale_after:.0f}"
                )
            return False
        
        self.consecutive_overflows = 0
        return True
    
    def should_reduce_lr(self) -> bool:
        """
        Check if learning rate should be reduced.
        
        Returns:
            True if LR should be reduced
        """
        if not self.config.enable_adaptive_lr:
            return False
        
        if self.consecutive_overflows >= self.config.lr_reduction_threshold:
            self.epoch_metrics.lr_reductions += 1
            self.consecutive_overflows = 0
            return True
        
        return False
    
    def get_lr_factor(self) -> float:
        """Get cumulative LR reduction factor."""
        return self.current_lr_factor
    
    def reduce_lr(self, optimizer, factor: float = None):
        """
        Reduce learning rate for all parameter groups.
        
        Args:
            optimizer: PyTorch optimizer
            factor: Reduction factor (default: config value)
        """
        factor = factor or self.config.lr_reduction_factor
        
        for pg in optimizer.param_groups:
            old_lr = pg['lr']
            new_lr = max(old_lr * factor, self.config.min_lr)
            pg['lr'] = new_lr
        
        self.current_lr_factor *= factor
        logger.warning(f"Reduced LR by {factor}x due to instability")
    
    def should_tighten_grad_clip(self) -> bool:
        """Check if gradient clipping should be tightened."""
        if not self.config.enable_adaptive_grad_clip:
            return False
        
        # Tighten if many INF gradients
        if self.epoch_metrics.inf_grad_ratio() > 0.1:
            self.epoch_metrics.grad_clip_tightenings += 1
            return True
        
        return False
    
    def get_grad_clip_factor(self) -> float:
        """Get cumulative grad clip tightening factor."""
        return self.current_grad_clip_factor
    
    def tighten_grad_clip(self, current_max_norm: float) -> float:
        """
        Get tightened gradient clipping value.
        
        Args:
            current_max_norm: Current max_norm value
            
        Returns:
            New max_norm value
        """
        factor = self.config.grad_clip_tightening_factor
        new_norm = max(current_max_norm * factor, self.config.min_grad_clip)
        self.current_grad_clip_factor *= factor
        logger.warning(f"Tightened grad clip: {current_max_norm} → {new_norm}")
        return new_norm
    
    def should_fail_fast(self) -> bool:
        """
        Check if training should be aborted due to persistent instability.
        
        Returns:
            True if training should stop
        """
        if self.epoch_metrics.nan_ratio() > self.config.max_nan_ratio_per_epoch:
            logger.error(
                f"FAIL FAST: NaN ratio {self.epoch_metrics.nan_ratio():.1%} "
                f"exceeds threshold {self.config.max_nan_ratio_per_epoch:.1%}"
            )
            return True
        return False
    
    def log_epoch_summary(self, epoch: int):
        """Log summary of stability metrics for the epoch."""
        m = self.epoch_metrics
        
        summary = (
            f"\n{'='*60}\n"
            f"STABILITY SUMMARY - Epoch {epoch}\n"
            f"{'='*60}\n"
            f"  Total steps: {m.total_steps}\n"
            f"  NaN losses: {m.nan_loss_count} ({m.nan_ratio():.1%})\n"
            f"  INF losses: {m.inf_loss_count}\n"
            f"  INF gradients: {m.inf_grad_count} ({m.inf_grad_ratio():.1%})\n"
            f"  Scaler overflows: {m.scaler_overflow_count} ({m.overflow_ratio():.1%})\n"
            f"  Avg grad norm: {m.avg_grad_norm():.2f}\n"
            f"  Avg loss: {m.avg_loss():.4f}\n"
            f"  LR reductions: {m.lr_reductions}\n"
            f"  Grad clip tightenings: {m.grad_clip_tightenings}\n"
            f"{'='*60}"
        )
        
        # Use appropriate log level based on severity
        if m.nan_ratio() > 0.05 or m.inf_grad_ratio() > 0.2:
            logger.warning(summary)
        else:
            logger.info(summary)
        
        # Print to stdout as well
        print(summary)
        
        # Save to file if configured
        if self.config.save_metrics_path:
            self._save_metrics()
    
    def _log_instability(self, msg: str):
        """Log instability event."""
        if self._step_in_epoch % self.config.log_every_n_steps == 0:
            logger.warning(f"[Stability] {msg}")
    
    def _metrics_to_dict(self, metrics: EpochMetrics) -> dict:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_steps": metrics.total_steps,
            "nan_loss_count": metrics.nan_loss_count,
            "inf_loss_count": metrics.inf_loss_count,
            "inf_grad_count": metrics.inf_grad_count,
            "scaler_overflow_count": metrics.scaler_overflow_count,
            "nan_ratio": metrics.nan_ratio(),
            "inf_grad_ratio": metrics.inf_grad_ratio(),
            "overflow_ratio": metrics.overflow_ratio(),
            "avg_grad_norm": metrics.avg_grad_norm(),
            "avg_loss": metrics.avg_loss(),
            "lr_reductions": metrics.lr_reductions,
            "grad_clip_tightenings": metrics.grad_clip_tightenings,
        }
    
    def _save_metrics(self):
        """Save all epoch metrics to file."""
        if self.config.save_metrics_path:
            self.config.save_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.save_metrics_path, 'w') as f:
                json.dump(self.all_epoch_metrics, f, indent=2)


# =============================================================================
# TRAINING INTEGRATION HELPERS
# =============================================================================

def create_stability_aware_training_loop(
    base_lr: float = 1e-4,
    base_max_norm: float = 1.0,
    use_bf16: bool = False,
) -> dict:
    """
    Create recommended settings for stability-aware training.
    
    Returns dict of settings to pass to training script.
    """
    
    settings = {
        # Base settings
        "optim.lr": base_lr,
        "optim.max_norm": base_max_norm,
        
        # Use bf16 if available (more stable than fp16)
        "autocast": True,
        "autocast_dtype": "bfloat16" if use_bf16 else "float16",
        
        # Stability recommendations
        "_stability_notes": [
            "Monitor grad_norm in logs - should stay finite",
            "If INF gradients > 20% of steps, reduce LR",
            "Consider bf16 over fp16 for better stability",
            "Gradient clipping (max_norm) is critical",
        ]
    }
    
    return settings


def patch_musicgen_train_for_stability():
    """
    Returns code snippet to patch MusicGen training for stability monitoring.
    
    This is meant to be added to musicgen_train.py.
    """
    
    patch_code = '''
# === STABILITY MONITORING PATCH ===
# Add to musicgen_train.py after imports

from Training.training_checks.stability_monitor import StabilityMonitor, StabilityConfig

# Initialize monitor
stability_config = StabilityConfig(
    max_nan_ratio_per_epoch=0.1,
    max_inf_grad_ratio_per_epoch=0.3,
    enable_adaptive_lr=True,
    lr_reduction_factor=0.5,
    save_metrics_path=EXPERIMENTS_DIR / "stability_metrics.json",
)
stability_monitor = StabilityMonitor(stability_config)

# In training loop, after loss computation:
# if not stability_monitor.check_loss(loss):
#     continue  # Skip this step
#
# After gradient computation:
# stability_monitor.check_grad_norm(grad_norm)
#
# After scaler.update():
# stability_monitor.check_scaler_update(old_scale, new_scale)
#
# At epoch end:
# stability_monitor.log_epoch_summary(epoch)
# stability_monitor.reset_epoch()
#
# Check for fail-fast:
# if stability_monitor.should_fail_fast():
#     raise RuntimeError("Training too unstable")
'''
    
    return patch_code


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demonstrate stability monitor usage."""
    
    print("=" * 60)
    print("STABILITY MONITOR - Usage Example")
    print("=" * 60)
    
    # Create monitor
    monitor = StabilityMonitor(
        max_nan_ratio_per_epoch=0.1,
        enable_adaptive_lr=True,
    )
    
    # Simulate some training steps
    print("\nSimulating training with some instability events...")
    
    import random
    random.seed(42)
    
    for step in range(100):
        # Simulate loss
        if random.random() < 0.05:  # 5% NaN
            loss = float('nan')
        elif random.random() < 0.02:  # 2% INF
            loss = float('inf')
        else:
            loss = 8.0 + random.gauss(0, 0.1)
        
        loss_tensor = torch.tensor(loss)
        valid = monitor.check_loss(loss_tensor)
        
        if valid:
            # Simulate grad norm
            if random.random() < 0.1:  # 10% INF grad
                grad_norm = float('inf')
            else:
                grad_norm = 5.0 + random.gauss(0, 2)
            
            monitor.check_grad_norm(grad_norm)
            
            # Simulate scaler
            if random.random() < 0.08:  # 8% overflow
                monitor.check_scaler_update(65536, 32768)
            else:
                monitor.check_scaler_update(65536, 65536)
        
        # Check adaptive actions
        if monitor.should_reduce_lr():
            print(f"  Step {step}: Would reduce LR")
    
    # Log summary
    monitor.log_epoch_summary(epoch=1)
    
    # Print integration code
    print("\n" + "=" * 60)
    print("INTEGRATION CODE")
    print("=" * 60)
    print(patch_musicgen_train_for_stability())


if __name__ == "__main__":
    main()
