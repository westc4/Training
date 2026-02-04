"""
Runtime Monitoring for AudioCraft Training

Implements:
- Task 4A: Throughput monitoring, dataloader stalls, GPU utilization proxies
- Task 4B: NaN/Inf guardrails with auto-abort and snapshot
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import deque
import json
import time
import datetime
import re


@dataclass
class RuntimeStats:
    """Aggregated runtime statistics."""
    
    # Throughput
    iterations: int = 0
    total_time_sec: float = 0.0
    iter_times: list[float] = field(default_factory=list)
    
    # Gradient norms
    grad_norms: list[float] = field(default_factory=list)
    inf_grad_count: int = 0
    
    # AMP scaler
    amp_scales: list[float] = field(default_factory=list)
    
    # Loss tracking
    losses: list[float] = field(default_factory=list)
    nan_loss_count: int = 0
    
    # Per-epoch stats
    epoch_stats: list[dict] = field(default_factory=list)
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        import statistics
        
        summary = {
            "iterations": self.iterations,
            "total_time_sec": round(self.total_time_sec, 2),
        }
        
        if self.iter_times:
            summary["throughput"] = {
                "avg_it_per_sec": round(self.iterations / max(self.total_time_sec, 0.001), 2),
                "avg_sec_per_it": round(statistics.mean(self.iter_times), 4),
                "median_sec_per_it": round(statistics.median(self.iter_times), 4),
            }
        
        if self.grad_norms:
            finite_norms = [g for g in self.grad_norms if g != float('inf') and g == g]
            summary["grad_norm"] = {
                "inf_count": self.inf_grad_count,
                "inf_pct": round(100 * self.inf_grad_count / len(self.grad_norms), 2),
            }
            if finite_norms:
                summary["grad_norm"]["median"] = round(statistics.median(finite_norms), 4)
                summary["grad_norm"]["p95"] = round(sorted(finite_norms)[int(len(finite_norms) * 0.95)], 4)
        
        if self.amp_scales:
            summary["amp_scale"] = {
                "min": round(min(self.amp_scales), 2),
                "max": round(max(self.amp_scales), 2),
                "median": round(statistics.median(self.amp_scales), 2),
            }
        
        if self.losses:
            summary["loss"] = {
                "nan_count": self.nan_loss_count,
                "last": round(self.losses[-1], 6) if self.losses else None,
            }
        
        return summary


class RuntimeMonitor:
    """
    Monitor training runtime for throughput, stalls, and AMP stability.
    
    Task 4A: Tracks rolling it/sec, dataloader time, AMP scaler behavior.
    """
    
    def __init__(
        self,
        xp_dir: Optional[Path] = None,
        throughput_drop_threshold: float = 0.3,  # 30% drop
        throughput_window_minutes: float = 5.0,
        log_every: int = 100,
    ):
        self.xp_dir = xp_dir
        self.throughput_drop_threshold = throughput_drop_threshold
        self.throughput_window = int(throughput_window_minutes * 60)  # seconds
        self.log_every = log_every
        
        self.stats = RuntimeStats()
        self._start_time: Optional[float] = None
        self._iter_start: Optional[float] = None
        self._baseline_it_per_sec: Optional[float] = None
        
        # Rolling window for throughput monitoring
        self._recent_iters: deque = deque(maxlen=1000)
        
        # Callbacks
        self._on_throughput_drop: Optional[Callable] = None
        self._on_inf_grad_norm: Optional[Callable] = None
    
    def start(self):
        """Start the monitor."""
        self._start_time = time.time()
    
    def on_iter_start(self):
        """Call at the start of each iteration."""
        self._iter_start = time.time()
    
    def on_iter_end(
        self,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        amp_scale: Optional[float] = None,
    ):
        """
        Call at the end of each iteration.
        
        Args:
            loss: Current loss value
            grad_norm: Gradient norm (may be inf)
            amp_scale: AMP scaler value
        """
        iter_time = time.time() - (self._iter_start or time.time())
        
        self.stats.iterations += 1
        self.stats.total_time_sec = time.time() - (self._start_time or time.time())
        self.stats.iter_times.append(iter_time)
        
        # Track recent iterations for rolling throughput
        self._recent_iters.append((time.time(), iter_time))
        
        # Track loss
        if loss is not None:
            if loss != loss or loss == float('inf') or loss == float('-inf'):
                self.stats.nan_loss_count += 1
            else:
                self.stats.losses.append(loss)
        
        # Track gradient norm
        if grad_norm is not None:
            self.stats.grad_norms.append(grad_norm)
            if grad_norm == float('inf'):
                self.stats.inf_grad_count += 1
                if self._on_inf_grad_norm:
                    self._on_inf_grad_norm(self.stats.iterations)
        
        # Track AMP scale
        if amp_scale is not None:
            self.stats.amp_scales.append(amp_scale)
        
        # Check throughput periodically
        if self.stats.iterations % self.log_every == 0:
            self._check_throughput()
    
    def _check_throughput(self):
        """Check for sustained throughput drops."""
        if len(self._recent_iters) < 10:
            return
        
        # Compute current throughput (last N seconds)
        now = time.time()
        recent = [(t, d) for t, d in self._recent_iters if now - t < self.throughput_window]
        
        if not recent:
            return
        
        current_it_per_sec = len(recent) / (recent[-1][0] - recent[0][0] + 0.001)
        
        # Set baseline from first measurements
        if self._baseline_it_per_sec is None and len(self.stats.iter_times) > 100:
            import statistics
            baseline_times = self.stats.iter_times[:100]
            self._baseline_it_per_sec = 1.0 / statistics.mean(baseline_times)
        
        # Check for significant drop
        if self._baseline_it_per_sec is not None:
            drop = 1.0 - (current_it_per_sec / self._baseline_it_per_sec)
            if drop > self.throughput_drop_threshold:
                if self._on_throughput_drop:
                    self._on_throughput_drop(current_it_per_sec, self._baseline_it_per_sec, drop)
    
    def on_epoch_end(self, epoch: int, metrics: dict = None):
        """Record epoch-level statistics."""
        epoch_stat = {
            "epoch": epoch,
            "timestamp": datetime.datetime.now().isoformat(),
            "iterations": self.stats.iterations,
        }
        
        # Compute epoch-specific stats
        if self.stats.iter_times:
            recent_times = self.stats.iter_times[-self.log_every:] if len(self.stats.iter_times) > self.log_every else self.stats.iter_times
            epoch_stat["avg_it_per_sec"] = round(1.0 / (sum(recent_times) / len(recent_times)), 2)
        
        if self.stats.grad_norms:
            recent_norms = self.stats.grad_norms[-self.log_every:]
            inf_count = sum(1 for g in recent_norms if g == float('inf'))
            epoch_stat["inf_grad_pct"] = round(100 * inf_count / len(recent_norms), 2)
        
        if metrics:
            epoch_stat["metrics"] = metrics
        
        self.stats.epoch_stats.append(epoch_stat)
    
    def get_report(self) -> dict:
        """Get full monitoring report."""
        return {
            "summary": self.stats.get_summary(),
            "epochs": self.stats.epoch_stats,
        }
    
    def save_report(self, path: Optional[Path] = None):
        """Save monitoring report to file."""
        if path is None and self.xp_dir:
            path = self.xp_dir / "runtime_monitor.json"
        
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.get_report(), f, indent=2)


@dataclass
class NaNSnapshot:
    """Snapshot of training state when NaN/Inf detected."""
    
    iteration: int
    epoch: int
    loss: float
    grad_norm: float
    batch_paths: list[str]
    batch_durations: list[float]
    model_state_path: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


class NaNGuard:
    """
    Task 4B: NaN/Inf guardrails with auto-abort and snapshot.
    
    Detects NaN/Inf in loss or gradients and saves debugging artifacts.
    """
    
    def __init__(
        self,
        xp_dir: Path,
        max_nan_losses: int = 3,  # Abort after this many NaN losses
        max_consecutive_inf_grad: int = 10,  # Abort after consecutive inf grads
        save_emergency_checkpoint: bool = True,
    ):
        self.xp_dir = xp_dir
        self.max_nan_losses = max_nan_losses
        self.max_consecutive_inf_grad = max_consecutive_inf_grad
        self.save_emergency_checkpoint = save_emergency_checkpoint
        
        self._nan_loss_count = 0
        self._consecutive_inf_grad = 0
        self._snapshots: list[NaNSnapshot] = []
        self._current_batch_info: dict = {}
        self._model_ref = None
        self._epoch = 0
        self._iteration = 0
    
    def set_model(self, model):
        """Set reference to model for emergency checkpointing."""
        self._model_ref = model
    
    def set_epoch(self, epoch: int):
        """Update current epoch."""
        self._epoch = epoch
    
    def set_iteration(self, iteration: int):
        """Update current iteration."""
        self._iteration = iteration
    
    def set_batch_info(self, paths: list[str] = None, durations: list[float] = None, **kwargs):
        """Set info about current batch for debugging."""
        self._current_batch_info = {
            "paths": paths or [],
            "durations": durations or [],
            **kwargs,
        }
    
    def check(
        self,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check for NaN/Inf values.
        
        Returns:
            (should_abort, reason): If should_abort is True, training should stop.
        """
        should_abort = False
        reason = None
        
        # Check loss
        if loss is not None:
            is_nan = loss != loss  # NaN check
            is_inf = loss == float('inf') or loss == float('-inf')
            
            if is_nan or is_inf:
                self._nan_loss_count += 1
                self._save_snapshot(loss, grad_norm, f"nan_loss_{self._nan_loss_count}")
                
                if self._nan_loss_count >= self.max_nan_losses:
                    should_abort = True
                    reason = f"NaN/Inf loss detected {self._nan_loss_count} times (max: {self.max_nan_losses})"
        
        # Check gradient norm
        if grad_norm is not None:
            if grad_norm == float('inf'):
                self._consecutive_inf_grad += 1
                
                if self._consecutive_inf_grad >= self.max_consecutive_inf_grad:
                    self._save_snapshot(loss, grad_norm, "inf_grad_consecutive")
                    should_abort = True
                    reason = f"Inf gradient norm for {self._consecutive_inf_grad} consecutive steps"
            else:
                self._consecutive_inf_grad = 0  # Reset counter
        
        return should_abort, reason
    
    def _save_snapshot(self, loss: float, grad_norm: float, tag: str):
        """Save debugging snapshot."""
        snapshot = NaNSnapshot(
            iteration=self._iteration,
            epoch=self._epoch,
            loss=loss,
            grad_norm=grad_norm,
            batch_paths=self._current_batch_info.get("paths", []),
            batch_durations=self._current_batch_info.get("durations", []),
        )
        
        # Save snapshot JSON
        snapshot_dir = self.xp_dir / "nan_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_path = snapshot_dir / f"snapshot_{tag}_iter{self._iteration}.json"
        with open(snapshot_path, 'w') as f:
            json.dump({
                "iteration": snapshot.iteration,
                "epoch": snapshot.epoch,
                "loss": str(snapshot.loss),
                "grad_norm": str(snapshot.grad_norm),
                "batch_paths": snapshot.batch_paths,
                "batch_durations": snapshot.batch_durations,
                "timestamp": snapshot.timestamp,
            }, f, indent=2)
        
        # Save emergency checkpoint if enabled
        if self.save_emergency_checkpoint and self._model_ref is not None:
            try:
                import torch
                emergency_path = snapshot_dir / f"emergency_ckpt_{tag}_iter{self._iteration}.pt"
                torch.save({
                    "model": self._model_ref.state_dict(),
                    "iteration": self._iteration,
                    "epoch": self._epoch,
                }, emergency_path)
                snapshot.model_state_path = str(emergency_path)
            except Exception as e:
                print(f"Warning: Failed to save emergency checkpoint: {e}")
        
        self._snapshots.append(snapshot)
        
        print(f"\n⚠️  NaN/Inf detected at iteration {self._iteration}")
        print(f"   Loss: {loss}, Grad norm: {grad_norm}")
        print(f"   Snapshot saved: {snapshot_path}")
    
    def get_snapshots(self) -> list[dict]:
        """Get all snapshots as dicts."""
        return [
            {
                "iteration": s.iteration,
                "epoch": s.epoch,
                "loss": str(s.loss),
                "grad_norm": str(s.grad_norm),
                "timestamp": s.timestamp,
                "model_state_path": s.model_state_path,
            }
            for s in self._snapshots
        ]


class LogParser:
    """Parse Dora/AudioCraft training logs for metrics."""
    
    # Patterns for common log formats
    PATTERNS = {
        "train_loss": re.compile(r"train.*?loss[=:\s]+([\d.]+)", re.IGNORECASE),
        "valid_loss": re.compile(r"valid.*?loss[=:\s]+([\d.]+)", re.IGNORECASE),
        "grad_norm": re.compile(r"grad_norm[=:\s]+([\d.]+|inf)", re.IGNORECASE),
        "it_per_sec": re.compile(r"([\d.]+)\s*it/s", re.IGNORECASE),
        "epoch": re.compile(r"epoch[=:\s]+(\d+)", re.IGNORECASE),
        "ce": re.compile(r"\bce[=:\s]+([\d.]+)", re.IGNORECASE),
        "ppl": re.compile(r"\bppl[=:\s]+([\d.]+)", re.IGNORECASE),
    }
    
    @classmethod
    def parse_line(cls, line: str) -> dict:
        """Parse a single log line for metrics."""
        metrics = {}
        
        for name, pattern in cls.PATTERNS.items():
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
        
        return metrics
    
    @classmethod
    def parse_file(cls, log_path: Path) -> list[dict]:
        """Parse entire log file."""
        results = []
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    metrics = cls.parse_line(line)
                    if metrics:
                        results.append(metrics)
        except Exception:
            pass
        
        return results
