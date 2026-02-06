"""
Tests for Preflight Checks Module

Run with: pytest -q Training/training_checks/tests/
"""

import pytest
import json
import tempfile
from pathlib import Path

# Import the modules we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training_checks.preflight import (
    PreflightConfig,
    PreflightResults,
    check_manifest_exists,
    check_manifest_schema,
    check_segment_duration,
    check_training_config,
)
from training_checks.early_stopping import EarlyStopping, EarlyStoppingConfig


class TestManifestValidation:
    """Test manifest validation checks (Task 1A)."""
    
    def test_missing_manifest_fails(self, tmp_path):
        """Missing manifest should trigger fatal error."""
        config = PreflightConfig(
            train_jsonl=tmp_path / "nonexistent" / "train.jsonl",
            valid_jsonl=tmp_path / "nonexistent" / "valid.jsonl",
        )
        results = PreflightResults()
        
        ok = check_manifest_exists(config, results)
        
        assert not ok
        assert len(results.fatal_errors) > 0
        assert "manifest_train_exists" in results.checks
        assert results.checks["manifest_train_exists"]["status"] == "FATAL"
    
    def test_empty_manifest_fails(self, tmp_path):
        """Empty manifest should trigger fatal error."""
        # Create empty files
        train_jsonl = tmp_path / "train.jsonl"
        valid_jsonl = tmp_path / "valid.jsonl"
        train_jsonl.write_text("")
        valid_jsonl.write_text("")
        
        config = PreflightConfig(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
        )
        results = PreflightResults()
        
        ok = check_manifest_exists(config, results)
        
        assert not ok
        assert any("empty" in err.lower() for err in results.fatal_errors)
    
    def test_valid_manifest_passes(self, tmp_path):
        """Valid manifest should pass."""
        # Create valid JSONL files
        train_jsonl = tmp_path / "train.jsonl"
        valid_jsonl = tmp_path / "valid.jsonl"
        
        sample_entry = json.dumps({
            "path": str(tmp_path / "audio.wav"),
            "duration": 30.0,
            "sample_rate": 32000,
            "channels": 1,
        })
        
        train_jsonl.write_text(sample_entry + "\n")
        valid_jsonl.write_text(sample_entry + "\n")
        
        config = PreflightConfig(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
        )
        results = PreflightResults()
        
        ok = check_manifest_exists(config, results)
        
        assert ok
        assert len(results.fatal_errors) == 0
    
    def test_schema_missing_fields_fails(self, tmp_path):
        """Manifest entries missing required fields should fail."""
        train_jsonl = tmp_path / "train.jsonl"
        valid_jsonl = tmp_path / "valid.jsonl"
        
        # Missing "duration" field
        bad_entry = json.dumps({"path": "/some/path.wav", "sample_rate": 32000})
        train_jsonl.write_text(bad_entry + "\n")
        valid_jsonl.write_text(bad_entry + "\n")
        
        config = PreflightConfig(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            manifest_sample_size=10,
        )
        results = PreflightResults()
        
        # First check existence
        check_manifest_exists(config, results)
        # Then check schema
        ok = check_manifest_schema(config, results)
        
        assert not ok
        assert any("schema" in err.lower() for err in results.fatal_errors)
    
    def test_sample_rate_mismatch_fails(self, tmp_path):
        """Sample rate mismatch should fail."""
        train_jsonl = tmp_path / "train.jsonl"
        valid_jsonl = tmp_path / "valid.jsonl"
        
        # Create audio file
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake audio content")
        
        # Wrong sample rate
        entry = json.dumps({
            "path": str(audio_path),
            "duration": 30.0,
            "sample_rate": 44100,  # Expected 32000
            "channels": 1,
        })
        train_jsonl.write_text(entry + "\n")
        valid_jsonl.write_text(entry + "\n")
        
        config = PreflightConfig(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            expected_sample_rate=32000,
            manifest_sample_size=10,
        )
        results = PreflightResults()
        
        check_manifest_exists(config, results)
        ok = check_manifest_schema(config, results)
        
        assert not ok
        assert any("sample_rate" in err.lower() or "mismatch" in err.lower() for err in results.fatal_errors)


class TestSegmentDuration:
    """Test segment duration feasibility checks (Task 1B)."""
    
    def test_short_clips_warning(self, tmp_path):
        """Many short clips should trigger warning."""
        train_jsonl = tmp_path / "train.jsonl"
        valid_jsonl = tmp_path / "valid.jsonl"
        
        # Create entries with short durations (15% < segment_duration=60)
        entries = []
        for i in range(100):
            dur = 50.0 if i < 15 else 120.0  # 15% short
            entries.append(json.dumps({
                "path": f"/audio/{i}.wav",
                "duration": dur,
                "sample_rate": 32000,
            }))
        
        train_jsonl.write_text("\n".join(entries))
        valid_jsonl.write_text("\n".join(entries[:10]))
        
        config = PreflightConfig(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            segment_duration=60.0,
            short_clip_warn_pct=10.0,
            short_clip_fatal_pct=30.0,
            manifest_sample_size=100,
        )
        results = PreflightResults()
        
        ok = check_segment_duration(config, results)
        
        assert ok  # Should pass but with warning
        assert len(results.warnings) > 0
    
    def test_too_many_short_clips_fails(self, tmp_path):
        """Too many short clips should fail unless allowed."""
        train_jsonl = tmp_path / "train.jsonl"
        valid_jsonl = tmp_path / "valid.jsonl"
        
        # Create entries with 50% short durations
        entries = []
        for i in range(100):
            dur = 30.0 if i < 50 else 120.0  # 50% short
            entries.append(json.dumps({
                "path": f"/audio/{i}.wav",
                "duration": dur,
                "sample_rate": 32000,
            }))
        
        train_jsonl.write_text("\n".join(entries))
        valid_jsonl.write_text("\n".join(entries[:10]))
        
        config = PreflightConfig(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            segment_duration=60.0,
            short_clip_fatal_pct=30.0,
            allow_short_clips=False,
            manifest_sample_size=100,
        )
        results = PreflightResults()
        
        ok = check_segment_duration(config, results)
        
        assert not ok
        assert len(results.fatal_errors) > 0
    
    def test_allow_short_clips_bypasses_fatal(self, tmp_path):
        """allow_short_clips=True should prevent fatal error."""
        train_jsonl = tmp_path / "train.jsonl"
        valid_jsonl = tmp_path / "valid.jsonl"
        
        # All short clips
        entries = [json.dumps({"path": f"/audio/{i}.wav", "duration": 10.0, "sample_rate": 32000}) for i in range(20)]
        train_jsonl.write_text("\n".join(entries))
        valid_jsonl.write_text("\n".join(entries[:5]))
        
        config = PreflightConfig(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            segment_duration=60.0,
            allow_short_clips=True,
            manifest_sample_size=20,
        )
        results = PreflightResults()
        
        ok = check_segment_duration(config, results)
        
        assert ok  # Should pass with allow_short_clips=True


class TestTrainingConfig:
    """Test training configuration checks (Task 3A)."""
    
    def test_epoch_math_full_dataset(self, tmp_path):
        """Full dataset mode should compute correct updates per epoch."""
        train_jsonl = tmp_path / "train.jsonl"
        valid_jsonl = tmp_path / "valid.jsonl"
        
        # 1000 samples
        entries = [json.dumps({"path": f"/audio/{i}.wav", "duration": 30.0, "sample_rate": 32000}) for i in range(1000)]
        train_jsonl.write_text("\n".join(entries))
        valid_jsonl.write_text("\n".join(entries[:10]))
        
        config = PreflightConfig(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            batch_size=8,
            world_size=4,
            target_hours=None,  # Full dataset mode
        )
        results = PreflightResults()
        
        ok = check_training_config(config, results)
        
        assert ok
        assert results.resolved_config["training"]["epoch_mode"] == "full_dataset"
        assert results.resolved_config["training"]["global_batch_size"] == 32
        assert results.resolved_config["training"]["updates_per_epoch"] == 1000 // 32
    
    def test_epoch_math_target_hours(self, tmp_path):
        """Target hours mode should compute correct updates per epoch."""
        train_jsonl = tmp_path / "train.jsonl"
        valid_jsonl = tmp_path / "valid.jsonl"
        
        # 10000 samples (enough for target hours calculation)
        entries = [json.dumps({"path": f"/audio/{i}.wav", "duration": 30.0, "sample_rate": 32000}) for i in range(10000)]
        train_jsonl.write_text("\n".join(entries))
        valid_jsonl.write_text("\n".join(entries[:10]))
        
        config = PreflightConfig(
            train_jsonl=train_jsonl,
            valid_jsonl=valid_jsonl,
            segment_duration=30.0,
            batch_size=8,
            world_size=4,
            target_hours=10,  # 10 hours = 1200 samples at 30s each
        )
        results = PreflightResults()
        
        ok = check_training_config(config, results)
        
        assert ok
        assert results.resolved_config["training"]["epoch_mode"] == "target_hours"
        assert results.resolved_config["training"]["target_samples"] == (10 * 3600) // 30


class TestEarlyStopping:
    """Test early stopping functionality (Task 5B)."""
    
    def test_improvement_resets_patience(self):
        """Improvement should reset patience counter."""
        config = EarlyStoppingConfig(
            enabled=True,
            metric="valid_loss",
            patience=5,
            min_delta=0.001,
            mode="min",
        )
        es = EarlyStopping(config)
        
        # Epoch 1: baseline
        should_stop, _ = es.check(1, {"valid_loss": 1.0})
        assert not should_stop
        
        # Epoch 2: improvement
        should_stop, _ = es.check(2, {"valid_loss": 0.8})
        assert not should_stop
        assert es.state.epochs_without_improvement == 0
        
        # Epoch 3: no improvement
        should_stop, _ = es.check(3, {"valid_loss": 0.85})
        assert not should_stop
        assert es.state.epochs_without_improvement == 1
        
        # Epoch 4: improvement again
        should_stop, _ = es.check(4, {"valid_loss": 0.6})
        assert not should_stop
        assert es.state.epochs_without_improvement == 0
    
    def test_patience_exceeded_stops(self):
        """Exceeding patience should trigger stop."""
        config = EarlyStoppingConfig(
            enabled=True,
            metric="valid_loss",
            patience=3,
            min_delta=0.001,
            mode="min",
        )
        es = EarlyStopping(config)
        
        # Baseline
        es.check(1, {"valid_loss": 1.0})
        
        # No improvement for patience epochs
        for epoch in range(2, 6):
            should_stop, reason = es.check(epoch, {"valid_loss": 1.0})
            if epoch < 5:
                assert not should_stop
            else:
                assert should_stop
                assert "patience" in reason.lower() or "improvement" in reason.lower()
    
    def test_ratio_explosion_stops(self):
        """Ratio explosion should trigger immediate stop."""
        config = EarlyStoppingConfig(
            enabled=True,
            ratio_explosion_threshold=100.0,
        )
        es = EarlyStopping(config)
        
        # Normal ratio
        should_stop, _ = es.check(1, {"ratio1": 50.0})
        assert not should_stop
        
        # Exploding ratio
        should_stop, reason = es.check(2, {"ratio1": 150.0})
        assert should_stop
        assert "ratio" in reason.lower()
    
    def test_max_epochs_stops(self):
        """max_epochs should trigger stop."""
        config = EarlyStoppingConfig(
            enabled=True,
            max_epochs=5,
        )
        es = EarlyStopping(config)
        
        for epoch in range(1, 7):
            should_stop, reason = es.check(epoch, {"loss": 1.0})
            if epoch < 5:
                assert not should_stop
            else:
                assert should_stop
                assert "max epochs" in reason.lower()


class TestPreflightResults:
    """Test PreflightResults aggregation."""
    
    def test_fatal_error_sets_passed_false(self):
        """Adding a fatal error should set passed=False."""
        results = PreflightResults()
        assert results.passed
        
        results.add_fatal("test_check", "Something went wrong")
        
        assert not results.passed
        assert len(results.fatal_errors) == 1
    
    def test_warning_keeps_passed_true(self):
        """Warnings should not affect passed status."""
        results = PreflightResults()
        
        results.add_warning("test_check", "Minor issue")
        
        assert results.passed
        assert len(results.warnings) == 1
    
    def test_ok_adds_info(self):
        """OK checks should add to info list."""
        results = PreflightResults()
        
        results.add_ok("test_check", "All good")
        
        assert results.passed
        assert len(results.info) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
