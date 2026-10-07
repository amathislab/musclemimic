"""Unit tests for preemption-robust auto-resume functionality."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from musclemimic.runner.checkpointing import (
    config_hash,
    find_latest_checkpoint,
    resolve_checkpoint_dir,
    validate_checkpoint_compatibility,
    write_manifest,
)


class TestConfigHash:
    """Tests for config_hash function."""

    def test_same_config_produces_same_hash(self):
        """Same config should produce same hash."""
        cfg1 = OmegaConf.create({"lr": 0.001, "batch_size": 32})
        cfg2 = OmegaConf.create({"lr": 0.001, "batch_size": 32})
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_different_config_produces_different_hash(self):
        """Different config should produce different hash."""
        cfg1 = OmegaConf.create({"lr": 0.001, "batch_size": 32})
        cfg2 = OmegaConf.create({"lr": 0.002, "batch_size": 32})
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_excludes_volatile_fields(self):
        """Volatile fields like resume_from shouldn't affect hash."""
        cfg1 = OmegaConf.create({"lr": 0.001, "resume_from": None})
        cfg2 = OmegaConf.create({"lr": 0.001, "resume_from": "/path/to/ckpt"})
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_excludes_checkpoint_dir(self):
        """checkpoint_dir shouldn't affect hash."""
        cfg1 = OmegaConf.create({"lr": 0.001, "checkpoint_dir": "/path/a"})
        cfg2 = OmegaConf.create({"lr": 0.001, "checkpoint_dir": "/path/b"})
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_excludes_auto_resume_fields(self):
        """auto_resume, run_id, checkpoint_root shouldn't affect hash."""
        cfg1 = OmegaConf.create({
            "lr": 0.001,
            "auto_resume": True,
            "run_id": None,
            "checkpoint_root": None,
        })
        cfg2 = OmegaConf.create({
            "lr": 0.001,
            "auto_resume": False,
            "run_id": "my-run",
            "checkpoint_root": "/stable/root",
        })
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_excludes_validation_config(self):
        """Validation-only changes should not block training checkpoint resume."""
        cfg1 = OmegaConf.create({
            "lr": 0.001,
            "validation": {"active": False, "num_steps": 500},
        })
        cfg2 = OmegaConf.create({
            "lr": 0.001,
            "validation": {"active": True, "num_steps": 1000},
        })
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_hash_is_12_chars(self):
        """Hash should be 12 characters."""
        cfg = OmegaConf.create({"lr": 0.001})
        h = config_hash(cfg)
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_order_independence(self):
        """Key order shouldn't affect hash."""
        cfg1 = OmegaConf.create({"a": 1, "b": 2, "c": 3})
        cfg2 = OmegaConf.create({"c": 3, "b": 2, "a": 1})
        assert config_hash(cfg1) == config_hash(cfg2)


def _create_complete_checkpoint(checkpoint_dir: Path) -> None:
    """Create a checkpoint with metadata to mark it as complete."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Create Orbax metadata marker file
    (checkpoint_dir / "_CHECKPOINT_METADATA").touch()
    metadata_dir = checkpoint_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    (metadata_dir / "metadata").touch()


class TestFindLatestCheckpoint:
    """Tests for find_latest_checkpoint function."""

    def test_returns_none_for_nonexistent_dir(self):
        """Should return None if directory doesn't exist."""
        result = find_latest_checkpoint("/nonexistent/path")
        assert result is None

    def test_returns_none_for_empty_dir(self):
        """Should return None if directory has no checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_checkpoint(tmpdir)
            assert result is None

    def test_finds_single_checkpoint(self):
        """Should find single complete checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint_100"
            _create_complete_checkpoint(ckpt_path)
            result = find_latest_checkpoint(tmpdir)
            assert result == str(ckpt_path)

    def test_finds_latest_checkpoint(self):
        """Should find latest complete checkpoint by step number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [100, 500, 300, 200]:
                _create_complete_checkpoint(Path(tmpdir) / f"checkpoint_{step}")
            result = find_latest_checkpoint(tmpdir)
            assert result == str(Path(tmpdir) / "checkpoint_500")

    def test_ignores_incomplete_checkpoints(self):
        """Should ignore checkpoints without metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create incomplete checkpoint (no metadata)
            incomplete = Path(tmpdir) / "checkpoint_500"
            incomplete.mkdir()
            # Create complete checkpoint with lower step
            _create_complete_checkpoint(Path(tmpdir) / "checkpoint_100")
            result = find_latest_checkpoint(tmpdir)
            # Should pick the complete one, not the higher incomplete one
            assert result == str(Path(tmpdir) / "checkpoint_100")

    def test_ignores_non_checkpoint_dirs(self):
        """Should ignore directories that don't match checkpoint_* pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_complete_checkpoint(Path(tmpdir) / "checkpoint_100")
            (Path(tmpdir) / "other_dir").mkdir()
            (Path(tmpdir) / "checkpoint_invalid").mkdir()  # No number
            result = find_latest_checkpoint(tmpdir)
            assert result == str(Path(tmpdir) / "checkpoint_100")


class TestWriteManifest:
    """Tests for write_manifest function."""

    def test_writes_manifest(self):
        """Should write manifest.json with correct fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"lr": 0.001, "batch_size": 32})
            write_manifest(tmpdir, cfg, "abc123def456")

            manifest_path = Path(tmpdir) / "manifest.json"
            assert manifest_path.exists()

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["config_hash"] == "abc123def456"
            assert "created_at" in manifest
            assert "experiment_config" in manifest
            assert manifest["experiment_config"]["lr"] == 0.001

    def test_idempotent(self):
        """Should not overwrite existing manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg1 = OmegaConf.create({"lr": 0.001})
            write_manifest(tmpdir, cfg1, "hash1")

            # Try to write again with different data
            cfg2 = OmegaConf.create({"lr": 0.999})
            write_manifest(tmpdir, cfg2, "hash2")

            # Should still have original data
            manifest_path = Path(tmpdir) / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["config_hash"] == "hash1"
            assert manifest["experiment_config"]["lr"] == 0.001


class TestValidateCheckpointCompatibility:
    """Tests for validate_checkpoint_compatibility function."""

    def test_returns_true_for_no_manifest(self):
        """Should return True if no manifest exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_checkpoint_compatibility(tmpdir, "any_hash")
            assert result is True

    def test_returns_true_for_matching_hash(self):
        """Should return True if hash matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"lr": 0.001})
            write_manifest(tmpdir, cfg, "matching_hash")
            result = validate_checkpoint_compatibility(tmpdir, "matching_hash")
            assert result is True

    def test_returns_false_for_mismatched_hash(self, capsys):
        """Should return False and print warning if hash mismatches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = OmegaConf.create({"lr": 0.001})
            write_manifest(tmpdir, cfg, "original_hash")
            result = validate_checkpoint_compatibility(tmpdir, "different_hash")
            assert result is False

            captured = capsys.readouterr()
            assert "WARNING: Config hash mismatch" in captured.out
            assert "original_hash" in captured.out
            assert "different_hash" in captured.out


class TestResolveCheckpointDir:
    """Tests for resolve_checkpoint_dir function."""

    def test_auto_resume_uses_launch_dir(self):
        """With auto_resume=true, relative paths resolve to launch_dir."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=True,
        )
        assert result == "/project/checkpoints/abc123"

    def test_auto_resume_false_uses_result_dir(self):
        """With auto_resume=false, relative paths resolve to result_dir."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=False,
        )
        # Returns base without experiment_id; caller adds unique suffix
        assert result == "/outputs/2026-01-07/run1/checkpoints"

    def test_checkpoint_root_overrides(self):
        """checkpoint_root takes precedence over configured_ckpt_dir."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=True,
            checkpoint_root="custom_ckpts",
        )
        assert result == "/project/custom_ckpts/abc123"

    def test_absolute_paths_unchanged(self):
        """Absolute paths are not modified."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="/absolute/checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=True,
        )
        assert result == "/absolute/checkpoints/abc123"

    def test_absolute_checkpoint_root(self):
        """Absolute checkpoint_root is not modified."""
        result = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/2026-01-07/run1",
            experiment_id="abc123",
            auto_resume=True,
            checkpoint_root="/stable/root",
        )
        assert result == "/stable/root/abc123"

    def test_different_experiment_ids_different_dirs(self):
        """Different experiment_ids produce different directories."""
        result1 = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/run1",
            experiment_id="exp_a",
            auto_resume=True,
        )
        result2 = resolve_checkpoint_dir(
            configured_ckpt_dir="checkpoints",
            launch_dir="/project",
            result_dir="/outputs/run1",
            experiment_id="exp_b",
            auto_resume=True,
        )
        assert result1 != result2
        assert result1 == "/project/checkpoints/exp_a"
        assert result2 == "/project/checkpoints/exp_b"
