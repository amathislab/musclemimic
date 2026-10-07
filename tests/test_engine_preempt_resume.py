from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from musclemimic.runner import engine


class _DummyHooks:
    def build_video_recorder(self, result_dir, config):
        return None


def _make_config(
    *,
    auto_resume: bool = True,
    resume_from: str | None = None,
    exact_resume: bool = False,
):
    return OmegaConf.create(
        {
            "wandb": {"mode": "disabled", "project": "musclemimic"},
            "experiment": {
                "resume_from": resume_from,
                "auto_resume": auto_resume,
                "exact_resume": exact_resume,
                "run_id": None,
                "checkpoint_root": None,
                "checkpoint_dir": "checkpoints",
                "online_logging_interval": 1,
                "validation": {"active": False},
            },
        }
    )


def _patch_run_experiment_dependencies(monkeypatch, tmp_path: Path, captured: dict):
    runtime = SimpleNamespace(
        output_dir=str(tmp_path / "outputs" / "run1"),
        cwd=str(tmp_path / "launch"),
    )
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: SimpleNamespace(runtime=runtime)))

    monkeypatch.setattr(engine, "setup_jax_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "setup_wandb", lambda _config: (False, None))
    monkeypatch.setattr(engine, "instantiate_env", lambda _config: SimpleNamespace(th=None))
    monkeypatch.setattr(engine, "should_share_validation_trajectory", lambda _config: False)
    captured["events"] = []

    def fake_prepare_trajectory_data_shards(_env, _config, _n_devices, *, label="training"):
        captured["events"].append("trajectory_shards")
        return "device-shards"

    def fake_build_resume_runtime_state_target(*_args, **_kwargs):
        captured["events"].append("runtime_target")
        return {"step": 0}

    def fake_instantiate_validation_env(_config, share_trajectory=False):
        captured["events"].append("validation_env")
        return SimpleNamespace(th=None)

    monkeypatch.setattr(engine, "build_resume_runtime_state_target", fake_build_resume_runtime_state_target)
    monkeypatch.setattr(engine, "prepare_trajectory_data_shards", fake_prepare_trajectory_data_shards)
    monkeypatch.setattr(engine, "instantiate_validation_env", fake_instantiate_validation_env)
    algorithm = SimpleNamespace(
        init_agent_conf=lambda _env, config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(engine, "pick_algorithm", lambda _config: algorithm)
    monkeypatch.setattr(engine, "build_metrics_handler", lambda _config, _env: object())
    monkeypatch.setattr(engine, "build_logging_callback", lambda *_args, **_kwargs: lambda *_a, **_k: None)
    monkeypatch.setattr(engine, "config_hash", lambda _cfg: "hash123")
    monkeypatch.setattr(
        engine,
        "resolve_checkpoint_dir",
        lambda **_kwargs: str(tmp_path / "checkpoints" / "hash123"),
    )
    monkeypatch.setattr(engine, "validate_checkpoint_compatibility", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(engine, "write_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "compute_training_rngs", lambda _config: [0])

    def fake_run_training(*_args, **kwargs):
        captured["trajectory_data_shards"] = kwargs.get("trajectory_data_shards")
        captured["validation_trajectory_data_shards"] = kwargs.get("validation_trajectory_data_shards")
        captured["resume_agent_state"] = kwargs.get("resume_agent_state")
        return {"ok": True}

    monkeypatch.setattr(engine, "run_training", fake_run_training)

    def fake_resume_or_fresh(
        env,
        agent_conf,
        algorithm_cls,
        config,
        mh,
        val_env=None,
        apply_resume_resets=True,
        preserve_checkpoint_target=False,
        runtime_state_target=None,
    ):
        captured["apply_resume_resets"] = apply_resume_resets
        captured["preserve_checkpoint_target"] = preserve_checkpoint_target
        captured["resume_from"] = config.experiment.resume_from
        captured["runtime_state_target"] = runtime_state_target
        return "training", "resume-state"

    monkeypatch.setattr(engine, "resume_or_fresh", fake_resume_or_fresh)


def _write_checkpoint_metadata(checkpoint_path: Path, metadata: dict) -> None:
    metadata_dir = checkpoint_path / "metadata"
    metadata_dir.mkdir(parents=True)
    with open(metadata_dir / "metadata", "w") as f:
        json.dump(metadata, f)


def test_run_experiment_local_auto_resume_resets_env_by_default(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    latest = tmp_path / "checkpoints" / "hash123" / "checkpoint_42"
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: str(latest))

    config = _make_config(auto_resume=True, resume_from="/external/checkpoint_10")
    engine.run_experiment(config, _DummyHooks())

    assert captured["apply_resume_resets"] is True
    assert captured["preserve_checkpoint_target"] is True
    assert captured["resume_from"] == str(latest)
    assert captured["runtime_state_target"] is None
    assert captured["resume_agent_state"] == "resume-state"
    assert captured["trajectory_data_shards"] == "device-shards"
    assert captured["validation_trajectory_data_shards"] is None
    assert captured["events"] == ["trajectory_shards", "validation_env"]


def test_run_experiment_attaches_shared_validation_trajectory(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    full_trajectory = object()
    sharded_trajectory = object()
    training_env = SimpleNamespace(th=SimpleNamespace(traj=full_trajectory))
    validation_env = SimpleNamespace(th=None)

    monkeypatch.setattr(engine, "instantiate_env", lambda _config: training_env)
    monkeypatch.setattr(engine, "should_share_validation_trajectory", lambda _config: True)

    def log_trajectory_summary(label, env):
        captured.setdefault("trajectory_summaries", []).append((label, env.th.traj))

    def prepare_trajectory_shards(env, _config, _n_devices):
        assert captured["trajectory_summaries"] == [
            ("Train", full_trajectory),
            ("Validation", full_trajectory),
        ]
        env.th.traj = sharded_trajectory
        return "device-shards"

    def instantiate_validation(_config, *, share_trajectory):
        captured["share_trajectory"] = share_trajectory
        return validation_env

    def attach_trajectory(train_env, val_env):
        captured["attached"] = (train_env, val_env)
        val_env.th = train_env.th

    monkeypatch.setattr(engine, "log_trajectory_summary", log_trajectory_summary)
    monkeypatch.setattr(engine, "prepare_trajectory_data_shards", prepare_trajectory_shards)
    monkeypatch.setattr(engine, "instantiate_validation_env", instantiate_validation)
    monkeypatch.setattr(engine, "attach_shared_validation_trajectory", attach_trajectory)

    engine.run_experiment(_make_config(auto_resume=False), _DummyHooks())

    assert captured["share_trajectory"] is True
    assert captured["attached"] == (training_env, validation_env)
    assert validation_env.th.traj is sharded_trajectory
    assert captured["validation_trajectory_data_shards"] is None


def test_run_experiment_cleans_up_after_training_error(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)

    def fail_training(*_args, **_kwargs):
        raise RuntimeError("training failed")

    def flush_videos(_hooks, _use_wandb, *, wait):
        captured["flush_wait"] = wait

    monkeypatch.setattr(engine, "run_training", fail_training)
    monkeypatch.setattr(engine, "flush_validation_videos", flush_videos)
    with pytest.raises(RuntimeError, match="training failed"):
        engine.run_experiment(_make_config(auto_resume=False), _DummyHooks())

    assert captured["flush_wait"] is False


def test_run_experiment_exact_local_auto_resume_skips_resume_resets(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    latest = tmp_path / "checkpoints" / "hash123" / "checkpoint_42"
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: str(latest))

    config = _make_config(auto_resume=True, resume_from="/external/checkpoint_10", exact_resume=True)
    engine.run_experiment(config, _DummyHooks())

    assert captured["apply_resume_resets"] is False
    assert captured["preserve_checkpoint_target"] is True
    assert captured["resume_from"] == str(latest)
    assert captured["runtime_state_target"] == {"step": 0}
    assert captured["resume_agent_state"] == "resume-state"
    assert captured["trajectory_data_shards"] == "device-shards"
    assert captured["events"] == ["runtime_target", "trajectory_shards", "validation_env"]


def test_run_experiment_explicit_resume_keeps_resume_resets(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: None)

    explicit = "/external/checkpoint_10"
    config = _make_config(auto_resume=True, resume_from=explicit)
    engine.run_experiment(config, _DummyHooks())

    assert captured["apply_resume_resets"] is True
    assert captured["preserve_checkpoint_target"] is False
    assert captured["resume_from"] == explicit
    assert captured["runtime_state_target"] is None
    assert captured["events"] == ["trajectory_shards", "validation_env"]


def test_run_experiment_auto_resume_starts_fresh_wandb_run(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    latest = tmp_path / "checkpoints" / "hash123" / "checkpoint_42"
    _write_checkpoint_metadata(
        latest,
        {
            "global_timestep": 123456,
            "wandb_run_id": "source-run",
        },
    )
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: str(latest))

    def fake_setup_wandb(config):
        captured["fork_from"] = config.wandb.get("fork_from", None)
        captured["wandb_id"] = config.wandb.get("id", None)
        captured["wandb_resume"] = config.wandb.get("resume", None)
        return True, SimpleNamespace(id="fresh-run")

    monkeypatch.setattr(engine, "setup_wandb", fake_setup_wandb)

    config = _make_config(auto_resume=True)
    with open_dict(config.wandb):
        config.wandb.mode = "online"
        config.wandb.name = "preempted-run"

    engine.run_experiment(config, _DummyHooks())

    assert captured["fork_from"] is None
    assert captured["wandb_id"] is None
    assert captured["wandb_resume"] is None
    assert captured["apply_resume_resets"] is True
    assert captured["preserve_checkpoint_target"] is True
    assert captured["resume_from"] == str(latest)
    assert captured["runtime_state_target"] is None
    assert config.experiment.wandb_run_id == "fresh-run"


def test_run_experiment_auto_resume_without_wandb_source_disables_wandb_resume(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    latest = tmp_path / "checkpoints" / "hash123" / "checkpoint_42"
    _write_checkpoint_metadata(latest, {"global_timestep": 123456})
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: str(latest))
    monkeypatch.setenv("WANDB_RUN_ID", "stale-run")
    monkeypatch.setenv("WANDB_RESUME", "allow")

    def fake_setup_wandb(config):
        captured["fork_from"] = config.wandb.get("fork_from", None)
        captured["wandb_id"] = config.wandb.get("id", None)
        captured["wandb_resume"] = config.wandb.get("resume", None)
        captured["env_run_id"] = os.environ.get("WANDB_RUN_ID")
        captured["env_resume"] = os.environ.get("WANDB_RESUME")
        return True, SimpleNamespace(id="fresh-run")

    monkeypatch.setattr(engine, "setup_wandb", fake_setup_wandb)

    config = _make_config(auto_resume=True)
    with open_dict(config.wandb):
        config.wandb.mode = "online"
        config.wandb.id = "stale-run"
        config.wandb.resume = "allow"

    engine.run_experiment(config, _DummyHooks())

    assert captured["fork_from"] is None
    assert captured["wandb_id"] is None
    assert captured["wandb_resume"] is None
    assert captured["env_run_id"] is None
    assert captured["env_resume"] is None


def test_run_experiment_explicit_resume_does_not_reconfigure_wandb(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    _patch_run_experiment_dependencies(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(engine, "find_latest_checkpoint", lambda _path: None)

    def fake_setup_wandb(config):
        captured["fork_from"] = config.wandb.get("fork_from", None)
        return True, SimpleNamespace(id="fresh-run")

    monkeypatch.setattr(engine, "setup_wandb", fake_setup_wandb)

    config = _make_config(auto_resume=True, resume_from="/external/checkpoint_10")
    with open_dict(config.wandb):
        config.wandb.mode = "online"

    engine.run_experiment(config, _DummyHooks())

    assert captured["fork_from"] is None
    assert captured["apply_resume_resets"] is True
    assert captured["preserve_checkpoint_target"] is False
    assert captured["resume_from"] == "/external/checkpoint_10"
