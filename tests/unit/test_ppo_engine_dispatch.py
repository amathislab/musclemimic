"""The new PPO coordinator is opt-in and leaves the legacy engine intact."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from musclemimic.runner import engine, ppo_experiment


def test_distributed_ppo_dispatches_before_legacy_engine(monkeypatch):
    sentinel = object()
    observed = {}

    def run(config, hooks):
        observed["config"] = config
        observed["hooks"] = hooks
        return sentinel

    monkeypatch.setattr(ppo_experiment, "run_ppo_experiment", run)
    config = OmegaConf.create(
        {
            "experiment": {
                "algorithm": "PPOJax",
                "distributed": {"num_devices": 2},
            }
        }
    )
    hooks = SimpleNamespace()

    assert engine.run_experiment(config, hooks) is sentinel
    assert observed == {"config": config, "hooks": hooks}


def test_metrics_handler_symbol_remains_patchable(monkeypatch):
    replacement = object()
    monkeypatch.setattr(engine, "MetricsHandler", lambda config, env: (replacement, config, env))
    config = OmegaConf.create({"experiment": {"validation": {"active": True}}})
    env = object()

    assert engine.build_metrics_handler(config, env) == (replacement, config, env)


def test_setup_wandb_uses_stable_optional_run_identity(monkeypatch):
    observed = {}
    run = object()
    fake_wandb = SimpleNamespace(
        login=lambda: observed.setdefault("logged_in", True),
        init=lambda **kwargs: observed.setdefault("params", kwargs) and run,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = OmegaConf.create(
        {
            "wandb": {"mode": "online", "project": "terra", "tags": ["ppo"]},
            "experiment": {"wandb_run_id": "ppo-study-001"},
        }
    )

    assert engine.setup_wandb(config) == (True, run)
    assert observed["logged_in"] is True
    assert observed["params"]["id"] == "ppo-study-001"
    assert observed["params"]["resume"] == "allow"


def test_ppo_binds_checkpoint_run_id_to_wandb_identity():
    config = OmegaConf.create({"experiment": {"run_id": "ppo-study-001"}})

    assert ppo_experiment._bind_ppo_wandb_run_id(config) == "ppo-study-001"
    assert config.experiment.wandb_run_id == "ppo-study-001"


def test_ppo_auto_resume_rejects_changed_manifest(monkeypatch):
    def reject(*_args):
        raise ValueError("experiment config hash changed")

    monkeypatch.setattr(
        ppo_experiment,
        "require_ppo_local_resume_compatibility",
        reject,
    )

    with pytest.raises(ValueError, match="experiment config hash changed"):
        ppo_experiment._require_compatible_local_resume("checkpoints/ppo", "new-hash")


def test_ppo_auto_resume_validates_manifest_without_a_finalized_checkpoint(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        ppo_experiment,
        "_require_compatible_local_resume",
        lambda checkpoint_dir, experiment_hash: observed.append((checkpoint_dir, experiment_hash)),
    )

    ppo_experiment._require_compatible_existing_local_manifest(
        str(tmp_path),
        "current-hash",
    )

    assert observed == [(str(tmp_path), "current-hash")]
