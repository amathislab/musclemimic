"""Tests for experiment logging setup."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from musclemimic.runner import experiment_logging


class FakeWandb:
    def __init__(self):
        self.params = None
        self.login_count = 0

    def login(self):
        self.login_count += 1
        return None

    def init(self, **params):
        self.params = params
        return SimpleNamespace(id="new-run")


def test_setup_wandb_uses_fork_without_resume_or_id(monkeypatch):
    wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    config = OmegaConf.create(
        {
            "wandb": {
                "mode": "online",
                "project": "musclemimic",
                "name": "forked",
                "id": "old-run",
                "resume": "allow",
                "fork_from": "source-run?_step=123456",
            },
            "experiment": {},
        }
    )

    use_wandb, run = experiment_logging.setup_wandb(config)

    assert use_wandb is True
    assert run.id == "new-run"
    assert wandb.params["fork_from"] == "source-run?_step=123456"
    assert "id" not in wandb.params
    assert "resume" not in wandb.params


def test_setup_wandb_passes_entity(monkeypatch):
    wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    config = OmegaConf.create(
        {
            "wandb": {
                "mode": "online",
                "project": "musclemimic",
                "entity": "chengkun",
            },
            "experiment": {},
        }
    )

    use_wandb, _ = experiment_logging.setup_wandb(config)

    assert use_wandb is True
    assert wandb.params["entity"] == "chengkun"


def test_setup_wandb_offline_skips_login_and_passes_mode(monkeypatch):
    wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    config = OmegaConf.create(
        {
            "wandb": {
                "mode": "offline",
                "project": "musclemimic",
            },
            "experiment": {},
        }
    )

    use_wandb, _ = experiment_logging.setup_wandb(config)

    assert use_wandb is True
    assert wandb.login_count == 0
    assert wandb.params["mode"] == "offline"


def test_checkpoint_resume_clears_wandb_resume_state(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "stale-run")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    config = OmegaConf.create(
        {
            "wandb": {
                "mode": "online",
                "id": "stale-run",
                "resume": "allow",
                "fork_from": "source-run",
            }
        }
    )

    experiment_logging.configure_wandb_for_checkpoint_resume(config, "/tmp/checkpoint_1")

    assert os.environ.get("WANDB_RUN_ID") is None
    assert os.environ.get("WANDB_RESUME") is None
    assert config.wandb.id is None
    assert config.wandb.resume is None
    assert config.wandb.fork_from is None


def test_metrics_handler_is_only_created_for_active_validation(monkeypatch):
    created = object()
    monkeypatch.setattr(experiment_logging, "MetricsHandler", lambda _config, _env: created)
    active = OmegaConf.create({"experiment": {"validation": {"active": True}}})
    inactive = OmegaConf.create({"experiment": {"validation": {"active": False}}})

    assert experiment_logging.build_metrics_handler(active, object()) is created
    assert experiment_logging.build_metrics_handler(inactive, object()) is None


def test_logging_callback_preserves_absent_episode_statistics():
    logs = []
    logger = SimpleNamespace(
        enrich=lambda *_args: None,
        log=lambda payload, *, step: logs.append((payload, step)),
    )
    callback = experiment_logging.build_logging_callback(object(), OmegaConf.create({"experiment": {}}), logger, None)

    callback({"max_timestep": 4, "episodes/completed": 0})
    callback({"max_timestep": 8, "mean_episode_return": 2.0, "mean_episode_length": 3.0})

    assert "Mean Episode Return" not in logs[0][0]
    assert "Mean Episode Length" not in logs[0][0]
    assert logs[0][0]["episodes/completed"] == 0
    assert logs[1][0]["Mean Episode Return"] == 2.0
    assert logs[1][0]["Mean Episode Length"] == 3.0
