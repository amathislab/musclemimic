from __future__ import annotations

import inspect

from omegaconf import OmegaConf

from musclemimic.runner import ppo_logging


class _ZeroMetrics:
    def __getattr__(self, _name):
        return 0.0


def test_ppo_log_selects_metrics_for_validation_mode(monkeypatch):
    observed = {}

    def flatten(*_args, **kwargs):
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(ppo_logging, "flatten_validation_metrics", flatten)
    zero_metrics = _ZeroMetrics()
    diagnostics = _ZeroMetrics()

    validation_identity = {}
    if "validation_number" in inspect.signature(ppo_logging.format_ppo_log).parameters:
        validation_identity["validation_number"] = 1

    ppo_logging.format_ppo_log(
        zero_metrics,
        zero_metrics,
        diagnostics,
        OmegaConf.create(
            {
                "update_epochs": 1,
                "validation": {"evaluate_all": False},
                "adaptive_termination": {"enabled": False},
                "reward_curriculum": {"enabled": False},
                "adaptive_sampling": {"enabled": False},
                "use_moe": False,
                "schedule": "fixed",
            }
        ),
        global_timestep=100,
        raw_timestep=100,
        validation_metrics=object(),
        checkpoint_update=1,
        **validation_identity,
    )

    assert observed["include_trajectory_set_metrics"] is False
