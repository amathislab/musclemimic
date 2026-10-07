"""Tests for PPO training functions."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.ppo import train as ppo_train

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _training() -> ppo_train.PPOTraining:
    return ppo_train.PPOTraining(
        env=object(),
        val_env=object(),
        agent_conf=object(),
        agent_state_cls=object,
        config=OmegaConf.create({}),
        metrics_handler=None,
        resume_info=None,
        apply_resume_resets=True,
        preserve_checkpoint_target=False,
        axis_name=None,
        shard_spec=None,
        validation_shard_spec=None,
        validation_shares_training_trajectory=False,
        curriculum_params=None,
        debug_flags=object(),
    )


@pytest.mark.parametrize(
    "config_path",
    [
        "fullbody/conf_fullbody.yaml",
        "bimanual/conf_bimanual.yaml",
    ],
)
def test_experiment_configs_use_canonical_checkpoint_schedule_key(config_path):
    config = OmegaConf.load(REPOSITORY_ROOT / config_path)

    assert config.experiment.checkpoints_on_validation is True
    assert "save_checkpoints_on_validation" not in config.experiment


def test_training_restores_trajectory_overrides_after_trace_failure(monkeypatch):
    override_events = []

    @contextmanager
    def trajectory_overrides(*_args):
        override_events.append("enter")
        try:
            yield
        finally:
            override_events.append("exit")

    monkeypatch.setattr(ppo_train, "with_trajectory_data_overrides", trajectory_overrides)

    def fail_wrapping(*_args, **_kwargs):
        raise RuntimeError("trace failed")

    monkeypatch.setattr(ppo_train, "wrap_env", fail_wrapping)
    training = _training()

    with pytest.raises(RuntimeError, match="trace failed"):
        training.update(
            runtime_state=object(),
            trajectory_data_override=object(),
        )
    with pytest.raises(RuntimeError, match="trace failed"):
        training.update_chunk(
            runtime_state=object(),
            trajectory_data_override=object(),
            num_updates=ppo_train.jnp.asarray(2),
            max_updates=3,
        )

    assert override_events == ["enter", "exit", "enter", "exit"]


def test_validation_disables_reward_normalization_collectives(monkeypatch):
    training = _training()
    training.config.update({"normalize_env": True, "validation": {"evaluate_all": False}})
    wrapping = []

    def wrap(env, config, **kwargs):
        wrapping.append(kwargs)
        return env

    monkeypatch.setattr(ppo_train, "wrap_env", wrap)
    monkeypatch.setattr(ppo_train, "with_trajectory_data_overrides", lambda *args: nullcontext())
    monkeypatch.setattr(ppo_train, "run_validation_step", lambda *args, **kwargs: None)
    training.validate(object(), object(), None)
    assert wrapping == [{"axis_name": None, "normalize_reward": False}]


def test_training_chunk_runs_updates_in_one_compiled_scan(monkeypatch):
    override_events = []

    @contextmanager
    def trajectory_overrides(*_args):
        override_events.append("enter")
        try:
            yield
        finally:
            override_events.append("exit")

    monkeypatch.setattr(ppo_train, "with_trajectory_data_overrides", trajectory_overrides)
    monkeypatch.setattr(ppo_train, "wrap_env", lambda *_args, **_kwargs: object())

    def update(_self, state, _env):
        next_state = state + 1
        return ppo_train.PPOTrainStep(
            runtime_state=next_state,
            training_metrics=next_state,
            ppo_metrics=next_state * 2,
            diagnostics=next_state * 3,
            validation_reset_weights=next_state[None],
            counter=next_state,
        )

    monkeypatch.setattr(ppo_train.PPOTraining, "_update", update)
    training = _training()
    compiled = ppo_train.jax.jit(
        lambda state, num_updates: training.update_chunk(
            state,
            None,
            num_updates,
            3,
        ),
    )

    result = compiled(ppo_train.jnp.asarray(0), ppo_train.jnp.asarray(2))

    assert int(result.runtime_state) == 2
    assert result.training_metrics.tolist() == [1, 2, 0]
    assert result.ppo_metrics.tolist() == [2, 4, 0]
    assert result.diagnostics.tolist() == [3, 6, 0]
    assert result.validation_reset_weights.tolist() == [2]
    assert override_events == ["enter", "exit"]


def test_build_training_allows_single_device_static_trajectory():
    experiment = OmegaConf.create(
        {
            "distributed": {"num_devices": 1},
            "trajectory": {"sharding": "static"},
            "n_trajectories": 4,
            "validation": {},
            "num_envs": 8,
            "num_steps": 4,
            "num_minibatches": 2,
            "adaptive_termination": {"enabled": False},
            "debug": {},
        }
    )

    training = ppo_train.build_training(
        object(),
        SimpleNamespace(config=OmegaConf.create({"experiment": experiment})),
        object,
    )

    assert training.shard_spec.counts == (4,)
    assert training.shard_spec.offsets == (0, 4)
    assert training.config.num_envs == 8
    assert training.config.global_num_envs == 8


def test_build_training_keeps_unsharded_sampling_state():
    experiment = OmegaConf.create(
        {
            "distributed": {"num_devices": 1},
            "trajectory": {"sharding": "none"},
            "validation": {},
            "num_envs": 8,
            "num_steps": 4,
            "num_minibatches": 2,
            "adaptive_termination": {"enabled": False},
            "debug": {},
        }
    )

    training = ppo_train.build_training(
        object(),
        SimpleNamespace(config=OmegaConf.create({"experiment": experiment})),
        object,
    )

    assert training.shard_spec is None
    assert training.validation_shard_spec is None
