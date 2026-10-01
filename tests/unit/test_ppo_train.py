"""Tests for PPO training functions."""

from __future__ import annotations

from pathlib import Path

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
    restore_token = object()
    restored = []

    monkeypatch.setattr(
        ppo_train,
        "install_trajectory_data_overrides",
        lambda *_args: restore_token,
    )
    monkeypatch.setattr(
        ppo_train,
        "restore_trajectory_data_overrides",
        restored.append,
    )

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

    assert restored == [restore_token, restore_token]


def test_training_chunk_runs_updates_in_one_compiled_scan(monkeypatch):
    restored = []
    monkeypatch.setattr(
        ppo_train,
        "install_trajectory_data_overrides",
        lambda *_args: "restore-token",
    )
    monkeypatch.setattr(
        ppo_train,
        "restore_trajectory_data_overrides",
        restored.append,
    )
    monkeypatch.setattr(ppo_train, "wrap_env", lambda *_args, **_kwargs: object())

    def update(_self, state, _env):
        next_state = state + 1
        return ppo_train.PPOTrainStep(
            runtime_state=next_state,
            training_metrics=next_state,
            ppo_metrics=next_state * 2,
            # Static scalar leaves are valid JAX outputs and must be promoted
            # before the first result is concatenated with the scan results.
            diagnostics=3.0,
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
    assert result.diagnostics.tolist() == [3, 3, 0]
    assert result.validation_reset_weights.tolist() == [2]
    assert restored == ["restore-token"]
