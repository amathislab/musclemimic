"""Tests for pure FlashSAC interaction chunks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.flashsac import train as flashsac_train
from musclemimic.algorithms.flashsac.update import FlashSACUpdateMetrics


@struct.dataclass
class _Replay:
    size: jax.Array


@struct.dataclass
class _Agent:
    replay_buffer: _Replay


@struct.dataclass
class _Runtime:
    value: jax.Array
    reward_ema: jax.Array
    agent: _Agent


def _training() -> flashsac_train.FlashSACTraining:
    config = OmegaConf.create({})
    return flashsac_train.FlashSACTraining(
        env=object(),
        val_env=None,
        agent_conf=SimpleNamespace(config=OmegaConf.create({"experiment": config})),
        config=config,
        metrics_handler=None,
        resume_info=None,
        preserve_checkpoint_target=False,
        axis_name=None,
        shard_spec=None,
        validation_shard_spec=None,
        validation_shares_training_trajectory=True,
    )


def _runtime() -> _Runtime:
    return _Runtime(
        value=jnp.asarray(0, dtype=jnp.int32),
        reward_ema=jnp.asarray(0.0),
        agent=_Agent(_Replay(jnp.asarray(0, dtype=jnp.int32))),
    )


@pytest.mark.parametrize("updates_per_interaction", (2, 4))
@pytest.mark.parametrize("initial_update", (0, 2))
@pytest.mark.parametrize("compiled", (False, True))
def test_update_summary_preserves_periodic_actor_measurements(updates_per_interaction, initial_update, compiled):
    update_steps = jnp.arange(initial_update, initial_update + updates_per_interaction)
    actor_mask = update_steps % 4 == 0
    values = jnp.arange(updates_per_interaction, dtype=jnp.float32) + 4.0
    actor_values = jnp.where(actor_mask, values, 0.0)
    metrics = FlashSACUpdateMetrics(
        actor_loss=actor_values,
        critic_loss=values,
        temperature_loss=actor_values * 2.0,
        temperature=values / 10.0,
        entropy=actor_values * 3.0,
        q_value=actor_values * 4.0,
        has_actor_update=actor_mask,
        has_critic_update=jnp.ones(updates_per_interaction, dtype=bool),
    )
    summarize = flashsac_train.summarize_update_metrics
    if compiled:
        summarize = jax.jit(summarize)

    summary = summarize(metrics)

    has_actor_update = bool(jnp.any(actor_mask))
    expected_actor_mean = float(jnp.sum(actor_values) / jnp.maximum(jnp.sum(actor_mask), 1))
    assert bool(summary.has_actor_update) is has_actor_update
    assert bool(summary.has_critic_update) is True
    assert float(summary.actor_loss) == pytest.approx(expected_actor_mean)
    assert float(summary.temperature_loss) == pytest.approx(expected_actor_mean * 2.0)
    assert float(summary.entropy) == pytest.approx(expected_actor_mean * 3.0)
    assert float(summary.q_value) == pytest.approx(expected_actor_mean * 4.0)
    assert float(summary.critic_loss) == pytest.approx(float(jnp.mean(values)))
    assert float(summary.temperature) == pytest.approx(float(values[-1] / 10.0))


def test_update_summary_preserves_warmup_validity_and_finite_placeholders():
    zeros = jnp.zeros(4, dtype=jnp.float32)
    no_updates = jnp.zeros(4, dtype=bool)
    metrics = FlashSACUpdateMetrics(
        actor_loss=zeros,
        critic_loss=zeros,
        temperature_loss=zeros,
        temperature=zeros,
        entropy=zeros,
        q_value=zeros,
        has_actor_update=no_updates,
        has_critic_update=no_updates,
    )

    summary = jax.jit(flashsac_train.summarize_update_metrics)(metrics)

    assert not bool(summary.has_actor_update)
    assert not bool(summary.has_critic_update)
    assert all(np.isfinite(np.asarray(value)).all() for value in summary)
    assert float(summary.actor_loss) == 0.0
    assert float(summary.critic_loss) == 0.0


def test_training_chunk_runs_interactions_in_one_compiled_scan(monkeypatch):
    override_events = []

    @contextmanager
    def trajectory_overrides(*_args):
        override_events.append("enter")
        try:
            yield
        finally:
            override_events.append("exit")

    monkeypatch.setattr(flashsac_train, "with_trajectory_data_overrides", trajectory_overrides)
    monkeypatch.setattr(flashsac_train, "wrap_env", lambda *_args, **_kwargs: object())

    def update(_self, state, _env):
        value = state.value + 1
        next_state = state.replace(
            value=value,
            reward_ema=value.astype(jnp.float32) / 10.0,
            agent=state.agent.replace(
                replay_buffer=state.agent.replay_buffer.replace(size=value * 4),
            ),
        )
        scalar = value.astype(jnp.float32)
        return flashsac_train.FlashSACStep(
            runtime_state=next_state,
            collection_metrics=flashsac_train.FlashSACCollectionMetrics(
                mean_reward=scalar,
                done_count=value,
                absorbing_count=jnp.asarray(0),
                tracking_termination_count=jnp.asarray(0),
                horizon_end_count=jnp.asarray(0),
                trajectory_end_count=jnp.asarray(0),
                numerical_count=jnp.asarray(0),
                site_position_count=jnp.asarray(0),
                episode_return=scalar,
                episode_length=scalar,
                reward_components={"reward/total": scalar},
            ),
            update_metrics=FlashSACUpdateMetrics(
                actor_loss=scalar * 2,
                critic_loss=scalar * 2,
                temperature_loss=scalar * 2,
                temperature=scalar * 2,
                entropy=scalar * 2,
                q_value=scalar * 2,
                has_actor_update=jnp.asarray(True),
                has_critic_update=jnp.asarray(True),
            ),
            learning_rate=scalar / 100.0,
        )

    monkeypatch.setattr(flashsac_train.FlashSACTraining, "_update", update)
    training = _training()
    compiled = jax.jit(
        lambda state, count: training.update_chunk(
            state,
            None,
            count,
            4,
        )
    )

    chunk = compiled(_runtime(), jnp.asarray(3))

    assert int(chunk.runtime_state.value) == 3
    assert chunk.collection_metrics.mean_reward.tolist() == [1.0, 2.0, 3.0, 0.0]
    assert chunk.update_metrics.actor_loss.tolist() == [2.0, 4.0, 6.0, 0.0]
    np.testing.assert_allclose(chunk.reward_ema, [0.1, 0.2, 0.3, 0.0])
    assert chunk.replay_size.tolist() == [4, 8, 12, 0]
    np.testing.assert_allclose(chunk.learning_rate, [0.01, 0.02, 0.03, 0.0])
    assert override_events == ["enter", "exit"]


def test_training_chunk_restores_trajectory_override_after_trace_failure(monkeypatch):
    override_events = []

    @contextmanager
    def trajectory_overrides(*_args):
        override_events.append("enter")
        try:
            yield
        finally:
            override_events.append("exit")

    monkeypatch.setattr(flashsac_train, "with_trajectory_data_overrides", trajectory_overrides)

    def fail_wrapping(*_args, **_kwargs):
        raise RuntimeError("trace failed")

    monkeypatch.setattr(flashsac_train, "wrap_env", fail_wrapping)

    with pytest.raises(RuntimeError, match="trace failed"):
        _training().update_chunk(_runtime(), None, jnp.asarray(1), 1)

    assert override_events == ["enter", "exit"]


def test_validation_disables_reward_normalization(monkeypatch):
    @contextmanager
    def trajectory_overrides(*_args):
        yield

    wrap_arguments = {}

    def wrap_validation_env(env, _config, **kwargs):
        wrap_arguments.update(kwargs)
        return env

    monkeypatch.setattr(
        flashsac_train,
        "with_trajectory_data_overrides",
        trajectory_overrides,
    )
    monkeypatch.setattr(flashsac_train, "wrap_env", wrap_validation_env)
    monkeypatch.setattr(
        flashsac_train,
        "run_validation_step",
        lambda *_args, **_kwargs: "validation-result",
    )
    training = replace(
        _training(),
        config=OmegaConf.create(
            {
                "normalize_env": True,
                "validation": {"evaluate_all": False},
            }
        ),
        metrics_handler=object(),
    )

    result = training.validate(
        policy_state=object(),
        validation_rng=jax.random.key(0),
    )

    assert result == "validation-result"
    assert wrap_arguments["normalize_reward"] is False
