"""Tests for host-side FlashSAC metric formatting."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from musclemimic.algorithms.flashsac.train import FlashSACCollectionMetrics
from musclemimic.algorithms.flashsac.update import FlashSACUpdateMetrics
from musclemimic.runner import flashsac_logging
from musclemimic.runner.flashsac_logging import FlashSACLogWindow, format_flashsac_log


def _format_log(*, has_actor_update: bool, has_critic_update: bool) -> dict[str, object]:
    collection_metrics = SimpleNamespace(
        done_count=4.0,
        absorbing_count=2.0,
        tracking_termination_count=1.0,
        horizon_end_count=1.0,
        trajectory_end_count=1.0,
        numerical_count=1.0,
        site_position_count=1.0,
        episode_return=1.0,
        episode_length=2.0,
        mean_reward=3.0,
        reward_components={},
    )
    update_metrics = FlashSACUpdateMetrics(
        actor_loss=jnp.asarray(1.0),
        critic_loss=jnp.asarray(2.0),
        temperature_loss=jnp.asarray(3.0),
        temperature=jnp.asarray(4.0),
        entropy=jnp.asarray(5.0),
        q_value=jnp.asarray(6.0),
        has_actor_update=jnp.asarray(has_actor_update),
        has_critic_update=jnp.asarray(has_critic_update),
    )
    return format_flashsac_log(
        collection_metrics,
        update_metrics,
        reward_ema=0.0,
        replay_size=10,
        learning_rate=3e-4,
        global_timestep=4,
        raw_timestep=4,
        config={},
    )


def test_log_omits_update_metrics_before_replay_warmup():
    log = _format_log(has_actor_update=False, has_critic_update=False)

    assert "flashsac/critic_loss" not in log
    assert "flashsac/temperature" not in log
    assert "flashsac/actor_loss" not in log
    assert "flashsac/temperature_loss" not in log
    assert "flashsac/entropy" not in log
    assert "flashsac/q_value" not in log


def test_log_omits_actor_metrics_when_only_critic_was_updated():
    log = _format_log(has_actor_update=False, has_critic_update=True)

    assert log["flashsac/critic_loss"] == 2.0
    assert log["flashsac/temperature"] == 4.0
    assert "flashsac/actor_loss" not in log
    assert "flashsac/temperature_loss" not in log
    assert "flashsac/entropy" not in log
    assert "flashsac/q_value" not in log


def test_log_includes_actor_metrics_after_actor_update():
    log = _format_log(has_actor_update=True, has_critic_update=True)

    assert log["flashsac/actor_loss"] == 1.0
    assert log["flashsac/temperature_loss"] == 3.0
    assert log["flashsac/entropy"] == 5.0
    assert log["flashsac/q_value"] == 6.0


def test_log_includes_termination_causes():
    log = _format_log(has_actor_update=False, has_critic_update=False)

    assert log["episodes/trajectory_ended"] == 1
    assert log["episodes/tracking_terminated"] == 1
    assert log["episodes/tracking_termination_rate"] == 0.25
    assert log["episodes/horizon_ended"] == 1
    assert log["episodes/numerical_terminated"] == 1
    assert log["episodes/site_position_terminated"] == 1


def test_log_window_omits_stale_episode_and_optimizer_measurements_then_resets():
    window = FlashSACLogWindow()
    collection = FlashSACCollectionMetrics(
        mean_reward=0.5,
        **dict.fromkeys(flashsac_logging.COLLECTION_COUNT_FIELDS, 0),
        episode_return=float("nan"),
        episode_length=float("nan"),
        reward_components={"reward/total": 0.5},
    )
    no_update = FlashSACUpdateMetrics(
        actor_loss=float("nan"),
        critic_loss=float("nan"),
        temperature_loss=float("nan"),
        temperature=float("nan"),
        entropy=float("nan"),
        q_value=float("nan"),
        has_actor_update=False,
        has_critic_update=False,
    )
    window.add(collection, no_update)
    window.add(collection, no_update)
    result = window.pop()
    log = format_flashsac_log(
        result.collection,
        result.updates,
        window_interactions=result.num_interactions,
        reward_ema=0.5,
        replay_size=10,
        learning_rate=3e-4,
        global_timestep=8,
        raw_timestep=8,
        config={},
    )

    assert log["flashsac/log_window_interactions"] == 2
    assert log["episodes/completed"] == 0
    assert log["reward/mean_step"] == 0.5
    assert "mean_episode_return" not in log
    assert "mean_episode_length" not in log
    assert "episodes/early_termination_rate" not in log
    assert "episodes/tracking_termination_rate" not in log
    assert "flashsac/actor_loss" not in log
    assert "flashsac/critic_loss" not in log
    with pytest.raises(ValueError, match="empty FlashSAC log window"):
        window.pop()

    window.add(collection._replace(mean_reward=1.5), no_update)
    fresh = window.pop()
    assert fresh.num_interactions == 1
    assert fresh.collection.mean_reward == 1.5

    for value in (2.0, 4.0):
        window.add(
            collection,
            FlashSACUpdateMetrics(
                actor_loss=value,
                critic_loss=value,
                temperature_loss=value,
                temperature=value,
                entropy=value,
                q_value=value,
                has_actor_update=True,
                has_critic_update=True,
            ),
        )
    window.add(collection, no_update._replace(critic_loss=6.0, temperature=6.0, has_critic_update=True))
    mixed = window.pop()
    assert mixed.num_interactions == 3
    for name in flashsac_logging.ACTOR_METRIC_FIELDS:
        assert getattr(mixed.updates, name) == 3.0
    assert mixed.updates.critic_loss == 4.0
    assert mixed.updates.temperature == 6.0


def test_sampled_validation_omits_trajectory_set_metrics(monkeypatch):
    observed = {}

    def flatten(*_args, **kwargs):
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(flashsac_logging, "flatten_validation_metrics", flatten)
    collection_metrics = SimpleNamespace(
        done_count=0.0,
        absorbing_count=0.0,
        tracking_termination_count=0.0,
        horizon_end_count=0.0,
        trajectory_end_count=0.0,
        numerical_count=0.0,
        site_position_count=0.0,
        episode_return=1.0,
        episode_length=2.0,
        mean_reward=3.0,
        reward_components={},
    )
    update_metrics = FlashSACUpdateMetrics(
        actor_loss=jnp.asarray(1.0),
        critic_loss=jnp.asarray(2.0),
        temperature_loss=jnp.asarray(3.0),
        temperature=jnp.asarray(4.0),
        entropy=jnp.asarray(5.0),
        q_value=jnp.asarray(6.0),
        has_actor_update=jnp.asarray(False),
        has_critic_update=jnp.asarray(False),
    )

    flashsac_logging.format_flashsac_log(
        collection_metrics,
        update_metrics,
        reward_ema=0.0,
        replay_size=10,
        learning_rate=3e-4,
        global_timestep=4,
        raw_timestep=4,
        config={"validation": {"evaluate_all": False}},
        validation_metrics=object(),
        validation_number=1,
        checkpoint_update=1,
    )

    assert observed["include_trajectory_set_metrics"] is False
