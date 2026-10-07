"""Host-side formatting for FlashSAC metrics."""

from __future__ import annotations

from typing import Any, NamedTuple

from musclemimic.algorithms.flashsac.train import FlashSACCollectionMetrics
from musclemimic.algorithms.flashsac.update import FlashSACUpdateMetrics
from musclemimic.utils.metrics import flatten_validation_metrics

COLLECTION_COUNT_FIELDS = (
    "done_count",
    "absorbing_count",
    "tracking_termination_count",
    "horizon_end_count",
    "trajectory_end_count",
    "numerical_count",
    "site_position_count",
)
ACTOR_METRIC_FIELDS = ("actor_loss", "temperature_loss", "entropy", "q_value")


class FlashSACWindowMetrics(NamedTuple):
    collection: FlashSACCollectionMetrics
    updates: FlashSACUpdateMetrics
    num_interactions: int


class FlashSACLogWindow:
    """Host-only totals for interactions since the previous emitted log."""

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._interactions = 0
        self._counts = dict.fromkeys(COLLECTION_COUNT_FIELDS, 0)
        self._reward_sum = 0.0
        self._reward_components: dict[str, float] = {}
        self._return_sum = 0.0
        self._length_sum = 0.0
        self._actor_count = 0
        self._critic_count = 0
        self._actor_sums = dict.fromkeys(ACTOR_METRIC_FIELDS, 0.0)
        self._critic_loss_sum = 0.0
        self._temperature = 0.0

    def add(self, collection: FlashSACCollectionMetrics, updates: FlashSACUpdateMetrics) -> None:
        """Accumulate one host-resident interaction, including all completed episodes."""
        self._interactions += 1
        for name in COLLECTION_COUNT_FIELDS:
            self._counts[name] += int(getattr(collection, name))
        self._reward_sum += float(collection.mean_reward)
        for name, value in collection.reward_components.items():
            self._reward_components[name] = self._reward_components.get(name, 0.0) + float(value)

        done_count = int(collection.done_count)
        if done_count:
            self._return_sum += float(collection.episode_return) * done_count
            self._length_sum += float(collection.episode_length) * done_count
        if bool(updates.has_actor_update):
            self._actor_count += 1
            for name in ACTOR_METRIC_FIELDS:
                self._actor_sums[name] += float(getattr(updates, name))
        if bool(updates.has_critic_update):
            self._critic_count += 1
            self._critic_loss_sum += float(updates.critic_loss)
            self._temperature = float(updates.temperature)

    def pop(self) -> FlashSACWindowMetrics:
        """Return window sums and weighted means, then begin an empty window."""
        if not self._interactions:
            raise ValueError("Cannot summarize an empty FlashSAC log window")
        done_count = max(self._counts["done_count"], 1)
        collection = FlashSACCollectionMetrics(
            **self._counts,
            mean_reward=self._reward_sum / self._interactions,
            episode_return=self._return_sum / done_count,
            episode_length=self._length_sum / done_count,
            reward_components={name: value / self._interactions for name, value in self._reward_components.items()},
        )
        updates = FlashSACUpdateMetrics(
            **{name: value / max(self._actor_count, 1) for name, value in self._actor_sums.items()},
            critic_loss=self._critic_loss_sum / max(self._critic_count, 1),
            temperature=self._temperature,
            has_actor_update=self._actor_count > 0,
            has_critic_update=self._critic_count > 0,
        )
        result = FlashSACWindowMetrics(collection, updates, self._interactions)
        self._reset()
        return result


def format_flashsac_log(
    collection_metrics: Any,
    update_metrics: Any,
    *,
    reward_ema: Any,
    replay_size: Any,
    learning_rate: Any,
    global_timestep: int,
    raw_timestep: int,
    validation_metrics: Any | None = None,
    checkpoint_update: int = -1,
    validation_number: int | None = None,
    window_interactions: int = 1,
    config: Any,
) -> dict[str, Any]:
    """Convert host-resident FlashSAC values to the shared log schema."""
    done_count = float(collection_metrics.done_count)
    absorbing_count = float(collection_metrics.absorbing_count)
    tracking_termination_count = float(collection_metrics.tracking_termination_count)
    horizon_end_count = float(collection_metrics.horizon_end_count)
    trajectory_end_count = float(collection_metrics.trajectory_end_count)
    numerical_count = float(collection_metrics.numerical_count)
    site_position_count = float(collection_metrics.site_position_count)
    has_actor_update = bool(update_metrics.has_actor_update)
    has_critic_update = bool(update_metrics.has_critic_update)
    log = {
        "max_timestep": int(global_timestep),
        "jax_raw_timestep": int(raw_timestep),
        "learning_rate": float(learning_rate),
        "flashsac/replay_size": int(replay_size),
        "flashsac/log_window_interactions": int(window_interactions),
        "reward/mean_step": float(collection_metrics.mean_reward),
        "reward/ema_step": float(reward_ema),
        "episodes/completed": int(done_count),
        "episodes/early_terminated": int(absorbing_count),
        "episodes/tracking_terminated": int(tracking_termination_count),
        "episodes/horizon_ended": int(horizon_end_count),
        "episodes/trajectory_ended": int(trajectory_end_count),
        "episodes/numerical_terminated": int(numerical_count),
        "episodes/site_position_terminated": int(site_position_count),
    }
    if done_count > 0:
        log["mean_episode_return"] = float(collection_metrics.episode_return)
        log["mean_episode_length"] = float(collection_metrics.episode_length)
        log["episodes/early_termination_rate"] = absorbing_count / done_count
        log["episodes/tracking_termination_rate"] = tracking_termination_count / done_count
    if has_critic_update:
        log["flashsac/critic_loss"] = float(update_metrics.critic_loss)
        log["flashsac/temperature"] = float(update_metrics.temperature)
    if has_actor_update:
        log["flashsac/actor_loss"] = float(update_metrics.actor_loss)
        log["flashsac/temperature_loss"] = float(update_metrics.temperature_loss)
        log["flashsac/entropy"] = float(update_metrics.entropy)
        log["flashsac/q_value"] = float(update_metrics.q_value)
    log.update({key: float(value) for key, value in collection_metrics.reward_components.items()})
    if validation_metrics is not None:
        if validation_number is None or validation_number < 1:
            raise ValueError("validation_number must identify the scheduled validation")
        validation_config = config.get("validation", {})
        log["has_validation_update"] = True
        log["_checkpoint_update"] = int(checkpoint_update)
        log["_validation_number"] = int(validation_number)
        log.update(
            flatten_validation_metrics(
                validation_metrics,
                validation_config.get("measures"),
                validation_config.get("quantities"),
                include_trajectory_set_metrics=bool(
                    validation_config.get("evaluate_all", False)
                ),
            )
        )
    return log
