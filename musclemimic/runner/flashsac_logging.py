"""Host-side formatting for FlashSAC metrics."""

from __future__ import annotations

from typing import Any

from musclemimic.utils.metrics import flatten_validation_metrics


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
    config: Any,
) -> dict[str, Any]:
    """Convert host-resident FlashSAC values to the shared log schema."""
    done_count = float(collection_metrics.done_count)
    absorbing_count = float(collection_metrics.absorbing_count)
    has_actor_update = bool(update_metrics.has_actor_update)
    has_critic_update = bool(update_metrics.has_critic_update)
    log = {
        "mean_episode_return": float(collection_metrics.episode_return),
        "mean_episode_length": float(collection_metrics.episode_length),
        "max_timestep": int(global_timestep),
        "jax_raw_timestep": int(raw_timestep),
        "learning_rate": float(learning_rate),
        "flashsac/replay_size": int(replay_size),
        "reward/mean_step": float(collection_metrics.mean_reward),
        "reward/ema_step": float(reward_ema),
        "episodes/completed": int(done_count),
        "episodes/early_terminated": int(absorbing_count),
        "episodes/early_termination_rate": absorbing_count / max(done_count, 1.0),
    }
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
