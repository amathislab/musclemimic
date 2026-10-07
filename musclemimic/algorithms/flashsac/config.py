"""FlashSAC configuration and static agent inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax
import jax
import optax
from omegaconf import DictConfig, OmegaConf

from musclemimic.algorithms.common.base_algorithm import AgentConfBase

from .networks import FlashSACActor, FlashSACDoubleCritic

FLASHSAC_DEFAULTS = {
    "buffer_capacity": 131_072,
    "min_buffer_size": 16_384,
    "sample_batch_size": 2_048,
    "replay_storage_dtype": "bfloat16",
    "updates_per_interaction": 1,
    "update_chunk_size": 10,
    "gamma": 0.99,
    "normalize_reward": True,
    "normalized_return_max": 5.0,
    "actor_update_period": 2,
    "actor_hidden_dim": 128,
    "actor_num_blocks": 2,
    "critic_hidden_dim": 256,
    "critic_num_blocks": 2,
    "critic_num_bins": 101,
    "critic_min_value": -5.0,
    "critic_max_value": 5.0,
    "target_update_tau": 0.01,
    "initial_temperature": 0.01,
    "target_sigma": 0.15,
    "learning_rate": 3e-4,
    "end_learning_rate": 1.5e-4,
    "warmup_updates": 0,
    "max_grad_norm": 1.0,
    "noise_zeta_exponent": 2.0,
    "noise_max_repeat": 16,
}


def get_flashsac_config(experiment: DictConfig) -> DictConfig:
    """Merge FlashSAC settings and derive interaction counts."""
    values = dict(FLASHSAC_DEFAULTS)
    flashsac = experiment.get("flashsac_config")
    if flashsac is not None:
        values.update(OmegaConf.to_container(flashsac, resolve=True) or {})

    merged = OmegaConf.to_container(experiment, resolve=True)
    merged.update(values)
    merged["num_interactions"] = int(merged["total_timesteps"]) // int(merged["num_envs"])
    merged["num_updates"] = merged["num_interactions"] * int(merged["updates_per_interaction"])
    # The shared checkpoint schema counts one vectorized environment
    # interaction as one FlashSAC training update.
    merged["num_steps"] = 1
    merged["num_minibatches"] = 1
    merged["update_epochs"] = 1

    _validate_config(merged)
    config = OmegaConf.create(merged)
    OmegaConf.set_struct(config, False)
    validation = config.get("validation", {})
    if validation.get("active", False):
        requested = int(validation.get("num", 1))
        if requested < 1:
            raise ValueError("validation.num must be positive")
        config.validation_interval = max(1, int(config.num_interactions) // requested)
        config.validation.num = max(1, int(config.num_interactions) // int(config.validation_interval))
    else:
        config.validation_interval = int(config.num_interactions) + 1
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if int(config["num_interactions"]) < 1:
        raise ValueError("total_timesteps must be at least num_envs")
    if int(config["buffer_capacity"]) < int(config["num_envs"]):
        raise ValueError("buffer_capacity must be at least num_envs")
    if int(config["min_buffer_size"]) < 1:
        raise ValueError("min_buffer_size must be positive")
    if int(config["min_buffer_size"]) > int(config["buffer_capacity"]):
        raise ValueError("min_buffer_size cannot exceed buffer_capacity")
    for key in (
        "sample_batch_size",
        "updates_per_interaction",
        "update_chunk_size",
        "actor_update_period",
        "actor_hidden_dim",
        "critic_hidden_dim",
        "critic_num_bins",
        "noise_max_repeat",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(config["critic_num_bins"]) < 2:
        raise ValueError("critic_num_bins must be at least 2")
    if float(config["critic_min_value"]) >= float(config["critic_max_value"]):
        raise ValueError("critic_min_value must be smaller than critic_max_value")
    if not 0.0 < float(config["target_update_tau"]) <= 1.0:
        raise ValueError("target_update_tau must be in (0, 1]")
    if float(config["initial_temperature"]) <= 0.0:
        raise ValueError("initial_temperature must be positive")
    if float(config["target_sigma"]) <= 0.0:
        raise ValueError("target_sigma must be positive")
    if bool(config.get("normalize_env", False)):
        raise ValueError("FlashSAC requires normalize_env=false; it normalizes replay rewards itself")
    if bool(config.get("exact_resume", False)):
        raise ValueError("FlashSAC does not support exact_resume; use lightweight checkpoint resume")


@dataclass(frozen=True)
class FlashSACAgentConf(AgentConfBase):
    """Networks, optimizers, and static observation projections."""

    config: DictConfig
    actor: FlashSACActor
    critic: FlashSACDoubleCritic
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation
    temperature_optimizer: optax.GradientTransformation
    learning_rate_schedule: Any
    actor_observation_indices: jax.Array
    critic_observation_indices: jax.Array

    def serialize(self) -> dict[str, Any]:
        return {
            "config": OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True),
            "actor": flax.serialization.to_state_dict(self.actor),
            "critic": flax.serialization.to_state_dict(self.critic),
        }

    @classmethod
    def from_dict(cls, _data: dict[str, Any]) -> FlashSACAgentConf:
        raise NotImplementedError("FlashSAC uses Orbax experiment checkpoints")
