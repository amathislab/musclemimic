"""
PPO configuration and state dataclasses.
"""

from dataclasses import dataclass
from typing import Any

import flax
from flax import struct
from omegaconf import DictConfig, OmegaConf

from musclemimic.algorithms import AgentConfBase, AgentStateBase
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.algorithms.common.networks import ActorCritic

# PPO-specific parameters
PPO_PARAMS = (
    "num_steps",
    "update_epochs",
    "num_minibatches",
    "gamma",
    "gae_lambda",
    "clip_eps",
    "clip_eps_vf",
    "init_std",
    "learnable_std",
    "ent_coef",
    "vf_coef",
)


def get_ppo_config(exp: DictConfig) -> DictConfig:
    """Get PPO-specific config with backward compatibility.

    Merges ppo_config section into experiment config for backward compatibility.
    Top-level values take precedence (for variant overrides), with ppo_config
    providing defaults.

    Args:
        exp: The experiment config (config.experiment)

    Returns:
        DictConfig with PPO params accessible at top level
    """
    ppo_cfg = exp.get("ppo_config", None)
    if ppo_cfg is None:
        # If params are at top level, take exp as is
        return exp

    # New-style config: merge ppo_config values into a copy of exp
    # This allows code to access exp.gamma instead of exp.ppo_config.gamma
    merged = OmegaConf.to_container(exp, resolve=True)
    ppo_dict = OmegaConf.to_container(ppo_cfg, resolve=True)

    # ppo_config provides defaults; top-level values override (for variant configs)
    for param in PPO_PARAMS:
        if param in ppo_dict and param not in merged:
            # Only use ppo_config value if not set at top level
            merged[param] = ppo_dict[param]
        elif param in ppo_dict and param in merged:
            # Top-level exists - keep it (variant override takes precedence)
            pass

    # Compute derived values before creating config
    if "total_timesteps" in merged and "num_steps" in merged and "num_envs" in merged:
        merged["num_updates"] = merged["total_timesteps"] // merged["num_steps"] // merged["num_envs"]
        merged["minibatch_size"] = merged["num_envs"] * merged["num_steps"] // merged.get("num_minibatches", 1)

    # Return with struct=False to allow adding more derived values later
    cfg = OmegaConf.create(merged)
    OmegaConf.set_struct(cfg, False)
    return cfg


@dataclass(frozen=True)
class PPOAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCritic
    tx: Any

    def serialize(self):
        """
        Serialize the agent configuration and network configuration.

        Returns:
            Serialized agent configuration as a dictionary.
        """
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_network = flax.serialization.to_state_dict(self.network)
        return {"config": conf_dict, "network": serialized_network}

    @classmethod
    def from_dict(cls, d):
        # Import here to avoid circular dependency
        from musclemimic.algorithms.common.optimizer import get_optimizer

        config = OmegaConf.create(d["config"])
        tx = get_optimizer(config.experiment)
        return cls(
            config=config,
            network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
            tx=tx,
        )


@struct.dataclass
class PPOAgentState(AgentStateBase):
    train_state: TrainState

    def serialize(self):
        serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        return {"train_state": serialized_train_state}

    @classmethod
    def from_dict(cls, d, agent_conf):
        train_state = TrainState(
            apply_fn=agent_conf.network,
            tx=agent_conf.tx,
            **d["train_state"],
        )
        return cls(train_state)
