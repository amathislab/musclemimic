"""Dynamic state carried by FlashSAC training."""

from __future__ import annotations

from typing import Any

import flax
import jax
from flax import struct

from musclemimic.algorithms.common.base_algorithm import AgentStateBase

from .replay_buffer import ReplayBuffer


@struct.dataclass
class FlashSACLearnerState:
    """FlashSAC learning state without device-resident replay storage."""

    actor_params: Any
    actor_batch_stats: Any
    actor_optimizer_state: Any
    critic_params: Any
    critic_batch_stats: Any
    critic_optimizer_state: Any
    target_critic_params: Any
    target_critic_batch_stats: Any
    log_temperature: jax.Array
    temperature_optimizer_state: Any
    reward_return_mean: jax.Array
    reward_return_var: jax.Array
    reward_return_count: jax.Array
    reward_return_max: jax.Array
    update_step: jax.Array

    def with_replay_buffer(self, replay_buffer: ReplayBuffer) -> FlashSACTrainState:
        """Attach replay storage initialized on the active training shard."""
        return FlashSACTrainState(
            actor_params=self.actor_params,
            actor_batch_stats=self.actor_batch_stats,
            actor_optimizer_state=self.actor_optimizer_state,
            critic_params=self.critic_params,
            critic_batch_stats=self.critic_batch_stats,
            critic_optimizer_state=self.critic_optimizer_state,
            target_critic_params=self.target_critic_params,
            target_critic_batch_stats=self.target_critic_batch_stats,
            log_temperature=self.log_temperature,
            temperature_optimizer_state=self.temperature_optimizer_state,
            replay_buffer=replay_buffer,
            reward_return_mean=self.reward_return_mean,
            reward_return_var=self.reward_return_var,
            reward_return_count=self.reward_return_count,
            reward_return_max=self.reward_return_max,
            update_step=self.update_step,
        )


@struct.dataclass
class FlashSACTrainState:
    """Actor, critics, optimizers, replay, and reward statistics."""

    actor_params: Any
    actor_batch_stats: Any
    actor_optimizer_state: Any
    critic_params: Any
    critic_batch_stats: Any
    critic_optimizer_state: Any
    target_critic_params: Any
    target_critic_batch_stats: Any
    log_temperature: jax.Array
    temperature_optimizer_state: Any
    replay_buffer: ReplayBuffer
    reward_return_mean: jax.Array
    reward_return_var: jax.Array
    reward_return_count: jax.Array
    reward_return_max: jax.Array
    update_step: jax.Array

    def without_replay_buffer(self) -> FlashSACLearnerState:
        """Return the learner fields that can be checkpointed independently."""
        return FlashSACLearnerState(
            actor_params=self.actor_params,
            actor_batch_stats=self.actor_batch_stats,
            actor_optimizer_state=self.actor_optimizer_state,
            critic_params=self.critic_params,
            critic_batch_stats=self.critic_batch_stats,
            critic_optimizer_state=self.critic_optimizer_state,
            target_critic_params=self.target_critic_params,
            target_critic_batch_stats=self.target_critic_batch_stats,
            log_temperature=self.log_temperature,
            temperature_optimizer_state=self.temperature_optimizer_state,
            reward_return_mean=self.reward_return_mean,
            reward_return_var=self.reward_return_var,
            reward_return_count=self.reward_return_count,
            reward_return_max=self.reward_return_max,
            update_step=self.update_step,
        )


@struct.dataclass
class FlashSACRuntimeState:
    """Environment and exploration state carried between interactions."""

    agent: FlashSACTrainState
    env_state: Any
    observation: jax.Array
    exploration_noise: jax.Array
    noise_steps_remaining: jax.Array
    rng: jax.Array
    validation_rng: jax.Array
    interaction_step: jax.Array
    reward_ema: jax.Array
    discounted_return: jax.Array
    last_episode_return: jax.Array
    last_episode_length: jax.Array


@struct.dataclass
class FlashSACPolicyState:
    """Actor-only state used by evaluation processes."""

    actor_params: Any
    actor_batch_stats: Any


@struct.dataclass
class FlashSACAgentState(AgentStateBase):
    """Serializable FlashSAC learning state."""

    train_state: Any

    def serialize(self, *, include_replay_buffer: bool = True) -> dict[str, Any]:
        """Serialize learner state, optionally excluding replay array storage."""
        train_state = self.train_state
        if not include_replay_buffer and isinstance(train_state, FlashSACTrainState):
            train_state = train_state.without_replay_buffer()
        state_dict = flax.serialization.to_state_dict(train_state)
        if not include_replay_buffer:
            state_dict.pop("replay_buffer", None)
        return state_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any], agent_conf: Any) -> FlashSACAgentState:
        from .checkpoint import restore_train_state

        return cls(train_state=restore_train_state(data, agent_conf))
