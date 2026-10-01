"""FlashSAC adapter for MuscleMimic experiments."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import optax
from omegaconf import open_dict

from musclemimic.algorithms.common.base_algorithm import JaxRLAlgorithmBase
from musclemimic.algorithms.common.env_utils import expand_obs_indices_for_history

from .checkpoint import load_checkpoint_for_resume, restore_train_state
from .config import FlashSACAgentConf, get_flashsac_config
from .networks import FlashSACActor, FlashSACDoubleCritic
from .state import (
    FlashSACAgentState,
    FlashSACLearnerState,
    FlashSACPolicyState,
    FlashSACTrainState,
)
from .train import build_training

if TYPE_CHECKING:
    from musclemimic.utils.metrics import MetricsHandler

logger = logging.getLogger(__name__)


class FlashSACJax(JaxRLAlgorithmBase):
    """Norm-constrained off-policy learning for vectorized MJX tasks."""

    _agent_conf = FlashSACAgentConf
    _agent_state = FlashSACAgentState

    @classmethod
    def init_agent_conf(cls, env: Any, config: Any) -> FlashSACAgentConf:
        if int(config.experiment.get("n_seeds", 1)) != 1:
            raise ValueError("FlashSAC supports one seed per process")

        experiment = get_flashsac_config(config.experiment)
        adaptive = experiment.get("adaptive_sampling", {})
        trajectory = experiment.get("trajectory", {}) or {}
        needs_trajectory_count = (
            adaptive.get("enabled", False)
            or str(trajectory.get("sharding", "none") or "none").lower() != "none"
        )
        if needs_trajectory_count:
            if experiment.get("n_trajectories") is None:
                base_env = env
                if hasattr(env, "unwrapped"):
                    unwrapped = env.unwrapped
                    base_env = unwrapped() if callable(unwrapped) else unwrapped
                if getattr(base_env, "th", None) is None:
                    raise ValueError(
                        "adaptive sampling or motion sharding requires an environment trajectory handler"
                    )
                with open_dict(experiment):
                    experiment.n_trajectories = int(base_env.th.n_trajectories)
        with open_dict(config):
            config.experiment = experiment

        actor_indices = cls._observation_indices(env, experiment, "actor_obs_group")
        critic_indices = cls._observation_indices(env, experiment, "critic_obs_group")
        action_dim = int(env.info.action_space.shape[0])
        actor = FlashSACActor(
            action_dim=action_dim,
            hidden_dim=int(experiment.actor_hidden_dim),
            num_blocks=int(experiment.actor_num_blocks),
        )
        critic = FlashSACDoubleCritic(
            hidden_dim=int(experiment.critic_hidden_dim),
            num_blocks=int(experiment.critic_num_blocks),
            num_bins=int(experiment.critic_num_bins),
            min_value=float(experiment.critic_min_value),
            max_value=float(experiment.critic_max_value),
        )

        schedule = cls._learning_rate_schedule(experiment)
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(float(experiment.max_grad_norm)),
            optax.adam(schedule),
        )
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(float(experiment.max_grad_norm)),
            optax.adam(schedule),
        )
        temperature_optimizer = optax.adam(schedule)

        logger.info(
            "[flashsac] interactions=%d, gradient_updates=%d, batch=%d, replay=%d",
            int(experiment.num_interactions),
            int(experiment.num_updates),
            int(experiment.sample_batch_size),
            int(experiment.buffer_capacity),
        )
        return FlashSACAgentConf(
            config=config,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temperature_optimizer=temperature_optimizer,
            learning_rate_schedule=schedule,
            actor_observation_indices=actor_indices,
            critic_observation_indices=critic_indices,
        )

    @staticmethod
    def _observation_indices(env: Any, experiment: Any, group_key: str) -> jax.Array:
        group = experiment.get(group_key)
        if group is None:
            indices = jnp.arange(env.mdp_info.observation_space.shape[0])
        else:
            indices = jnp.asarray(env.obs_container.get_obs_ind_by_group(group))
        if int(experiment.get("len_obs_history", 1)) > 1:
            indices = expand_obs_indices_for_history(indices, env, experiment)
        return indices

    @staticmethod
    def _learning_rate_schedule(experiment: Any) -> optax.Schedule:
        total_updates = max(1, int(experiment.num_updates))
        warmup_updates = int(experiment.warmup_updates)
        peak = float(experiment.learning_rate)
        end = float(experiment.end_learning_rate)
        if peak <= 0.0 or end < 0.0:
            raise ValueError("FlashSAC learning rates must be non-negative and peak must be positive")
        if warmup_updates > 0:
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=peak,
                warmup_steps=warmup_updates,
                decay_steps=max(total_updates, warmup_updates + 1),
                end_value=end,
            )
        return optax.cosine_decay_schedule(
            init_value=peak,
            decay_steps=total_updates,
            alpha=end / peak,
        )

    @classmethod
    def build_training(
        cls,
        env: Any,
        agent_conf: FlashSACAgentConf,
        *,
        metrics_handler: MetricsHandler | None = None,
        resume_info: dict[str, Any] | None = None,
        val_env: Any | None = None,
        preserve_checkpoint_target: bool = False,
    ) -> Any:
        return build_training(
            env,
            agent_conf,
            metrics_handler=metrics_handler,
            resume_info=resume_info,
            val_env=val_env,
            preserve_checkpoint_target=preserve_checkpoint_target,
        )

    @classmethod
    def build_resumed_training(
        cls,
        env: Any,
        agent_conf: FlashSACAgentConf,
        checkpoint_path: str,
        metrics_handler: MetricsHandler | None = None,
        val_env: Any | None = None,
        apply_resume_resets: bool = True,
        preserve_checkpoint_target: bool = False,
        runtime_state_target: Any | None = None,
    ) -> tuple[Any, FlashSACLearnerState | FlashSACTrainState]:
        if not apply_resume_resets or runtime_state_target is not None:
            raise ValueError("FlashSAC supports lightweight resume only")
        agent_state, resume_info = load_checkpoint_for_resume(
            checkpoint_path,
            agent_conf,
        )
        training = cls.build_training(
            env,
            agent_conf,
            metrics_handler=metrics_handler,
            resume_info=resume_info,
            val_env=val_env,
            preserve_checkpoint_target=preserve_checkpoint_target,
        )
        return training, agent_state.train_state

    @classmethod
    def build_runtime_state_restore_target(cls, *_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("FlashSAC does not support exact_resume")

    @classmethod
    def build_inference_state(
        cls,
        policy_state: dict[str, Any],
        agent_conf: FlashSACAgentConf,
    ) -> FlashSACAgentState:
        """Build actor-only state for evaluation and validation video."""
        del agent_conf
        return FlashSACAgentState(
            train_state=FlashSACPolicyState(
                actor_params=policy_state["actor_params"],
                actor_batch_stats=policy_state["actor_batch_stats"],
            )
        )

    @classmethod
    def restore_agent_state(
        cls,
        state_data: dict[str, Any],
        agent_conf: FlashSACAgentConf,
    ) -> FlashSACAgentState:
        """Restore complete FlashSAC state for standalone evaluation."""
        return FlashSACAgentState(
            train_state=restore_train_state(state_data, agent_conf),
        )

    @classmethod
    def play_policy(
        cls,
        env: Any,
        agent_conf: FlashSACAgentConf,
        agent_state: FlashSACAgentState,
        n_envs: int,
        n_steps: int | None = None,
        render: bool = True,
        record: bool = False,
        rng: jax.Array | None = None,
        deterministic: bool = False,
        use_mujoco: bool = False,
        wrap_env: bool = True,
        train_state_seed: int | None = None,
        sequential_mjx: bool = False,
    ) -> None:
        from musclemimic.algorithms.common.inference import play_policy

        play_policy(
            env,
            agent_conf,
            agent_state,
            n_envs,
            n_steps,
            render,
            record,
            rng,
            deterministic,
            use_mujoco,
            wrap_env,
            train_state_seed,
            sequential_mjx,
        )

    @classmethod
    def play_policy_mujoco(
        cls,
        env: Any,
        agent_conf: FlashSACAgentConf,
        agent_state: FlashSACAgentState,
        n_steps: int | None = None,
        render: bool = True,
        record: bool = False,
        rng: jax.Array | None = None,
        deterministic: bool = False,
        train_state_seed: int | None = None,
    ) -> None:
        cls.play_policy(
            env,
            agent_conf,
            agent_state,
            n_envs=1,
            n_steps=n_steps,
            render=render,
            record=record,
            rng=rng,
            deterministic=deterministic,
            use_mujoco=True,
            wrap_env=False,
            train_state_seed=train_state_seed,
        )
