"""Pure FlashSAC initialization and interaction functions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf, open_dict

from musclemimic.algorithms.common.adaptive_sampling import (
    compute_adaptive_weights,
    compute_per_traj_difficulty_stats,
)
from musclemimic.algorithms.common.distributed import (
    DATA_AXIS_NAME,
    device_count,
    mark_data_varying,
)
from musclemimic.algorithms.common.distributed import (
    pmax as _max,
)
from musclemimic.algorithms.common.distributed import (
    pmean as _mean,
)
from musclemimic.algorithms.common.distributed import (
    psum as _sum,
)
from musclemimic.algorithms.common.env_state_utils import (
    get_carry_normalized,
    get_carry_unnormalized,
    preserve_data_axis_variation,
    update_carry_ema_normalized,
    update_carry_ema_unnormalized,
    update_carry_weights_normalized,
    update_carry_weights_unnormalized,
)
from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.algorithms.common.trajectory_sampling import (
    reset_from_trajectory_weights,
)
from musclemimic.algorithms.common.trajectory_sharding import (
    ShardSpec,
    apply_shard_mask_and_renormalize,
    trajectory_sharding_mode,
    with_trajectory_data_overrides,
)
from musclemimic.core.wrappers import LogEnvState

from .config import FlashSACAgentConf
from .evaluation import run_validation_step
from .networks import normalize_parameters
from .replay_buffer import create_replay_buffer
from .reward_normalization import update_discounted_return
from .state import (
    FlashSACLearnerState,
    FlashSACPolicyState,
    FlashSACRuntimeState,
    FlashSACTrainState,
)
from .update import FlashSACUpdateMetrics, update_from_batch

if TYPE_CHECKING:
    from musclemimic.utils.metrics import MetricsHandler


REWARD_INFO_TO_LOG_KEY = (
    ("reward_total", "reward/total"),
    ("reward_qpos", "reward/qpos"),
    ("reward_qvel", "reward/qvel"),
    ("reward_root_pos", "reward/root_pos"),
    ("reward_root_quat", "reward/root_quat"),
    ("reward_rpos", "reward/rpos"),
    ("reward_rquat", "reward/rquat"),
    ("reward_rvel_rot", "reward/rvel_rot"),
    ("reward_rvel_lin", "reward/rvel_lin"),
    ("reward_root_vel", "reward/root_vel"),
    ("penalty_total", "reward/penalty"),
    ("penalty_activation_energy", "reward/penalty_activation_energy"),
    ("err_root_xyz", "err/root_xyz"),
    ("err_root_quat", "err/root_quat"),
    ("err_root_yaw", "err/root_yaw"),
    ("err_joint_pos", "err/joint_pos"),
    ("err_joint_vel", "err/joint_vel"),
    ("err_site_abs", "err/site_abs"),
    ("err_rpos", "err/rpos"),
)
REWARD_NORMALIZER_EPSILON = 1e-4


def _install_adaptive_sampling_groups(config: Any, env: Any) -> None:
    """Copy optional environment motion families into the sampler config."""
    adaptive = config.get("adaptive_sampling", {})
    trajectory_group_ids = getattr(env, "trajectory_group_ids", None)
    if trajectory_group_ids is None or not bool(adaptive.get("balance_motion_groups", True)):
        return
    if len(trajectory_group_ids) != int(env.th.n_trajectories):
        raise ValueError("trajectory_group_ids must contain one group id per trajectory")
    trajectory_group_names = getattr(env, "trajectory_group_names", None)
    with open_dict(config.adaptive_sampling):
        config.adaptive_sampling.group_ids = list(trajectory_group_ids)
        config.adaptive_sampling.group_names = list(trajectory_group_names or ("all",))


class FlashSACCollectionMetrics(NamedTuple):
    mean_reward: jax.Array
    done_count: jax.Array
    absorbing_count: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    reward_components: dict[str, jax.Array]


class FlashSACStep(NamedTuple):
    runtime_state: FlashSACRuntimeState
    collection_metrics: FlashSACCollectionMetrics
    update_metrics: FlashSACUpdateMetrics
    learning_rate: jax.Array


class FlashSACChunk(NamedTuple):
    """Outputs from several interactions executed in one compiled scan."""

    runtime_state: FlashSACRuntimeState
    collection_metrics: FlashSACCollectionMetrics
    update_metrics: FlashSACUpdateMetrics
    reward_ema: jax.Array
    replay_size: jax.Array
    learning_rate: jax.Array


@dataclass(frozen=True)
class FlashSACProgress:
    completed_interactions: int
    remaining_interactions: int
    base_global_timestep: int
    target_global_timestep: int


def _aligned_interaction_target(
    base_timestep: int,
    requested_target: int,
    num_envs: int,
) -> int:
    """Align a timestep target to a reachable vectorized interaction."""
    remaining = max(0, requested_target - base_timestep)
    return base_timestep + remaining // num_envs * num_envs


def initialize_learner_state(
    rng: jax.Array,
    agent_conf: FlashSACAgentConf,
    config: Any | None = None,
) -> FlashSACLearnerState:
    """Initialize networks, optimizers, and reward statistics."""
    config = agent_conf.config.experiment if config is None else config
    actor_rng, critic_rng = jax.random.split(rng)
    actor_observation = jnp.zeros((2, agent_conf.actor_observation_indices.size), dtype=jnp.float32)
    critic_observation = jnp.zeros((2, agent_conf.critic_observation_indices.size), dtype=jnp.float32)
    action = jnp.zeros((2, agent_conf.actor.action_dim), dtype=jnp.float32)

    actor_variables = agent_conf.actor.init(actor_rng, actor_observation, training=False)
    critic_variables = agent_conf.critic.init(critic_rng, critic_observation, action, training=False)
    actor_params = normalize_parameters(actor_variables["params"])
    critic_params = normalize_parameters(critic_variables["params"])
    log_temperature = jnp.asarray(math.log(float(config.initial_temperature)), dtype=jnp.float32)

    return FlashSACLearnerState(
        actor_params=actor_params,
        actor_batch_stats=actor_variables["batch_stats"],
        actor_optimizer_state=agent_conf.actor_optimizer.init(actor_params),
        critic_params=critic_params,
        critic_batch_stats=critic_variables["batch_stats"],
        critic_optimizer_state=agent_conf.critic_optimizer.init(critic_params),
        target_critic_params=critic_params,
        target_critic_batch_stats=critic_variables["batch_stats"],
        log_temperature=log_temperature,
        temperature_optimizer_state=agent_conf.temperature_optimizer.init(log_temperature),
        reward_return_mean=jnp.asarray(0.0, dtype=jnp.float32),
        reward_return_var=jnp.asarray(1.0, dtype=jnp.float32),
        reward_return_count=jnp.asarray(0.0, dtype=jnp.float32),
        reward_return_max=jnp.asarray(0.0, dtype=jnp.float32),
        update_step=jnp.asarray(0, dtype=jnp.int32),
    )


def initialize_agent_state(
    rng: jax.Array,
    observation_dim: int,
    action_dim: int,
    agent_conf: FlashSACAgentConf,
    config: Any | None = None,
) -> FlashSACTrainState:
    """Initialize networks, optimizers, replay, and reward statistics."""
    config = agent_conf.config.experiment if config is None else config
    learner = initialize_learner_state(
        rng,
        agent_conf,
        config,
    )
    replay_buffer = create_replay_buffer(
        capacity=int(config.buffer_capacity),
        observation_dim=observation_dim,
        action_dim=action_dim,
        storage_dtype=str(config.replay_storage_dtype),
    )
    return learner.with_replay_buffer(replay_buffer)


def _zeta_log_probabilities(exponent: float, max_repeat: int) -> jax.Array:
    repeats = jnp.arange(1, max_repeat + 1, dtype=jnp.float32)
    probabilities = repeats ** (-exponent)
    return jnp.log(probabilities / jnp.sum(probabilities))


def select_collection_actions(
    replay_size: jax.Array,
    min_buffer_size: int,
    policy_actions: jax.Array,
    uniform_rng: jax.Array,
) -> jax.Array:
    """Use uniform exploration until replay is ready to train."""
    uniform_actions = jax.random.uniform(
        uniform_rng,
        policy_actions.shape,
        minval=-1.0,
        maxval=1.0,
        dtype=policy_actions.dtype,
    )
    return jnp.where(replay_size < min_buffer_size, uniform_actions, policy_actions)


def mean_reward_components(
    info: dict[str, jax.Array],
    axis_name: str | None = None,
) -> dict[str, jax.Array]:
    """Average the reward diagnostics shared with PPO logging."""
    metrics = {
        log_key: jnp.mean(info[info_key])
        for info_key, log_key in REWARD_INFO_TO_LOG_KEY
        if info_key in info
    }
    if axis_name is not None:
        metrics = jax.tree.map(lambda value: _mean(value, axis_name), metrics)
    return metrics


@dataclass(frozen=True, eq=False)
class FlashSACTraining:
    """Static inputs for pure FlashSAC computations."""

    env: Any
    val_env: Any | None
    agent_conf: FlashSACAgentConf
    config: Any
    metrics_handler: MetricsHandler | None
    resume_info: dict[str, Any] | None
    preserve_checkpoint_target: bool
    axis_name: str | None
    shard_spec: ShardSpec | None
    validation_shard_spec: ShardSpec | None
    validation_shares_training_trajectory: bool

    def initialize(
        self,
        rng: jax.Array,
        agent_state_data: FlashSACLearnerState | FlashSACTrainState | None = None,
        trajectory_data_override: Any | None = None,
    ) -> FlashSACRuntimeState:
        """Initialize a fresh environment around fresh or restored learning state."""
        with with_trajectory_data_overrides(
            self.env,
            self.val_env,
            trajectory_data_override,
            None,
        ):
            env = wrap_env(self.env, self.config, axis_name=self.axis_name)
            observation_dim = int(env.info.observation_space.shape[0])
            action_dim = int(env.info.action_space.shape[0])
            rng, agent_rng = jax.random.split(rng)
            if agent_state_data is None:
                agent = initialize_agent_state(
                    agent_rng,
                    observation_dim,
                    action_dim,
                    self.agent_conf,
                    self.config,
                )
            elif isinstance(agent_state_data, FlashSACLearnerState):
                agent = agent_state_data.with_replay_buffer(
                    create_replay_buffer(
                        capacity=int(self.config.buffer_capacity),
                        observation_dim=observation_dim,
                        action_dim=action_dim,
                        storage_dtype=str(self.config.replay_storage_dtype),
                    )
                )
            else:
                agent = agent_state_data
            if self.axis_name is not None:
                rng = jax.random.fold_in(rng, jax.lax.axis_index(self.axis_name))
            reset_rng, training_rng, validation_rng = jax.random.split(rng, 3)
            reset_keys = jax.random.split(reset_rng, int(self.config.num_envs))
            reset_weights = self._training_reset_weights()
            if reset_weights is None:
                observations, env_state = env.reset(reset_keys)
            else:
                observations, env_state = reset_from_trajectory_weights(
                    env,
                    reset_keys,
                    reset_weights,
                )
            adaptive = self.config.get("adaptive_sampling", {})
            if adaptive.get("enabled", False):
                num_trajectories = int(self.config.n_trajectories)
                done_ema = jnp.broadcast_to(
                    jnp.full(
                        (num_trajectories,),
                        adaptive.get("ema_done_init", 10.0),
                    ),
                    (int(self.config.num_envs), num_trajectories),
                )
                early_ema = jnp.broadcast_to(
                    jnp.full(
                        (num_trajectories,),
                        adaptive.get("ema_early_init", 5.0),
                    ),
                    (int(self.config.num_envs), num_trajectories),
                )
                if self.config.normalize_env:
                    env_state = update_carry_ema_normalized(env_state, done_ema, early_ema)
                else:
                    env_state = update_carry_ema_unnormalized(env_state, done_ema, early_ema)
            env_state = mark_data_varying(env_state, self.axis_name)
            completed = 0 if self.resume_info is None else int(self.resume_info["update_number"])
            return FlashSACRuntimeState(
                agent=agent,
                env_state=env_state,
                observation=observations,
                exploration_noise=jnp.zeros((int(self.config.num_envs), action_dim), dtype=jnp.float32),
                noise_steps_remaining=jnp.zeros((int(self.config.num_envs),), dtype=jnp.int32),
                rng=training_rng,
                validation_rng=validation_rng,
                interaction_step=jnp.asarray(completed, dtype=jnp.int32),
                reward_ema=jnp.asarray(0.0, dtype=jnp.float32),
                discounted_return=jnp.zeros((int(self.config.num_envs),), dtype=jnp.float32),
                last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
                last_episode_length=jnp.asarray(0.0, dtype=jnp.float32),
            )

    def update(
        self,
        runtime_state: FlashSACRuntimeState,
        trajectory_data_override: Any | None = None,
    ) -> FlashSACStep:
        """Collect one vectorized interaction and run configured gradient updates."""
        with with_trajectory_data_overrides(
            self.env,
            self.val_env,
            trajectory_data_override,
            None,
        ):
            env = wrap_env(self.env, self.config, axis_name=self.axis_name)
            return self._update(runtime_state, env)

    def update_chunk(
        self,
        runtime_state: FlashSACRuntimeState,
        trajectory_data_override: Any | None,
        num_interactions: jax.Array,
        max_interactions: int,
    ) -> FlashSACChunk:
        """Run up to ``max_interactions`` without returning to the host."""
        if max_interactions < 1:
            raise ValueError("max_interactions must be positive")

        with with_trajectory_data_overrides(
            self.env,
            self.val_env,
            trajectory_data_override,
            None,
        ):
            env = wrap_env(self.env, self.config, axis_name=self.axis_name)

            def run_interaction(state: FlashSACRuntimeState):
                step = self._update(state, env)
                next_state = step.runtime_state
                metrics = (
                    step.collection_metrics,
                    step.update_metrics,
                    next_state.reward_ema,
                    next_state.agent.replay_buffer.size,
                    step.learning_rate,
                )
                return next_state, metrics

            runtime_state, first_metrics = run_interaction(runtime_state)
            zero_metrics = jax.tree.map(jnp.zeros_like, first_metrics)

            def scan_interaction(state, interaction_index):
                return jax.lax.cond(
                    interaction_index < num_interactions,
                    run_interaction,
                    lambda current_state: (current_state, zero_metrics),
                    state,
                )

            runtime_state, remaining_metrics = jax.lax.scan(
                scan_interaction,
                runtime_state,
                xs=jnp.arange(1, max_interactions),
            )
            metrics = jax.tree.map(
                lambda first, remaining: jnp.concatenate(
                    (first[None], remaining),
                    axis=0,
                ),
                first_metrics,
                remaining_metrics,
            )
            collection_metrics, update_metrics, reward_ema, replay_size, learning_rate = metrics
            return FlashSACChunk(
                runtime_state=runtime_state,
                collection_metrics=collection_metrics,
                update_metrics=update_metrics,
                reward_ema=reward_ema,
                replay_size=replay_size,
                learning_rate=learning_rate,
            )

    def validate(
        self,
        policy_state: FlashSACPolicyState | FlashSACTrainState,
        validation_rng: jax.Array,
        trajectory_data_override: Any | None = None,
        validation_trajectory_data_override: Any | None = None,
    ) -> Any:
        """Run one validation rollout and advance its independent RNG."""
        if self.metrics_handler is None:
            raise RuntimeError("FlashSAC validation requires a metrics handler")
        with with_trajectory_data_overrides(
            self.env,
            self.val_env,
            trajectory_data_override,
            validation_trajectory_data_override,
        ):
            validation_env = self.val_env if self.val_env is not None else self.env
            validation_env = wrap_env(
                validation_env,
                self.config,
                axis_name=self.axis_name,
                normalize_reward=False,
            )
            trajectory_ids = None
            trajectory_lengths = None
            trajectory_mask = None
            if bool(self.config.validation.get("evaluate_all", False)):
                if self.validation_shard_spec is None or self.axis_name is None:
                    raise ValueError("Exact validation requires a distinct trajectory dataset.")
                device_id = jax.lax.axis_index(self.axis_name)
                offsets = jnp.asarray(
                    self.validation_shard_spec.offsets,
                    dtype=jnp.int32,
                )
                counts = jnp.asarray(
                    self.validation_shard_spec.counts,
                    dtype=jnp.int32,
                )
                local_slot = jnp.arange(
                    int(self.config.validation.num_envs),
                    dtype=jnp.int32,
                )
                local_count = counts[device_id]
                trajectory_mask = local_slot < local_count
                first_trajectory = offsets[device_id]
                trajectory_ids = jnp.where(
                    trajectory_mask,
                    first_trajectory + local_slot,
                    first_trajectory,
                )
                trajectory_lengths = validation_env.th.len_trajectory(trajectory_ids)
            return run_validation_step(
                self.agent_conf,
                policy_state,
                validation_rng,
                validation_env,
                self.config,
                self.metrics_handler,
                reset_weights=self._validation_reset_weights(),
                trajectory_ids=trajectory_ids,
                trajectory_lengths=trajectory_lengths,
                trajectory_mask=trajectory_mask,
                axis_name=self.axis_name,
            )

    def _training_reset_weights(self) -> jax.Array | None:
        adaptive = self.config.get("adaptive_sampling", {})
        if self.shard_spec is None and not adaptive.get("enabled", False):
            return None
        if self.shard_spec is None:
            count = int(self.config.n_trajectories)
            return jnp.full((count,), 1.0 / count)
        if self.axis_name is None:
            raise ValueError("axis_name is required when a trajectory shard is configured")
        return self.shard_spec.uniform_weights_for_device(jax.lax.axis_index(self.axis_name))

    def _update_adaptive_sampling(
        self,
        env_state: Any,
        transition_state: Any,
        done: jax.Array,
        absorbing: jax.Array,
        info: dict[str, jax.Array],
        env: Any,
    ) -> Any:
        """Update TERRA's progress-aware motion sampler after one interaction."""
        adaptive = self.config.get("adaptive_sampling", {})
        if not adaptive.get("enabled", False):
            return env_state

        num_trajectories = int(self.config.n_trajectories)
        trajectory_state = transition_state.additional_carry.traj_state
        split_points = jnp.asarray(env.th.traj.data.split_points)
        trajectory_length = (
            split_points[trajectory_state.traj_no + 1]
            - split_points[trajectory_state.traj_no]
        )
        trajectory_end = trajectory_state.subtraj_step_no >= trajectory_length - 1
        excluded = jnp.asarray(
            info.get("numerics_nonfinite_transition", jnp.zeros_like(done)),
            dtype=bool,
        )
        _, done_counts, difficulty_counts = compute_per_traj_difficulty_stats(
            trajectory_state,
            done,
            absorbing,
            trajectory_end,
            split_points,
            num_trajectories,
            late_failure_floor=adaptive.get("late_failure_floor", 0.25),
            excluded_transitions=excluded,
        )
        done_counts = _sum(done_counts, self.axis_name)
        difficulty_counts = _sum(difficulty_counts, self.axis_name)

        carry = (
            get_carry_normalized(env_state)
            if self.config.normalize_env
            else get_carry_unnormalized(env_state)
        )
        group_ids_config = adaptive.get("group_ids", None)
        group_ids = None
        num_groups = 1
        if group_ids_config is not None:
            group_ids_tuple = tuple(int(value) for value in group_ids_config)
            num_groups = max(group_ids_tuple) + 1
            if set(group_ids_tuple) != set(range(num_groups)):
                raise ValueError("adaptive sampling group ids must be contiguous from zero")
            group_ids = jnp.asarray(group_ids_tuple, dtype=jnp.int32)

        weights, new_ema_done, new_ema_early, _rates = compute_adaptive_weights(
            done_counts,
            difficulty_counts,
            carry.ema_done_counts[0],
            carry.ema_early_counts[0],
            beta=adaptive.get("beta", 0.2),
            alpha=adaptive.get("alpha", 1.0),
            floor_mix=adaptive.get("floor_mix", 0.1),
            eps_div=adaptive.get("eps_div", 1e-6),
            eps_pow=adaptive.get("eps_pow", 1e-6),
            group_ids=group_ids,
            num_groups=num_groups,
            difficulty_mix=adaptive.get("difficulty_mix", None),
            max_weight_ratio=adaptive.get("max_weight_ratio", None),
        )
        if self.shard_spec is not None:
            if self.axis_name is None:
                raise ValueError("axis_name is required when a trajectory shard is configured")
            weights = apply_shard_mask_and_renormalize(
                weights,
                self.shard_spec,
                jax.lax.axis_index(self.axis_name),
            )
        batched_weights = jnp.broadcast_to(
            weights,
            (int(self.config.num_envs), num_trajectories),
        )
        batched_done_ema = jnp.broadcast_to(new_ema_done, batched_weights.shape)
        batched_early_ema = jnp.broadcast_to(new_ema_early, batched_weights.shape)
        if self.config.normalize_env:
            env_state = update_carry_weights_normalized(env_state, batched_weights)
            return update_carry_ema_normalized(
                env_state,
                batched_done_ema,
                batched_early_ema,
            )
        env_state = update_carry_weights_unnormalized(env_state, batched_weights)
        return update_carry_ema_unnormalized(
            env_state,
            batched_done_ema,
            batched_early_ema,
        )

    def _validation_reset_weights(self) -> jax.Array | None:
        if self.validation_shares_training_trajectory:
            return self._training_reset_weights()
        if self.validation_shard_spec is None:
            return None
        if self.axis_name is None:
            raise ValueError("axis_name is required when a trajectory shard is configured")
        return self.validation_shard_spec.uniform_weights_for_device(
            jax.lax.axis_index(self.axis_name),
        )

    def progress(self) -> FlashSACProgress:
        """Compute remaining interactions for a fresh or resumed run."""
        num_envs = int(self.config.global_num_envs)
        if self.resume_info is None:
            target = _aligned_interaction_target(
                0,
                int(self.config.total_timesteps),
                num_envs,
            )
            return FlashSACProgress(
                completed_interactions=0,
                remaining_interactions=int(self.config.num_interactions),
                base_global_timestep=0,
                target_global_timestep=target,
            )

        completed = int(self.resume_info["update_number"])
        base_timestep = int(self.resume_info["global_timestep"])
        stored_target = int(self.resume_info.get("target_global_timestep", 0) or 0)
        if bool(self.config.get("total_timesteps_is_absolute", False)):
            target = int(self.config.total_timesteps)
        elif self.preserve_checkpoint_target and stored_target > 0:
            target = stored_target
        else:
            target = base_timestep + int(self.config.total_timesteps)
        target = _aligned_interaction_target(base_timestep, target, num_envs)
        remaining = (target - base_timestep) // num_envs
        return FlashSACProgress(completed, remaining, base_timestep, target)

    def _update(self, state: FlashSACRuntimeState, env: Any) -> FlashSACStep:
        config = self.config
        noise_rng, repeat_rng, uniform_rng, update_rng = jax.random.split(state.rng, 4)
        actor_observations = state.observation[:, self.agent_conf.actor_observation_indices]
        mean, standard_deviation = self.agent_conf.actor.apply(
            {
                "params": state.agent.actor_params,
                "batch_stats": state.agent.actor_batch_stats,
            },
            actor_observations,
            training=False,
        )

        refresh_noise = state.noise_steps_remaining <= 0
        sampled_noise = jax.random.normal(noise_rng, mean.shape)
        sampled_repeats = (
            jax.random.categorical(
                repeat_rng,
                _zeta_log_probabilities(
                    float(config.noise_zeta_exponent),
                    int(config.noise_max_repeat),
                ),
                shape=(int(config.num_envs),),
            ).astype(jnp.int32)
            + 1
        )
        exploration_noise = jnp.where(refresh_noise[:, None], sampled_noise, state.exploration_noise)
        repeat_count = jnp.where(refresh_noise, sampled_repeats, state.noise_steps_remaining)
        policy_actions = jnp.tanh(mean + standard_deviation * exploration_noise)
        actions = select_collection_actions(
            state.agent.replay_buffer.size,
            int(config.min_buffer_size),
            policy_actions,
            uniform_rng,
        )

        (
            next_observations,
            rewards,
            absorbing,
            done,
            info,
            env_state,
            transition_state,
            transition_observations,
        ) = env.step_with_transition(state.env_state, actions)
        env_state = preserve_data_axis_variation(
            env_state,
            state.env_state,
            self.axis_name,
        )
        env_state = self._update_adaptive_sampling(
            env_state,
            transition_state,
            done,
            absorbing,
            info,
            env,
        )
        terminated = absorbing
        if getattr(env, "th", None) is not None:
            trajectory_state = transition_state.additional_carry.traj_state
            split_points = jnp.asarray(env.th.traj.data.split_points)
            trajectory_length = (
                split_points[trajectory_state.traj_no + 1]
                - split_points[trajectory_state.traj_no]
            )
            trajectory_end = trajectory_state.subtraj_step_no >= trajectory_length - 1
            terminated = jnp.logical_or(absorbing, trajectory_end)
        replay_buffer = state.agent.replay_buffer.add(
            state.observation,
            actions,
            rewards,
            transition_observations,
            terminated,
        )

        discounted_return_samples, next_discounted_return = update_discounted_return(
            state.discounted_return,
            rewards,
            done,
            gamma=float(config.gamma),
        )
        batch_mean = _mean(jnp.mean(discounted_return_samples), self.axis_name)
        batch_second_moment = _mean(
            jnp.mean(jnp.square(discounted_return_samples)),
            self.axis_name,
        )
        batch_variance = jnp.maximum(
            batch_second_moment - jnp.square(batch_mean),
            0.0,
        )
        batch_count = float(config.global_num_envs)
        total_count = state.agent.reward_return_count + batch_count
        delta = batch_mean - state.agent.reward_return_mean
        return_mean = state.agent.reward_return_mean + delta * batch_count / total_count
        first_moment = state.agent.reward_return_var * (state.agent.reward_return_count + REWARD_NORMALIZER_EPSILON)
        second_moment = batch_variance * batch_count
        combined_moment = (
            first_moment
            + second_moment
            + jnp.square(delta) * state.agent.reward_return_count * batch_count / total_count
        )
        agent = state.agent.replace(
            replay_buffer=replay_buffer,
            reward_return_mean=return_mean,
            reward_return_var=combined_moment / total_count,
            reward_return_count=total_count,
            reward_return_max=jnp.maximum(
                state.agent.reward_return_max,
                _max(jnp.max(jnp.abs(discounted_return_samples)), self.axis_name),
            ),
        )

        # The scan keeps a fixed pytree shape before replay warmup; validity flags
        # prevent these placeholders from being emitted as real measurements.
        zero_scalar = jnp.asarray(0.0, dtype=jnp.float32)
        zero_bool = jnp.asarray(False, dtype=jnp.bool_)
        no_update_metrics = FlashSACUpdateMetrics(
            actor_loss=zero_scalar,
            critic_loss=zero_scalar,
            temperature_loss=zero_scalar,
            temperature=zero_scalar,
            entropy=zero_scalar,
            q_value=zero_scalar,
            has_actor_update=zero_bool,
            has_critic_update=zero_bool,
        )

        def run_update(agent_and_rng, _unused):
            current_agent, rng = agent_and_rng
            rng, sample_rng, gradient_rng = jax.random.split(rng, 3)

            def learn(learning_agent):
                batch = learning_agent.replay_buffer.sample(sample_rng, int(config.sample_batch_size))
                return update_from_batch(
                    learning_agent,
                    batch,
                    gradient_rng,
                    self.agent_conf,
                    axis_name=self.axis_name,
                )

            current_agent, metrics = jax.lax.cond(
                current_agent.replay_buffer.size >= int(config.min_buffer_size),
                learn,
                lambda waiting_agent: (waiting_agent, no_update_metrics),
                current_agent,
            )
            return (current_agent, rng), metrics

        (agent, next_rng), update_metrics = jax.lax.scan(
            run_update,
            (agent, update_rng),
            None,
            length=int(config.updates_per_interaction),
        )
        update_metrics = jax.tree_util.tree_map(lambda values: values[-1], update_metrics)

        log_state = transition_state.find(LogEnvState)
        done_count = _sum(jnp.sum(done), self.axis_name)
        absorbing_count = _sum(jnp.sum(absorbing), self.axis_name)
        episode_return = _sum(
            jnp.sum(jnp.where(done, log_state.metrics.returned_episode_returns, 0.0)),
            self.axis_name,
        ) / jnp.maximum(done_count, 1)
        episode_length = _sum(
            jnp.sum(jnp.where(done, log_state.metrics.returned_episode_lengths, 0)),
            self.axis_name,
        ) / jnp.maximum(done_count, 1)
        mean_reward = _mean(jnp.mean(rewards), self.axis_name)
        reward_ema = jnp.where(
            state.interaction_step == 0,
            mean_reward,
            0.99 * state.reward_ema + 0.01 * mean_reward,
        )

        runtime_state = state.replace(
            agent=agent,
            env_state=env_state,
            observation=next_observations,
            exploration_noise=exploration_noise,
            noise_steps_remaining=repeat_count - 1,
            rng=next_rng,
            interaction_step=state.interaction_step + 1,
            reward_ema=reward_ema,
            discounted_return=next_discounted_return,
            last_episode_return=jnp.where(done_count > 0, episode_return, state.last_episode_return),
            last_episode_length=jnp.where(done_count > 0, episode_length, state.last_episode_length),
        )
        collection_metrics = FlashSACCollectionMetrics(
            mean_reward=mean_reward,
            done_count=done_count,
            absorbing_count=absorbing_count,
            episode_return=runtime_state.last_episode_return,
            episode_length=runtime_state.last_episode_length,
            reward_components=mean_reward_components(info, self.axis_name),
        )
        return FlashSACStep(
            runtime_state=runtime_state,
            collection_metrics=collection_metrics,
            update_metrics=update_metrics,
            learning_rate=self.agent_conf.learning_rate_schedule(agent.update_step),
        )


def build_training(
    env: Any,
    agent_conf: FlashSACAgentConf,
    *,
    metrics_handler: MetricsHandler | None = None,
    resume_info: dict[str, Any] | None = None,
    val_env: Any | None = None,
    preserve_checkpoint_target: bool = False,
) -> FlashSACTraining:
    """Build the static FlashSAC training inputs."""
    config = OmegaConf.create(
        OmegaConf.to_container(
            agent_conf.config.experiment,
            resolve=True,
            throw_on_missing=False,
        )
    )
    _install_adaptive_sampling_groups(config, env)
    n_devices = device_count(config)
    validation_shares_training_trajectory = (
        val_env is None
        or val_env is env
        or (
            getattr(val_env, "th", None) is not None
            and getattr(env, "th", None) is not None
            and val_env.th.traj is env.th.traj
        )
    )

    shard_spec = None
    validation_shard_spec = None
    static_trajectory_sharding = trajectory_sharding_mode(config) == "static"
    if static_trajectory_sharding:
        num_trajectories = config.get("n_trajectories")
        if num_trajectories is None:
            raise ValueError("trajectory.sharding=static requires experiment.n_trajectories")
        shard_spec = ShardSpec.build(int(num_trajectories), n_devices)
        if not validation_shares_training_trajectory:
            validation_num_trajectories = config.validation.get("n_trajectories")
            if validation_num_trajectories is None:
                raise ValueError("A distinct sharded validation dataset requires experiment.validation.n_trajectories")
            validation_shard_spec = ShardSpec.build(
                int(validation_num_trajectories),
                n_devices,
            )

    evaluate_all = bool(config.get("validation", {}).get("evaluate_all", False))
    if evaluate_all and not validation_shares_training_trajectory:
        validation_num_trajectories = config.validation.get("n_trajectories")
        if validation_num_trajectories is None and getattr(val_env, "th", None) is not None:
            validation_num_trajectories = int(val_env.th.n_trajectories)
        if validation_num_trajectories is None:
            raise ValueError("Exact validation requires validation.n_trajectories.")
        validation_shard_spec = ShardSpec.build(
            int(validation_num_trajectories),
            n_devices,
        )

    if evaluate_all:
        if validation_shard_spec is None:
            raise ValueError("validation.evaluate_all=true requires a distinct validation dataset")
        required_slots = max(validation_shard_spec.counts)
        configured_slots = int(config.validation.num_envs)
        if configured_slots < required_slots:
            raise ValueError(
                "Exact validation needs at least one environment per local trajectory; "
                f"got {configured_slots}, need {required_slots}."
            )

    global_sizes = {
        "num_envs": int(config.num_envs),
        "buffer_capacity": int(config.buffer_capacity),
        "min_buffer_size": int(config.min_buffer_size),
        "sample_batch_size": int(config.sample_batch_size),
    }
    for name, size in global_sizes.items():
        if size % n_devices != 0:
            raise ValueError(f"{name} ({size}) must be divisible by num_devices ({n_devices})")

    with open_dict(config):
        for name, global_size in global_sizes.items():
            config[f"global_{name}"] = global_size
            config[name] = global_size // n_devices

    return FlashSACTraining(
        env=env,
        val_env=val_env,
        agent_conf=agent_conf,
        config=config,
        metrics_handler=metrics_handler,
        resume_info=resume_info,
        preserve_checkpoint_target=preserve_checkpoint_target,
        axis_name=DATA_AXIS_NAME,
        shard_spec=shard_spec,
        validation_shard_spec=validation_shard_spec,
        validation_shares_training_trajectory=validation_shares_training_trajectory,
    )
