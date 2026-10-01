"""PPO initialization, update, and validation functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf, open_dict

from musclemimic.algorithms.common.checkpoint_utils import (
    TrainingConfig,
    compute_resume_state,
)
from musclemimic.algorithms.common.curriculum import (
    CurriculumParams,
    create_curriculum_params,
    create_curriculum_state,
    create_reward_curriculum_state,
    validate_curriculum_config,
)
from musclemimic.algorithms.common.ppo_distributed import PMAP_AXIS_NAME, device_count
from musclemimic.algorithms.common.env_state_utils import (
    update_carry_ema_normalized,
    update_carry_ema_unnormalized,
    update_carry_reward_weights_normalized,
    update_carry_reward_weights_unnormalized,
    update_carry_threshold_normalized,
    update_carry_threshold_unnormalized,
    update_carry_weights_normalized,
    update_carry_weights_unnormalized,
)
from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.algorithms.common.ppo_trajectory_sharding import (
    ShardSpec,
    install_trajectory_data_overrides,
    restore_trajectory_data_overrides,
    trajectory_sharding_mode,
)
from musclemimic.algorithms.ppo.checkpoint import agent_state_from_training_input
from musclemimic.algorithms.ppo.evaluation import run_validation_step
from musclemimic.algorithms.ppo.learning import (
    initialize_train_state,
    update_observation_stats,
)
from musclemimic.algorithms.ppo.rollout import reset_from_trajectory_weights
from musclemimic.algorithms.ppo.state import PPORuntimeState
from musclemimic.algorithms.ppo.update import run_ppo_update
from musclemimic.utils.debug_tools import DebugFlags

if TYPE_CHECKING:
    from musclemimic.algorithms.ppo.config import PPOAgentConf
    from musclemimic.utils.metrics import MetricsHandler


class PPOTrainStep(NamedTuple):
    """Output of one compiled training step."""

    runtime_state: PPORuntimeState
    training_metrics: Any
    ppo_metrics: Any
    diagnostics: Any
    validation_reset_weights: jax.Array | None
    counter: jax.Array


class PPOTrainChunk(NamedTuple):
    """Output of several PPO updates executed in one compiled scan."""

    runtime_state: PPORuntimeState
    training_metrics: Any
    ppo_metrics: Any
    diagnostics: Any
    validation_reset_weights: jax.Array | None


@dataclass(frozen=True)
class TrainingProgress:
    """Update counts for a fresh or resumed run."""

    completed_updates: int
    remaining_updates: int
    base_global_timestep: int
    base_optimizer_step: int
    target_global_timestep: int


def _install_adaptive_sampling_groups(config: Any, env: Any) -> None:
    """Copy optional environment motion families into the sampler config."""
    trajectory_group_ids = getattr(env, "trajectory_group_ids", None)
    if trajectory_group_ids is None or not bool(
        config.adaptive_sampling.get("balance_motion_groups", True)
    ):
        return

    if len(trajectory_group_ids) != int(env.th.n_trajectories):
        raise ValueError(
            "trajectory_group_ids must contain one group id per trajectory"
        )
    trajectory_group_names = getattr(env, "trajectory_group_names", None)
    with open_dict(config.adaptive_sampling):
        config.adaptive_sampling.group_ids = list(trajectory_group_ids)
        config.adaptive_sampling.group_names = list(
            trajectory_group_names or ("all",)
        )


@dataclass(frozen=True, eq=False)
class PPOTraining:
    """Static inputs and pure functions used by the PPO training loop."""

    env: Any
    val_env: Any | None
    agent_conf: PPOAgentConf
    agent_state_cls: type
    config: Any
    metrics_handler: MetricsHandler | None
    resume_info: dict[str, Any] | None
    apply_resume_resets: bool
    preserve_checkpoint_target: bool
    axis_name: str | None
    shard_spec: ShardSpec | None
    validation_shard_spec: ShardSpec | None
    validation_shares_training_trajectory: bool
    curriculum_params: CurriculumParams | None
    debug_flags: DebugFlags

    @property
    def base_learning_rate(self) -> float:
        return float(self.config.lr)

    @property
    def use_adaptive_learning_rate(self) -> bool:
        return self.config.get("schedule", "fixed") == "adaptive" and self.config.get("desired_kl") is not None

    @property
    def adaptive_termination_enabled(self) -> bool:
        return bool(self.config.get("adaptive_termination", {}).get("enabled", False))

    def initialize(
        self,
        rng: jax.Array,
        agent_state_data: Any | None = None,
        trajectory_data_override: Any | None = None,
    ) -> PPORuntimeState:
        """Initialize fresh state or restore saved state."""
        restore_tokens = install_trajectory_data_overrides(
            self.env,
            self.val_env,
            trajectory_data_override,
            None,
        )
        try:
            env = wrap_env(self.env, self.config, axis_name=self.axis_name)
            agent_state = None if agent_state_data is None else agent_state_from_training_input(agent_state_data)
            return self._initialize_runtime_state(rng, env, agent_state)
        finally:
            restore_trajectory_data_overrides(restore_tokens)

    def update(
        self,
        runtime_state: PPORuntimeState,
        trajectory_data_override: Any | None = None,
    ) -> PPOTrainStep:
        """Run one rollout and optimizer update."""
        restore_tokens = install_trajectory_data_overrides(
            self.env,
            self.val_env,
            trajectory_data_override,
            None,
        )
        try:
            env = wrap_env(self.env, self.config, axis_name=self.axis_name)
            return self._update(runtime_state, env)
        finally:
            restore_trajectory_data_overrides(restore_tokens)

    def update_chunk(
        self,
        runtime_state: PPORuntimeState,
        trajectory_data_override: Any | None,
        num_updates: jax.Array,
        max_updates: int,
    ) -> PPOTrainChunk:
        """Run up to ``max_updates`` without returning to the host."""
        if max_updates < 1:
            raise ValueError("max_updates must be positive")

        restore_tokens = install_trajectory_data_overrides(
            self.env,
            self.val_env,
            trajectory_data_override,
            None,
        )
        try:
            env = wrap_env(self.env, self.config, axis_name=self.axis_name)

            def run_update(state: PPORuntimeState):
                step = self._update(state, env)
                metrics = (
                    step.training_metrics,
                    step.ppo_metrics,
                    step.diagnostics,
                )
                carry = (step.runtime_state, step.validation_reset_weights)
                return carry, metrics

            (runtime_state, reset_weights), first_metrics = run_update(runtime_state)
            zero_metrics = jax.tree.map(jnp.zeros_like, first_metrics)

            def scan_update(carry, update_index):
                return jax.lax.cond(
                    update_index < num_updates,
                    lambda state_and_weights: run_update(state_and_weights[0]),
                    lambda state_and_weights: (state_and_weights, zero_metrics),
                    carry,
                )

            (final_state, reset_weights), remaining_metrics = jax.lax.scan(
                scan_update,
                (runtime_state, reset_weights),
                xs=jnp.arange(1, max_updates),
            )
            metrics = jax.tree.map(
                lambda first, remaining: jnp.concatenate(
                    (jnp.asarray(first)[None], remaining),
                    axis=0,
                ),
                first_metrics,
                remaining_metrics,
            )
            training_metrics, ppo_metrics, diagnostics = metrics
            return PPOTrainChunk(
                runtime_state=final_state,
                training_metrics=training_metrics,
                ppo_metrics=ppo_metrics,
                diagnostics=diagnostics,
                validation_reset_weights=reset_weights,
            )
        finally:
            restore_trajectory_data_overrides(restore_tokens)

    def validate(
        self,
        train_state: Any,
        validation_rng: jax.Array,
        reset_weights: jax.Array | None,
        trajectory_data_override: Any | None = None,
        validation_trajectory_data_override: Any | None = None,
    ) -> Any:
        """Run one scheduled validation and advance its independent RNG."""
        restore_tokens = install_trajectory_data_overrides(
            self.env,
            self.val_env,
            trajectory_data_override,
            validation_trajectory_data_override,
        )
        try:
            validation_env = self.val_env if self.val_env is not None else self.env
            validation_env = wrap_env(
                validation_env,
                self.config,
                axis_name=self.axis_name,
            )
            return run_validation_step(
                train_state,
                validation_rng,
                jnp.asarray(True),
                validation_env,
                self.config,
                self.metrics_handler,
                reset_weights=reset_weights,
                axis_name=self.axis_name,
            )
        finally:
            restore_trajectory_data_overrides(restore_tokens)

    def progress(self, runtime_state: PPORuntimeState, *, resumed: bool) -> TrainingProgress:
        """Compute the remaining updates after initialization."""
        train_state = runtime_state.train_state
        completed, remaining, base_timestep, base_step, target = _compute_resume_info(
            self.config,
            train_state if resumed else None,
            self.resume_info,
            train_state,
            preserve_checkpoint_target=self.preserve_checkpoint_target,
            apply_resume_resets=self.apply_resume_resets,
        )
        return TrainingProgress(
            completed_updates=completed,
            remaining_updates=remaining,
            base_global_timestep=base_timestep,
            base_optimizer_step=base_step,
            target_global_timestep=target,
        )

    def _update(self, runtime_state: PPORuntimeState, env: Any) -> PPOTrainStep:
        result = run_ppo_update(
            runtime_state,
            env=env,
            network=self.agent_conf.network,
            config=self.config,
            debug_flags=self.debug_flags,
            base_learning_rate=self.base_learning_rate,
            use_adaptive_learning_rate=self.use_adaptive_learning_rate,
            adaptive_termination_enabled=self.adaptive_termination_enabled,
            curriculum_params=self.curriculum_params,
            shard_spec=self.shard_spec,
            axis_name=self.axis_name,
        )
        return PPOTrainStep(
            runtime_state=result.runtime_state,
            training_metrics=result.training_metrics,
            ppo_metrics=result.ppo_metrics,
            diagnostics=result.diagnostics,
            validation_reset_weights=self._validation_reset_weights(
                result.adaptive_sampling_weights,
            ),
            counter=result.counter,
        )

    def _initialize_runtime_state(
        self,
        rng: jax.Array,
        env: Any,
        agent_state: Any | None,
    ) -> PPORuntimeState:
        config = self.config
        network = self.agent_conf.network
        train_state = initialize_train_state(
            rng,
            env,
            network,
            self.agent_conf.tx,
            agent_state,
            config,
            apply_resume_resets=self.apply_resume_resets,
        )

        if self.axis_name is not None:
            rng = jax.random.fold_in(rng, jax.lax.axis_index(self.axis_name))
        if agent_state is None:
            rng, _, _ = jax.random.split(rng, 3)

        adaptive_termination = config.get("adaptive_termination", {})
        terminal_params = config.env_params.get("terminal_state_params", {})
        default_threshold = terminal_params.get("mean_site_deviation_threshold", 0.3)
        if self.adaptive_termination_enabled:
            initial_threshold = adaptive_termination.get(
                "init_threshold",
                default_threshold,
            )
            curriculum_state = create_curriculum_state(
                initial_threshold,
                self.curriculum_params.init_ema_val,
            )
        else:
            initial_threshold = default_threshold
            curriculum_state = create_curriculum_state(
                initial_threshold,
                adaptive_termination.get("init_ema_val", 1.0),
            )

        reward_curriculum = config.get("reward_curriculum", {})
        reward_params = config.env_params.get("reward_params", {})
        reward_curriculum_enabled = reward_curriculum.get("enabled", False)
        qvel_weight = (
            reward_curriculum.get("qvel_w_sum_init", 0.2)
            if reward_curriculum_enabled
            else reward_params.get("qvel_w_sum", 0.2)
        )
        root_velocity_weight = (
            reward_curriculum.get("root_vel_w_sum_init", 0.2)
            if reward_curriculum_enabled
            else reward_params.get("root_vel_w_sum", 0.2)
        )
        reward_curriculum_state = create_reward_curriculum_state(reward_curriculum).replace(
            qvel_w_sum=jnp.asarray(qvel_weight, dtype=jnp.float32),
            root_vel_w_sum=jnp.asarray(root_velocity_weight, dtype=jnp.float32),
        )

        initial_weights, batched_weights, done_ema, early_ema = self._initial_sampling_state()
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, config.num_envs)
        if initial_weights is None:
            observation, env_state = env.reset(reset_keys)
        else:
            observation, env_state = reset_from_trajectory_weights(
                env,
                reset_keys,
                initial_weights,
            )

        if agent_state is None:
            train_state = update_observation_stats(
                train_state,
                observation,
                network=network,
                axis_name=self.axis_name,
            )

        if config.normalize_env:
            env_state = update_carry_threshold_normalized(env_state, initial_threshold)
            env_state = update_carry_reward_weights_normalized(
                env_state,
                reward_curriculum_state.qvel_w_sum,
                reward_curriculum_state.root_vel_w_sum,
            )
        else:
            env_state = update_carry_threshold_unnormalized(env_state, initial_threshold)
            env_state = update_carry_reward_weights_unnormalized(
                env_state,
                reward_curriculum_state.qvel_w_sum,
                reward_curriculum_state.root_vel_w_sum,
            )

        if batched_weights is not None:
            if config.normalize_env:
                env_state = update_carry_weights_normalized(env_state, batched_weights)
                env_state = update_carry_ema_normalized(
                    env_state,
                    done_ema,
                    early_ema,
                )
            else:
                env_state = update_carry_weights_unnormalized(env_state, batched_weights)
                env_state = update_carry_ema_unnormalized(
                    env_state,
                    done_ema,
                    early_ema,
                )

        checkpoint_runtime_state = getattr(agent_state, "runtime_state", None) if agent_state is not None else None
        if checkpoint_runtime_state is not None and not self.apply_resume_resets:
            return PPORuntimeState.from_checkpoint_state(
                train_state,
                checkpoint_runtime_state,
            )

        rng, training_rng = jax.random.split(rng)
        _, validation_rng = jax.random.split(rng)
        learning_rate = config.get("resume_lr_override", self.base_learning_rate)
        return PPORuntimeState(
            train_state=train_state,
            env_state=env_state,
            last_obs=observation,
            rng=training_rng,
            lr=jnp.asarray(learning_rate, dtype=jnp.float32),
            val_rng=validation_rng,
            curriculum_state=curriculum_state,
            reward_curriculum_state=reward_curriculum_state,
        )

    def _initial_sampling_state(self) -> tuple[Any, Any, Any, Any]:
        config = self.config
        adaptive_sampling = config.get("adaptive_sampling", {})
        if not adaptive_sampling.get("enabled", False) and self.shard_spec is None:
            return None, None, None, None

        num_trajectories = config.get("n_trajectories")
        if num_trajectories is None:
            raise ValueError("adaptive_sampling or motion sharding requires experiment.n_trajectories")
        if self.shard_spec is None:
            weights = jnp.full((num_trajectories,), 1.0 / num_trajectories)
        else:
            weights = self.shard_spec.uniform_weights_for_device(jax.lax.axis_index(PMAP_AXIS_NAME))
        batched_weights = jnp.broadcast_to(
            weights,
            (config.num_envs, num_trajectories),
        )
        done_ema = jnp.broadcast_to(
            jnp.full(
                (num_trajectories,),
                adaptive_sampling.get("ema_done_init", 10.0),
            ),
            (config.num_envs, num_trajectories),
        )
        early_ema = jnp.broadcast_to(
            jnp.full(
                (num_trajectories,),
                adaptive_sampling.get("ema_early_init", 5.0),
            ),
            (config.num_envs, num_trajectories),
        )
        return weights, batched_weights, done_ema, early_ema

    def _validation_reset_weights(
        self,
        adaptive_sampling_weights: jax.Array | None,
    ) -> jax.Array | None:
        if self.validation_shares_training_trajectory:
            if self.config.get("adaptive_sampling", {}).get("enabled", False):
                return adaptive_sampling_weights
            if self.shard_spec is not None:
                return self.shard_spec.uniform_weights_for_device(jax.lax.axis_index(PMAP_AXIS_NAME))
            return None
        if self.validation_shard_spec is not None:
            return self.validation_shard_spec.uniform_weights_for_device(jax.lax.axis_index(PMAP_AXIS_NAME))
        return None


def build_training(
    env: Any,
    agent_conf: PPOAgentConf,
    agent_state_cls: type,
    *,
    metrics_handler: MetricsHandler | None = None,
    resume_info: dict[str, Any] | None = None,
    val_env: Any | None = None,
    apply_resume_resets: bool = True,
    preserve_checkpoint_target: bool = False,
) -> PPOTraining:
    """Collect the static inputs used by PPO training."""
    config = OmegaConf.create(
        OmegaConf.to_container(
            agent_conf.config.experiment,
            resolve=True,
            throw_on_missing=False,
        )
    )
    _install_adaptive_sampling_groups(config, env)
    n_devices = device_count(config)
    axis_name = PMAP_AXIS_NAME if n_devices > 1 else None
    validation_shares_training_trajectory = val_env is None or val_env is env or val_env.th.traj is env.th.traj

    shard_spec = None
    validation_shard_spec = None
    if trajectory_sharding_mode(config) == "static":
        if axis_name is None:
            raise ValueError("trajectory.sharding=static requires distributed.num_devices > 1")
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

    if axis_name is not None:
        global_num_envs = int(config.num_envs)
        if global_num_envs % n_devices != 0:
            raise ValueError(f"num_envs ({global_num_envs}) must be divisible by num_devices ({n_devices})")
        local_num_envs = global_num_envs // n_devices
        local_batch_size = local_num_envs * int(config.num_steps)
        num_minibatches = int(config.num_minibatches)
        if local_batch_size % num_minibatches != 0:
            raise ValueError(
                f"Per-device batch size {local_batch_size} is not divisible by num_minibatches {num_minibatches}"
            )
        with open_dict(config):
            config.num_envs = local_num_envs
            config.global_num_envs = global_num_envs
            config.minibatch_size = local_batch_size // num_minibatches
        if config.get("validation", {}).get("active", False):
            global_validation_num_envs = int(config.validation.num_envs)
            if global_validation_num_envs % n_devices != 0:
                raise ValueError(
                    "validation.num_envs "
                    f"({global_validation_num_envs}) must be divisible by num_devices ({n_devices})"
                )
            with open_dict(config.validation):
                config.validation.global_num_envs = global_validation_num_envs
                config.validation.num_envs = global_validation_num_envs // n_devices

    adaptive_termination = config.get("adaptive_termination", {})
    curriculum_params = None
    if adaptive_termination.get("enabled", False):
        validate_curriculum_config(adaptive_termination)
        curriculum_params = create_curriculum_params(adaptive_termination)

    return PPOTraining(
        env=env,
        val_env=val_env,
        agent_conf=agent_conf,
        agent_state_cls=agent_state_cls,
        config=config,
        metrics_handler=metrics_handler,
        resume_info=resume_info,
        apply_resume_resets=apply_resume_resets,
        preserve_checkpoint_target=preserve_checkpoint_target,
        axis_name=axis_name,
        shard_spec=shard_spec,
        validation_shard_spec=validation_shard_spec,
        validation_shares_training_trajectory=validation_shares_training_trajectory,
        curriculum_params=curriculum_params,
        debug_flags=DebugFlags.from_config(config.debug),
    )


def _resolve_target_global_timestep(
    config: Any,
    resume_info: dict[str, Any],
    base_global_timestep: int,
    preserve_checkpoint_target: bool = False,
) -> int:
    """Resolve the absolute target budget for the current resume path."""
    configured_total_timesteps = int(config.total_timesteps)
    if bool(config.get("total_timesteps_is_absolute", False)):
        return configured_total_timesteps

    stored_target = int(resume_info.get("target_global_timestep", 0) or 0)
    if preserve_checkpoint_target and stored_target > 0:
        return stored_target
    return configured_total_timesteps + int(base_global_timestep)


def _compute_resume_info(
    config: Any,
    agent_state: Any | None,
    resume_info: dict[str, Any] | None,
    train_state: Any,
    preserve_checkpoint_target: bool = False,
    apply_resume_resets: bool = True,
) -> tuple[int, int, int, int, int]:
    """Compute update counts and absolute timestep targets for this run."""
    completed_updates = 0
    remaining_updates = int(config.num_updates)
    base_global_timestep = 0
    target_global_timestep = int(config.total_timesteps)

    if agent_state is not None and resume_info is not None:
        current_config = TrainingConfig.from_experiment_config(config)
        base_global_timestep = int(resume_info["global_timestep"])
        target_global_timestep = _resolve_target_global_timestep(
            config,
            resume_info,
            base_global_timestep,
            preserve_checkpoint_target,
        )
        completed_updates, remaining_updates, config_changed = compute_resume_state(
            checkpoint_metadata=resume_info,
            current_config=current_config,
            total_timesteps=target_global_timestep,
        )
        print("\n" + "=" * 60)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 60)
        print(f"Completed updates:       {completed_updates}")
        print(f"Remaining updates:       {remaining_updates}")
        print(f"Global timestep:         {base_global_timestep:,}")
        print(f"Target global timestep:  {target_global_timestep:,}")
        if config_changed:
            current_num_envs = int(getattr(config, "global_num_envs", config.num_envs))
            print("\nTraining config changed:")
            print(f"   Checkpoint: num_envs={resume_info['num_envs']}, num_steps={resume_info['num_steps']}")
            print(f"   Current:    num_envs={current_num_envs}, num_steps={config.num_steps}")
        print("=" * 60 + "\n")

        base_optimizer_step = int(resume_info["optimizer_step"])
        if apply_resume_resets and config.get("reset_lr_schedule_on_resume", False):
            base_optimizer_step = 0
        return (
            completed_updates,
            remaining_updates,
            base_global_timestep,
            base_optimizer_step,
            target_global_timestep,
        )

    return (
        completed_updates,
        remaining_updates,
        base_global_timestep,
        int(train_state.step),
        target_global_timestep,
    )
