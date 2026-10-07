"""
PPO algorithm class.
"""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from omegaconf import ListConfig, open_dict

from musclemimic.algorithms import ActorCritic, JaxRLAlgorithmBase
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.algorithms.common.env_utils import expand_obs_indices_for_history, wrap_env
from musclemimic.algorithms.common.inference import play_policy, play_policy_mujoco
from musclemimic.algorithms.common.moe_networks import SoftMoEActorCritic
from musclemimic.algorithms.common.optimizer import get_optimizer
from musclemimic.algorithms.ppo.checkpoint import (
    agent_state_to_training_input,
    build_runtime_state_restore_target,
    create_agent_state_from_orbax,
    load_checkpoint_for_resume,
)
from musclemimic.algorithms.ppo.config import PPOAgentConf, PPOAgentState, get_ppo_config
from musclemimic.algorithms.ppo.train import build_training

if TYPE_CHECKING:
    from musclemimic.utils.metrics import MetricsHandler


logger = logging.getLogger(__name__)


class PPOJax(JaxRLAlgorithmBase):
    """PPO algorithm for JAX-based training."""

    _agent_conf = PPOAgentConf
    _agent_state = PPOAgentState

    @classmethod
    def init_agent_conf(cls, env: Any, config: Any) -> PPOAgentConf:
        """
        Initialize agent configuration.

        Args:
            env: environment instance
            config: hydra config

        Returns:
            PPOAgentConf with network and optimizer
        """
        # Merge ppo_config into experiment for backward compatibility
        exp = get_ppo_config(config.experiment)
        adaptive_cfg = exp.get("adaptive_sampling", {})
        traj_cfg = exp.get("trajectory", {}) or {}
        _needs_n_traj = (
            adaptive_cfg.get("enabled", False) or str(traj_cfg.get("sharding", "none") or "none").lower() != "none"
        )
        if _needs_n_traj and exp.get("n_trajectories", None) is None:
            base_env = env
            if hasattr(env, "unwrapped"):
                unwrapped_attr = env.unwrapped
                base_env = unwrapped_attr() if callable(unwrapped_attr) else unwrapped_attr
            if hasattr(base_env, "th") and base_env.th is not None:
                with open_dict(exp):
                    exp.n_trajectories = int(base_env.th.n_trajectories)
            else:
                raise ValueError(
                    "adaptive_sampling or motion sharding is enabled but the env "
                    "has no trajectory handler to derive n_trajectories from"
                )

        # compute derived config values
        with open_dict(exp):
            exp.num_updates = exp.total_timesteps // exp.num_steps // exp.num_envs
            exp.minibatch_size = exp.num_envs * exp.num_steps // exp.num_minibatches
            requested_validations = int(exp.validation.num)
            if requested_validations <= 0:
                raise ValueError("experiment.validation.num must be positive.")
            # Validation can run at most once per PPO update.
            exp.validation_interval = max(1, exp.num_updates // requested_validations)
            exp.validation.num = int(exp.num_updates // exp.validation_interval)

        # utd override
        with open_dict(exp):
            if "utd" in exp and exp.utd is not None:
                exp.update_epochs = int(exp.utd)

        # compute training metrics
        batch_size = int(exp.num_envs) * int(exp.num_steps)
        gradient_steps = int(exp.update_epochs) * int(exp.num_minibatches)

        # standard utd: gradient_updates / samples_collected
        utd = float(gradient_steps) / float(batch_size)
        # sample reuse: how many times each sample is used (= epochs)
        sample_reuse = float(exp.update_epochs)
        # update frequency
        unrolls_per_1m = 1_000_000.0 / float(batch_size)
        grad_steps_per_1m = unrolls_per_1m * float(gradient_steps)

        with open_dict(exp):
            exp.effective_utd = utd
            exp.sample_reuse = sample_reuse
            exp.unrolls_per_1m = unrolls_per_1m
            exp.updates_per_1m = unrolls_per_1m
            exp.grad_steps_per_1m = grad_steps_per_1m

        # Update config.experiment with merged + derived values for downstream use
        with open_dict(config):
            config.experiment = exp

        msg = (
            f"[ppo] roll out size={batch_size}, minibatch size={int(exp.minibatch_size)}, "
            f"grad_steps={gradient_steps}, utd={utd:.4f}, "
            f"sample_reuse={sample_reuse:.0f}x, unrolls/1M={unrolls_per_1m:.1f}, "
            f"grad_steps/1M={grad_steps_per_1m:.1f}"
        )
        if logger.isEnabledFor(logging.INFO) and (logger.handlers or logging.getLogger().handlers):
            logger.info(msg)
        else:
            print(msg)

        network = cls._create_network(env, config)
        tx = get_optimizer(config.experiment)

        return cls._agent_conf(config, network, tx)

    @classmethod
    def _create_network(cls, env: Any, config: Any) -> ActorCritic | SoftMoEActorCritic:
        """Create actor-critic network."""
        exp = config.experiment

        # parse hidden layers
        actor_hidden = cls._parse_hidden_layers(exp.actor_hidden_layers)
        critic_hidden = cls._parse_hidden_layers(exp.critic_hidden_layers)

        # observation indices for actor/critic
        if hasattr(exp, "actor_obs_group") and exp.actor_obs_group is not None:
            actor_obs_ind = env.obs_container.get_obs_ind_by_group(exp.actor_obs_group)
        else:
            actor_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])

        if hasattr(exp, "critic_obs_group") and exp.critic_obs_group is not None:
            critic_obs_ind = env.obs_container.get_obs_ind_by_group(exp.critic_obs_group)
        else:
            critic_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])

        if hasattr(exp, "len_obs_history") and exp.len_obs_history > 1:
            actor_obs_ind = expand_obs_indices_for_history(actor_obs_ind, env, exp)
            critic_obs_ind = expand_obs_indices_for_history(critic_obs_ind, env, exp)

        use_moe = exp.get("use_moe", False)
        use_layernorm = exp.get("use_layernorm", False)
        layernorm_eps = exp.get("layernorm_eps", 1e-5)

        # Residual network options
        use_residual = exp.get("use_residual", False)
        residual_type = exp.get("residual_type", "gated")
        residual_gate_init = exp.get("residual_gate_init", -2.0)

        if use_moe:
            moe_config = exp.get("moe_config", {})
            return SoftMoEActorCritic(
                action_dim=env.info.action_space.shape[0],
                activation=exp.activation,
                init_std=exp.init_std,
                learnable_std=exp.learnable_std,
                hidden_layer_dims=actor_hidden,  # moe uses same for both
                actor_obs_ind=actor_obs_ind,
                critic_obs_ind=critic_obs_ind,
                num_experts=moe_config.get("num_experts", 8),
                moe_at_layers=moe_config.get("moe_at_layers", (1,)),
                apply_moe_to=moe_config.get("apply_moe_to", "both"),
                temperature=moe_config.get("temperature", 1.0),
                load_balance_loss_weight=moe_config.get("load_balance_loss_weight", 0.01),
                use_layernorm=use_layernorm,
                layernorm_eps=layernorm_eps,
            )
        else:
            return ActorCritic(
                env.info.action_space.shape[0],
                activation=exp.activation,
                init_std=exp.init_std,
                learnable_std=exp.learnable_std,
                hidden_layer_dims=actor_hidden,
                critic_hidden_layer_dims=critic_hidden if critic_hidden != actor_hidden else None,
                actor_obs_ind=actor_obs_ind,
                critic_obs_ind=critic_obs_ind,
                use_layernorm=use_layernorm,
                layernorm_eps=layernorm_eps,
                use_residual=use_residual,
                residual_type=residual_type,
                residual_gate_init=residual_gate_init,
            )

    @classmethod
    def _parse_hidden_layers(cls, layers: Any) -> list[int]:
        """Parse hidden layers from config value."""
        if isinstance(layers, list | ListConfig):
            return list(layers)
        return ast.literal_eval(layers)

    @classmethod
    def _get_optimizer(cls, config: Any):
        """Get optimizer (delegates to optimizer module)."""
        return get_optimizer(config.experiment)

    @classmethod
    def build_training(
        cls,
        env: Any,
        agent_conf: PPOAgentConf,
        *,
        metrics_handler: MetricsHandler | None = None,
        resume_info: dict[str, Any] | None = None,
        val_env: Any | None = None,
        apply_resume_resets: bool = True,
        preserve_checkpoint_target: bool = False,
    ):
        """Prepare PPO training."""
        return build_training(
            env,
            agent_conf,
            cls._agent_state,
            metrics_handler=metrics_handler,
            resume_info=resume_info,
            val_env=val_env,
            apply_resume_resets=apply_resume_resets,
            preserve_checkpoint_target=preserve_checkpoint_target,
        )

    @classmethod
    def _wrap_env(cls, env: Any, config: Any) -> Any:
        """Wrap environment (delegates to env_utils module)."""
        return wrap_env(env, config)

    @classmethod
    def build_runtime_state_restore_target(
        cls,
        rng: jax.Array,
        env: Any,
        agent_conf: PPOAgentConf,
        *,
        local_num_envs: int | None = None,
    ) -> Any:
        """Build the typed exact-resume runtime target for PPO."""
        return build_runtime_state_restore_target(
            rng,
            env,
            agent_conf,
            local_num_envs=local_num_envs,
        )

    @classmethod
    def build_inference_state(
        cls,
        policy_state: dict[str, Any],
        agent_conf: PPOAgentConf,
    ) -> PPOAgentState:
        """Build policy-only PPO state for evaluation processes."""
        train_state = TrainState(
            apply_fn=agent_conf.network.apply,
            tx=agent_conf.tx,
            params=policy_state["params"],
            run_stats=policy_state["run_stats"],
            opt_state=None,
            step=0,
        )
        return PPOAgentState(train_state=train_state)

    @classmethod
    def restore_agent_state(
        cls,
        state_data: dict[str, Any],
        agent_conf: PPOAgentConf,
    ) -> PPOAgentState:
        """Restore a typed PPO state for standalone evaluation."""
        restored = create_agent_state_from_orbax(state_data)
        return PPOAgentState(
            train_state=restored.train_state.replace(
                apply_fn=agent_conf.network.apply,
                tx=agent_conf.tx,
            )
        )

    @classmethod
    def play_policy(
        cls,
        env: Any,
        agent_conf: PPOAgentConf,
        agent_state: PPOAgentState,
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
        """Run policy for visualization (delegates to inference module)."""
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
        agent_conf: PPOAgentConf,
        agent_state: PPOAgentState,
        n_steps: int | None = None,
        render: bool = True,
        record: bool = False,
        rng: jax.Array | None = None,
        deterministic: bool = False,
        train_state_seed: int | None = None,
    ) -> None:
        """Run policy with mujoco backend (delegates to inference module)."""
        play_policy_mujoco(
            env,
            agent_conf,
            agent_state,
            n_steps,
            render,
            record,
            rng,
            deterministic,
            train_state_seed,
        )

    @classmethod
    def build_resumed_training(
        cls,
        env: Any,
        agent_conf: PPOAgentConf,
        checkpoint_path: str,
        metrics_handler: MetricsHandler | None = None,
        val_env: Any | None = None,
        apply_resume_resets: bool = True,
        preserve_checkpoint_target: bool = False,
        runtime_state_target: Any | None = None,
    ) -> tuple[Any, Any]:
        """Load a checkpoint and prepare the resumed training run."""
        if runtime_state_target is None and not apply_resume_resets:
            raise ValueError("runtime_state_target is required when apply_resume_resets=False")
        loaded_state, resume_info = load_checkpoint_for_resume(
            checkpoint_path,
            agent_conf,
            runtime_state_target=runtime_state_target,
            require_runtime_state=not apply_resume_resets,
        )
        resume_state = agent_state_to_training_input(loaded_state)

        training = cls.build_training(
            env,
            agent_conf,
            metrics_handler=metrics_handler,
            resume_info=resume_info,
            val_env=val_env,
            apply_resume_resets=apply_resume_resets,
            preserve_checkpoint_target=preserve_checkpoint_target,
        )
        return training, resume_state
