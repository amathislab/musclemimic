"""
Checkpoint loading utilities for PPO.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf, open_dict

from musclemimic.algorithms.common.checkpoint_manager import UnifiedCheckpointManager
from musclemimic.algorithms.common.curriculum import (
    create_curriculum_params,
    create_curriculum_state,
    create_reward_curriculum_state,
)
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.algorithms.common.ppo_distributed import device_count
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
from musclemimic.algorithms.common.ppo_trajectory_sharding import trajectory_sharding_mode
from musclemimic.algorithms.ppo.checkpoint_integrity import validate_ppo_checkpoint_train_state


def _dataclass_field_names(value: Any) -> tuple[str, ...]:
    fields = getattr(value, "__dataclass_fields__", None)
    return tuple(fields.keys()) if fields is not None else ()


def _array_shape_dtype(value: Any) -> tuple[tuple[int, ...], Any] | None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return None
    return tuple(int(dim) for dim in shape), dtype


def _shape_dtype_target(value: Any) -> Any:
    array = _array_shape_dtype(value)
    if array is not None:
        import jax

        shape, dtype = array
        return jax.ShapeDtypeStruct(shape, dtype)

    if isinstance(value, dict):
        return {key: _shape_dtype_target(child) for key, child in value.items()}

    field_names = _dataclass_field_names(value)
    if field_names:
        return value.replace(**{name: _shape_dtype_target(getattr(value, name)) for name in field_names})

    return value


def _runtime_batch_shape(
    runtime_state: Any,
    num_envs: int,
    num_devices: int,
) -> tuple[int, ...] | None:
    if not isinstance(runtime_state, dict):
        return None
    last_obs = runtime_state.get("last_obs")
    shape = getattr(last_obs, "shape", None)
    if shape is None or num_envs <= 0:
        return None
    shape = tuple(int(dim) for dim in shape)
    is_sharded_batch = len(shape) >= 2 and shape[0] * shape[1] == num_envs and (num_devices > 1 or shape[0] != num_envs)
    if is_sharded_batch:
        return shape[:2]
    if len(shape) >= 1 and shape[0] == num_envs:
        return shape[:1]
    return None


def _validate_exact_resume_topology(
    metadata: Any,
    runtime_state: Any,
    agent_conf: Any,
) -> tuple[int, ...]:
    """Reject an exact resume whose replica-local state cannot match this run."""
    num_devices = device_count(agent_conf.config)
    checkpoint_devices = int(getattr(metadata, "num_devices", 1) or 1)
    if checkpoint_devices != num_devices:
        raise ValueError(
            "Exact resume requires the checkpoint device count to match: "
            f"checkpoint={checkpoint_devices}, configured={num_devices}."
        )

    configured_sharding = trajectory_sharding_mode(agent_conf.config)
    checkpoint_sharding = str(getattr(metadata, "trajectory_sharding", "none") or "none")
    if checkpoint_sharding != configured_sharding:
        raise ValueError(
            "Exact resume requires the checkpoint trajectory sharding mode to match: "
            f"checkpoint={checkpoint_sharding!r}, configured={configured_sharding!r}."
        )

    global_num_envs = int(agent_conf.config.experiment.num_envs)
    checkpoint_num_envs = int(getattr(metadata, "num_envs", -1) or -1)
    if checkpoint_num_envs != global_num_envs:
        raise ValueError(
            "Exact resume requires the global environment count to match: "
            f"checkpoint={checkpoint_num_envs}, configured={global_num_envs}."
        )
    if global_num_envs % num_devices != 0:
        raise ValueError("Exact resume requires num_envs to be divisible by num_devices.")

    batch_shape = _runtime_batch_shape(runtime_state, global_num_envs, num_devices)
    expected_batch_shape = (
        (global_num_envs,) if num_devices == 1 else (num_devices, global_num_envs // num_devices)
    )
    if batch_shape != expected_batch_shape:
        raise ValueError(
            "Exact-resume runtime state has an incompatible replica/environment shape: "
            f"checkpoint={batch_shape}, expected={expected_batch_shape}."
        )
    return batch_shape


def _restore_target_for_checkpoint_schema(
    raw_value: Any,
    target_value: Any,
    runtime_batch_shape: tuple[int, ...] | None = None,
) -> Any:
    """Trim a typed restore target to fields that exist in the checkpoint."""
    if target_value is None:
        return None

    raw_array = _array_shape_dtype(raw_value)
    target_array = _array_shape_dtype(target_value)
    if raw_array is not None and target_array is not None:
        import jax

        raw_shape, _ = raw_array
        target_shape, target_dtype = target_array
        if any(dim == 0 for dim in target_shape):
            if runtime_batch_shape is not None and len(target_shape) >= len(runtime_batch_shape):
                restore_shape = (*runtime_batch_shape, *target_shape[len(runtime_batch_shape) :])
            else:
                restore_shape = target_shape
        else:
            restore_shape = raw_shape
        return jax.ShapeDtypeStruct(restore_shape, target_dtype)

    if isinstance(raw_value, dict):
        if isinstance(target_value, dict):
            return {
                key: _restore_target_for_checkpoint_schema(raw_child, target_value[key], runtime_batch_shape)
                for key, raw_child in raw_value.items()
                if key in target_value
            }

        field_names = _dataclass_field_names(target_value)
        if field_names:
            return {
                key: _restore_target_for_checkpoint_schema(
                    raw_child,
                    getattr(target_value, key),
                    runtime_batch_shape,
                )
                for key, raw_child in raw_value.items()
                if key in field_names
            }

    raw_fields = _dataclass_field_names(raw_value)
    target_fields = _dataclass_field_names(target_value)
    if raw_fields and target_fields:
        return target_value.replace(
            **{
                name: _restore_target_for_checkpoint_schema(
                    getattr(raw_value, name),
                    getattr(target_value, name),
                    runtime_batch_shape,
                )
                for name in raw_fields
                if name in target_fields
            }
        )

    return target_value


def _fill_missing_from_runtime_target(restored_value: Any, target_value: Any) -> Any:
    """Rebuild restored runtime state with new fields initialized from the current target."""
    if target_value is None:
        return restored_value

    target_fields = _dataclass_field_names(target_value)
    if target_fields:
        restored_is_dict = isinstance(restored_value, dict)
        restored_fields = _dataclass_field_names(restored_value)
        values = {}
        for name in target_fields:
            target_child = getattr(target_value, name)
            if restored_is_dict and name in restored_value:
                values[name] = _fill_missing_from_runtime_target(restored_value[name], target_child)
            elif restored_fields and name in restored_fields:
                values[name] = _fill_missing_from_runtime_target(getattr(restored_value, name), target_child)
            else:
                values[name] = target_child
        return target_value.replace(**values)

    if isinstance(restored_value, dict) and isinstance(target_value, dict):
        merged = dict(restored_value)
        for key, target_child in target_value.items():
            if key in merged:
                merged[key] = _fill_missing_from_runtime_target(merged[key], target_child)
            else:
                merged[key] = target_child
        return merged

    return restored_value


def _replace_target_owned_runtime_leaves(restored_value: Any, target_value: Any, path: tuple[str, ...] = ()) -> Any:
    """Use current runtime target for PyTree metadata and MJX scratch buffers."""
    if target_value is None:
        return restored_value

    restored_array = _array_shape_dtype(restored_value)
    target_array = _array_shape_dtype(target_value)
    target_fields = _dataclass_field_names(target_value)
    restored_fields = _dataclass_field_names(restored_value)
    is_container = (
        isinstance(restored_value, dict)
        or isinstance(target_value, dict)
        or bool(restored_fields)
        or bool(target_fields)
    )
    if path:
        is_contact_buffer = path[-1].startswith("contact__")
        is_static_metadata = restored_array is None and target_array is None and not is_container
        if is_contact_buffer or is_static_metadata:
            return target_value

    if restored_array is not None or target_array is not None:
        return restored_value

    if target_fields:
        restored_is_dict = isinstance(restored_value, dict)
        values = {}
        for name in target_fields:
            target_child = getattr(target_value, name)
            if restored_is_dict and name in restored_value:
                restored_child = restored_value[name]
            elif restored_fields and name in restored_fields:
                restored_child = getattr(restored_value, name)
            else:
                restored_child = target_child
            values[name] = _replace_target_owned_runtime_leaves(restored_child, target_child, (*path, name))
        return target_value.replace(**values)

    if isinstance(restored_value, dict) and isinstance(target_value, dict):
        merged = dict(restored_value)
        for key, target_child in target_value.items():
            restored_child = restored_value[key] if key in restored_value else target_child
            merged[key] = _replace_target_owned_runtime_leaves(restored_child, target_child, (*path, str(key)))
        return merged

    return restored_value


def create_agent_state_from_orbax(orbax_data: dict[str, Any]) -> SimpleNamespace:
    """
    Create agent state from orbax checkpoint data.

    Args:
        orbax_data: dict with params, opt_state, step, run_stats

    Returns:
        SimpleNamespace with train_state attribute
    """
    params = orbax_data.get("params", {})
    opt_state = orbax_data.get("opt_state", {})
    step = orbax_data.get("step", 0)
    run_stats = orbax_data.get("run_stats", {})

    class _DummyModule:
        def apply(self, *args, **kwargs):
            raise NotImplementedError("dummy module for checkpoint loading")

    ts = TrainState(
        apply_fn=_DummyModule().apply,
        params=params,
        tx=None,
        opt_state=opt_state,
        step=step,
        run_stats=run_stats or {},
    )

    agent_state = SimpleNamespace()
    agent_state.train_state = ts
    if "runtime_state" in orbax_data:
        agent_state.runtime_state = orbax_data["runtime_state"]
    return agent_state


def agent_state_to_training_input(agent_state: SimpleNamespace) -> dict[str, Any]:
    data = {"train_state": agent_state.train_state}
    if hasattr(agent_state, "runtime_state"):
        data["runtime_state"] = agent_state.runtime_state
    return data


def agent_state_from_training_input(data: dict[str, Any]) -> SimpleNamespace:
    agent_state = SimpleNamespace()
    agent_state.train_state = data["train_state"]
    if "runtime_state" in data:
        agent_state.runtime_state = data["runtime_state"]
    return agent_state


def create_train_state_restore_target(
    raw_state_data: dict[str, Any],
    tx: Any,
    runtime_state_target: Any | None = None,
    runtime_batch_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build a typed Orbax restore target for TrainState checkpoint data."""
    import jax

    params = _shape_dtype_target(raw_state_data["params"])
    target = {
        "params": params,
        "opt_state": jax.eval_shape(tx.init, params),
        "step": _shape_dtype_target(raw_state_data.get("step", 0)),
        "run_stats": _shape_dtype_target(raw_state_data.get("run_stats", {})),
    }
    if "runtime_state" in raw_state_data:
        target["runtime_state"] = (
            _restore_target_for_checkpoint_schema(
                raw_state_data["runtime_state"],
                runtime_state_target,
                runtime_batch_shape,
            )
            if runtime_state_target is not None
            else raw_state_data["runtime_state"]
        )
    return target


def load_checkpoint_for_resume(
    checkpoint_path: str,
    agent_conf: Any,
    runtime_state_target: Any | None = None,
    require_runtime_state: bool = False,
) -> tuple[SimpleNamespace, dict[str, int]]:
    """
    Load checkpoint and prepare resume info.

    Args:
        checkpoint_path: path to checkpoint
        agent_conf: agent configuration (config will be modified for lr override)

    Returns:
        (loaded_state, resume_info dict)
    """
    checkpoint_path_obj = Path(checkpoint_path)

    manager = UnifiedCheckpointManager(
        checkpoint_dir=str(checkpoint_path_obj.parent),
        max_to_keep=5,
    )

    try:
        (_, raw_state_data), metadata = manager.load_checkpoint(checkpoint_path)
        validate_ppo_checkpoint_train_state(
            raw_state_data,
            context=f"resume from {checkpoint_path}",
        )
        tx = getattr(agent_conf, "tx", None)
        if tx is None:
            raise ValueError("agent_conf.tx is required to restore optimizer state")
        if require_runtime_state and "runtime_state" not in raw_state_data:
            raise ValueError(
                "Local auto-resume requires a checkpoint with runtime_state. "
                "Start a fresh run or resume from a checkpoint saved by the current code."
            )
        if require_runtime_state and runtime_state_target is None:
            raise ValueError("runtime_state_target is required for local auto-resume")

        if require_runtime_state:
            runtime_batch_shape = _validate_exact_resume_topology(
                metadata,
                raw_state_data["runtime_state"],
                agent_conf,
            )
        else:
            runtime_batch_shape = _runtime_batch_shape(
                raw_state_data.get("runtime_state"),
                int(getattr(metadata, "num_envs", -1) or -1),
                device_count(agent_conf.config),
            )
        train_state_target = create_train_state_restore_target(
            raw_state_data,
            tx,
            runtime_state_target=runtime_state_target,
            runtime_batch_shape=runtime_batch_shape,
        )
        del raw_state_data
        (_, loaded_state_data), metadata = manager.load_checkpoint(
            checkpoint_path,
            train_state_target=train_state_target,
        )
        if runtime_state_target is not None and "runtime_state" in loaded_state_data:
            loaded_state_data["runtime_state"] = _fill_missing_from_runtime_target(
                loaded_state_data["runtime_state"], runtime_state_target
            )
            loaded_state_data["runtime_state"] = _replace_target_owned_runtime_leaves(
                loaded_state_data["runtime_state"], runtime_state_target
            )
        import jax

        loaded_state_data = jax.device_get(loaded_state_data)
        loaded_state = create_agent_state_from_orbax(loaded_state_data)

        # inject lr override if available
        cur_lr = getattr(metadata, "learning_rate", None)
        if cur_lr is not None and cur_lr > 0:
            with open_dict(agent_conf.config):
                agent_conf.config.experiment.resume_lr_override = float(cur_lr)

        resume_info = {
            "update_number": int(getattr(metadata, "update_number", -1) or -1),
            "global_timestep": int(getattr(metadata, "global_timestep", -1) or -1),
            "target_global_timestep": int(getattr(metadata, "target_global_timestep", 0) or 0),
            "optimizer_step": int(loaded_state_data.get("step", 0)),
            "num_envs": int(getattr(metadata, "num_envs", -1) or -1),
            "num_steps": int(getattr(metadata, "num_steps", -1) or -1),
            "num_minibatches": int(getattr(metadata, "num_minibatches", -1) or -1),
            "update_epochs": int(getattr(metadata, "update_epochs", -1) or -1),
            "num_devices": int(getattr(metadata, "num_devices", 1) or 1),
            "trajectory_sharding": str(getattr(metadata, "trajectory_sharding", "none") or "none"),
        }

        return loaded_state, resume_info
    finally:
        manager.close()


def build_runtime_state(
    env_state: Any,
    observation: jax.Array,
    rng: jax.Array,
    learning_rate: jax.Array,
    evaluation_rng: jax.Array,
    curriculum_state: Any,
    reward_curriculum_state: Any,
) -> dict[str, Any]:
    """Build the dynamic state required for exact training continuation."""
    return {
        "env_state": env_state,
        "last_obs": observation,
        "rng": rng,
        "lr": learning_rate,
        "val_rng": evaluation_rng,
        "curriculum_state": curriculum_state,
        "reward_curriculum_state": reward_curriculum_state,
    }


def _add_device_axis(runtime_state: Any, num_devices: int) -> Any:
    """Build a host-backed multi-device restore target without GPU replication."""

    def _broadcast(value):
        if value is None:
            return None
        host_value = np.asarray(jax.device_get(value))
        return np.broadcast_to(
            host_value[None],
            (num_devices, *host_value.shape),
        )

    return jax.tree_util.tree_map(
        _broadcast,
        runtime_state,
        is_leaf=lambda value: value is None,
    )


def build_runtime_state_restore_target(
    rng: jax.Array,
    env: Any,
    agent_conf: Any,
    *,
    local_num_envs: int | None = None,
) -> Any:
    """Create the typed runtime-state target for exact local auto-resume."""
    config = OmegaConf.create(
        OmegaConf.to_container(
            agent_conf.config.experiment,
            resolve=True,
            throw_on_missing=False,
        )
    )
    num_devices = device_count(agent_conf.config)
    if num_devices > 1:
        if int(config.num_envs) % num_devices != 0:
            raise ValueError("num_envs must be divisible by num_devices to restore runtime_state.")
        with open_dict(config):
            config.global_num_envs = int(config.num_envs)
            config.num_envs = int(config.num_envs) // num_devices
    if local_num_envs is not None:
        with open_dict(config):
            config.num_envs = int(local_num_envs)

    wrapped_env = wrap_env(env, config)
    adaptive_termination = config.get("adaptive_termination", {})
    terminal_params = config.env_params.get("terminal_state_params", {})
    default_threshold = terminal_params.get(
        "mean_site_deviation_threshold",
        0.3,
    )
    if adaptive_termination.get("enabled", False):
        curriculum_params = create_curriculum_params(adaptive_termination)
        initial_threshold = adaptive_termination.get(
            "init_threshold",
            default_threshold,
        )
        curriculum_state = create_curriculum_state(
            initial_threshold,
            curriculum_params.init_ema_val,
        )
    else:
        initial_threshold = default_threshold
        curriculum_state = create_curriculum_state(
            initial_threshold,
            adaptive_termination.get("init_ema_val", 1.0),
        )

    reward_curriculum = config.get("reward_curriculum", {})
    reward_params = config.env_params.get("reward_params", {})
    if reward_curriculum.get("enabled", False):
        initial_qvel_weight = reward_curriculum.get("qvel_w_sum_init", 0.2)
        initial_root_velocity_weight = reward_curriculum.get(
            "root_vel_w_sum_init",
            0.2,
        )
    else:
        initial_qvel_weight = reward_params.get("qvel_w_sum", 0.2)
        initial_root_velocity_weight = reward_params.get("root_vel_w_sum", 0.2)
    reward_curriculum_state = create_reward_curriculum_state(reward_curriculum).replace(
        qvel_w_sum=jnp.asarray(initial_qvel_weight, dtype=jnp.float32),
        root_vel_w_sum=jnp.asarray(
            initial_root_velocity_weight,
            dtype=jnp.float32,
        ),
    )

    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, config.num_envs)
    if trajectory_sharding_mode(config) == "static":
        trajectory_ids = jnp.zeros(config.num_envs, dtype=jnp.int32)
        observation, env_state = wrapped_env.reset_to(
            reset_keys,
            trajectory_ids,
        )
    else:
        observation, env_state = wrapped_env.reset(reset_keys)
    if config.normalize_env:
        env_state = update_carry_threshold_normalized(
            env_state,
            initial_threshold,
        )
        env_state = update_carry_reward_weights_normalized(
            env_state,
            reward_curriculum_state.qvel_w_sum,
            reward_curriculum_state.root_vel_w_sum,
        )
    else:
        env_state = update_carry_threshold_unnormalized(
            env_state,
            initial_threshold,
        )
        env_state = update_carry_reward_weights_unnormalized(
            env_state,
            reward_curriculum_state.qvel_w_sum,
            reward_curriculum_state.root_vel_w_sum,
        )

    adaptive_sampling = config.get("adaptive_sampling", {})
    needs_trajectory_weights = adaptive_sampling.get("enabled", False) or trajectory_sharding_mode(config) == "static"
    if needs_trajectory_weights:
        num_trajectories = config.get("n_trajectories", None)
        if num_trajectories is None:
            raise ValueError("n_trajectories is required to restore runtime_state with trajectory weights.")
        uniform_weights = jnp.ones(num_trajectories) / num_trajectories
        batched_weights = jnp.broadcast_to(
            uniform_weights,
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
        if config.normalize_env:
            env_state = update_carry_weights_normalized(
                env_state,
                batched_weights,
            )
            env_state = update_carry_ema_normalized(
                env_state,
                done_ema,
                early_ema,
            )
        else:
            env_state = update_carry_weights_unnormalized(
                env_state,
                batched_weights,
            )
            env_state = update_carry_ema_unnormalized(
                env_state,
                done_ema,
                early_ema,
            )

    rng, training_rng = jax.random.split(rng)
    _, evaluation_rng = jax.random.split(rng)
    runtime_state = build_runtime_state(
        env_state,
        observation,
        training_rng,
        jnp.asarray(float(config.lr), dtype=jnp.float32),
        evaluation_rng,
        curriculum_state,
        reward_curriculum_state,
    )
    if num_devices > 1:
        return _add_device_axis(runtime_state, num_devices)
    return runtime_state
