"""PPO environment construction with explicit global-to-local batch semantics."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms.common.ppo_distributed import device_count
from musclemimic.algorithms.common.ppo_trajectory_sharding import trajectory_sharding_mode

VALIDATION_DATASET_CONFIG_KEYS = (
    "amass_dataset_conf",
    "c3d_dataset_conf",
    "lafan1_dataset_conf",
    "custom_dataset_conf",
)


def _local_batch_size(global_size: int, n_devices: int, *, setting: str) -> int:
    if global_size % n_devices != 0:
        raise ValueError(f"{setting} ({global_size}) must be divisible by num_devices ({n_devices})")
    return global_size // n_devices


def training_env_params(config: Any) -> dict[str, Any]:
    """Resolve per-device training environment parameters."""
    params = dict(OmegaConf.to_container(config.experiment.env_params, resolve=True))
    n_devices = device_count(config)
    global_num_envs = int(config.experiment.get("num_envs", params.get("num_envs", 1)))
    params["num_envs"] = _local_batch_size(
        global_num_envs,
        n_devices,
        setting="num_envs",
    )
    return params


def validation_env_params(config: Any) -> dict[str, Any]:
    """Resolve validation overrides and interpret validation.num_envs globally."""
    validation = config.experiment.validation
    params = dict(OmegaConf.to_container(config.experiment.env_params, resolve=True))
    explicit_overrides = validation.get("env_params")
    if explicit_overrides is not None:
        params.update(dict(OmegaConf.to_container(explicit_overrides, resolve=True)))

    params["terminal_state_type"] = validation.get(
        "terminal_state_type",
        "NoTerminalStateHandler",
    )
    params["terminal_state_params"] = OmegaConf.to_container(
        validation.get("terminal_state_params", {}),
        resolve=True,
    )
    init_state_type = validation.get("init_state_type")
    if init_state_type is not None:
        params["init_state_type"] = init_state_type
        params["init_state_params"] = OmegaConf.to_container(
            validation.get("init_state_params", {}),
            resolve=True,
        )
    if validation.get("start_from_beginning", False):
        params.setdefault("th_params", {})["start_from_random_step"] = False
    validation_th_params = validation.get("th_params")
    if validation_th_params is not None:
        params["th_params"] = {
            **(params.get("th_params", {}) or {}),
            **dict(OmegaConf.to_container(validation_th_params, resolve=True)),
        }

    n_devices = device_count(config)
    global_num_envs = int(validation.get("num_envs", config.experiment.env_params.num_envs))
    params["num_envs"] = _local_batch_size(
        global_num_envs,
        n_devices,
        setting="validation.num_envs",
    )
    return params


def validation_task_factory_params(config: Any) -> dict[str, Any]:
    """Apply validation dataset overrides without reordering their motion list."""
    params = dict(
        OmegaConf.to_container(
            config.experiment.task_factory.params,
            resolve=True,
        )
    )
    validation = config.experiment.validation
    overrides = {key: validation.get(key) for key in VALIDATION_DATASET_CONFIG_KEYS if validation.get(key) is not None}
    if overrides:
        for key in VALIDATION_DATASET_CONFIG_KEYS:
            params.pop(key, None)
    for key, value in overrides.items():
        params[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    for key in ("trajectory_cache_root", "trajectory_cache_key"):
        value = validation.get(key)
        if value is not None:
            params[key] = value
    return params


def validation_uses_training_trajectory(config: Any) -> bool:
    """Return whether validation resolves to the exact training dataset config."""
    validation = config.experiment.get("validation", {})
    if not validation.get("active", False) or not validation.get("share_trajectory_handler", False):
        return False
    training_params = dict(OmegaConf.to_container(config.experiment.task_factory.params, resolve=True))
    validation_params = validation_task_factory_params(config)
    return {
        key: training_params.get(key) for key in VALIDATION_DATASET_CONFIG_KEYS if training_params.get(key) is not None
    } == {
        key: validation_params.get(key)
        for key in VALIDATION_DATASET_CONFIG_KEYS
        if validation_params.get(key) is not None
    }


def _convert_trajectory_to_jax(env: Any, config: Any) -> None:
    if trajectory_sharding_mode(config) == "static":
        return
    handler = getattr(env, "th", None)
    if getattr(env, "mjx_enabled", False) and handler is not None and handler.is_numpy:
        handler.to_jax()


def instantiate_training_env(config: Any) -> Any:
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(
        **training_env_params(config),
        **config.experiment.task_factory.params,
    )
    _convert_trajectory_to_jax(env, config)
    return env


def instantiate_validation_env(config: Any, training_env: Any | None = None) -> Any | None:
    if not config.experiment.get("validation", {}).get("active", False):
        return None
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    params = validation_task_factory_params(config)
    if validation_uses_training_trajectory(config):
        if training_env is None or getattr(training_env, "th", None) is None:
            raise ValueError("shared PPO validation trajectory requires a populated training environment")
        params["trajectory_handler"] = training_env.th
    env = factory.make(**validation_env_params(config), **params)
    _convert_trajectory_to_jax(env, config)
    return env


__all__ = [
    "instantiate_training_env",
    "instantiate_validation_env",
    "training_env_params",
    "validation_env_params",
    "validation_uses_training_trajectory",
]
