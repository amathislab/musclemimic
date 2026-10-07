"""Training and validation environment construction."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms.common.distributed import device_count
from musclemimic.utils.logging import setup_logger

from .trajectory_sharing import VALIDATION_DATASET_CONFIG_KEYS
from .validation_config import apply_validation_reward_termination

logger = setup_logger(__name__)

VALIDATION_SIMULATION_OVERRIDE_KEYS = ("nconmax", "naconmax", "njmax")


def _training_env_params(config: Any, *, localize_num_envs: bool) -> dict[str, Any]:
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    if not localize_num_envs:
        return env_params

    n_devices = device_count(config)
    if n_devices <= 1:
        return env_params

    global_num_envs = int(config.experiment.get("num_envs", env_params.get("num_envs", 1)))
    if global_num_envs % n_devices != 0:
        raise ValueError(
            f"num_envs ({global_num_envs}) must be divisible by num_devices "
            f"({n_devices}). Set distributed.num_devices or round num_envs up "
            "to the next multiple."
        )
    env_params["num_envs"] = global_num_envs // n_devices
    return env_params


def _validation_env_params(config: Any) -> dict[str, Any]:
    validation = config.experiment.validation
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)

    for key in VALIDATION_SIMULATION_OVERRIDE_KEYS:
        value = validation.get(key)
        if value is not None:
            env_params[key] = value

    env_params["terminal_state_type"] = validation.get(
        "terminal_state_type",
        "NoTerminalStateHandler",
    )
    env_params["terminal_state_params"] = validation.get("terminal_state_params", {})
    apply_validation_reward_termination(env_params, validation)

    init_state_type = validation.get("init_state_type")
    if init_state_type is not None:
        init_state_params = validation.get("init_state_params")
        env_params["init_state_type"] = init_state_type
        env_params["init_state_params"] = (
            OmegaConf.to_container(init_state_params, resolve=True) if init_state_params is not None else {}
        )

    if validation.get("start_from_beginning", False):
        env_params.setdefault("th_params", {})["start_from_random_step"] = False

    env_params["num_envs"] = validation.get(
        "num_envs",
        config.experiment.env_params.num_envs,
    )

    validation_th_params = validation.get("th_params")
    if validation_th_params is not None:
        base_th_params = env_params.get("th_params", {}) or {}
        env_params["th_params"] = {
            **base_th_params,
            **OmegaConf.to_container(validation_th_params, resolve=True),
        }
    return env_params


def _validation_task_factory_params(config: Any, *, share_trajectory: bool) -> dict[str, Any]:
    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    if share_trajectory:
        for key in VALIDATION_DATASET_CONFIG_KEYS:
            task_params.pop(key, None)
        logger.info("Skipping validation trajectory load (will share from training env).")
        return task_params

    validation = config.experiment.validation
    dataset_overrides = {
        key: validation.get(key) for key in VALIDATION_DATASET_CONFIG_KEYS if validation.get(key) is not None
    }
    if dataset_overrides:
        for key in VALIDATION_DATASET_CONFIG_KEYS:
            task_params.pop(key, None)

    for key, dataset_config in dataset_overrides.items():
        task_params[key] = (
            OmegaConf.to_container(dataset_config, resolve=True)
            if OmegaConf.is_config(dataset_config)
            else dataset_config
        )
        logger.info(f"Using validation-specific {key}.")
    return task_params


def instantiate_env(config: Any, *, localize_num_envs: bool = True) -> Any:
    """Create the training environment with the correct per-device batch size."""
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(
        **_training_env_params(config, localize_num_envs=localize_num_envs),
        **config.experiment.task_factory.params,
    )
    return env


def instantiate_validation_env(config: Any, *, share_trajectory: bool = False) -> Any | None:
    """Create the validation environment, optionally without loading trajectory data."""
    if not config.experiment.get("validation", {}).get("active", False):
        return None

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(
        **_validation_env_params(config),
        **_validation_task_factory_params(config, share_trajectory=share_trajectory),
    )
    return env
