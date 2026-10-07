"""Helpers for applying validation-only environment overrides."""

from typing import Any

from omegaconf import OmegaConf


def to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True) or {})
    return dict(value)


def apply_validation_reward_termination(env_params: dict[str, Any], validation_config: Any | None) -> None:
    """Apply validation-only reward termination settings to copied env params."""
    if validation_config is None:
        return

    reward_termination_type = validation_config.get("reward_termination_type", None)
    if reward_termination_type is not None:
        env_params["reward_termination_type"] = reward_termination_type

    reward_termination_params = validation_config.get("reward_termination_params", None)
    if reward_termination_params is not None:
        env_params["reward_termination_params"] = to_plain_dict(reward_termination_params)
