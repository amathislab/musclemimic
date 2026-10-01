"""Validation trajectory sharing policy and attachment."""

from __future__ import annotations

from typing import Any

from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__)

VALIDATION_DATASET_CONFIG_KEYS = (
    "amass_dataset_conf",
    "c3d_dataset_conf",
    "lafan1_dataset_conf",
    "custom_dataset_conf",
)


def should_share_validation_trajectory(config: Any) -> bool:
    """Return whether validation should reuse the training motion library."""
    validation_config = config.experiment.get("validation")
    if not validation_config or not validation_config.get("active", False):
        return False
    if not validation_config.get("share_trajectory_handler", True):
        return False

    # A validation dataset override is authoritative. Equal motion names can
    # still represent different retargeting parameters or trajectory layouts.
    return not any(validation_config.get(key) is not None for key in VALIDATION_DATASET_CONFIG_KEYS)


def attach_shared_validation_trajectory(training_env: Any, validation_env: Any) -> None:
    """Attach the training trajectory to an existing validation environment."""
    training_handler = training_env.th
    if training_handler is None:
        raise ValueError("Cannot share a training trajectory without a trajectory handler.")

    if validation_env.th is not None and validation_env.th.traj is training_handler.traj:
        logger.info("Validation trajectory data already shared.")
        return

    validation_env.load_trajectory(
        traj=training_handler.traj,
        warn=False,
        cache_type=training_handler.cache_type,
        site_names=training_handler.cache_site_names,
        materialized=True,
        motion_names=training_handler.motion_names,
    )
    logger.info("Attached training trajectory to validation env.")
