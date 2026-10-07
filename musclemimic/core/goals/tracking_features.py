"""Heading-local features for full-body trajectory tracking goals."""

from types import ModuleType
from typing import Any

from loco_mujoco.core.utils.math import calc_site_velocities
from musclemimic.core.observations.egocentric import (
    root_velocity_in_heading_frame,
    rotation_error_in_frame,
    world_to_heading,
)

ROOT_TRACKING_ERROR_DIM = 12
SITE_TRACKING_ERROR_DIM = 12
FUTURE_ROOT_INTENT_DIM = 12
FUTURE_SITE_INTENT_DIM = 3
FUTURE_METADATA_DIM = 2
SUPPORT_INTENT_DIM_PER_SITE = 2


def tracking_goal_dimension(
    n_relative_sites: int,
    n_horizons: int,
    n_support_sites: int,
) -> int:
    """Return the flattened tracking-goal dimension.

    The first horizon contains the full next-frame tracking error. Each later
    horizon contains a compact intent followed by ``[time_offset, is_valid]``.
    Every horizon can also contain floor-relative height and vertical velocity
    for each reference support site.
    """
    if n_relative_sites < 0:
        raise ValueError("n_relative_sites must be non-negative")
    if n_horizons < 1:
        raise ValueError("n_horizons must be at least one")
    if n_support_sites < 0:
        raise ValueError("n_support_sites must be non-negative")

    support_dim = SUPPORT_INTENT_DIM_PER_SITE * n_support_sites
    immediate_dim = ROOT_TRACKING_ERROR_DIM + SITE_TRACKING_ERROR_DIM * n_relative_sites + support_dim
    future_dim = FUTURE_ROOT_INTENT_DIM + FUTURE_SITE_INTENT_DIM * n_relative_sites + support_dim + FUTURE_METADATA_DIM
    return immediate_dim + future_dim * (n_horizons - 1)


def fixed_root_tracking_goal_dimension(
    n_relative_sites: int,
    n_horizons: int,
) -> int:
    """Return the flattened tracking-goal dimension for a fixed-root model."""
    if n_relative_sites < 0:
        raise ValueError("n_relative_sites must be non-negative")
    if n_horizons < 1:
        raise ValueError("n_horizons must be at least one")

    immediate_dim = SITE_TRACKING_ERROR_DIM * n_relative_sites
    future_dim = FUTURE_SITE_INTENT_DIM * n_relative_sites + FUTURE_METADATA_DIM
    return immediate_dim + future_dim * (n_horizons - 1)


def site_kinematics(
    data: Any,
    site_ids: Any,
    site_body_ids: Any,
    body_root_ids: Any,
    backend: ModuleType,
    trajectory_site_indices: Any = None,
) -> tuple[Any, Any, Any]:
    """Return site world positions, rotations, and spatial velocities."""
    positions = site_positions(data, site_ids, trajectory_site_indices)
    data_indices = site_ids if trajectory_site_indices is None else trajectory_site_indices
    rotations = data.site_xmat[data_indices].reshape(-1, 3, 3)
    root_body_ids = body_root_ids[site_body_ids]
    velocities = calc_site_velocities(
        site_ids,
        data,
        site_body_ids,
        root_body_ids,
        backend,
        flg_local=False,
        trajectory_site_indices=trajectory_site_indices,
    )
    return positions, rotations, velocities


def site_positions(
    data: Any,
    site_ids: Any,
    trajectory_site_indices: Any = None,
) -> Any:
    """Return site world positions in model site order."""
    data_indices = site_ids if trajectory_site_indices is None else trajectory_site_indices
    return data.site_xpos[data_indices]


def reference_support_features(
    target_site_positions: Any,
    target_site_velocities: Any,
    support_site_indices: Any,
    floor_height: Any,
    backend: ModuleType,
) -> Any:
    """Encode continuous reference support cues as height and vertical velocity.

    Heights use world Z relative to the flat floor. Both quantities remain
    invariant to global XY translation and heading rotation while preserving
    the vertical motion needed to distinguish support, takeoff, and landing.
    """
    support_positions = target_site_positions[support_site_indices]
    support_velocities = target_site_velocities[support_site_indices]
    heights = support_positions[:, 2] - floor_height
    vertical_velocities = support_velocities[:, 5]
    return backend.ravel(backend.stack([heights, vertical_velocities], axis=-1))


def root_tracking_error_features(
    current_position: Any,
    current_rotation: Any,
    current_qvel: Any,
    target_position: Any,
    target_rotation: Any,
    target_qvel: Any,
    heading_rotation: Any,
    backend: ModuleType,
) -> Any:
    """Encode target-minus-current root errors."""
    position_error = world_to_heading(
        target_position - current_position,
        heading_rotation,
        backend,
    )
    rotation_error = rotation_error_in_frame(
        current_rotation,
        target_rotation,
        heading_rotation,
        backend,
    )
    current_velocity = root_velocity_in_heading_frame(
        current_qvel,
        current_rotation,
        heading_rotation,
        backend,
    )
    target_velocity = root_velocity_in_heading_frame(
        target_qvel,
        target_rotation,
        heading_rotation,
        backend,
    )
    return backend.concatenate([position_error, rotation_error, target_velocity - current_velocity])


def site_tracking_error_features(
    current_positions: Any,
    current_rotations: Any,
    current_velocities: Any,
    target_positions: Any,
    target_rotations: Any,
    target_velocities: Any,
    heading_rotation: Any,
    backend: ModuleType,
) -> Any:
    """Encode root-relative target-minus-current site errors."""
    current_relative_positions = current_positions[1:] - current_positions[0]
    target_relative_positions = target_positions[1:] - target_positions[0]
    position_errors = world_to_heading(
        target_relative_positions - current_relative_positions,
        heading_rotation,
        backend,
    )

    rotation_errors = rotation_error_in_frame(
        current_rotations[1:],
        target_rotations[1:],
        heading_rotation,
        backend,
    )

    current_relative_velocities = current_velocities[1:] - current_velocities[0]
    target_relative_velocities = target_velocities[1:] - target_velocities[0]
    velocity_errors_world = target_relative_velocities - current_relative_velocities
    angular_errors = world_to_heading(
        velocity_errors_world[:, :3],
        heading_rotation,
        backend,
    )
    linear_errors = world_to_heading(
        velocity_errors_world[:, 3:],
        heading_rotation,
        backend,
    )
    velocity_errors = backend.concatenate([angular_errors, linear_errors], axis=-1)

    return backend.concatenate(
        [
            backend.ravel(position_errors),
            backend.ravel(rotation_errors),
            backend.ravel(velocity_errors),
        ]
    )


def future_intent_features(
    current_root_position: Any,
    current_root_rotation: Any,
    target_root_position: Any,
    target_root_rotation: Any,
    target_root_qvel: Any,
    target_site_positions: Any,
    target_support_features: Any | None,
    heading_rotation: Any,
    time_offset_seconds: Any,
    is_valid: Any,
    backend: ModuleType,
) -> Any:
    """Encode a compact future target in the current simulated heading frame."""
    root_displacement = world_to_heading(
        target_root_position - current_root_position,
        heading_rotation,
        backend,
    )
    root_rotation = rotation_error_in_frame(
        current_root_rotation,
        target_root_rotation,
        heading_rotation,
        backend,
    )
    root_velocity = root_velocity_in_heading_frame(
        target_root_qvel,
        target_root_rotation,
        heading_rotation,
        backend,
    )
    components = [
        root_displacement,
        root_rotation,
        root_velocity,
    ]
    components.append(
        future_site_intent_features(
            target_site_positions,
            heading_rotation,
            time_offset_seconds,
            is_valid,
            backend,
            additional_features=target_support_features,
        )
    )
    return backend.concatenate(components)


def future_site_intent_features(
    target_site_positions: Any,
    heading_rotation: Any,
    time_offset_seconds: Any,
    is_valid: Any,
    backend: ModuleType,
    *,
    additional_features: Any | None = None,
) -> Any:
    """Encode future site positions relative to the anchor site and heading."""
    relative_site_positions = target_site_positions[1:] - target_site_positions[0]
    relative_site_positions = world_to_heading(
        relative_site_positions,
        heading_rotation,
        backend,
    )
    metadata = backend.stack([time_offset_seconds, is_valid])
    components = [backend.ravel(relative_site_positions)]
    if additional_features is not None:
        components.append(additional_features)
    components.append(metadata)
    return backend.concatenate(components)
