"""Shared horizon handling for compact trajectory tracking goals."""

from collections.abc import Sequence
from itertools import pairwise
from numbers import Integral
from types import ModuleType
from typing import Any

import jax.numpy as jnp

DEFAULT_LOOKAHEAD_STEPS = (1, 5, 10, 20, 40, 80)


def validate_lookahead_steps(steps: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize tracking horizons measured in control steps."""
    steps = tuple(steps)
    if any(isinstance(step, bool) or not isinstance(step, Integral) for step in steps):
        raise TypeError("lookahead_steps must contain integers")
    steps = tuple(int(step) for step in steps)
    if not steps or steps[0] != 1:
        raise ValueError("lookahead_steps must start with 1")
    if any(step <= 0 for step in steps):
        raise ValueError("lookahead_steps must be positive")
    if any(current >= following for current, following in pairwise(steps)):
        raise ValueError("lookahead_steps must be strictly increasing")
    return steps


def rescale_lookahead_steps(
    steps: Sequence[int],
    old_control_dt: float,
    new_control_dt: float,
) -> tuple[int, ...]:
    """Preserve future intent times after changing the control timestep.

    The first horizon remains the next frame. Later horizons use the nearest
    distinct frame at the new control rate.
    """
    steps = validate_lookahead_steps(steps)
    if old_control_dt <= 0 or new_control_dt <= 0:
        raise ValueError("control timesteps must be positive")

    scale = old_control_dt / new_control_dt
    scaled_steps = [1]
    for step in steps[1:]:
        scaled_step = round(step * scale)
        scaled_steps.append(max(scaled_step, scaled_steps[-1] + 1))
    return tuple(scaled_steps)


class TrackingGoalMixin:
    """Provide validated horizons and trajectory indexing for tracking goals."""

    lookahead_steps: tuple[int, ...]

    def __init__(
        self,
        *args: Any,
        lookahead_steps: Sequence[int] = DEFAULT_LOOKAHEAD_STEPS,
        **kwargs: Any,
    ) -> None:
        self.lookahead_steps = validate_lookahead_steps(lookahead_steps)
        super().__init__(*args, **kwargs)

    def _trajectory_site_indices_for_backend(self, backend: ModuleType) -> Any:
        trajectory_site_indices = None
        if self._site_mapper.requires_mapping:
            trajectory_site_indices = self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
            if backend is jnp:
                trajectory_site_indices = backend.asarray(trajectory_site_indices)
        return trajectory_site_indices

    @staticmethod
    def _bounded_target_step(
        current_step: Any,
        step_offset: int,
        trajectory_length: Any,
        backend: ModuleType,
    ) -> tuple[Any, Any]:
        requested_step = current_step + step_offset
        is_valid = requested_step < trajectory_length
        if backend is jnp:
            target_step = backend.clip(requested_step, 0, trajectory_length - 1)
        else:
            target_step = max(0, min(requested_step, trajectory_length - 1))
        return target_step, is_valid
