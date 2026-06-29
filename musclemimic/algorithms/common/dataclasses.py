from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state

from musclemimic.core.wrappers.mjx import Metrics
from musclemimic.environments.base import TrajState


class Transition(NamedTuple):
    done: jnp.ndarray
    absorbing: jnp.ndarray
    traj_end: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    traj_state: TrajState
    metrics: Metrics


class ValidationDataFields(NamedTuple):
    """MJX Data fields needed for validation metrics.

    Matches the interface of mjx.Data for the fields accessed by MetricsHandler.
    """

    qpos: jnp.ndarray  # (B, nq)
    qvel: jnp.ndarray  # (B, nv)
    xpos: jnp.ndarray  # (B, nbody, 3)
    xquat: jnp.ndarray  # (B, nbody, 4)
    cvel: jnp.ndarray  # (B, nbody, 6)
    subtree_com: jnp.ndarray  # (B, nbody, 3) - needed for calc_site_velocities
    site_xpos: jnp.ndarray  # (B, nsite, 3)
    site_xmat: jnp.ndarray  # (B, nsite, 3, 3)


class ValidationCarry(NamedTuple):
    """Additional carry fields needed for validation metrics."""

    traj_state: TrajState


class ValidationData(NamedTuple):
    """Container for validation metrics computation."""

    metrics: Metrics
    data: ValidationDataFields
    additional_carry: ValidationCarry


class MetricHandlerTransition(NamedTuple):
    """Transition for validation metric handling."""

    val_data: ValidationData


class TrainState(train_state.TrainState):
    run_stats: Any
