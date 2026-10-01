"""Dynamic state carried between PPO updates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

import jax
from flax import struct

from musclemimic.algorithms.common.curriculum import (
    CurriculumState,
    RewardCurriculumState,
)
from musclemimic.algorithms.common.dataclasses import TrainState


@struct.dataclass
class PPORuntimeState:
    """Values carried from one PPO update to the next.

    The field names match the existing checkpoint format.
    """

    train_state: TrainState
    env_state: Any
    last_obs: jax.Array
    rng: jax.Array
    lr: jax.Array
    val_rng: jax.Array
    curriculum_state: CurriculumState
    reward_curriculum_state: RewardCurriculumState

    @classmethod
    def from_checkpoint_state(
        cls,
        train_state: TrainState,
        checkpoint_state: Mapping[str, Any],
    ) -> Self:
        """Restore the runtime pytree from the existing checkpoint schema."""
        return cls(
            train_state=train_state,
            env_state=checkpoint_state["env_state"],
            last_obs=checkpoint_state["last_obs"],
            rng=checkpoint_state["rng"],
            lr=checkpoint_state["lr"],
            val_rng=checkpoint_state["val_rng"],
            curriculum_state=checkpoint_state["curriculum_state"],
            reward_curriculum_state=checkpoint_state["reward_curriculum_state"],
        )

    def to_checkpoint_state(self) -> dict[str, Any]:
        """Return the legacy-compatible runtime checkpoint mapping."""
        return {
            "env_state": self.env_state,
            "last_obs": self.last_obs,
            "rng": self.rng,
            "lr": self.lr,
            "val_rng": self.val_rng,
            "curriculum_state": self.curriculum_state,
            "reward_curriculum_state": self.reward_curriculum_state,
        }
