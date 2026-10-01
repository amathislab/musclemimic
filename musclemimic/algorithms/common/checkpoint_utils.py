"""Shared utilities for checkpoint state calculations.

Single source of truth for converting between:
- Optimizer steps (train_state.step)
- Update numbers (rollout iterations)
- Global timesteps (environment steps)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax


@dataclass
class TrainingConfig:
    """Training configuration snapshot for checkpoints."""

    num_envs: int
    num_steps: int
    num_minibatches: int
    update_epochs: int

    @classmethod
    def from_experiment_config(cls, exp_config):
        """Extract from OmegaConf experiment config."""
        num_envs = getattr(exp_config, "global_num_envs", None)
        if num_envs is None:
            num_envs = exp_config.num_envs
        return cls(
            num_envs=int(num_envs),
            num_steps=int(exp_config.num_steps),
            num_minibatches=int(exp_config.num_minibatches),
            update_epochs=int(exp_config.update_epochs),
        )


def optimizer_step_to_update(optimizer_step: int, num_minibatches: int, update_epochs: int) -> int:
    """Convert optimizer step to update number.

    This is the canonical formula used throughout the codebase.
    Matches the training loop counter calculation.

    Args:
        optimizer_step: Current optimizer step (train_state.step)
        num_minibatches: Number of minibatches per update
        update_epochs: Number of epochs per update

    Returns:
        Update number (rollout iteration count)
    """
    return ((optimizer_step + 1) // num_minibatches) // update_epochs


def update_to_global_timestep(update_number: int, num_steps: int, num_envs: int) -> int:
    """Convert update number to global environment timesteps.

    Args:
        update_number: Update/rollout iteration number
        num_steps: Steps per rollout
        num_envs: Parallel environments

    Returns:
        Total environment steps across all environments
    """
    return update_number * num_steps * num_envs


def compute_checkpoint_metadata(optimizer_step: int, config: TrainingConfig, learning_rate: float) -> dict:
    """Compute all checkpoint metadata from optimizer step and config.

    Single source of truth for metadata calculation.

    Args:
        optimizer_step: Current optimizer step
        config: Training configuration
        learning_rate: Current learning rate

    Returns:
        Dictionary with all metadata fields
    """
    update_num = optimizer_step_to_update(optimizer_step, config.num_minibatches, config.update_epochs)
    global_ts = update_to_global_timestep(update_num, config.num_steps, config.num_envs)

    return {
        "step": optimizer_step,
        "update_number": update_num,
        "global_timestep": global_ts,
        "learning_rate": learning_rate,
        "num_envs": config.num_envs,
        "num_steps": config.num_steps,
        "num_minibatches": config.num_minibatches,
        "update_epochs": config.update_epochs,
    }


def compute_resume_state(
    checkpoint_metadata: dict, current_config: TrainingConfig, total_timesteps: int
) -> tuple[int, int, bool]:
    """Compute resume state from checkpoint metadata and current config.

    Args:
        checkpoint_metadata: Loaded checkpoint metadata
        current_config: Current training configuration
        total_timesteps: Total timesteps for this training run

    Returns:
        Tuple of (completed_updates, remaining_updates, config_changed)

    Raises:
        ValueError: If checkpoint metadata is missing required fields
    """
    # Extract checkpoint state (strict - no defaults)
    try:
        completed_updates = int(checkpoint_metadata["update_number"])
        global_ts = int(checkpoint_metadata["global_timestep"])
        ckpt_num_envs = int(checkpoint_metadata["num_envs"])
        ckpt_num_steps = int(checkpoint_metadata["num_steps"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"Checkpoint missing required metadata fields. "
            f"Please create a new checkpoint with the updated format. Error: {e}"
        ) from e

    # Detect config change
    config_changed = ckpt_num_envs != current_config.num_envs or ckpt_num_steps != current_config.num_steps

    # Compute remaining updates with current config
    current_steps_per_update = current_config.num_steps * current_config.num_envs
    remaining_ts = max(total_timesteps - global_ts, 0)
    # Use ceiling division for remaining updates
    remaining_updates = (remaining_ts + current_steps_per_update - 1) // current_steps_per_update

    return completed_updates, remaining_updates, config_changed


def reset_lr_schedule_count(opt_state):
    """Reset ScaleByScheduleState.count to 0 while preserving other optimizer state.

    This allows the LR schedule to start fresh when resuming from a checkpoint,
    while keeping momentum/Adam statistics intact. Useful for finetuning where
    you want a fresh LR schedule but don't want to lose optimizer momentum.

    Args:
        opt_state: The optimizer state pytree (from optax)

    Returns:
        New optimizer state with schedule counters reset to 0
    """

    def reset_if_schedule_state(node):
        if isinstance(node, optax.ScaleByScheduleState):
            return optax.ScaleByScheduleState(count=jnp.zeros_like(node.count))
        return node

    return jax.tree_util.tree_map(
        reset_if_schedule_state,
        opt_state,
        is_leaf=lambda x: isinstance(x, optax.ScaleByScheduleState),
    )
