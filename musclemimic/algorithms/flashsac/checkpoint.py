"""FlashSAC checkpoint restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.checkpoint_manager import UnifiedCheckpointManager
from musclemimic.utils.logging import setup_logger

from .replay_buffer import ReplayBuffer
from .state import FlashSACAgentState, FlashSACLearnerState, FlashSACTrainState
from .train import initialize_learner_state

logger = setup_logger(__name__)
FLASHSAC_ALGO_VERSION = "FlashSACJax_v2"


def _reset_reward_normalizer_state(
    state: FlashSACLearnerState | FlashSACTrainState,
) -> FlashSACLearnerState | FlashSACTrainState:
    """Reset incompatible running statistics while preserving learner state."""
    return state.replace(
        reward_return_mean=jnp.asarray(0.0, dtype=jnp.float32),
        reward_return_var=jnp.asarray(1.0, dtype=jnp.float32),
        reward_return_count=jnp.asarray(0.0, dtype=jnp.float32),
        reward_return_max=jnp.asarray(0.0, dtype=jnp.float32),
    )


def restore_train_state(
    state_dict: dict[str, Any],
    agent_conf: Any,
) -> FlashSACLearnerState | FlashSACTrainState:
    """Reconstruct typed FlashSAC and Optax state from an Orbax mapping."""
    learner_state_dict = dict(state_dict)
    replay_state_dict = learner_state_dict.pop("replay_buffer", None)
    learner_template = initialize_learner_state(
        jax.random.key(0),
        agent_conf=agent_conf,
    )
    learner = flax.serialization.from_state_dict(learner_template, learner_state_dict)
    if replay_state_dict is None:
        return learner
    # Build replay directly from the restored arrays. A same-shaped zero target
    # would briefly duplicate large replay storage during legacy checkpoint load.
    return learner.with_replay_buffer(ReplayBuffer(**replay_state_dict))


def load_checkpoint_for_resume(
    checkpoint_path: str,
    agent_conf: Any,
) -> tuple[FlashSACAgentState, dict[str, Any]]:
    """Load off-policy learner state and progress metadata."""
    path = Path(checkpoint_path)
    manager = UnifiedCheckpointManager(
        checkpoint_dir=str(path.parent),
        max_to_keep=5,
    )
    try:
        (_, state_dict), metadata = manager.load_checkpoint(str(path))
    finally:
        manager.close()

    train_state = restore_train_state(state_dict, agent_conf)
    if metadata.algo_version != FLASHSAC_ALGO_VERSION:
        train_state = _reset_reward_normalizer_state(train_state)
        logger.warning(
            "Reset FlashSAC reward normalizer while migrating checkpoint %s to %s",
            metadata.algo_version,
            FLASHSAC_ALGO_VERSION,
        )

    agent_state = FlashSACAgentState(train_state=train_state)
    resume_info = {
        "optimizer_step": int(metadata.step),
        "update_number": int(metadata.update_number),
        "global_timestep": int(metadata.global_timestep),
        "target_global_timestep": int(metadata.target_global_timestep),
        "num_envs": int(metadata.num_envs),
        "num_steps": int(metadata.num_steps),
        "num_minibatches": int(metadata.num_minibatches),
        "update_epochs": int(metadata.update_epochs),
    }
    return agent_state, resume_info
