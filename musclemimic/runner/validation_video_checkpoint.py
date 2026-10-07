"""Read the exact policy checkpoint used by a validation video worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp

from .checkpointing import is_checkpoint_complete

DEFAULT_CHECKPOINT_WAIT_TIMEOUT_S = 300.0
CHECKPOINT_POLL_INTERVAL_S = 0.25
_POLICY_STATE_FIELDS = {
    "PPOJax": ("params", "run_stats"),
    "FlashSACJax": ("actor_params", "actor_batch_stats"),
}


@dataclass(frozen=True)
class ValidationPolicyCheckpoint:
    """Configuration and policy-only state restored for video inference."""

    config: dict[str, Any]
    policy_state: dict[str, Any]


def wait_for_checkpoint(
    checkpoint_path: str | Path,
    *,
    timeout_s: float = DEFAULT_CHECKPOINT_WAIT_TIMEOUT_S,
    poll_interval_s: float = CHECKPOINT_POLL_INTERVAL_S,
) -> Path:
    """Wait until the exact Orbax checkpoint is finalized and readable."""
    path = Path(checkpoint_path).resolve()
    deadline = monotonic() + float(timeout_s)
    while not is_checkpoint_complete(path):
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"Checkpoint was not finalized within {timeout_s:g}s: {path}")
        sleep(min(float(poll_interval_s), remaining))
    return path


def load_validation_policy_checkpoint(
    checkpoint_path: str | Path,
    *,
    timeout_s: float = DEFAULT_CHECKPOINT_WAIT_TIMEOUT_S,
) -> ValidationPolicyCheckpoint:
    """Restore config and the configured algorithm's policy-only state."""
    path = wait_for_checkpoint(checkpoint_path, timeout_s=timeout_s)
    train_state_path = path / "train_state"

    state_handler = ocp.PyTreeCheckpointHandler()
    config_handler = ocp.JsonCheckpointHandler()
    try:
        config = dict(config_handler.restore(path / "config"))
        algorithm_name = config["experiment"]["algorithm"]
        try:
            policy_fields = _POLICY_STATE_FIELDS[algorithm_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported checkpoint algorithm for validation video: {algorithm_name}") from exc

        state_metadata = state_handler.metadata(train_state_path).tree
        missing_fields = [field for field in policy_fields if field not in state_metadata]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Checkpoint is missing policy state fields ({missing}): {path}")

        policy_metadata = {field: state_metadata[field] for field in policy_fields}
        restore_args = jax.tree_util.tree_map(
            lambda _metadata: ocp.RestoreArgs(restore_type=np.ndarray),
            policy_metadata,
        )
        policy_state = state_handler.restore(
            train_state_path,
            args=ocp.args.PyTreeRestore(
                item=policy_metadata,
                restore_args=restore_args,
                partial_restore=True,
            ),
        )
    finally:
        state_handler.close()
        config_handler.close()

    return ValidationPolicyCheckpoint(
        config=config,
        policy_state=policy_state,
    )
