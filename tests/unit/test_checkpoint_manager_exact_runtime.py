"""Orbax round trip for exact multi-device PPO runtime state."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.common.checkpoint_manager import (
    CheckpointMetadata,
    UnifiedCheckpointManager,
)


@struct.dataclass
class _TrainState:
    params: dict
    opt_state: dict
    step: jax.Array
    run_stats: dict


def _shape_target(tree):
    return jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype),
        tree,
    )


def test_exact_runtime_checkpoint_round_trip_preserves_replica_axis_and_empty_arrays(tmp_path):
    train_state = _TrainState(
        params={"weight": jnp.arange(3, dtype=jnp.float32)},
        opt_state={"count": jnp.asarray(7, dtype=jnp.int32)},
        step=jnp.asarray(11, dtype=jnp.int32),
        run_stats={"mean": jnp.zeros(3, dtype=jnp.float32)},
    )
    runtime_state = {
        "last_obs": jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3),
        "rng": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.uint32),
        "empty_contact": jnp.empty((2, 4, 0), dtype=jnp.float32),
    }
    metadata = CheckpointMetadata(
        step=11,
        update_number=3,
        global_timestep=96,
        target_global_timestep=192,
        learning_rate=3e-4,
        num_envs=8,
        num_steps=4,
        num_minibatches=2,
        update_epochs=1,
        num_devices=2,
        trajectory_sharding="none",
    )
    manager = UnifiedCheckpointManager(str(tmp_path), async_save=False)
    try:
        path = manager.save_checkpoint(
            3,
            SimpleNamespace(config=OmegaConf.create({"experiment": {}})),
            SimpleNamespace(train_state=train_state, runtime_state=runtime_state),
            metadata,
        )
        target = {
            "params": _shape_target(train_state.params),
            "opt_state": _shape_target(train_state.opt_state),
            "step": jax.ShapeDtypeStruct((), jnp.int32),
            "run_stats": _shape_target(train_state.run_stats),
            "runtime_state": _shape_target(runtime_state),
        }
        (_, restored), restored_metadata = manager.load_checkpoint(path, train_state_target=target)
    finally:
        manager.close()

    np.testing.assert_array_equal(restored["runtime_state"]["last_obs"], runtime_state["last_obs"])
    np.testing.assert_array_equal(restored["runtime_state"]["rng"], runtime_state["rng"])
    assert restored["runtime_state"]["empty_contact"].shape == (2, 4, 0)
    assert restored_metadata.num_devices == 2
    assert restored_metadata.trajectory_sharding == "none"
