"""Restore checkpoints whose saved device is absent from the current process."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("orbax.checkpoint")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Released checkpoints store device arrays with the saving device in their
# sharding metadata. Saving on cpu:1 and restoring with one host device
# reproduces a cpu:0 checkpoint restored under a GPU-only backend.
SAVE_LEGACY_CHECKPOINT = textwrap.dedent(
    """
    import sys

    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    weights = jax.device_put(jnp.arange(4, dtype=jnp.float32), jax.devices()[1])
    options = ocp.CheckpointManagerOptions(step_prefix="checkpoint", enable_async_checkpointing=False)
    manager = ocp.CheckpointManager(sys.argv[1], options=options, item_names=("train_state", "config", "metadata"))
    manager.save(
        7,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardSave({"params": {"w": weights}}),
            config=ocp.args.JsonSave({"experiment": {"num_envs": 2}}),
            metadata=ocp.args.JsonSave({"step": 7}),
        ),
    )
    manager.wait_until_finished()
    manager.close()
    """
)

RESTORE_CHECKPOINT = textwrap.dedent(
    """
    import sys

    import jax
    import numpy as np

    from musclemimic.algorithms.common.checkpoint_manager import OrbaxCheckpointManager

    checkpoint_path = sys.argv[1]
    manager = OrbaxCheckpointManager(str(sys.argv[2]), async_save=False)
    target = {"params": {"w": jax.ShapeDtypeStruct((4,), np.float32)}}
    for train_state_target in (None, target):
        (_, state), metadata = manager.load_checkpoint(checkpoint_path, train_state_target=train_state_target)
        np.testing.assert_array_equal(np.asarray(state["params"]["w"]), np.arange(4, dtype=np.float32))
        assert metadata.step == 7
    print("restored")
    """
)


def _run_with_host_devices(script: str, device_count: int, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={device_count}",
            "PYTHONPATH": os.pathsep.join(filter(None, [str(REPOSITORY_ROOT), env.get("PYTHONPATH")])),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPOSITORY_ROOT,
        timeout=300,
    )


def test_restore_places_arrays_saved_on_absent_device(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    saved = _run_with_host_devices(SAVE_LEGACY_CHECKPOINT, 2, str(checkpoint_dir))
    assert saved.returncode == 0, saved.stderr

    restored = _run_with_host_devices(RESTORE_CHECKPOINT, 1, str(checkpoint_dir / "checkpoint_7"), str(checkpoint_dir))
    assert restored.returncode == 0, restored.stderr
    assert "restored" in restored.stdout
