import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from jax.sharding import Mesh, PartitionSpec as P

from loco_mujoco.trajectory.handler import TrajState, TrajectoryHandler


@struct.dataclass
class _FakeCarry:
    traj_state: TrajState


@struct.dataclass
class _ExplicitResetCarry:
    key: jnp.ndarray
    traj_state: TrajState


def _make_handler(lengths):
    handler = object.__new__(TrajectoryHandler)
    handler.len_trajectory = lambda traj_no: lengths[int(np.asarray(traj_no))]
    return handler


@pytest.mark.parametrize("backend", [np, jnp])
def test_reached_trajectory_end_detects_last_step(backend):
    handler = _make_handler([3])
    last_step = backend.asarray(2, dtype=backend.int32)
    prev_step = backend.asarray(1, dtype=backend.int32)

    at_end = handler.reached_trajectory_end(TrajState(0, last_step, 0), backend)
    before_end = handler.reached_trajectory_end(TrajState(0, prev_step, 0), backend)

    assert bool(np.asarray(at_end))
    assert not bool(np.asarray(before_end))


@pytest.mark.parametrize("backend", [np, jnp])
def test_update_state_advances_within_current_trajectory_without_wrap(backend):
    handler = _make_handler([4, 6])
    carry = _FakeCarry(
        traj_state=TrajState(
            traj_no=backend.asarray(0, dtype=backend.int32),
            subtraj_step_no=backend.asarray(2, dtype=backend.int32),
            subtraj_step_no_init=backend.asarray(1, dtype=backend.int32),
        )
    )

    next_carry = handler.update_state(None, None, None, carry, backend)

    assert int(np.asarray(next_carry.traj_state.traj_no)) == 0
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no)) == 3
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no_init)) == 1


@pytest.mark.parametrize("backend", [np, jnp])
def test_update_state_clamps_at_terminal_step_instead_of_switching_trajectory(backend):
    handler = _make_handler([4, 6])
    carry = _FakeCarry(
        traj_state=TrajState(
            traj_no=backend.asarray(0, dtype=backend.int32),
            subtraj_step_no=backend.asarray(3, dtype=backend.int32),
            subtraj_step_no_init=backend.asarray(0, dtype=backend.int32),
        )
    )

    next_carry = handler.update_state(None, None, None, carry, backend)

    assert int(np.asarray(next_carry.traj_state.traj_no)) == 0
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no)) == 3
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no_init)) == 0


@pytest.mark.parametrize("backend", [np, jnp])
def test_reset_state_to_step_does_not_sample_trajectory_length(backend):
    handler = _make_handler([4, 6])
    handler.len_trajectory = lambda _traj_no: (_ for _ in ()).throw(
        AssertionError("explicit resets must not trace random trajectory sampling")
    )
    carry = _ExplicitResetCarry(
        key=backend.asarray([1, 2], dtype=backend.uint32),
        traj_state=TrajState(0, 0, 0),
    )

    _, next_carry = handler.reset_state_to_step(
        None,
        None,
        None,
        carry,
        backend,
        backend.asarray(1, dtype=backend.int32),
        backend.asarray(5, dtype=backend.int32),
    )

    assert int(np.asarray(next_carry.traj_state.traj_no)) == 1
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no)) == 5
    assert int(np.asarray(next_carry.traj_state.subtraj_step_no_init)) == 5


def test_reset_state_to_step_compiles_with_data_parallel_vma():
    handler = _make_handler([4, 6])

    def reset_lane(traj_idx, subtraj_step):
        carry = _ExplicitResetCarry(
            key=jnp.asarray([1, 2], dtype=jnp.uint32),
            traj_state=TrajState(0, 0, 0),
        )
        _, next_carry = handler.reset_state_to_step(
            None,
            None,
            None,
            carry,
            jnp,
            traj_idx,
            subtraj_step,
        )
        return jnp.stack(
            (
                next_carry.traj_state.traj_no,
                next_carry.traj_state.subtraj_step_no,
                next_carry.traj_state.subtraj_step_no_init,
            )
        )

    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    reset_batch = jax.vmap(reset_lane)
    compiled_reset = jax.jit(
        jax.shard_map(
            reset_batch,
            mesh=mesh,
            in_specs=(P("data"), P("data")),
            out_specs=P("data"),
            check_vma=True,
        )
    )
    trajectory_ids = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    start_steps = jnp.asarray([0, 5, 2, 3], dtype=jnp.int32)

    result = compiled_reset(trajectory_ids, start_steps)

    np.testing.assert_array_equal(np.asarray(result[:, 0]), np.asarray(trajectory_ids))
    np.testing.assert_array_equal(np.asarray(result[:, 1]), np.asarray(start_steps))
    np.testing.assert_array_equal(np.asarray(result[:, 2]), np.asarray(start_steps))
