from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from loco_mujoco.trajectory.handler import TrajectoryHandler, TrajState
from musclemimic.environments.base import LocoEnv


@struct.dataclass
class _FakeCarry:
    traj_state: TrajState


@struct.dataclass
class _ResetCarry:
    key: jax.Array
    traj_state: TrajState
    selected_traj_idx: jax.Array
    selected_subtraj_step: jax.Array
    sampling_weights: jax.Array | None = None


def _make_handler(lengths):
    handler = object.__new__(TrajectoryHandler)
    handler.len_trajectory = lambda traj_no: lengths[int(np.asarray(traj_no))]
    return handler


def test_motion_names_are_indexed_with_trajectories():
    handler = object.__new__(TrajectoryHandler)
    handler.traj = type("FakeTrajectory", (), {"data": type("FakeData", (), {"split_points": [0, 2, 5]})()})()

    handler.motion_names = handler._normalize_motion_names(["motion-a", "motion-b"])

    assert handler.motion_names == ("motion-a", "motion-b")
    assert handler.get_motion_name(1) == "motion-b"


def test_motion_names_must_match_trajectory_count():
    handler = object.__new__(TrajectoryHandler)
    handler.traj = type("FakeTrajectory", (), {"data": type("FakeData", (), {"split_points": [0, 2, 5]})()})()

    with pytest.raises(ValueError, match="Expected 2 motion names"):
        handler._normalize_motion_names(["motion-a"])


def _make_sampling_handler(lengths, random_start_sampling):
    split_points = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    handler = object.__new__(TrajectoryHandler)
    handler.traj = SimpleNamespace(data=SimpleNamespace(split_points=split_points))
    handler.random_start_sampling = random_start_sampling
    handler.start_from_random_step = True
    return handler


def test_cpu_step_info_exposes_motion_name():
    env = object.__new__(LocoEnv)
    env.th = SimpleNamespace(
        len_trajectory=lambda _traj_no: 10,
        get_motion_name=lambda _traj_no: "KIT/walk_01",
    )
    carry = SimpleNamespace(
        traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=2),
    )

    info = env._update_info_dictionary({}, np.empty(0), None, carry)

    assert info["motion_name"] == "KIT/walk_01"


def test_loco_done_info_distinguishes_trajectory_end_from_horizon():
    env = object.__new__(LocoEnv)
    env._mdp_info = SimpleNamespace(horizon=100)
    env._model = None
    env._reward_function = SimpleNamespace(requires_trajectory=False)
    env._goal = SimpleNamespace(
        requires_trajectory=True,
        is_done=lambda *_args: False,
    )
    env.th = SimpleNamespace(
        reached_trajectory_end=lambda traj_state, backend: backend.greater_equal(
            traj_state.subtraj_step_no,
            2,
        )
    )
    carry = SimpleNamespace(
        cur_step_in_episode=1,
        traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=2),
    )
    info = {"done_horizon": False, "done_trajectory": False}

    done = env._is_done(np.empty(0), False, info, None, carry)

    assert done
    assert info["done_trajectory"]
    assert not info["done_horizon"]


def test_loco_reset_info_owns_fixed_shape_termination_keys():
    env = object.__new__(LocoEnv)

    cpu_info = env._reset_info_dictionary(np.empty(0), None, jax.random.key(0))
    mjx_info = env._mjx_reset_info_dictionary(jnp.empty(0), None, jax.random.key(0))

    for info in (cpu_info, mjx_info):
        assert not bool(np.asarray(info["done_numerical"]))
        assert not bool(np.asarray(info["done_trajectory"]))
        assert not bool(np.asarray(info["termination_site_position"]))
        assert np.asarray(info["done_trajectory"]).shape == ()
        assert np.asarray(info["termination_site_position"]).shape == ()


def test_loco_mjx_done_info_is_fixed_scalar_under_jit():
    env = object.__new__(LocoEnv)
    env._mdp_info = SimpleNamespace(horizon=100)
    env._model = None
    env._reward_function = SimpleNamespace(requires_trajectory=False)
    env._goal = SimpleNamespace(
        requires_trajectory=True,
        mjx_is_done=lambda *_args: jnp.asarray(False),
    )
    env.th = SimpleNamespace(
        reached_trajectory_end=lambda traj_state, backend: backend.greater_equal(
            traj_state.subtraj_step_no,
            2,
        )
    )

    def evaluate(step):
        info = {"done_horizon": jnp.asarray(False), "done_trajectory": jnp.asarray(False)}
        carry = SimpleNamespace(
            cur_step_in_episode=jnp.asarray(1),
            traj_state=TrajState(jnp.asarray(0), step, jnp.asarray(0)),
        )
        done = env._mjx_is_done(jnp.empty(0), False, info, None, carry)
        return done, info["done_trajectory"]

    before_end = jax.jit(evaluate)(jnp.asarray(1))
    at_end = jax.jit(evaluate)(jnp.asarray(2))

    assert not any(bool(np.asarray(value)) for value in before_end)
    assert all(bool(np.asarray(value)) for value in at_end)


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
def test_trajectory_random_start_keeps_existing_weight_semantics(backend):
    handler = _make_sampling_handler([2, 6, 2], "trajectory")
    weights = backend.asarray([0.1, 0.2, 0.7], dtype=backend.float32)

    probabilities = handler._start_trajectory_probabilities(weights, backend)

    np.testing.assert_allclose(np.asarray(probabilities), np.asarray(weights), atol=1e-6)


@pytest.mark.parametrize("backend", [np, jnp])
def test_state_random_start_samples_trajectories_by_length(backend):
    handler = _make_sampling_handler([2, 6, 2], "state")

    probabilities = handler._start_trajectory_probabilities(None, backend)

    np.testing.assert_allclose(np.asarray(probabilities), [0.2, 0.6, 0.2], atol=1e-6)


@pytest.mark.parametrize("backend", [np, jnp])
def test_state_random_start_treats_sampling_weights_as_state_priorities(backend):
    handler = _make_sampling_handler([2, 6, 2], "state")
    weights = backend.asarray([0.5, 0.25, 0.25], dtype=backend.float32)

    probabilities = handler._start_trajectory_probabilities(weights, backend)

    np.testing.assert_allclose(np.asarray(probabilities), [1.0 / 3.0, 0.5, 1.0 / 6.0], atol=1e-6)


@pytest.mark.parametrize("backend", [np, jnp])
def test_state_random_start_does_not_length_bias_fixed_step_resets(backend):
    handler = _make_sampling_handler([2, 6, 2], "state")
    handler.start_from_random_step = False
    weights = backend.asarray([0.1, 0.2, 0.7], dtype=backend.float32)

    probabilities = handler._start_trajectory_probabilities(weights, backend)

    np.testing.assert_allclose(np.asarray(probabilities), np.asarray(weights), atol=1e-6)


@pytest.mark.parametrize(
    ("mode", "selected_traj_idx", "selected_subtraj_step"),
    [
        ("random_frame", -1, -1),
        ("random_trajectory", -1, -1),
        ("fixed", -1, -1),
        ("selected", 1, 3),
    ],
)
@pytest.mark.parametrize("random_start_sampling", ["trajectory", "state"])
@pytest.mark.parametrize("sampling_weights", [None, (0.25, 0.75)], ids=["unweighted", "weighted"])
def test_reset_modes_are_valid_under_checked_shard_map(
    mode,
    selected_traj_idx,
    selected_subtraj_step,
    random_start_sampling,
    sampling_weights,
):
    handler = _make_sampling_handler([4, 6], random_start_sampling)
    handler.traj.data.split_points = jnp.asarray(
        handler.traj.data.split_points,
        dtype=jnp.int32,
    )
    handler.random_start = mode != "fixed"
    handler.use_fixed_start = mode == "fixed"
    handler.fixed_start_conf = (1, 2) if mode == "fixed" else None
    handler.start_from_random_step = mode == "random_frame"
    num_devices = jax.local_device_count()
    mesh = jax.make_mesh(
        (num_devices,),
        ("data",),
        devices=jax.local_devices(),
    )
    partition = jax.sharding.PartitionSpec
    batch_size = 4

    def reset_batch_from_key(key):
        reset_keys = jax.random.split(key, batch_size)

        def reset_one(reset_key):
            carry = _ResetCarry(
                key=reset_key,
                traj_state=TrajState(0, 0, 0),
                selected_traj_idx=jnp.asarray(selected_traj_idx, dtype=jnp.int32),
                selected_subtraj_step=jnp.asarray(selected_subtraj_step, dtype=jnp.int32),
                sampling_weights=(
                    None
                    if sampling_weights is None
                    else jnp.asarray(sampling_weights, dtype=jnp.float32)
                ),
            )
            _, next_carry = handler.reset_state(None, None, None, carry, jnp)
            return jnp.stack(
                (
                    next_carry.traj_state.traj_no,
                    next_carry.traj_state.subtraj_step_no,
                )
            )

        return jax.vmap(reset_one)(reset_keys)

    def reset_batch(seed):
        key = jax.random.fold_in(seed, jax.lax.axis_index("data"))
        return reset_batch_from_key(key)

    mapped_reset = jax.jit(
        jax.shard_map(
            reset_batch,
            mesh=mesh,
            in_specs=partition(),
            out_specs=partition("data"),
            check_vma=True,
        )
    )
    base_seed = jax.random.PRNGKey(0)
    seed = jax.device_put(
        base_seed,
        jax.sharding.NamedSharding(mesh, partition()),
    )

    reset_indices = mapped_reset(seed)
    expected_indices = np.concatenate(
        [
            np.asarray(reset_batch_from_key(jax.random.fold_in(base_seed, device_index)))
            for device_index in range(num_devices)
        ],
        axis=0,
    )

    assert reset_indices.shape == (num_devices * batch_size, 2)
    np.testing.assert_array_equal(reset_indices, expected_indices)


@pytest.mark.parametrize("vary_key", [False, True])
def test_reset_selected_indices_accepts_data_sharded_inputs(vary_key):
    handler = _make_sampling_handler([4, 6], "trajectory")
    handler.traj.data.split_points = jnp.asarray(
        handler.traj.data.split_points,
        dtype=jnp.int32,
    )
    handler.random_start = True
    handler.use_fixed_start = False
    handler.fixed_start_conf = None
    mesh = jax.make_mesh(
        (1,),
        ("data",),
        devices=jax.local_devices()[:1],
    )
    partition = jax.sharding.PartitionSpec
    batch_size = 4

    def reset_batch(seed, trajectory_ids, start_steps):
        if vary_key:
            seed = jax.random.fold_in(seed, jax.lax.axis_index("data"))
        reset_keys = jax.random.split(seed, batch_size)

        def reset_one(reset_key, trajectory_id, start_step):
            carry = _ResetCarry(
                key=reset_key,
                traj_state=TrajState(0, 0, 0),
                selected_traj_idx=trajectory_id,
                selected_subtraj_step=start_step,
            )
            _, next_carry = handler.reset_state(None, None, None, carry, jnp)
            return jnp.stack(
                (
                    next_carry.traj_state.traj_no,
                    next_carry.traj_state.subtraj_step_no,
                )
            )

        return jax.vmap(reset_one)(reset_keys, trajectory_ids, start_steps)

    mapped_reset = jax.jit(
        jax.shard_map(
            reset_batch,
            mesh=mesh,
            in_specs=(partition(), partition("data"), partition("data")),
            out_specs=partition("data"),
            check_vma=True,
        )
    )
    replicated = jax.sharding.NamedSharding(mesh, partition())
    data_sharded = jax.sharding.NamedSharding(mesh, partition("data"))
    seed = jax.device_put(jax.random.PRNGKey(0), replicated)
    trajectory_ids = jax.device_put(jnp.asarray([0, 1, 0, 1], dtype=jnp.int32), data_sharded)
    start_steps = jax.device_put(jnp.asarray([0, 2, -1, 3], dtype=jnp.int32), data_sharded)

    reset_indices = mapped_reset(seed, trajectory_ids, start_steps)

    np.testing.assert_array_equal(
        reset_indices,
        np.asarray([[0, 0], [1, 2], [0, 0], [1, 3]], dtype=np.int32),
    )
