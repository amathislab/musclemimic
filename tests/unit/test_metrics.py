from types import SimpleNamespace
from typing import ClassVar

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.utils.metrics import MetricsHandler, ValidationSummary, flatten_validation_metrics


class DummyMapper:
    def __init__(self, mapping, requires_mapping=True):
        self.mapping = mapping
        self.requires_mapping = requires_mapping

    def model_ids_to_traj_indices(self, model_site_ids):
        return jnp.array([self.mapping[int(mid)] for mid in model_site_ids], dtype=int)


@struct.dataclass
class FakeTrajData:
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    xpos: jnp.ndarray
    xquat: jnp.ndarray
    cvel: jnp.ndarray
    subtree_com: jnp.ndarray
    site_xpos: jnp.ndarray
    site_xmat: jnp.ndarray

    def get(self, traj_no, subtraj_step_no):
        return SimpleNamespace(
            qpos=self.qpos[subtraj_step_no],
            qvel=self.qvel[subtraj_step_no],
            xpos=self.xpos[subtraj_step_no],
            xquat=self.xquat[subtraj_step_no],
            cvel=self.cvel[subtraj_step_no],
            subtree_com=self.subtree_com[subtraj_step_no],
            site_xpos=self.site_xpos[subtraj_step_no],
            site_xmat=self.site_xmat[subtraj_step_no],
        )


def make_handler(
    rel_site_ids, mapper, traj_data, *, vec_site_vel=None, vec_rel_site=None, rel_qpos_ids=None, rel_qvel_ids=None
):
    """Build a MetricsHandler instance without running its heavy __init__."""
    handler = object.__new__(MetricsHandler)
    handler.rel_site_ids = jnp.array(rel_site_ids, dtype=int)
    handler.rel_body_ids = jnp.array([0, 1])
    handler.rel_qpos_ids = jnp.array(rel_qpos_ids if rel_qpos_ids is not None else [0, 1])
    handler.rel_qvel_ids = jnp.array(rel_qvel_ids if rel_qvel_ids is not None else [0, 1])
    handler._quat_in_qpos = jnp.array([False] * len(handler.rel_qpos_ids))
    handler._not_quat_in_qpos = jnp.invert(handler._quat_in_qpos)
    handler._site_bodyid = jnp.arange(np.max(rel_site_ids) + 1)
    handler._body_rootid = jnp.zeros(np.max(rel_site_ids) + 1, dtype=int)
    handler._site_mapper = mapper
    handler._trajectory_handler = SimpleNamespace(traj=SimpleNamespace(data=traj_data))
    handler._root_qpos_ids_xy = None
    handler.get_traj_indices = lambda env_states: 0

    # Allow tests to intercept the vectorized helpers
    handler._vec_calc_site_velocities = vec_site_vel or (lambda *args, **kwargs: None)
    handler._vec_calc_rel_site_quantities = vec_rel_site or (lambda *args, **kwargs: None)
    return handler


def make_env_state(site_xpos, site_xmat):
    data = SimpleNamespace(
        site_xpos=jnp.asarray(site_xpos),
        site_xmat=jnp.asarray(site_xmat),
        xpos=jnp.zeros((2, 3)),
        xquat=jnp.zeros((2, 4)),
        cvel=jnp.zeros((2, 6)),
        subtree_com=jnp.zeros((2, 3)),
    )
    traj_state = SimpleNamespace(traj_no=0, subtraj_step_no=0)
    additional_carry = SimpleNamespace(traj_state=traj_state)
    return SimpleNamespace(data=data, additional_carry=additional_carry)


def test_get_body_positions_and_orientations_no_mapping():
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    rel_body_ids = [0, 1]

    env_state = SimpleNamespace(
        data=SimpleNamespace(
            xpos=jnp.array([[0, 0, 0], [1, 0, 0]]),
            xquat=jnp.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            cvel=jnp.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
            site_xpos=jnp.zeros((2, 3)),
            site_xmat=jnp.zeros((2, 3, 3)),
            qpos=jnp.zeros((2,)),
            qvel=jnp.zeros((2,)),
        ),
        additional_carry=SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0)),
    )

    traj_data = FakeTrajData(
        qpos=jnp.zeros((1, 2)),
        qvel=jnp.zeros((1, 2)),
        xpos=jnp.array([[[10, 0, 0], [20, 0, 0]]]),
        xquat=jnp.array([[[1, 0, 0, 0], [0, 1, 0, 0]]]),
        cvel=jnp.array([[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]),
        subtree_com=jnp.zeros((1, 2, 3)),
        site_xpos=jnp.zeros((1, 2, 3)),
        site_xmat=jnp.zeros((1, 2, 9)),
    )

    handler = make_handler(rel_site_ids=[0, 1], mapper=mapper, traj_data=traj_data)
    handler.rel_body_ids = jnp.array(rel_body_ids)

    body_pos_env, body_pos_traj = handler.get_body_positions(env_state)
    body_rot_env, body_rot_traj = handler.get_body_orientations(env_state)

    np.testing.assert_allclose(body_pos_env, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(body_pos_traj, [[10, 0, 0], [20, 0, 0]])
    # Rotation vectors: [1,0,0,0] -> 0, [0,1,0,0] -> pi about X
    np.testing.assert_allclose(np.linalg.norm(body_rot_env, axis=-1), [0.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(body_rot_traj, axis=-1), [0.0, np.pi])


def test_get_site_positions_no_mapping():
    rel_site_ids = [0, 1]
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    env_state = make_env_state(
        site_xpos=[[0, 0, 0], [5, 0, 0]],
        site_xmat=np.tile(np.eye(3), (2, 1, 1)),
    )
    traj_data = FakeTrajData(
        qpos=jnp.zeros((1, 2)),
        qvel=jnp.zeros((1, 2)),
        cvel=jnp.zeros((1, 2, 6)),
        xpos=jnp.zeros((1, 2, 3)),
        subtree_com=jnp.zeros((1, 2, 3)),
        xquat=jnp.zeros((1, 2, 4)),
        site_xpos=jnp.asarray([[[10, 0, 0], [20, 0, 0]]]),
        site_xmat=jnp.zeros((1, 2, 9)),
    )

    handler = make_handler(rel_site_ids, mapper, traj_data)
    env_pos, traj_pos = handler.get_site_positions(env_state)

    np.testing.assert_allclose(env_pos, [[0, 0, 0], [5, 0, 0]])
    np.testing.assert_allclose(traj_pos, [[10, 0, 0], [20, 0, 0]])


def test_get_site_orientations_no_mapping():
    rel_site_ids = [0, 1]
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    env_state = make_env_state(
        site_xpos=np.zeros((2, 3)),
        site_xmat=np.array([np.eye(3), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])]),
    )
    traj_data = FakeTrajData(
        qpos=jnp.zeros((1, 2)),
        qvel=jnp.zeros((1, 2)),
        cvel=jnp.zeros((1, 2, 6)),
        xpos=jnp.zeros((1, 2, 3)),
        subtree_com=jnp.zeros((1, 2, 3)),
        xquat=jnp.zeros((1, 2, 4)),
        site_xpos=jnp.zeros((1, 2, 3)),
        site_xmat=jnp.array(
            [
                [
                    np.eye(3).reshape(9),
                    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(9),
                ]
            ]
        ),
    )

    handler = make_handler(rel_site_ids, mapper, traj_data)
    env_rot, traj_rot = handler.get_site_orientations(env_state)

    np.testing.assert_allclose(np.linalg.norm(env_rot, axis=-1), [0.0, np.pi])
    np.testing.assert_allclose(np.linalg.norm(traj_rot, axis=-1), [0.0, np.pi])


def test_get_joint_positions_and_velocities_no_mapping():
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    handler = make_handler(
        rel_site_ids=[0, 1],
        mapper=mapper,
        traj_data=FakeTrajData(
            qpos=jnp.array([[1.0, 2.0]]),
            qvel=jnp.array([[0.1, 0.2]]),
            xpos=jnp.zeros((1, 2, 3)),
            xquat=jnp.zeros((1, 2, 4)),
            cvel=jnp.zeros((1, 2, 6)),
            subtree_com=jnp.zeros((1, 2, 3)),
            site_xpos=jnp.zeros((1, 2, 3)),
            site_xmat=jnp.zeros((1, 2, 9)),
        ),
        rel_qpos_ids=[0, 1],
        rel_qvel_ids=[0, 1],
    )
    env_state = SimpleNamespace(
        data=SimpleNamespace(
            qpos=jnp.array([1.0, 2.0]),
            qvel=jnp.array([0.1, 0.2]),
            xpos=jnp.zeros((2, 3)),
            xquat=jnp.zeros((2, 4)),
            cvel=jnp.zeros((2, 6)),
            site_xpos=jnp.zeros((2, 3)),
            site_xmat=jnp.zeros((2, 3, 3)),
        ),
        additional_carry=SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0)),
    )

    qpos_env, qpos_traj = handler.get_joint_positions(env_state)
    qvel_env, qvel_traj = handler.get_joint_velocities(env_state)

    np.testing.assert_allclose(qpos_env, [1.0, 2.0])
    np.testing.assert_allclose(qpos_traj, [1.0, 2.0])
    np.testing.assert_allclose(qvel_env, [0.1, 0.2])
    np.testing.assert_allclose(qvel_traj, [0.1, 0.2])


def test_get_site_positions_uses_mapper_for_traj_order():
    rel_site_ids = [0, 2]
    mapper = DummyMapper(mapping={0: 1, 2: 0}, requires_mapping=True)

    # env has 3 sites in model order; trajectory stores only 2 mimic sites reordered
    env_state = make_env_state(
        site_xpos=[[0, 0, 0], [10, 0, 0], [20, 0, 0]],
        site_xmat=np.zeros((3, 3, 3)),  # unused here
    )
    traj_data = FakeTrajData(
        qpos=jnp.zeros((1, 1)),
        qvel=jnp.zeros((1, 1)),
        cvel=jnp.zeros((1, 2, 6)),
        xpos=jnp.zeros((1, 2, 3)),
        subtree_com=jnp.zeros((1, 2, 3)),
        xquat=jnp.zeros((1, 2, 4)),
        site_xpos=jnp.asarray([[[100, 0, 0], [200, 0, 0]]]),  # shape (T=1, mimic=2, 3)
        site_xmat=jnp.zeros((1, 2, 9)),
    )

    handler = make_handler(rel_site_ids, mapper, traj_data)
    env_pos, traj_pos = handler.get_site_positions(env_state)

    np.testing.assert_allclose(env_pos, [[0, 0, 0], [20, 0, 0]])
    np.testing.assert_allclose(traj_pos, [[200, 0, 0], [100, 0, 0]])


def test_get_site_orientations_respects_mapper():
    rel_site_ids = [0, 2]
    mapper = DummyMapper(mapping={0: 1, 2: 0}, requires_mapping=True)

    # env orientations: site 0 identity, site 2 180deg about Z
    env_state = make_env_state(
        site_xpos=np.zeros((3, 3)),
        site_xmat=np.array(
            [
                np.eye(3),
                np.eye(3),
                np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            ]
        ),
    )

    traj_data = FakeTrajData(
        qpos=jnp.zeros((1, 1)),
        qvel=jnp.zeros((1, 1)),
        site_xpos=jnp.zeros((1, 2, 3)),
        site_xmat=jnp.array(
            [
                [
                    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(9),  # mapped to model site 2
                    np.eye(3).reshape(9),  # mapped to model site 0
                ]
            ]
        ),
        cvel=jnp.zeros((1, 2, 6)),
        xpos=jnp.zeros((1, 2, 3)),
        subtree_com=jnp.zeros((1, 2, 3)),
        xquat=jnp.zeros((1, 2, 4)),
    )

    handler = make_handler(rel_site_ids, mapper, traj_data)
    env_rot, traj_rot = handler.get_site_orientations(env_state)

    # rotvec norms distinguish 0 vs pi rotation; order must match rel_site_ids
    np.testing.assert_allclose(np.linalg.norm(env_rot, axis=-1), [0.0, np.pi])
    np.testing.assert_allclose(np.linalg.norm(traj_rot, axis=-1), [0.0, np.pi])


def test_get_site_velocities_passes_mapped_indices_to_traj():
    calls = []

    def fake_vec_site_vel(site_ids, data, parent_ids, root_ids, backend, flg_local, traj_site_indices):
        calls.append((np.array(site_ids), None if traj_site_indices is None else np.array(traj_site_indices)))
        return backend.arange(len(site_ids)).reshape(len(site_ids), 1)

    rel_site_ids = [0, 2]
    mapper = DummyMapper(mapping={0: 1, 2: 0}, requires_mapping=True)
    env_state = make_env_state(site_xpos=np.zeros((3, 3)), site_xmat=np.zeros((3, 3, 3)))
    traj_data = FakeTrajData(
        qpos=jnp.zeros((1, 1)),
        qvel=jnp.zeros((1, 1)),
        site_xpos=jnp.zeros((1, 2, 3)),
        site_xmat=jnp.zeros((1, 2, 9)),
        cvel=jnp.zeros((1, 2, 6)),
        xpos=jnp.zeros((1, 2, 3)),
        subtree_com=jnp.zeros((1, 2, 3)),
        xquat=jnp.zeros((1, 2, 4)),
    )

    handler = make_handler(rel_site_ids, mapper, traj_data, vec_site_vel=fake_vec_site_vel)
    site_vel, traj_site_vel = handler.get_site_velocities(env_state)

    # First call is env (no mapping); second is traj (mapping applied)
    assert calls[0][1] is None
    np.testing.assert_array_equal(calls[1][1], [1, 0])
    np.testing.assert_array_equal(site_vel.squeeze(), [0, 1])
    np.testing.assert_array_equal(traj_site_vel.squeeze(), [0, 1])


def test_get_relative_site_quantities_passes_mapping_to_traj():
    calls = []

    def fake_vec_rel_site(data, site_ids, parent_ids, root_ids, backend, traj_site_indices=None):
        calls.append((np.array(site_ids), None if traj_site_indices is None else np.array(traj_site_indices)))
        base = backend.arange(len(site_ids)).reshape(len(site_ids), 1)
        return base, base, base

    rel_site_ids = [0, 2]
    mapper = DummyMapper(mapping={0: 1, 2: 0}, requires_mapping=True)
    env_state = make_env_state(site_xpos=np.zeros((3, 3)), site_xmat=np.zeros((3, 3, 3)))
    traj_data = FakeTrajData(
        qpos=jnp.zeros((1, 2)),
        qvel=jnp.zeros((1, 2)),
        site_xpos=jnp.zeros((1, 2, 3)),
        site_xmat=jnp.zeros((1, 2, 9)),
        cvel=jnp.zeros((1, 2, 6)),
        xpos=jnp.zeros((1, 2, 3)),
        subtree_com=jnp.zeros((1, 2, 3)),
        xquat=jnp.zeros((1, 2, 4)),
    )

    handler = make_handler(rel_site_ids, mapper, traj_data, vec_rel_site=fake_vec_rel_site)
    rel_site_pos, _rel_site_rotvec, _rel_site_vel, traj_rel_pos, _traj_rel_rot, _traj_rel_vel = (
        handler.get_relative_site_quantities(env_state)
    )

    assert calls[0][1] is None
    np.testing.assert_array_equal(calls[1][1], [1, 0])
    np.testing.assert_array_equal(rel_site_pos.squeeze(), [0, 1])
    np.testing.assert_array_equal(traj_rel_pos.squeeze(), [0, 1])


def test_no_mapping_paths_leave_traj_indices_none():
    calls_vel = []
    calls_rel = []

    def fake_vec_site_vel(site_ids, data, parent_ids, root_ids, backend, flg_local, traj_site_indices):
        calls_vel.append(traj_site_indices)
        return backend.zeros((len(site_ids), 1))

    def fake_vec_rel_site(data, site_ids, parent_ids, root_ids, backend, traj_site_indices=None):
        calls_rel.append(traj_site_indices)
        base = backend.zeros((len(site_ids), 1))
        return base, base, base

    rel_site_ids = [0, 1]
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    env_state = make_env_state(site_xpos=np.zeros((2, 3)), site_xmat=np.zeros((2, 3, 3)))
    traj_data = FakeTrajData(
        qpos=jnp.zeros((1, 2)),
        qvel=jnp.zeros((1, 2)),
        site_xpos=jnp.zeros((1, 2, 3)),
        site_xmat=jnp.zeros((1, 2, 9)),
        cvel=jnp.zeros((1, 2, 6)),
        xpos=jnp.zeros((1, 2, 3)),
        subtree_com=jnp.zeros((1, 2, 3)),
        xquat=jnp.zeros((1, 2, 4)),
    )

    handler = make_handler(
        rel_site_ids,
        mapper,
        traj_data,
        vec_site_vel=fake_vec_site_vel,
        vec_rel_site=fake_vec_rel_site,
    )

    handler.get_site_velocities(env_state)
    handler.get_relative_site_quantities(env_state)

    assert calls_vel == [None, None]
    assert calls_rel == [None, None]


def test_get_site_positions_handles_batched_traj_indices():
    rel_site_ids = [0, 2]
    mapper = DummyMapper(mapping={0: 1, 2: 0}, requires_mapping=True)

    env_state = make_env_state(
        site_xpos=[[0, 0, 0], [10, 0, 0], [20, 0, 0]],
        site_xmat=np.tile(np.eye(3), (3, 1, 1)),
    )
    traj_data = FakeTrajData(
        qpos=jnp.zeros((2, 1)),
        qvel=jnp.zeros((2, 1)),
        cvel=jnp.zeros((2, 2, 6)),
        xpos=jnp.zeros((2, 2, 3)),
        subtree_com=jnp.zeros((2, 2, 3)),
        xquat=jnp.zeros((2, 2, 4)),
        site_xpos=jnp.asarray(
            [
                [[100, 0, 0], [200, 0, 0]],  # t0: mimic order
                [[300, 0, 0], [400, 0, 0]],  # t1: mimic order
            ]
        ),
        site_xmat=jnp.zeros((2, 2, 9)),
    )

    handler = make_handler(rel_site_ids, mapper, traj_data)
    handler.get_traj_indices = lambda env_states: jnp.array([0, 1])

    env_pos, traj_pos = handler.get_site_positions(env_state)

    np.testing.assert_allclose(env_pos, [[0, 0, 0], [20, 0, 0]])
    expected = np.array(
        [
            [[200, 0, 0], [100, 0, 0]],  # mapped order for t0
            [[400, 0, 0], [300, 0, 0]],  # mapped order for t1
        ]
    )
    np.testing.assert_allclose(traj_pos, expected)


def test_get_site_orientations_handles_batched_traj_indices():
    rel_site_ids = [0, 2]
    mapper = DummyMapper(mapping={0: 1, 2: 0}, requires_mapping=True)

    env_state = make_env_state(
        site_xpos=np.zeros((3, 3)),
        site_xmat=np.array(
            [
                np.eye(3),
                np.eye(3),
                np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 180 deg about Z
            ]
        ),
    )
    traj_data = FakeTrajData(
        qpos=jnp.zeros((2, 1)),
        qvel=jnp.zeros((2, 1)),
        cvel=jnp.zeros((2, 2, 6)),
        xpos=jnp.zeros((2, 2, 3)),
        subtree_com=jnp.zeros((2, 2, 3)),
        xquat=jnp.zeros((2, 2, 4)),
        site_xpos=jnp.zeros((2, 2, 3)),
        site_xmat=jnp.array(
            [
                [
                    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(9),  # t0 mimic idx0 -> model site 2
                    np.eye(3).reshape(9),  # t0 mimic idx1 -> model site 0
                ],
                [
                    np.eye(3).reshape(9),  # t1 mimic idx0 -> model site 2
                    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(9),  # t1 mimic idx1 -> model site 0
                ],
            ]
        ),
    )

    handler = make_handler(rel_site_ids, mapper, traj_data)
    handler.get_traj_indices = lambda env_states: jnp.array([0, 1])

    env_rot, traj_rot = handler.get_site_orientations(env_state)

    np.testing.assert_allclose(np.linalg.norm(env_rot, axis=-1), [0.0, np.pi])
    expected_norms = np.array([[0.0, np.pi], [np.pi, 0.0]])
    np.testing.assert_allclose(np.linalg.norm(traj_rot, axis=-1), expected_norms)


def test_get_site_positions_handles_2d_traj_indices():
    rel_site_ids = [0, 2]
    mapper = DummyMapper(mapping={0: 1, 2: 0}, requires_mapping=True)

    num_steps, num_envs, n_sites = 2, 3, 3
    env_site_xpos = np.zeros((num_steps, num_envs, n_sites, 3))
    for s in range(num_steps):
        for e in range(num_envs):
            env_site_xpos[s, e, 0] = [1000 * s + 100 * e + 0, 0, 0]
            env_site_xpos[s, e, 2] = [1000 * s + 100 * e + 2, 0, 0]
    env_state = make_env_state(
        site_xpos=env_site_xpos,
        site_xmat=np.tile(np.eye(3), (num_steps, num_envs, n_sites, 1, 1)),
    )

    traj_site_xpos = jnp.array(
        [
            [[100, 0, 0], [200, 0, 0]],  # frame 0
            [[300, 0, 0], [400, 0, 0]],  # frame 1
            [[500, 0, 0], [600, 0, 0]],  # frame 2
            [[700, 0, 0], [800, 0, 0]],  # frame 3
        ]
    )
    traj_data = FakeTrajData(
        qpos=jnp.zeros((4, 1)),
        qvel=jnp.zeros((4, 1)),
        cvel=jnp.zeros((4, 2, 6)),
        xpos=jnp.zeros((4, 2, 3)),
        subtree_com=jnp.zeros((4, 2, 3)),
        xquat=jnp.zeros((4, 2, 4)),
        site_xpos=traj_site_xpos,
        site_xmat=jnp.zeros((4, 2, 9)),
    )

    handler = make_handler(rel_site_ids, mapper, traj_data)
    handler.get_traj_indices = lambda env_states: jnp.array([[0, 1, 2], [3, 2, 1]])

    env_pos, traj_pos = handler.get_site_positions(env_state)

    expected_env = np.array(
        [
            [
                [[0, 0, 0], [2, 0, 0]],
                [[100, 0, 0], [102, 0, 0]],
                [[200, 0, 0], [202, 0, 0]],
            ],
            [
                [[1000, 0, 0], [1002, 0, 0]],
                [[1100, 0, 0], [1102, 0, 0]],
                [[1200, 0, 0], [1202, 0, 0]],
            ],
        ]
    )
    np.testing.assert_allclose(env_pos, expected_env)

    expected_traj = np.array(
        [
            [
                [[200, 0, 0], [100, 0, 0]],  # frame 0 mapped
                [[400, 0, 0], [300, 0, 0]],  # frame 1 mapped
                [[600, 0, 0], [500, 0, 0]],  # frame 2 mapped
            ],
            [
                [[800, 0, 0], [700, 0, 0]],  # frame 3 mapped
                [[600, 0, 0], [500, 0, 0]],  # frame 2 mapped
                [[400, 0, 0], [300, 0, 0]],  # frame 1 mapped
            ],
        ]
    )
    np.testing.assert_allclose(traj_pos, expected_traj)


def test_get_site_orientations_handles_2d_traj_indices():
    rel_site_ids = [0, 2]
    mapper = DummyMapper(mapping={0: 1, 2: 0}, requires_mapping=True)

    num_steps, num_envs, n_sites = 2, 3, 3
    env_state = make_env_state(
        site_xpos=np.zeros((num_steps, num_envs, n_sites, 3)),
        site_xmat=np.tile(np.eye(3), (num_steps, num_envs, n_sites, 1, 1)),
    )

    def rot180_z():
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).reshape(9)

    traj_site_xmat = jnp.array(
        [
            [rot180_z(), np.eye(3).reshape(9)],  # frame 0: mimic0->pi, mimic1->0
            [np.eye(3).reshape(9), rot180_z()],  # frame 1: mimic0->0, mimic1->pi
            [rot180_z(), np.eye(3).reshape(9)],  # frame 2: mimic0->pi, mimic1->0
            [np.eye(3).reshape(9), rot180_z()],  # frame 3: mimic0->0, mimic1->pi
        ]
    )
    traj_data = FakeTrajData(
        qpos=jnp.zeros((4, 1)),
        qvel=jnp.zeros((4, 1)),
        cvel=jnp.zeros((4, 2, 6)),
        xpos=jnp.zeros((4, 2, 3)),
        subtree_com=jnp.zeros((4, 2, 3)),
        xquat=jnp.zeros((4, 2, 4)),
        site_xpos=jnp.zeros((4, 2, 3)),
        site_xmat=traj_site_xmat,
    )

    handler = make_handler(rel_site_ids, mapper, traj_data)
    handler.get_traj_indices = lambda env_states: jnp.array([[0, 1, 2], [3, 2, 1]])

    env_rot, traj_rot = handler.get_site_orientations(env_state)

    assert env_rot.shape == (num_steps, num_envs, 2, 3)
    np.testing.assert_allclose(np.linalg.norm(env_rot, axis=-1), 0.0)

    expected_norms = np.array(
        [
            [[0.0, np.pi], [np.pi, 0.0], [0.0, np.pi]],  # frames 0,1,2 mapped
            [[np.pi, 0.0], [0.0, np.pi], [np.pi, 0.0]],  # frames 3,2,1 mapped
        ]
    )
    np.testing.assert_allclose(np.linalg.norm(traj_rot, axis=-1), expected_norms)


# =====================================================================
# XY Offset Alignment Tests
# =====================================================================


@struct.dataclass
class FakeTrajDataWithSplitPoints:
    """Extended FakeTrajData with split_points for offset alignment tests."""

    qpos: jnp.ndarray
    qvel: jnp.ndarray
    xpos: jnp.ndarray
    xquat: jnp.ndarray
    cvel: jnp.ndarray
    subtree_com: jnp.ndarray
    site_xpos: jnp.ndarray
    site_xmat: jnp.ndarray
    split_points: jnp.ndarray


def make_handler_with_offset(rel_site_ids, mapper, traj_data, root_qpos_ids_xy=None):
    """Build a MetricsHandler with XY offset support."""
    handler = make_handler(rel_site_ids, mapper, traj_data)
    handler._root_qpos_ids_xy = jnp.array(root_qpos_ids_xy) if root_qpos_ids_xy is not None else None
    # Override get_traj_indices to use split_points
    handler.get_traj_indices = lambda env_states: (
        traj_data.split_points[env_states.additional_carry.traj_state.traj_no]
        + env_states.additional_carry.traj_state.subtraj_step_no
    )
    return handler


def make_env_state_with_init(site_xpos, site_xmat, subtraj_step_no_init=0):
    """Make env_state with subtraj_step_no_init for offset tests."""
    data = SimpleNamespace(
        site_xpos=jnp.asarray(site_xpos),
        site_xmat=jnp.asarray(site_xmat),
        xpos=jnp.zeros((2, 3)),
        xquat=jnp.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
        cvel=jnp.zeros((2, 6)),
        subtree_com=jnp.zeros((2, 3)),
        qpos=jnp.zeros(7),
        qvel=jnp.zeros(6),
    )
    traj_state = SimpleNamespace(traj_no=0, subtraj_step_no=1, subtraj_step_no_init=subtraj_step_no_init)
    additional_carry = SimpleNamespace(traj_state=traj_state)
    return SimpleNamespace(data=data, additional_carry=additional_carry)


def test_get_root_xy_offset_returns_none_when_no_root_indices():
    """Handler without root qpos indices should return None offset."""
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    traj_data = FakeTrajDataWithSplitPoints(
        qpos=jnp.array([[5.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]]),
        qvel=jnp.zeros((2, 6)),
        xpos=jnp.zeros((2, 2, 3)),
        xquat=jnp.zeros((2, 2, 4)),
        cvel=jnp.zeros((2, 2, 6)),
        subtree_com=jnp.zeros((2, 2, 3)),
        site_xpos=jnp.zeros((2, 2, 3)),
        site_xmat=jnp.zeros((2, 2, 9)),
        split_points=jnp.array([0]),
    )
    handler = make_handler_with_offset([0, 1], mapper, traj_data, root_qpos_ids_xy=None)
    env_state = make_env_state_with_init(np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)))

    offset = handler._get_root_xy_offset(env_state)
    assert offset is None


def test_get_root_xy_offset_computes_correctly():
    """Handler should compute XY offset from init trajectory qpos."""
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    # Frame 0: init position at (5, 3, z), Frame 1: current position
    traj_data = FakeTrajDataWithSplitPoints(
        qpos=jnp.array([[5.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]]),
        qvel=jnp.zeros((2, 6)),
        xpos=jnp.zeros((2, 2, 3)),
        xquat=jnp.zeros((2, 2, 4)),
        cvel=jnp.zeros((2, 2, 6)),
        subtree_com=jnp.zeros((2, 2, 3)),
        site_xpos=jnp.zeros((2, 2, 3)),
        site_xmat=jnp.zeros((2, 2, 9)),
        split_points=jnp.array([0]),
    )
    handler = make_handler_with_offset([0, 1], mapper, traj_data, root_qpos_ids_xy=[0, 1])
    env_state = make_env_state_with_init(np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)), subtraj_step_no_init=0)

    offset = handler._get_root_xy_offset(env_state)
    np.testing.assert_allclose(offset, [5.0, 3.0, 0.0])


def test_get_site_positions_applies_xy_offset():
    """Site positions should be corrected by XY offset from init trajectory."""
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    # Init qpos at frame 0: root at (10, 5, z)
    # Site positions at frame 1: sites at world positions (10, 5, 0) and (11, 6, 0)
    # After offset correction, traj sites should be at (0, 0, 0) and (1, 1, 0)
    traj_data = FakeTrajDataWithSplitPoints(
        qpos=jnp.array([[10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]]),
        qvel=jnp.zeros((2, 6)),
        xpos=jnp.zeros((2, 2, 3)),
        xquat=jnp.zeros((2, 2, 4)),
        cvel=jnp.zeros((2, 2, 6)),
        subtree_com=jnp.zeros((2, 2, 3)),
        site_xpos=jnp.array([[[10.0, 5.0, 0.0], [10.0, 5.0, 0.0]], [[10.0, 5.0, 0.0], [11.0, 6.0, 0.0]]]),
        site_xmat=jnp.zeros((2, 2, 9)),
        split_points=jnp.array([0]),
    )
    handler = make_handler_with_offset([0, 1], mapper, traj_data, root_qpos_ids_xy=[0, 1])

    # Env sites at (0, 0, 0) and (1, 1, 0)
    env_state = make_env_state_with_init(
        site_xpos=[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        site_xmat=np.tile(np.eye(3), (2, 1, 1)),
        subtraj_step_no_init=0,
    )

    env_pos, traj_pos = handler.get_site_positions(env_state)

    # Env positions unchanged
    np.testing.assert_allclose(env_pos, [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    # Traj positions after offset correction: (10,5,0)-(10,5,0)=(0,0,0), (11,6,0)-(10,5,0)=(1,1,0)
    np.testing.assert_allclose(traj_pos, [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])


def test_get_body_positions_applies_xy_offset():
    """Body positions should be corrected by XY offset from init trajectory."""
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    # Init qpos at frame 0: root at (20, 10, z)
    # Body positions at frame 1: bodies at world positions
    traj_data = FakeTrajDataWithSplitPoints(
        qpos=jnp.array([[20.0, 10.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]]),
        qvel=jnp.zeros((2, 6)),
        xpos=jnp.array([[[20.0, 10.0, 0.0], [20.0, 10.0, 1.0]], [[20.0, 10.0, 0.0], [21.0, 11.0, 1.0]]]),
        xquat=jnp.array([[[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]]]),
        cvel=jnp.zeros((2, 2, 6)),
        subtree_com=jnp.zeros((2, 2, 3)),
        site_xpos=jnp.zeros((2, 2, 3)),
        site_xmat=jnp.zeros((2, 2, 9)),
        split_points=jnp.array([0]),
    )
    handler = make_handler_with_offset([0, 1], mapper, traj_data, root_qpos_ids_xy=[0, 1])
    handler.rel_body_ids = jnp.array([0, 1])

    env_state = SimpleNamespace(
        data=SimpleNamespace(
            xpos=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            xquat=jnp.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            cvel=jnp.zeros((2, 6)),
            site_xpos=jnp.zeros((2, 3)),
            site_xmat=jnp.zeros((2, 3, 3)),
            qpos=jnp.zeros(7),
            qvel=jnp.zeros(6),
        ),
        additional_carry=SimpleNamespace(
            traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=1, subtraj_step_no_init=0)
        ),
    )

    env_pos, traj_pos = handler.get_body_positions(env_state)

    # Env positions unchanged
    np.testing.assert_allclose(env_pos, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    # Traj positions after offset: (20,10,0)-(20,10,0)=(0,0,0), (21,11,1)-(20,10,0)=(1,1,1)
    np.testing.assert_allclose(traj_pos, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_xy_offset_no_effect_without_subtraj_step_no_init():
    """Handler should gracefully handle missing subtraj_step_no_init."""
    mapper = DummyMapper(mapping={}, requires_mapping=False)
    traj_data = FakeTrajDataWithSplitPoints(
        qpos=jnp.array([[10.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
        qvel=jnp.zeros((1, 6)),
        xpos=jnp.zeros((1, 2, 3)),
        xquat=jnp.zeros((1, 2, 4)),
        cvel=jnp.zeros((1, 2, 6)),
        subtree_com=jnp.zeros((1, 2, 3)),
        site_xpos=jnp.array([[[10.0, 5.0, 0.0], [11.0, 6.0, 0.0]]]),
        site_xmat=jnp.zeros((1, 2, 9)),
        split_points=jnp.array([0]),
    )
    handler = make_handler_with_offset([0, 1], mapper, traj_data, root_qpos_ids_xy=[0, 1])
    handler.get_traj_indices = lambda env_states: 0

    # traj_state WITHOUT subtraj_step_no_init
    env_state = SimpleNamespace(
        data=SimpleNamespace(
            site_xpos=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            site_xmat=jnp.tile(jnp.eye(3), (2, 1, 1)),
            xpos=jnp.zeros((2, 3)),
            xquat=jnp.zeros((2, 4)),
            cvel=jnp.zeros((2, 6)),
        ),
        additional_carry=SimpleNamespace(traj_state=SimpleNamespace(traj_no=0, subtraj_step_no=0)),
    )

    # Should not crash, offset should be None
    offset = handler._get_root_xy_offset(env_state)
    assert offset is None

    # get_site_positions should still work, returning uncorrected traj positions
    _env_pos, traj_pos = handler.get_site_positions(env_state)
    np.testing.assert_allclose(traj_pos, [[10.0, 5.0, 0.0], [11.0, 6.0, 0.0]])


# =====================================================================
# Integration tests for MetricsHandler.__init__ (tests real init path)
# =====================================================================

MINIMAL_MJCF = """
<mujoco model="test_metrics_init">
  <worldbody>
    <body name="root">
      <joint name="root_joint" type="free"/>
      <geom name="torso" size="0.1"/>
      <site name="pelvis_mimic" pos="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_metrics_handler_init_populates_root_qpos_ids_from_info_props():
    """MetricsHandler.__init__ should call env._get_all_info_properties() to get root joint name."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    class MockEnv:
        th = None
        sites_for_mimic: ClassVar[list[str]] = ["pelvis_mimic"]

        def get_model(self):
            return model

        def _get_all_info_properties(self):
            return {"root_free_joint_xml_name": "root_joint"}

    env = MockEnv()
    config = OmegaConf.create({"experiment": {"validation": None}})

    handler = MetricsHandler(config, env)

    # Should have extracted root XY indices (0, 1) from the free joint
    assert handler._root_qpos_ids_xy is not None
    np.testing.assert_array_equal(handler._root_qpos_ids_xy, [0, 1])


def test_metrics_handler_init_no_root_indices_without_info_props_method():
    """MetricsHandler should handle env without _get_all_info_properties gracefully."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    class MockEnvNoInfoProps:
        th = None
        sites_for_mimic: ClassVar[list[str]] = ["pelvis_mimic"]

        def get_model(self):
            return model

        # No _get_all_info_properties method

    env = MockEnvNoInfoProps()
    config = OmegaConf.create({"experiment": {"validation": None}})

    handler = MetricsHandler(config, env)

    # Should be None since env doesn't have _get_all_info_properties
    assert handler._root_qpos_ids_xy is None


def test_metrics_handler_init_no_root_indices_when_info_props_missing_key():
    """MetricsHandler should handle missing root_free_joint_xml_name in info_props."""
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)

    class MockEnvEmptyInfoProps:
        th = None
        sites_for_mimic: ClassVar[list[str]] = ["pelvis_mimic"]

        def get_model(self):
            return model

        def _get_all_info_properties(self):
            return {}  # No root_free_joint_xml_name key

    env = MockEnvEmptyInfoProps()
    config = OmegaConf.create({"experiment": {"validation": None}})

    handler = MetricsHandler(config, env)

    # Should be None since root_free_joint_xml_name not in info_props
    assert handler._root_qpos_ids_xy is None


def test_flatten_validation_metrics_skips_error_metrics_for_disabled_quantities():
    summary = ValidationSummary(
        mean_episode_return=jnp.array(1.0),
        mean_episode_length=jnp.array(2.0),
        early_termination_count=jnp.array(0.0),
        early_termination_rate=jnp.array(0.0),
        err_root_xyz=jnp.array(0.3),
        err_root_quat=jnp.array(0.35),
        err_root_yaw=jnp.array(0.4),
        err_joint_pos=jnp.array(1.1),
        err_joint_vel=jnp.array(1.2),
        err_site_abs=jnp.array(1.3),
        err_rpos=jnp.array(1.4),
    )

    flattened = flatten_validation_metrics(
        summary,
        enabled_measures=["EuclideanDistance"],
        enabled_quantities=["JointPosition", "RelSitePosition"],
    )

    assert flattened["val_err_root_xyz"] == pytest.approx(0.3)
    assert flattened["val_err_root_quat"] == pytest.approx(0.35)
    assert flattened["val_err_root_yaw"] == pytest.approx(0.4)
    assert flattened["val_err_joint_pos"] == pytest.approx(1.1)
    assert flattened["val_err_rpos"] == pytest.approx(1.4)
    assert "val_err_joint_vel" not in flattened
    assert "val_err_site_abs" not in flattened


def test_flatten_validation_metrics_omits_exact_set_metrics_for_sampled_validation():
    summary = ValidationSummary(
        episode_count=jnp.array(4.0),
        horizon_timeout_count=jnp.array(3.0),
        horizon_timeout_rate=jnp.array(0.75),
        frame_coverage=jnp.array(0.5),
        total_frame=jnp.array(100.0),
    )

    flattened = flatten_validation_metrics(
        summary,
        include_trajectory_set_metrics=False,
    )

    assert flattened["val_episode_count"] == pytest.approx(4.0)
    assert "val_horizon_timeout_count" not in flattened
    assert "val_horizon_timeout_rate" not in flattened
    assert "val_frame_coverage" not in flattened
    assert "val_total_frame" not in flattened


def test_flatten_validation_metrics_keeps_exact_set_metrics_by_default():
    summary = ValidationSummary(
        horizon_timeout_count=jnp.array(3.0),
        horizon_timeout_rate=jnp.array(0.75),
        frame_coverage=jnp.array(0.5),
        total_frame=jnp.array(100.0),
    )

    flattened = flatten_validation_metrics(summary)

    assert flattened["val_horizon_timeout_count"] == pytest.approx(3.0)
    assert flattened["val_horizon_timeout_rate"] == pytest.approx(0.75)
    assert flattened["val_frame_coverage"] == pytest.approx(0.5)
    assert flattened["val_total_frame"] == pytest.approx(100.0)
