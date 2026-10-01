from itertools import pairwise

import mujoco
import numpy as np
import pytest

from loco_mujoco.smpl import retargeting as retargeting_module
from loco_mujoco.smpl.retargeting import _fps_after_frame_skip
from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryCacheType,
    TrajectoryData,
    TrajectoryHandler,
    TrajectoryInfo,
    TrajectoryModel,
    compute_trajectory_kinematic_caches,
    interpolate_trajectories,
    materialize_trajectory,
    recompute_trajectory_velocities,
)
from loco_mujoco.trajectory.materialized_cache import MaterializedTrajectoryCache


def _make_model():
    xml = """
    <mujoco>
      <worldbody>
        <body name="root" pos="0 0 1">
          <freejoint name="root_free"/>
          <geom name="root_geom" type="sphere" size="0.05" mass="1"/>
          <site name="root_site" pos="0 0 0"/>
          <body name="link" pos="0 0 0.2">
            <joint name="hinge" type="hinge" axis="0 0 1"/>
            <geom name="link_geom" type="capsule" size="0.02 0.1" mass="1"/>
            <site name="link_site" pos="0.1 0 0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _make_hinge_model():
    xml = """
    <mujoco>
      <worldbody>
        <body name="root" pos="0 0 0">
          <joint name="joint_a" type="hinge" axis="0 0 1"/>
          <joint name="joint_b" type="hinge" axis="0 1 0"/>
          <geom name="root_geom" type="sphere" size="0.05" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _make_limited_hinge_model():
    xml = """
    <mujoco>
      <worldbody>
        <body name="root" pos="0 0 0">
          <joint name="limited" type="hinge" axis="0 0 1" range="-30 30"/>
          <geom name="root_geom" type="sphere" size="0.05" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _make_two_site_hinge_model():
    xml = """
    <mujoco>
      <worldbody>
        <body name="body_a" pos="0 0 0">
          <joint name="joint_a" type="hinge" axis="0 0 1"/>
          <geom name="geom_a" type="sphere" size="0.05" mass="1"/>
          <site name="site_a" pos="0.1 0 0"/>
        </body>
        <body name="body_b" pos="0 0 0">
          <joint name="joint_b" type="hinge" axis="0 1 0"/>
          <geom name="geom_b" type="sphere" size="0.05" mass="1"/>
          <site name="site_b" pos="0 0.1 0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _make_info(model, frequency):
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    return TrajectoryInfo(
        joint_names=joint_names,
        model=TrajectoryModel(njnt=model.njnt, jnt_type=np.array(model.jnt_type)),
        frequency=frequency,
    )


def _make_qpos(model, n_frames, x_offset=0.0):
    qpos = np.zeros((n_frames, model.nq), dtype=np.float32)
    frames = np.arange(n_frames, dtype=np.float32)
    if model.nq >= 7:
        qpos[:, 0] = x_offset + 0.05 * frames**2
        qpos[:, 2] = 1.0
        qpos[:, 3] = 1.0
        qpos[:, 7] = 0.1 * frames**2
    else:
        qpos[:] = 0.1 * frames[:, None] ** 2
    return qpos


def _differentiate(model, qpos_a, qpos_b, dt):
    qvel = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, qvel, dt, qpos_a, qpos_b)
    return qvel


def _assert_qvel_matches_qpos(model, qpos, qvel, split_points, frequency):
    dt = 1.0 / frequency
    for start, end in pairwise(split_points):
        if end - start == 1:
            np.testing.assert_allclose(qvel[start], np.zeros(model.nv), atol=1e-6)
            continue

        fwd = np.array([
            _differentiate(model, qpos[t], qpos[t + 1], dt)
            for t in range(start, end - 1)
        ])
        np.testing.assert_allclose(qvel[start], fwd[0], atol=1e-5)
        if end - start > 2:
            expected_mid = 0.5 * (fwd[:-1] + fwd[1:])
            np.testing.assert_allclose(qvel[start + 1:end - 1], expected_mid, atol=1e-5)
        np.testing.assert_allclose(qvel[end - 1], fwd[-1], atol=1e-5)


def test_recompute_velocities_uses_centered_differences_without_crossing_segments():
    model = _make_model()
    frequency = 50.0
    info = _make_info(model, frequency)
    qpos = np.concatenate([_make_qpos(model, 4), _make_qpos(model, 3, x_offset=100.0)], axis=0)
    split_points = np.array([0, 4, 7])
    data = TrajectoryData(
        qpos=qpos,
        qvel=np.zeros((len(qpos), model.nv), dtype=np.float32),
        split_points=split_points,
    )

    updated = recompute_trajectory_velocities(data, info, model, backend=np)

    _assert_qvel_matches_qpos(model, qpos, updated.qvel, split_points, frequency)
    boundary_crossing = _differentiate(model, qpos[3], qpos[4], 1.0 / frequency)
    assert not np.allclose(updated.qvel[3], boundary_crossing)
    assert not np.allclose(updated.qvel[4], boundary_crossing)


def test_interpolation_uses_mujoco_scalar_first_quaternion_order_for_qpos():
    model = _make_model()
    info = _make_info(model, 10.0)
    qpos = np.zeros((2, model.nq), dtype=np.float32)
    qpos[:, 2] = 1.0
    qpos[0, 3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    qpos[1, 3:7] = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    data = TrajectoryData(
        qpos=qpos,
        qvel=np.zeros((2, model.nv), dtype=np.float32),
        split_points=np.array([0, 2]),
    )

    interpolated, _ = interpolate_trajectories(data, info, 30.0, backend=np)

    alpha = 2 / (len(interpolated.qpos) - 1)
    expected_angle = alpha * np.pi / 2
    expected_quat = np.array([
        np.cos(expected_angle / 2),
        0.0,
        0.0,
        np.sin(expected_angle / 2),
    ])
    np.testing.assert_allclose(interpolated.qpos[2, 3:7], expected_quat, atol=1e-6)


def test_interpolation_outputs_source_data_only():
    model = _make_model()
    info = _make_info(model, 100.0)
    data = TrajectoryData(
        qpos=_make_qpos(model, 5),
        qvel=np.zeros((5, model.nv), dtype=np.float32),
        cvel=np.ones((5, model.nbody, 6), dtype=np.float32),
        cvel_parent=np.ones((5, model.nsite, 6), dtype=np.float32),
        subtree_com_root=np.ones((5, 3), dtype=np.float32),
        site_xpos=np.ones((5, model.nsite, 3), dtype=np.float32),
        site_xmat=np.ones((5, model.nsite, 9), dtype=np.float32),
        split_points=np.array([0, 5]),
    )

    downsampled, _ = interpolate_trajectories(data, info, 50.0, backend=np)

    assert downsampled.qpos.shape[0] == 3
    assert downsampled.qvel.shape[0] == 3
    assert downsampled.cvel.size == 0
    assert downsampled.cvel_parent.size == 0
    assert downsampled.subtree_com_root.size == 0
    assert downsampled.site_xpos.size == 0


def test_slow_resampling_repeats_single_frame_segments_as_source_data():
    model = _make_model()
    info = _make_info(model, 50.0)
    qpos = _make_qpos(model, 1)
    qvel = np.arange(model.nv, dtype=np.float32).reshape(1, model.nv)
    data = TrajectoryData(
        qpos=qpos,
        qvel=qvel,
        cvel_parent=np.ones((1, model.nsite, 6), dtype=np.float32),
        split_points=np.array([0, 1]),
    )

    upsampled, _ = interpolate_trajectories(data, info, 100.0, backend=np)

    assert upsampled.qpos.shape[0] == 2
    np.testing.assert_allclose(upsampled.qpos, np.repeat(qpos, 2, axis=0))
    np.testing.assert_allclose(upsampled.qvel, np.repeat(qvel, 2, axis=0))
    assert upsampled.cvel_parent.size == 0


def test_materialize_full_cache_after_upsampling():
    model = _make_model()
    info = _make_info(model, 50.0)
    qpos = _make_qpos(model, 5)
    traj = Trajectory(
        info=info,
        data=TrajectoryData(
            qpos=qpos,
            qvel=np.zeros((len(qpos), model.nv), dtype=np.float32),
            split_points=np.array([0, len(qpos)]),
        ),
    )

    handler = TrajectoryHandler(
        model,
        traj=traj,
        control_dt=0.01,
        random_start=False,
        fixed_start_conf=(0, 0),
        cache_type=TrajectoryCacheType.FULL,
    )

    assert handler.traj.info.frequency == 100.0
    _assert_qvel_matches_qpos(
        model,
        np.asarray(handler.traj.data.qpos),
        np.asarray(handler.traj.data.qvel),
        np.asarray(handler.traj.data.split_points),
        handler.traj.info.frequency,
    )

    frame = 2
    mj_data = mujoco.MjData(model)
    mj_data.qpos[:] = np.asarray(handler.traj.data.qpos)[frame]
    mj_data.qvel[:] = np.asarray(handler.traj.data.qvel)[frame]
    mujoco.mj_forward(model, mj_data)

    np.testing.assert_allclose(handler.traj.data.cvel[frame], mj_data.cvel, atol=1e-5)
    np.testing.assert_allclose(handler.traj.data.subtree_com[frame], mj_data.subtree_com, atol=1e-5)
    np.testing.assert_allclose(handler.traj.data.site_xpos[frame], mj_data.site_xpos, atol=1e-5)
    assert handler.traj.data.cvel_parent.size == 0


def test_materialized_trajectory_cache_round_trip_skips_recomputation(tmp_path, monkeypatch):
    model = _make_model()
    info = _make_info(model, 100.0)
    qpos = _make_qpos(model, 5)
    trajectory = Trajectory(
        info=info,
        data=TrajectoryData(
            qpos=qpos,
            qvel=np.zeros((len(qpos), model.nv), dtype=np.float32),
            split_points=np.array([0, len(qpos)]),
        ),
    )
    handler = TrajectoryHandler(
        model,
        traj=trajectory,
        control_dt=0.01,
        random_start=False,
        fixed_start_conf=(0, 0),
        cache_type=TrajectoryCacheType.SPARSE,
        site_names=["root_site", "link_site"],
    )
    cache = MaterializedTrajectoryCache(
        tmp_path,
        "a" * 64,
        model=model,
        control_dt=0.01,
        cache_type=TrajectoryCacheType.SPARSE,
        site_names=["root_site", "link_site"],
        clip_joint_ranges=False,
    )

    with cache.writer_lock():
        cache.publish(handler.traj)
    loaded = cache.load()
    assert loaded is not None
    np.testing.assert_array_equal(loaded.data.site_xpos, handler.traj.data.site_xpos)

    monkeypatch.setattr(
        "loco_mujoco.trajectory.handler.materialize_trajectory",
        lambda *_args, **_kwargs: pytest.fail("warm cache must not be materialized again"),
    )
    warm_handler = TrajectoryHandler(
        model,
        materialized_traj=loaded,
        control_dt=0.01,
        random_start=False,
        fixed_start_conf=(0, 0),
        cache_type=TrajectoryCacheType.SPARSE,
        site_names=["root_site", "link_site"],
    )
    assert warm_handler.n_trajectories == 1


def test_materialized_trajectory_cache_variants_include_the_model(tmp_path):
    model = _make_model()
    changed_model = _make_model()
    changed_model.body_pos[2, 0] += 0.01
    kwargs = {
        "control_dt": 0.01,
        "cache_type": TrajectoryCacheType.SPARSE,
        "site_names": ["root_site", "link_site"],
        "clip_joint_ranges": False,
    }

    first = MaterializedTrajectoryCache(tmp_path, "b" * 64, model=model, **kwargs)
    changed = MaterializedTrajectoryCache(tmp_path, "b" * 64, model=changed_model, **kwargs)

    assert first.path != changed.path


def test_materialized_trajectory_cache_removes_interrupted_staging_directory(tmp_path):
    model = _make_model()
    cache = MaterializedTrajectoryCache(
        tmp_path,
        "d" * 64,
        model=model,
        control_dt=0.01,
        cache_type=TrajectoryCacheType.SPARSE,
        site_names=["root_site", "link_site"],
        clip_joint_ranges=False,
    )
    cache.cohort_dir.mkdir(parents=True)
    interrupted = cache.cohort_dir / f".{cache.variant_key}.stage-interrupted"
    interrupted.mkdir()
    (interrupted / "partial").write_text("partial")

    with cache.writer_lock():
        assert not interrupted.exists()

def test_shape_preserving_upsampling_does_not_overshoot_joint_extrema():
    model = _make_limited_hinge_model()
    info = _make_info(model, 50.0)
    limit = float(model.jnt_range[0, 1])
    qpos = np.array([[0.0], [0.9 * limit], [limit], [0.9 * limit]], dtype=np.float32)
    data = TrajectoryData(
        qpos=qpos,
        qvel=np.zeros((len(qpos), model.nv), dtype=np.float32),
        split_points=np.array([0, len(qpos)]),
    )

    upsampled, upsampled_info = interpolate_trajectories(
        data,
        info,
        100.0,
        backend=np,
        bounded_qpos_ranges={0: (-limit, limit)},
    )

    assert np.max(upsampled.qpos[:, 0]) <= limit + 1e-7
    assert np.min(upsampled.qpos[:, 0]) >= 0.0
    assert upsampled_info.metadata["joint_range_interpolation_rescue_frame_count"] > 0
    assert upsampled_info.metadata["joint_range_interpolation_rescue_value_count"] > 0


def test_materialize_rescues_cubic_limit_overshoot_before_projection():
    model = _make_limited_hinge_model()
    info = _make_info(model, 50.0)
    limit = float(model.jnt_range[0, 1])
    qpos = np.array([[0.0], [0.9 * limit], [limit], [0.9 * limit]], dtype=np.float32)
    trajectory = Trajectory(
        info=info,
        data=TrajectoryData(
            qpos=qpos,
            qvel=np.zeros((len(qpos), model.nv), dtype=np.float32),
            split_points=np.array([0, len(qpos)]),
        ),
    )

    materialized = materialize_trajectory(
        trajectory,
        model=model,
        control_dt=0.01,
        clip_joint_ranges=True,
    )

    metadata = materialized.info.metadata
    assert metadata["joint_range_interpolation_rescue_frame_count"] > 0
    assert metadata["joint_range_projection_frame_count"] == 0
    assert np.max(materialized.data.qpos[:, 0]) <= limit


def test_materialize_joint_range_projection_recomputes_velocity():
    model = _make_limited_hinge_model()
    info = _make_info(model, 50.0)
    limit = float(model.jnt_range[0, 1])
    qpos = np.array([[0.0], [0.9 * limit], [1.01 * limit], [0.9 * limit]], dtype=np.float32)
    trajectory = Trajectory(
        info=info,
        data=TrajectoryData(
            qpos=qpos,
            qvel=np.zeros((len(qpos), model.nv), dtype=np.float32),
            split_points=np.array([0, len(qpos)]),
        ),
    )

    projected = materialize_trajectory(
        trajectory,
        model=model,
        control_dt=0.02,
        clip_joint_ranges=True,
    )

    assert np.max(projected.data.qpos[:, 0]) <= limit
    assert np.min(projected.data.qpos[:, 0]) >= -limit
    projection = projected.info.metadata
    assert projection["joint_range_projection_frame_count"] > 0
    assert projection["joint_range_projection_value_count"] >= projection["joint_range_projection_frame_count"]
    assert projection["joint_range_projection_max_abs"] > 0
    assert len(projection["joint_range_projection_frames"]) == projection["joint_range_projection_frame_count"]
    assert projection["joint_range_projection_angular_frame_count"] > 0
    assert projection["joint_range_projection_angular_value_count"] > 0
    assert projection["joint_range_projection_angular_max_abs"] > 0
    assert projection["joint_range_projection_material_angular_frame_count"] > 0
    assert projection["joint_range_projection_material_angular_value_count"] > 0
    assert projection["joint_range_projection_material_angular_threshold"] == pytest.approx(
        np.deg2rad(0.05)
    )
    assert '"limited"' in projection["joint_range_projection_by_joint_json"]
    _assert_qvel_matches_qpos(
        model,
        np.asarray(projected.data.qpos),
        np.asarray(projected.data.qvel),
        np.asarray(projected.data.split_points),
        projected.info.frequency,
    )


def test_materialize_sparse_cache_uses_explicit_site_names():
    model = _make_model()
    info = _make_info(model, 50.0)
    site_names = ["link_site", "root_site"]
    traj = Trajectory(
        info=info,
        data=TrajectoryData(
            qpos=_make_qpos(model, 1),
            qvel=np.zeros((1, model.nv), dtype=np.float32),
            split_points=np.array([0, 1]),
        ),
    )

    materialized = materialize_trajectory(
        traj,
        model=model,
        control_dt=1.0 / 50.0,
        cache_type=TrajectoryCacheType.SPARSE,
        site_names=site_names,
    )

    mj_data = mujoco.MjData(model)
    mj_data.qpos[:] = materialized.data.qpos[0]
    mj_data.qvel[:] = materialized.data.qvel[0]
    mujoco.mj_forward(model, mj_data)
    site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names])
    parent_ids = model.site_bodyid[site_ids]
    root_body_id = np.unique(model.body_rootid[parent_ids]).item()

    assert materialized.info.site_names == site_names
    assert materialized.data.xpos.size == 0
    assert materialized.data.cvel.size == 0
    np.testing.assert_allclose(materialized.data.site_xpos[0], mj_data.site_xpos[site_ids], atol=1e-6)
    np.testing.assert_allclose(materialized.data.cvel_parent[0], mj_data.cvel[parent_ids], atol=1e-6)
    np.testing.assert_allclose(materialized.data.subtree_com_root[0], mj_data.subtree_com[root_body_id], atol=1e-6)


def test_sparse_cache_requires_explicit_site_names():
    model = _make_model()
    data = TrajectoryData(
        qpos=_make_qpos(model, 2),
        qvel=np.zeros((2, model.nv), dtype=np.float32),
        split_points=np.array([0, 2]),
    )

    with pytest.raises(ValueError, match="SPARSE trajectory cache requires site_names"):
        compute_trajectory_kinematic_caches(data, model, TrajectoryCacheType.SPARSE, backend=np)


def test_retarget_extend_motion_returns_source_cache_after_layout_filtering(monkeypatch):
    model = _make_hinge_model()
    info = TrajectoryInfo(
        joint_names=["joint_a"],
        model=TrajectoryModel(njnt=1, jnt_type=np.array([int(mujoco.mjtJoint.mjJNT_HINGE)])),
        frequency=50.0,
    )
    qpos = (0.1 * np.arange(5, dtype=np.float32)[:, None] ** 2).astype(np.float32)
    traj = Trajectory(
        info=info,
        data=TrajectoryData(
            qpos=qpos,
            qvel=np.zeros((len(qpos), 1), dtype=np.float32),
            split_points=np.array([0, len(qpos)]),
        ),
    )

    class FakeEnv:
        dt = 0.01

        def __init__(self, **_kwargs):
            self._model = model
            self.th = None

        def load_trajectory(self, trajectory, warn=False, cache_type=TrajectoryCacheType.FULL, site_names=None):
            del warn, site_names
            self.th = TrajectoryHandler(
                model,
                traj=trajectory,
                control_dt=self.dt,
                random_start=False,
                fixed_start_conf=(0, 0),
                cache_type=cache_type,
            )

    monkeypatch.setitem(retargeting_module.Mujoco.registered_envs, "FakeRetargetEnv", FakeEnv)

    extended = retargeting_module.extend_motion("FakeRetargetEnv", {}, traj)

    assert extended.data.qpos.shape[1] == model.nq
    assert extended.data.qvel.shape[1] == model.nv
    assert extended.data.cvel.size == 0
    assert extended.data.site_xpos.size == 0
    _assert_qvel_matches_qpos(
        model,
        np.asarray(extended.data.qpos),
        np.asarray(extended.data.qvel),
        np.asarray(extended.data.split_points),
        extended.info.frequency,
    )


def test_retargeting_helpers_preserve_fractional_fps():
    assert _fps_after_frame_skip(59.94, 2) == pytest.approx(29.97)
