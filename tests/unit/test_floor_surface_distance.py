"""Tests for collision-surface ground distance and trajectory alignment."""

import mujoco
import numpy as np
import pytest

from loco_mujoco.core.utils.ground import FloorSurfaceDistanceEvaluator


def _make_floor_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" pos="0 0 0.2"
                  size="2 2 0.1" contype="1" conaffinity="1"/>
            <body name="root_body" pos="0 0 0.5">
              <freejoint name="root"/>
              <geom name="visual_only" type="sphere" size="0.5"
                    contype="0" conaffinity="0"/>
              <geom name="collision" type="sphere" size="0.1"
                    contype="1" conaffinity="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _make_floor_model_with_excluded_subtree() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" pos="0 0 0.2"
                  size="2 2 0.1" contype="1" conaffinity="1"/>
            <body name="root_body" pos="0 0 0.5">
              <freejoint name="root"/>
              <geom name="kept_collision" type="sphere" size="0.1"
                    contype="1" conaffinity="1"/>
              <body name="excluded_root" pos="0 0 -0.3">
                <body name="excluded_descendant">
                  <geom name="excluded_collision" type="sphere" size="0.1"
                        contype="1" conaffinity="1"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _qpos_at_heights(model: mujoco.MjModel, heights_m: list[float]) -> np.ndarray:
    qpos = np.repeat(model.qpos0[None], len(heights_m), axis=0)
    qpos[:, 2] = heights_m
    return qpos


def test_distance_uses_collision_surfaces_and_floor_position():
    model = _make_floor_model()
    evaluator = FloorSurfaceDistanceEvaluator(model)
    qpos = _qpos_at_heights(model, [0.5, 0.25])

    distances_m, closest_geom_ids = evaluator.measure_trajectory(qpos)

    np.testing.assert_allclose(distances_m, [0.2, -0.05], atol=1e-12)
    collision_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "collision")
    np.testing.assert_array_equal(closest_geom_ids, collision_geom_id)


def test_distance_excludes_named_body_subtrees():
    model = _make_floor_model_with_excluded_subtree()
    qpos = _qpos_at_heights(model, [0.5])

    unfiltered_distances_m, _ = FloorSurfaceDistanceEvaluator(model).measure_trajectory(qpos)
    filtered_distances_m, closest_geom_ids = FloorSurfaceDistanceEvaluator(
        model,
        excluded_body_subtree_names=("excluded_root",),
    ).measure_trajectory(qpos)

    np.testing.assert_allclose(unfiltered_distances_m, [-0.1], atol=1e-12)
    np.testing.assert_allclose(filtered_distances_m, [0.2], atol=1e-12)
    kept_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "kept_collision")
    np.testing.assert_array_equal(closest_geom_ids, kept_geom_id)


def test_distance_rejects_missing_excluded_body_name():
    model = _make_floor_model()

    with pytest.raises(ValueError, match="No body named 'missing'"):
        FloorSurfaceDistanceEvaluator(model, excluded_body_subtree_names=("missing",))


@pytest.mark.parametrize(
    ("heights_m", "expected_translation_m"),
    [
        ([0.5, 0.7], -0.2),
        ([0.25, 0.45], 0.05),
    ],
)
def test_alignment_applies_one_translation_and_preserves_vertical_motion(
    heights_m: list[float],
    expected_translation_m: float,
):
    model = _make_floor_model()
    evaluator = FloorSurfaceDistanceEvaluator(model)
    qpos = _qpos_at_heights(model, heights_m)
    original_qpos = qpos.copy()
    distances_m, _closest_geom_ids = evaluator.measure_trajectory(qpos)

    aligned_qpos, translation_m = evaluator.align_trajectory(
        qpos,
        "root",
        surface_distances_m=distances_m,
    )

    assert translation_m == pytest.approx(expected_translation_m)
    np.testing.assert_array_equal(qpos, original_qpos)
    np.testing.assert_allclose(np.diff(aligned_qpos[:, 2]), np.diff(qpos[:, 2]), atol=1e-12)
    aligned_distances_m, _closest_geom_ids = evaluator.measure_trajectory(aligned_qpos)
    assert np.min(aligned_distances_m) == pytest.approx(0.0, abs=1e-12)


def test_alignment_rejects_a_missing_root_joint():
    model = _make_floor_model()
    evaluator = FloorSurfaceDistanceEvaluator(model)
    qpos = _qpos_at_heights(model, [0.5])

    with pytest.raises(ValueError, match="No joint named 'missing'"):
        evaluator.align_trajectory(qpos, "missing")


@pytest.mark.parametrize(
    ("heights_m", "minimum_clearance_m", "expected_translation_m"),
    [
        ([0.25, 0.45], 1e-6, 0.050001),
        ([0.5, 0.7], 1e-6, 0.0),
    ],
)
def test_minimum_clearance_guard_only_lifts_and_preserves_vertical_motion(
    heights_m: list[float],
    minimum_clearance_m: float,
    expected_translation_m: float,
):
    model = _make_floor_model()
    evaluator = FloorSurfaceDistanceEvaluator(model)
    qpos = _qpos_at_heights(model, heights_m)
    distances_m, _closest_geom_ids = evaluator.measure_trajectory(qpos)

    guarded_qpos, translation_m = evaluator.ensure_minimum_trajectory_clearance(
        qpos,
        "root",
        minimum_clearance_m,
        surface_distances_m=distances_m,
    )

    assert translation_m == pytest.approx(expected_translation_m)
    np.testing.assert_allclose(np.diff(guarded_qpos[:, 2]), np.diff(qpos[:, 2]), atol=1e-12)
    guarded_distances_m, _closest_geom_ids = evaluator.measure_trajectory(guarded_qpos)
    assert np.min(guarded_distances_m) >= minimum_clearance_m - 1e-12
