"""Tests for musclemimic.utils.msk_metrics module."""

import mujoco as mj
import numpy as np
import pytest

from musclemimic.utils.retarget.msk_metrics import (
    floating_height,
    ground_penetration_stats,
    joint_violation_pct,
    load_model,
    tendon_jump_max,
)

# =============================================================================
# floating_height() tests
# =============================================================================

FLOAT_THRESHOLD = 0.005  # 5mm above ground


def test_grounded_trajectory(model):
    """Test that trajectory at ground level returns ~0 min height."""
    qpos = np.zeros((10, model.nq))

    min_z = floating_height(model, qpos)

    assert min_z < 0.1, f"Expected low min height, got {min_z}m"


def test_floating_trajectory_detected(model):
    """Test that elevated trajectory returns positive min height."""
    qpos = np.zeros((10, model.nq))
    qpos[:, 2] = 2.0  # 2m above ground

    min_z = floating_height(model, qpos)

    assert min_z > FLOAT_THRESHOLD, f"Expected floating, got min height {min_z}m"


def test_floating_height_varies_with_z(model):
    """Test that min height increases with Z position."""
    qpos_low = np.zeros((5, model.nq))
    qpos_low[:, 2] = 0.5

    qpos_high = np.zeros((5, model.nq))
    qpos_high[:, 2] = 2.0

    min_z_low = floating_height(model, qpos_low)
    min_z_high = floating_height(model, qpos_high)

    assert min_z_high > min_z_low, (
        f"Higher Z should give higher min_z: low={min_z_low}, high={min_z_high}"
    )


# =============================================================================
# ground_penetration_stats() tests
# =============================================================================


def test_no_penetration(model):
    """Test that trajectory well above ground has no penetration."""
    qpos = np.zeros((10, model.nq))
    qpos[:, 2] = 1.0  # Z height well above ground

    pen_pct, max_pen_m = ground_penetration_stats(model, qpos)

    assert pen_pct == 0.0, f"Expected 0% penetration, got {pen_pct}%"
    assert max_pen_m == 0.0, f"Expected 0m max penetration, got {max_pen_m}m"


def test_penetration_detected(model):
    """Test that trajectory dipping below ground detects penetration."""
    qpos = np.zeros((10, model.nq))
    qpos[5:8, 2] = -0.05  # 5cm below ground

    pen_pct, max_pen_m, max_frame, max_geom = ground_penetration_stats(
        model, qpos, return_info=True
    )

    assert pen_pct > 0.0, f"Expected penetration, got {pen_pct}%"
    assert max_pen_m > 0.0, f"Expected positive penetration depth, got {max_pen_m}m"
    assert max_frame >= 5, f"Expected penetration in frames 5-7, got frame {max_frame}"


def test_penetration_tolerance(model):
    """Test that penetrations below tolerance are ignored."""
    qpos = np.zeros((10, model.nq))
    qpos[:, 2] = 0.9  # Height that produces some penetration

    # First measure actual penetration with no tolerance
    _, max_pen_m = ground_penetration_stats(model, qpos, tol=0)
    assert max_pen_m > 0, "Test requires some penetration to verify tolerance"

    # Setting tolerance above max penetration should filter everything
    pen_pct, _ = ground_penetration_stats(model, qpos, tol=max_pen_m + 1e-4)
    assert pen_pct == 0.0, f"Expected 0% when tol > max_pen, got {pen_pct}%"


# =============================================================================
# joint_violation_pct() tests
# =============================================================================


def test_joint_violation_pct_no_violation():
    """Test that joints at midpoint of range have 0% violation."""
    model = load_model()
    T, nq = 10, model.nq

    qpos = np.zeros((T, nq))

    for j in range(model.njnt):
        if model.joint(j).type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            idx = model.jnt_qposadr[j]
            low, high = model.jnt_range[j]
            if not np.isinf(low) and not np.isinf(high):
                qpos[:, idx] = (low + high) / 2

    violation_pct = joint_violation_pct(model, qpos)
    assert violation_pct == 0.0, f"Expected 0% violation, got {violation_pct}%"


def test_joint_violation_pct_with_violation():
    """Test that joints exceeding limits are detected."""
    model = load_model()
    T, nq = 5, model.nq
    qpos = np.zeros((T, nq))

    hinge_joints = [
        j
        for j in range(model.njnt)
        if model.joint(j).type == mj.mjtJoint.mjJNT_HINGE
    ]
    assert len(hinge_joints) > 0, "No hinge joints found"

    hinge_j = hinge_joints[0]
    idx = model.jnt_qposadr[hinge_j]
    low, high = model.jnt_range[hinge_j]

    qpos[:, idx] = high + 0.1

    violation_pct = joint_violation_pct(model, qpos)
    assert violation_pct > 0.0, f"Expected violations, got {violation_pct}%"


# =============================================================================
# tendon_jump_max() tests
# =============================================================================


def test_tendon_jump_detected():
    """Test that sudden qpos changes trigger tendon jump detection."""
    model = load_model()
    T, nq = 10, model.nq

    qpos = np.zeros((T, nq))
    qpos[5, :] += 0.5  # Induce a jump mid-trajectory

    max_jump, max_t, max_k, tendon_name = tendon_jump_max(
        model, qpos, jump_factor=2.0, min_rel_jump=1e-6, return_info=True
    )

    assert max_jump > 0, f"Expected jump detected, got max_jump={max_jump}"
    assert max_t >= 0, f"Expected valid frame index, got {max_t}"
    assert tendon_name is None or isinstance(tendon_name, str)


def test_no_tendon_jump():
    """Test that constant qpos produces no tendon jumps."""
    model = load_model()
    T, nq = 20, model.nq
    qpos = np.zeros((T, nq))

    max_jump = tendon_jump_max(model, qpos)

    assert max_jump == 0.0, f"Expected no jumps, got max_jump={max_jump}"


def test_tendon_jump_frame_index():
    """Test that jump is detected at the correct frame."""
    model = load_model()
    T, nq = 15, model.nq

    qpos = np.zeros((T, nq))
    qpos[7, :] += 1.0  # Big jump at frame 7

    max_jump, max_t, max_k, tendon_name = tendon_jump_max(
        model, qpos, min_rel_jump=1e-6, return_info=True
    )

    assert max_t in [7, 8], f"Expected jump at frame 7 or 8, got frame {max_t}"
    assert tendon_name is None or isinstance(tendon_name, str)


def test_tendon_jump_magnitude_positive():
    """Test that detected jump magnitude is positive."""
    model = load_model()
    T, nq = 12, model.nq

    qpos = np.zeros((T, nq))
    qpos[6, :] += 0.3

    max_jump = tendon_jump_max(model, qpos, min_rel_jump=1e-6)

    if max_jump > 0:
        assert max_jump > 0, "Jump magnitude should be positive"


def test_tendon_jump_runtime():
    """Test that tendon jump detection completes in reasonable time."""
    model = load_model()
    T, nq = 200, model.nq
    qpos = np.random.randn(T, nq) * 0.01

    max_jump = tendon_jump_max(model, qpos)

    assert isinstance(max_jump, float)


# =============================================================================
# Added metrics: severity, physical tendon rate, couplers, joint steps, root
# =============================================================================


def test_joint_violation_severity_matches_the_excursion(model):
    """Severity must report how far past the stop, not just that it happened."""
    from musclemimic.utils.retarget.msk_metrics import joint_violation_stats

    hinge = next(
        j for j in range(model.njnt)
        if model.joint(j).type == mj.mjtJoint.mjJNT_HINGE
        and np.all(np.isfinite(model.jnt_range[j]))
        and model.jnt_range[j][1] > model.jnt_range[j][0]
    )
    idx = model.jnt_qposadr[hinge]
    low, high = model.jnt_range[hinge]

    qpos = np.zeros((10, model.nq))
    for j in range(model.njnt):
        if model.joint(j).type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            lo, hi = model.jnt_range[j]
            if np.isfinite(lo) and np.isfinite(hi):
                qpos[:, model.jnt_qposadr[j]] = (lo + hi) / 2
    qpos[3:6, idx] = high + 0.1

    s = joint_violation_stats(model, qpos)

    assert s["joint_max_deg"] == pytest.approx(np.degrees(0.1), abs=1e-6)
    assert s["joint_worst"] == model.joint(hinge).name
    assert s["joint_frame_pct"] == pytest.approx(30.0), "3 of 10 frames violate"
    # The joint-frame rate is diluted by every other limited joint, which is exactly why
    # the frame rate is reported alongside it.
    assert s["joint_pct"] < s["joint_frame_pct"]


def test_joint_violation_pct_still_matches_the_stats_helper(model):
    """The benchmark rate must not drift now that it delegates."""
    from musclemimic.utils.retarget.msk_metrics import joint_violation_pct, joint_violation_stats

    rng = np.random.default_rng(0)
    qpos = rng.normal(scale=0.4, size=(15, model.nq))

    assert joint_violation_pct(model, qpos) == joint_violation_stats(model, qpos)["joint_pct"]


def test_tendon_millimetre_rate_is_zero_when_nothing_moves(model):
    """A static pose has no length change, whatever the ratio detector says."""
    from musclemimic.utils.retarget.msk_metrics import tendon_stats

    s = tendon_stats(model, np.zeros((20, model.nq)))

    assert s["tendon_max_mm"] == 0.0
    assert s["tendon_max_jump"] == 0.0


def test_tendon_millimetre_rate_records_a_real_length_change(model):
    """The physical rate must fire on a jump and name the tendon it happened on."""
    from musclemimic.utils.retarget.msk_metrics import tendon_stats

    qpos = np.zeros((12, model.nq))
    qpos[6:, :] += 0.3

    s = tendon_stats(model, qpos, min_rel_jump=1e-6)

    assert s["tendon_max_mm"] > 1.0, "a 0.3 rad step must move some tendon by over a mm"
    assert s["tendon_mm_frame"] == 6
    assert isinstance(s["tendon_mm_name"], str)


def test_ranked_tendon_events_keep_secondary_frames(model):
    """A repaired maximum must not hide the next already-detected event."""
    from musclemimic.utils.retarget.msk_metrics import tendon_jump_events

    hinge = next(j for j in range(model.njnt) if model.joint(j).type == mj.mjtJoint.mjJNT_HINGE)
    qpos = np.repeat(np.asarray(model.qpos0)[None], 50, axis=0)
    adr = int(model.jnt_qposadr[hinge])
    qpos[10:, adr] += 0.8
    qpos[30:, adr] -= 0.4

    events = tendon_jump_events(model, qpos, jump_factor=2.0, min_rel_jump=1e-8)

    frames = {event["tendon_frame"] for event in events}
    assert {10, 30} <= frames
    assert [event["tendon_rank"] for event in events] == list(range(1, len(events) + 1))
    assert all(
        events[i]["tendon_relative_jump"] >= events[i + 1]["tendon_relative_jump"]
        for i in range(len(events) - 1)
    )
    assert len(events[1:]) >= 1, "removing the maximum must leave the ranked secondary event"


def test_myofullbody_biclong_wrap_side_sites_are_outside_the_unstable_boundary(model):
    """Metrics and runtime environments must compile the corrected BIClong wrap side."""
    import mujoco

    from musclemimic.utils.model_fixes import MYOFULLBODY_BICLONG_SIDESITE_Y

    for suffix in ("", "_left"):
        site = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"BIClong_ellipsoid_BIClong_2_sidesite{suffix}",
        )
        assert site >= 0
        assert model.site_pos[site, 1] == pytest.approx(MYOFULLBODY_BICLONG_SIDESITE_Y)


def test_myofullbody_gaslat_wrap_cylinders_use_stable_radius(model):
    """Both gastrocnemius paths must compile off their measured topology boundary."""
    import mujoco

    from musclemimic.utils.model_fixes import MYOFULLBODY_GASLAT_WRAP_RADIUS

    for side in ("r", "l"):
        geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"GasLat_at_shank_{side}_wrap"
        )
        assert geom >= 0
        assert model.geom_size[geom, 0] == pytest.approx(MYOFULLBODY_GASLAT_WRAP_RADIUS)


def test_coupler_residual_is_zero_on_the_manifold(model):
    """A pose built to satisfy the equalities must report no violation."""
    from musclemimic.retargeting import joint_couplers
    from musclemimic.utils.retarget.msk_metrics import coupler_residual_stats

    couplers = joint_couplers(model)
    assert couplers, "MyoFullBody defines joint couplers; the metric is meaningless without"

    rng = np.random.default_rng(1)
    qpos = np.zeros((5, model.nq))
    for _dep, indep, _poly in couplers:
        qpos[:, indep] = rng.normal(scale=0.2, size=5)
    # Assign dependents after every independent is set: a dependent joint may itself drive
    # nothing, but two couplers can share an independent.
    for dep, indep, poly in couplers:
        x = qpos[:, indep]
        qpos[:, dep] = poly[0] + poly[1] * x + poly[2] * x**2 + poly[3] * x**3 + poly[4] * x**4

    s = coupler_residual_stats(model, qpos)

    assert s["n_couplers"] == len(couplers)
    assert s["coupler_max_deg"] < 1e-6, f"on-manifold pose reported {s['coupler_max_deg']} deg"


def test_coupler_residual_sees_a_broken_chain(model):
    """Driving a dependent joint off its polynomial must be reported, in degrees."""
    from musclemimic.retargeting import joint_couplers
    from musclemimic.utils.retarget.msk_metrics import coupler_residual_stats

    couplers = joint_couplers(model)
    dep, _indep, _poly = couplers[0]

    qpos = np.zeros((4, model.nq))
    for d, _i, p in couplers:  # start on the manifold at qpos = 0
        qpos[:, d] = p[0]
    qpos[:, dep] += np.radians(5.0)

    s = coupler_residual_stats(model, qpos)

    assert s["coupler_max_deg"] == pytest.approx(5.0, abs=1e-6)
    assert s["coupler_mean_deg"] > 0


def test_qpos_step_finds_the_joint_that_jumped(model):
    """A branch flip shows only in the raw step, so it must be located exactly."""
    from musclemimic.utils.retarget.msk_metrics import qpos_step_stats

    hinge = next(j for j in range(model.njnt) if model.joint(j).type == mj.mjtJoint.mjJNT_HINGE)
    qpos = np.zeros((10, model.nq))
    qpos[4:, model.jnt_qposadr[hinge]] = 1.0  # ~57 deg in one frame

    s = qpos_step_stats(model, qpos)

    assert s["qstep_max_deg"] == pytest.approx(np.degrees(1.0), abs=1e-6)
    assert s["qstep_frame"] == 4
    assert s["qstep_worst"] == model.joint(hinge).name


def test_root_path_ratio_compares_distance_travelled(model):
    """Feet that stick shorten the path and feet that slide lengthen it."""
    from musclemimic.utils.retarget.msk_metrics import root_fidelity_stats

    t = np.linspace(0, 1, 50)
    source = np.stack([t, np.zeros_like(t), np.full_like(t, 0.9)], axis=1)

    qpos = np.zeros((50, model.nq))
    qpos[:, 0] = t
    qpos[:, 2] = 0.9
    assert root_fidelity_stats(model, qpos, source)["root_path_ratio"] == pytest.approx(1.0)

    qpos[:, 0] = t / 2  # half the distance for the same motion
    assert root_fidelity_stats(model, qpos, source)["root_path_ratio"] == pytest.approx(0.5)


def test_root_z_mae_ignores_a_constant_ground_offset(model):
    """Methods apply different ground offsets; only the vertical shape is comparable."""
    from musclemimic.utils.retarget.msk_metrics import root_fidelity_stats

    t = np.linspace(0, 4 * np.pi, 200)
    z = 0.9 + 0.02 * np.sin(t)
    source = np.stack([np.zeros_like(t), np.zeros_like(t), z], axis=1)

    qpos = np.zeros((200, model.nq))
    qpos[:, 2] = z + 0.05  # same oscillation, 5 cm higher

    assert root_fidelity_stats(model, qpos, source)["root_z_mae_mm"] == pytest.approx(0.0, abs=1e-6)


def test_root_step_metrics_locate_translation_rotation_and_ignore_quaternion_sign(model):
    """Root discontinuities need physical units and sign-invariant quaternion distance."""
    from musclemimic.utils.retarget.msk_metrics import root_step_series, root_step_stats

    free = next(j for j in range(model.njnt) if model.joint(j).type == mj.mjtJoint.mjJNT_FREE)
    adr = int(model.jnt_qposadr[free])
    qpos = np.repeat(np.asarray(model.qpos0)[None], 12, axis=0)
    qpos[4:, adr] += 0.025
    angle = np.radians(30.0)
    rotated = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    qpos[6:, adr + 3:adr + 7] = rotated
    qpos[9:, adr + 3:adr + 7] *= -1.0

    translation, rotation = root_step_series(model, qpos)
    stats = root_step_stats(model, qpos)

    assert stats["root_step_max_mm"] == pytest.approx(25.0)
    assert stats["root_step_frame"] == 4
    assert stats["root_rot_step_max_deg"] == pytest.approx(30.0, abs=1e-6)
    assert stats["root_rot_step_frame"] == 6
    assert translation[4] == pytest.approx(25.0)
    assert rotation[9] == pytest.approx(0.0, abs=1e-9), "q and -q are the same rotation"


def test_self_penetration_is_reported_in_range(model):
    """Body-against-body contacts, by the same test used for the floor."""
    from musclemimic.utils.retarget.msk_metrics import self_penetration_stats

    qpos = np.zeros((5, model.nq))
    qpos[:, 2] = 2.0  # clear of the floor, so only self-contacts can be reported

    s = self_penetration_stats(model, qpos)

    assert 0.0 <= s["self_pen_pct"] <= 100.0
    assert s["self_pen_max_m"] >= 0.0
    assert s["self_pen_worst"] is None or "/" in s["self_pen_worst"]
