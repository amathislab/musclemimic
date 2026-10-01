"""
MSK metrics for retargeting comparison.

Finger disabling is applied before metric computation to keep qpos and tendon
indexing consistent. The module stays quiet unless run as ``__main__``.

Two families of metric live here. The first is the core benchmark set --
joint-limit violations, ground penetration, floating, tendon jump, RMSE and speed. The
second was added while evaluating OmniRetarget, after several defects turned out to be
invisible to the first: the joint-coupler residual, physical tendon length rate,
joint-violation severity and per-frame joint step. Each is documented with the failure it
exists to catch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import mujoco as mj
import numpy as np
from musclemimic_models import get_xml_path


def _get_converted_amass_path() -> Path:
    """Get converted AMASS path from env var, config, or the pipeline default.

    Delegates to the resolver the retargeting pipeline itself uses, which accepts both the
    ``CONVERTED_AMASS_PATH`` and ``MUSCLEMIMIC_CONVERTED_AMASS_PATH`` spellings and falls
    back to ``~/.musclemimic/caches/AMASS``. This module used to accept only the long name
    and raise otherwise, so metrics refused to run on a cache the retargeting had just
    written.
    """
    from loco_mujoco.smpl.retargeting import get_converted_amass_dataset_path

    return Path(get_converted_amass_dataset_path())


# Cache subdirectory per retargeting method, relative to <converted>/MyoFullBody. The
# benchmark methods use explicit, non-overlapping namespaces. ``mimic`` remains the legacy
# top-level cache.
METHOD_DIRS: Dict[str, str] = {
    "mimic": "",
    "smpl": "smpl",
    "gmr": "gmr",
    "terra": "terra",
    "omniretarget": "omniretarget",
}


def method_cache_path(method: str, motion_file: Path | str) -> Path:
    """Cache path of one motion under one retargeting method."""
    if method not in METHOD_DIRS:
        raise KeyError(f"unknown retargeting method {method!r}; known: {sorted(METHOD_DIRS)}")
    base = _get_converted_amass_path() / "MyoFullBody"
    sub = METHOD_DIRS[method]
    motion_file = Path(motion_file)
    if motion_file.suffix != ".npz":
        motion_file = motion_file.with_suffix(".npz")
    return (base / sub / motion_file) if sub else (base / motion_file)


def _get_base_and_gmr_dirs() -> tuple[Path, Path]:
    """Get BASE and GMR_DIR paths for MSK metrics comparison."""
    base = _get_converted_amass_path() / "MyoFullBody"
    gmr_dir = base / "gmr"
    return base, gmr_dir


def apply_spec_changes(spec: mj.MjSpec) -> mj.MjSpec:
    """Apply the same finger-disabling logic used by the retargeting pipeline."""

    from musclemimic.utils.model_fixes import apply_myofullbody_model_fixes

    spec = apply_myofullbody_model_fixes(spec)

    finger_joints = {
        # right
        "cmc_flexion_r",
        "cmc_abduction_r",
        "mp_flexion_r",
        "ip_flexion_r",
        "mcp2_flexion_r",
        "mcp2_abduction_r",
        "mcp3_flexion_r",
        "mcp3_abduction_r",
        "mcp4_flexion_r",
        "mcp4_abduction_r",
        "mcp5_flexion_r",
        "mcp5_abduction_r",
        "md2_flexion_r",
        "md3_flexion_r",
        "md4_flexion_r",
        "md5_flexion_r",
        "pm2_flexion_r",
        "pm3_flexion_r",
        "pm4_flexion_r",
        "pm5_flexion_r",
        # left
        "cmc_flexion_l",
        "cmc_abduction_l",
        "mp_flexion_l",
        "ip_flexion_l",
        "mcp2_flexion_l",
        "mcp2_abduction_l",
        "mcp3_flexion_l",
        "mcp3_abduction_l",
        "mcp4_flexion_l",
        "mcp4_abduction_l",
        "mcp5_flexion_l",
        "mcp5_abduction_l",
        "md2_flexion_l",
        "md3_flexion_l",
        "md4_flexion_l",
        "md5_flexion_l",
        "pm2_flexion_l",
        "pm3_flexion_l",
        "pm4_flexion_l",
        "pm5_flexion_l",
    }

    joints_to_remove = [j for j in spec.joints if j.name in finger_joints]
    for j in joints_to_remove:
        spec.delete(j)

    return spec


def load_model() -> mj.MjModel:
    xml = get_xml_path("myofullbody")
    spec = mj.MjSpec.from_file(str(xml))
    spec = apply_spec_changes(spec)  # Keep joint and tendon indexing consistent.
    return spec.compile()


def joint_violation_stats(
    model,
    qpos_traj,
    threshold: float = 1e-5,
) -> Dict[str, float]:
    """
    Joint-limit violation rate *and* severity.

    The core benchmark reports only the rate, which cannot distinguish a joint sitting
    1e-4 rad past its stop for half the clip from one driven 30 deg through it for three
    frames. Both read as a percentage of frames, and only the second is a broken pose. The
    severity columns are the worst excursion beyond a limit, in degrees, and the fraction
    of frames in which *any* joint violates (as opposed to the joint-frame rate, which is
    diluted by the model's 80-odd joints).

    Violation test matches the original check_qpos semantics:
      val < low - threshold OR val > high + threshold
    """
    assert qpos_traj.ndim == 2

    T = qpos_traj.shape[0]
    total_violations = 0
    total_checks = 0
    worst_deg = 0.0
    worst_joint = None
    any_joint = np.zeros(T, dtype=bool)

    for j in range(model.njnt):
        jt = model.joint(j).type
        if jt not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue

        low, high = model.jnt_range[j]
        qpos_idx = model.jnt_qposadr[j]
        vals = qpos_traj[:, qpos_idx]

        violations = (vals < (low - threshold)) | (vals > (high + threshold))
        total_violations += int(np.sum(violations))
        total_checks += T
        any_joint |= violations

        if violations.any():
            # Slide joints are metres, not radians; the model has none among the limited
            # DOFs, but converting them to degrees would be nonsense if it gained any.
            excess = np.maximum(low - vals, vals - high)[violations].max()
            excess = float(np.degrees(excess)) if jt == mj.mjtJoint.mjJNT_HINGE else float(excess)
            if excess > worst_deg:
                worst_deg, worst_joint = excess, model.joint(j).name

    return {
        "joint_pct": 100.0 * total_violations / max(1, total_checks),
        "joint_frame_pct": 100.0 * float(any_joint.mean()) if T else 0.0,
        "joint_max_deg": worst_deg,
        "joint_worst": worst_joint,
    }


def joint_violation_pct(
    model,
    qpos_traj,
    threshold: float = 1e-5,
):
    """Percentage of joint-limit violations with tolerance. See `joint_violation_stats`."""
    return joint_violation_stats(model, qpos_traj, threshold)["joint_pct"]


def ground_penetration_stats(
    model,
    qpos,
    floor: str = "floor",
    tol: float = 1e-3,
    return_info: bool = False,
):
    """
    Ground penetration over trajectory using MuJoCo contacts.

    Returns:
      pen_pct: % frames with any penetration deeper than tol
      max_pen_m: maximum penetration depth (meters, positive)

    If return_info=True, also returns:
      (max_pen_m, max_pen_frame, max_pen_geom_name)
    """
    data = mj.MjData(model)
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, floor)
    if floor_id < 0:
        raise ValueError(f"No geom named '{floor}' found")

    T = len(qpos)
    violated_frames = 0

    max_pen_m = 0.0
    max_pen_frame = -1
    max_pen_geom_name = None

    for t in range(T):
        data.qpos[:] = qpos[t]
        mj.mj_forward(model, data)

        frame_has_violation = False

        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == floor_id or c.geom2 == floor_id:
                # c.dist < 0 => penetration. depth in meters = -c.dist
                if c.dist < -tol:
                    frame_has_violation = True

                    depth_m = float(-c.dist)  # positive penetration depth
                    if depth_m > max_pen_m:
                        max_pen_m = depth_m
                        max_pen_frame = t
                        offending_geom = c.geom2 if c.geom1 == floor_id else c.geom1
                        max_pen_geom_name = model.geom(offending_geom).name

        if frame_has_violation:
            violated_frames += 1

    pen_pct = 100.0 * violated_frames / max(1, T)

    if return_info:
        return pen_pct, max_pen_m, max_pen_frame, max_pen_geom_name
    return pen_pct, max_pen_m


def floating_height(model, qpos):
    """
    Return the minimum geom z-height over the whole trajectory (meters),
    skipping the geom named "floor" if it exists.
    """
    data = mj.MjData(model)
    min_z = np.inf

    # Precompute non-floor geom ids once
    non_floor_geom_ids = []
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if name == "floor":
            continue
        non_floor_geom_ids.append(geom_id)

    # Fallback: if nothing left (unlikely), use all geoms
    use_ids = non_floor_geom_ids if non_floor_geom_ids else list(range(model.ngeom))

    for q in qpos:
        data.qpos[:] = q
        mj.mj_forward(model, data)
        min_z = min(min_z, float(np.min(data.geom_xpos[use_ids, 2])))

    return float(min_z)


def tendon_stats_and_events(
    model,
    qpos,
    jump_factor=10.0,
    min_rel_jump=1e-3,
    ema_alpha=0.01,
) -> tuple[dict[str, float], list[dict]]:
    """
    Tendon length behaviour and all flagged event frames, in one forward pass.

    **Jump detection** (the benchmark metric). Per-tendon relative step change

        dL_rel = |L[t] - L[t-1]| / max(L0, 1e-6)

    is compared against an exponential moving average of itself; a jump is flagged when

        dL_rel > max(jump_factor * EMA(dL_rel), min_rel_jump)

    and `tendon_max_jump` is the largest flagged dL_rel (0.0 if none).

    **Physical length rate** (added). `tendon_max_mm` is the largest |L[t] - L[t-1]| in
    millimetres, regardless of whether it was flagged. The jump metric is a ratio against
    each tendon's *own* moving average, so which tendon wins and by how much can flip on a
    sub-degree pose change: two OmniRetarget configurations read 0.0461 and 0.0792 -
    opposite sides of the 0.05 threshold, and different named tendons - while their worst
    physical length changes were 1.764 and 1.772 mm/frame, 0.45% apart. Without the
    millimetre column there is no way to tell a musculotendon regression from detector
    noise, and tuning against the ratio alone tunes against the noise.

    One event is retained for every flagged frame (the largest flagged tendon on that
    frame), ranked by relative jump. Keeping the whole ranked list matters during repair:
    once the former maximum is removed, the old scalar metric otherwise conceals the next
    pelvis-spanning event until a full evaluation is rerun.
    """

    data = mj.MjData(model)
    nt = model.ntendon
    T = qpos.shape[0]

    prev_len = None
    ema_abs_change = np.zeros(nt)

    max_jump = 0.0
    max_t = -1
    max_k = -1

    max_mm = 0.0
    max_mm_t = -1
    max_mm_k = -1
    events = []

    L0 = model.tendon_length0.astype(np.float64)

    for t in range(T):
        data.qpos[:] = qpos[t]
        mj.mj_forward(model, data)
        cur_len = data.ten_length.copy()

        if prev_len is not None:
            dL_abs = np.abs(cur_len - prev_len)
            dL_rel = dL_abs / np.maximum(L0, 1e-6)

            k_abs = int(np.argmax(dL_abs))
            if dL_abs[k_abs] * 1000.0 > max_mm:
                max_mm = float(dL_abs[k_abs]) * 1000.0
                max_mm_t, max_mm_k = t, k_abs

            # Initialize EMA on first delta
            if t == 1:
                ema_abs_change[:] = dL_rel
            else:
                ema_abs_change[:] = (1.0 - ema_alpha) * ema_abs_change + ema_alpha * dL_rel

            threshold = np.maximum(jump_factor * ema_abs_change, min_rel_jump)

            mask = dL_rel > threshold
            if np.any(mask):
                k_local = np.argmax(np.where(mask, dL_rel, -np.inf))
                val = float(dL_rel[k_local])
                events.append(
                    {
                        "tendon_frame": int(t),
                        "tendon_id": int(k_local),
                        "tendon_name": model.tendon(int(k_local)).name,
                        "tendon_relative_jump": val,
                        "tendon_physical_mm": float(dL_abs[k_local]) * 1000.0,
                    }
                )

                if val > max_jump:
                    max_jump = val
                    max_t = t
                    max_k = int(k_local)

        prev_len = cur_len

    stats = {
        "tendon_max_jump": max_jump,
        "tendon_jump_frame": max_t,
        "tendon_jump_id": max_k,
        "tendon_jump_name": model.tendon(max_k).name if max_k >= 0 else None,
        "tendon_max_mm": max_mm,
        "tendon_mm_frame": max_mm_t,
        "tendon_mm_name": model.tendon(max_mm_k).name if max_mm_k >= 0 else None,
    }
    events.sort(key=lambda event: event["tendon_relative_jump"], reverse=True)
    for rank, event in enumerate(events, 1):
        event["tendon_rank"] = rank
    return stats, events


def tendon_stats(
    model,
    qpos,
    jump_factor=10.0,
    min_rel_jump=1e-3,
    ema_alpha=0.01,
) -> dict[str, float]:
    """Scalar tendon statistics. See :func:`tendon_stats_and_events`."""
    return tendon_stats_and_events(model, qpos, jump_factor, min_rel_jump, ema_alpha)[0]


def tendon_jump_events(
    model,
    qpos,
    jump_factor=10.0,
    min_rel_jump=1e-3,
    ema_alpha=0.01,
) -> list[dict]:
    """All flagged tendon-event frames, ranked largest first.

    Each dict includes the frame, winning tendon id/name, relative jump, physical change
    in millimetres, and one-based rank. Use :func:`tendon_stats_and_events` when both the
    scalar summary and this list are needed so MuJoCo is traversed only once.
    """
    return tendon_stats_and_events(model, qpos, jump_factor, min_rel_jump, ema_alpha)[1]


def tendon_jump_max(
    model,
    qpos,
    jump_factor=10.0,
    min_rel_jump=1e-3,
    ema_alpha=0.01,
    return_info=False,
):
    """
    Maximum relative tendon length change classified as a jump. See `tendon_stats`.

    If return_info=True, also returns the timestep index, tendon index, and
    tendon name corresponding to the maximum jump.
    """
    s = tendon_stats(model, qpos, jump_factor, min_rel_jump, ema_alpha)
    if return_info:
        return s["tendon_max_jump"], s["tendon_jump_frame"], s["tendon_jump_id"], s["tendon_jump_name"]
    return s["tendon_max_jump"]


def self_penetration_stats(model, qpos, floor: str = "floor", tol: float = 1e-3) -> Dict[str, float]:
    """
    Interpenetration between the model's own geoms, by the same test used for the floor.

    The core benchmark checks the body against the ground and never against itself, so an arm
    driven through the torso -- a pose no controller can reproduce and one that changes
    muscle paths -- scores clean on every column. Uses MuJoCo's own contact distances, so
    pairs the model excludes (`contype`/`conaffinity`, explicit `<exclude>`, parent-child)
    are excluded here too: this reports only the contacts the model itself considers real.
    """
    data = mj.MjData(model)
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, floor)

    T = len(qpos)
    frames = 0
    max_depth = 0.0
    worst = None

    for t in range(T):
        data.qpos[:] = qpos[t]
        mj.mj_forward(model, data)
        hit = False
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 == floor_id or c.geom2 == floor_id:
                continue
            if c.dist < -tol:
                hit = True
                depth = float(-c.dist)
                if depth > max_depth:
                    max_depth = depth
                    worst = f"{model.geom(c.geom1).name}/{model.geom(c.geom2).name}"
        frames += int(hit)

    return {
        "self_pen_pct": 100.0 * frames / max(1, T),
        "self_pen_max_m": max_depth,
        "self_pen_worst": worst,
    }


def root_fidelity_stats(model, qpos, source_pelvis: np.ndarray | None) -> Dict[str, float]:
    """
    Whether the root travelled the way the source did.

    Every other tracking number here is pelvis-relative, which is what makes it comparable
    across methods with different ground-alignment and heading conventions -- and which
    means global root motion is otherwise unmeasured. For locomotion that is the part that
    has to be right: a retargeter whose planted feet slide covers a different distance than
    the actor did, and no joint, contact or tendon column reports it.

    `root_path_ratio` is horizontal path length, retargeted over source: 1.0 is correct
    travel, below 1 is a motion compressed by feet that stick, above 1 one stretched by feet
    that slide. `root_z_mae_mm` compares pelvis height after removing each trajectory's own
    median height, so the differing ground offsets cancel and only the shape of the vertical
    oscillation is compared.
    """
    empty = {"root_path_ratio": None, "root_z_mae_mm": None, "root_travel_m": None}
    xy = np.asarray(qpos)[:, :2]
    travel = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    empty["root_travel_m"] = travel
    if source_pelvis is None or len(source_pelvis) < 2:
        return empty

    src_xy = np.asarray(source_pelvis)[:, :2]
    src_travel = float(np.linalg.norm(np.diff(src_xy, axis=0), axis=1).sum())

    out = dict(empty)
    if src_travel > 1e-6:
        out["root_path_ratio"] = travel / src_travel

    # Compare over the overlapping span only; methods trim differently.
    n = min(len(qpos), len(source_pelvis))
    z = np.asarray(qpos)[:n, 2]
    src_z = np.asarray(source_pelvis)[:n, 2]
    out["root_z_mae_mm"] = float(np.abs((z - np.median(z)) - (src_z - np.median(src_z))).mean() * 1000.0)
    return out


def root_step_series(model, qpos) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame free-root translation (mm) and rotation (degrees) steps.

    The returned arrays have the same length as `qpos`; frame zero is 0 because it has no
    predecessor. Quaternion distance is geodesic, after normalization, and uses
    ``abs(dot(q[t], q[t-1]))`` so the equivalent `q` and `-q` encodings never create a
    false 360-degree jump.

    If the model has no free joint, both arrays are zero. A zero-norm quaternion is treated
    as the identity, which makes partially initialized diagnostic trajectories usable.
    """
    qpos = np.asarray(qpos, dtype=float)
    n = len(qpos)
    translation = np.zeros(n, dtype=float)
    rotation = np.zeros(n, dtype=float)
    if n < 2:
        return translation, rotation

    free = next(
        (j for j in range(model.njnt) if model.joint(j).type == mj.mjtJoint.mjJNT_FREE),
        None,
    )
    if free is None:
        return translation, rotation
    adr = int(model.jnt_qposadr[free])
    if qpos.ndim != 2 or qpos.shape[1] < adr + 7:
        raise ValueError(f"qpos must have at least {adr + 7} columns for the model's free joint, got {qpos.shape}")

    translation[1:] = np.linalg.norm(np.diff(qpos[:, adr : adr + 3], axis=0), axis=1) * 1000.0

    quat = qpos[:, adr + 3 : adr + 7].copy()
    norm = np.linalg.norm(quat, axis=1)
    zero = norm <= np.finfo(float).eps
    quat[~zero] /= norm[~zero, None]
    quat[zero] = (1.0, 0.0, 0.0, 0.0)
    dot = np.abs(np.sum(quat[1:] * quat[:-1], axis=1))
    rotation[1:] = np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))
    return translation, rotation


def root_step_stats(model, qpos) -> dict[str, float]:
    """Maximum and 99.9th-percentile frame steps of the free root.

    Translation is reported in millimetres and quaternion geodesic rotation in degrees.
    Frame fields identify the *new* frame at the maximum, matching `qstep_frame` and the
    tendon-event frame convention.
    """
    translation, rotation = root_step_series(model, qpos)
    if len(translation) < 2:
        return {
            "root_step_max_mm": 0.0,
            "root_step_p999_mm": 0.0,
            "root_step_frame": -1,
            "root_rot_step_max_deg": 0.0,
            "root_rot_step_p999_deg": 0.0,
            "root_rot_step_frame": -1,
        }

    trans_steps = translation[1:]
    rot_steps = rotation[1:]
    return {
        "root_step_max_mm": float(trans_steps.max()),
        "root_step_p999_mm": float(np.percentile(trans_steps, 99.9)),
        "root_step_frame": int(np.argmax(trans_steps)) + 1,
        "root_rot_step_max_deg": float(rot_steps.max()),
        "root_rot_step_p999_deg": float(np.percentile(rot_steps, 99.9)),
        "root_rot_step_frame": int(np.argmax(rot_steps)) + 1,
    }


def coupler_residual_stats(model, qpos) -> Dict[str, float]:
    """
    Violation of the model's own joint-equality couplers, in degrees.

    MyoFullBody defines 49 active ``mjEQ_JOINT`` equalities: the lumbar chain is driven off
    `flex_extension` / `lat_bending` / `axial_rotation` (e.g. ``L1_L2_FE = 0.255 *
    flex_extension``) and the shoulder girdle off `shoulder_elv`. They are part of the
    model's definition of a valid pose. MuJoCo enforces equalities through constraint
    *forces* during simulation, so a directly-assigned `qpos` never experiences them -
    which means a retargeter that optimises `qpos` kinematically can leave the coupled
    manifold entirely while every pose still looks correct in playback.

    This is the metric no core benchmark column stands in for, and it is a *validity* check
    rather than a quality one: a trajectory with a large residual is not a valid
    configuration of this model, so its muscle lengths, moment arms and tendon forces are
    not the model's own. It is also cheap - pure qpos arithmetic, no forward pass.

    Returns the mean and max over ``|dependent - poly(independent)|`` across all couplers
    and frames, plus the worst offending dependent joint.
    """
    from musclemimic.retargeting import joint_couplers

    couplers = joint_couplers(model)
    if not couplers:
        return {"coupler_mean_deg": 0.0, "coupler_max_deg": 0.0, "coupler_worst": None, "n_couplers": 0}

    qpos = np.asarray(qpos)
    per_coupler_max = []
    all_resid = []
    for dep, indep, poly in couplers:
        x = qpos[:, indep]
        y = qpos[:, dep]
        pred = poly[0] + poly[1] * x + poly[2] * x**2 + poly[3] * x**3 + poly[4] * x**4
        resid = np.degrees(np.abs(y - pred))
        all_resid.append(resid)
        per_coupler_max.append((float(resid.max()), dep))

    all_resid = np.stack(all_resid)  # (n_couplers, T)
    worst_val, worst_dep = max(per_coupler_max)

    # Name the joint by its qpos address; hinge joints are one address each, so the map is
    # unambiguous for every joint a coupler can drive.
    name = None
    for j in range(model.njnt):
        if int(model.jnt_qposadr[j]) == worst_dep:
            name = model.joint(j).name
            break

    return {
        "coupler_mean_deg": float(all_resid.mean()),
        "coupler_max_deg": worst_val,
        "coupler_worst": name,
        "n_couplers": len(couplers),
    }


def qpos_step_stats(model, qpos, skip_free: bool = True) -> Dict[str, float]:
    """
    Largest frame-to-frame joint step, in degrees, and the joint it happens on.

    Catches discontinuities that the tendon detector reads as clean. With the orientation
    cost on and axial damping off, `elv_angle` / `shoulder_rot` swap by ~1 rad in a single
    frame at low shoulder elevation - two joint-space branches for the same humerus
    orientation, which no task-space cost can adjudicate. `tendon_jump_max` reads 0.0215
    either way and the rendered video looks fine; only the raw qpos step sees it.

    The free root joint is excluded by default: its quaternion components are not angles,
    and its translation is dominated by locomotion speed rather than by smoothness.
    """
    qpos = np.asarray(qpos)
    if len(qpos) < 2:
        return {"qstep_max_deg": 0.0, "qstep_p999_deg": 0.0, "qstep_frame": -1, "qstep_worst": None}

    idx, names = [], []
    for j in range(model.njnt):
        jt = model.joint(j).type
        if skip_free and jt in (mj.mjtJoint.mjJNT_FREE, mj.mjtJoint.mjJNT_BALL):
            continue
        if jt != mj.mjtJoint.mjJNT_HINGE:
            continue
        idx.append(int(model.jnt_qposadr[j]))
        names.append(model.joint(j).name)

    steps = np.degrees(np.abs(np.diff(qpos[:, idx], axis=0)))  # (T-1, n_hinge)
    flat = int(np.argmax(steps))
    t, k = np.unravel_index(flat, steps.shape)
    return {
        "qstep_max_deg": float(steps[t, k]),
        # A single worst frame is a spike; the 99.9th percentile says whether the whole
        # trajectory is rough, which is the difference between one bad frame and a jitter.
        "qstep_p999_deg": float(np.percentile(steps, 99.9)),
        "qstep_frame": int(t) + 1,
        "qstep_worst": names[int(k)],
    }


def speed_stats(motion_file: Path | str):
    """
    Load retargeting speed stats + RMSE from *_analysis.npz.

    Expected:
      <motion_file>_analysis.npz with keys:
        - 'retarget_fps'
        - 'pos_error' (per-frame position error)
    Returns:
      (fps, sec_per_frame, pos_rmse)
    """
    motion_file = Path(motion_file)
    analysis_path = motion_file.with_name(motion_file.stem + "_analysis.npz")

    if not analysis_path.exists():
        return None, None, None

    try:
        d = np.load(analysis_path, allow_pickle=True)

        fps = float(d["retarget_fps"]) if "retarget_fps" in d else None
        sec_per_frame = None if fps is None else 1.0 / max(1e-12, fps)

        pos_rmse = None
        if "pos_error" in d:
            pe = np.asarray(d["pos_error"])
            pos_rmse = float(np.sqrt(np.mean(pe**2)))

        return fps, sec_per_frame, pos_rmse

    except Exception:
        return None, None, None


def speed_and_error_stats(motion_file: Path | str, n_output_frames: int, output_hz: float = 100.0):
    """
    Retargeting cost and landmark error from `*_analysis.npz`, with the frame-rate
    confound made explicit.

    `retarget_fps` is frames-solved per wall second, but each method solves at a *different*
    rate: `fit_smpl_motion` fits at ~25 Hz, `fit_gmr_motion` aligns the source to ~30 Hz,
    and `fit_terra_motion` solves every 100 Hz output frame. All three then produce a 100 Hz
    trajectory. So `1 / retarget_fps` -- the benchmark's $T_\\text{frame}$ -- is seconds per
    *solved* frame, and the three columns are denominated in different units: a method that
    solves at a quarter of the output rate looks four times cheaper than it is.

    `sec_per_motion_sec` fixes that by dividing total solve time by the duration of the
    motion, which is the same quantity for every method and the one that decides how long a
    dataset takes to prepare.

    Returns a dict; `None`-valued entries mean the sidecar is absent or lacks the key. GMR
    caches fetched from the published HuggingFace dataset have no sidecar at all, so they
    report `None` here -- a downloaded trajectory has no local retargeting cost to report,
    and inventing one from the download time would be a fabrication.
    """
    motion_file = Path(motion_file)
    analysis_path = motion_file.with_name(motion_file.stem + "_analysis.npz")

    empty = {
        "fps": None,
        "sec_per_frame": None,
        "pos_rmse": None,
        "pos_max": None,
        "solve_hz": None,
        "solve_seconds": None,
        "sec_per_motion_sec": None,
        "n_solved": None,
    }
    if not analysis_path.exists():
        return empty

    try:
        d = np.load(analysis_path, allow_pickle=True)
    except Exception:
        return empty

    out = dict(empty)
    fps = float(d["retarget_fps"]) if "retarget_fps" in d else None
    out["fps"] = fps
    if fps is not None:
        out["sec_per_frame"] = 1.0 / max(1e-12, fps)

    if "pos_error" in d:
        pe = np.asarray(d["pos_error"])
        out["pos_rmse"] = float(np.sqrt(np.mean(pe**2)))
        out["pos_max"] = float(pe.max())
        # `pos_error` is recorded once per solved frame, which is how the solve rate is
        # recoverable at all -- nothing else in the sidecar records it.
        out["n_solved"] = int(pe.shape[0])

    motion_seconds = n_output_frames / output_hz if output_hz else None
    if out["n_solved"] and fps:
        out["solve_seconds"] = out["n_solved"] / max(1e-12, fps)
        if motion_seconds:
            out["solve_hz"] = out["n_solved"] / motion_seconds
            out["sec_per_motion_sec"] = out["solve_seconds"] / motion_seconds

    return out


def compute_msk_metrics(
    motion_file: Path | str,
    compute: Iterable[str] = ("joint", "penetration", "float", "tendon", "speed"),
    methods: Iterable[str] = ("gmr", "mimic"),
) -> Dict[str, Dict[str, float]]:
    """
    Metrics for one motion under each named retargeting method.

    Defaults to the two-method ``("gmr", "mimic")`` benchmark comparison; pass
    e.g. ``methods=("smpl", "gmr", "terra")`` for a three-way one. Method names resolve
    through `METHOD_DIRS`.
    """
    motion_file = Path(motion_file)
    model = load_model()
    methods = list(methods)

    out: Dict[str, Dict[str, float]] = {}
    for method in methods:
        path = method_cache_path(method, motion_file)
        qpos = np.load(path, allow_pickle=True)["qpos"]
        row: Dict[str, float] = {}

        if "joint" in compute:
            row["joint_pct"] = joint_violation_pct(model, qpos)

        if "penetration" in compute:
            pen_pct, max_pen_m = ground_penetration_stats(model, qpos)
            row["pen_pct"] = pen_pct
            row["pen_max_depth_m"] = max_pen_m

        if "float" in compute:
            row["float_height"] = floating_height(model, qpos)

        if "tendon" in compute:
            max_jump, t, k, name = tendon_jump_max(model, qpos, return_info=True)
            row["tendon_max_jump_%"] = max_jump
            row["tendon_max_jump_frame"] = float(t)
            row["tendon_max_jump_tendon_id"] = float(k)
            row["tendon_max_jump_tendon_name"] = name

        if "speed" in compute:
            speed = speed_and_error_stats(path, len(qpos))
            row["fps"] = speed["fps"]
            row["sec_per_frame"] = speed["sec_per_frame"]
            row["pos_rmse"] = speed["pos_rmse"]

        out[method] = row

    return out


def _print(metrics):
    for k, row in metrics.items():
        print(f"\n========== {k.upper()} ==========")
        for n, v in row.items():
            print(f"{n:18s}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--methods", default="gmr,mimic", help="comma-separated, see METHOD_DIRS")
    args = parser.parse_args()

    _print(compute_msk_metrics(args.motion_file, methods=args.methods.split(",")))
