"""
MSK metrics for GMR vs. mimic retargeting.

Finger disabling is applied before metric computation to keep qpos and tendon
indexing consistent. The module stays quiet unless run as ``__main__``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

import mujoco as mj
import numpy as np
from musclemimic_models import get_xml_path

import loco_mujoco
from loco_mujoco.core.utils.ground import FloorSurfaceDistanceEvaluator


def _get_converted_amass_path() -> Path:
    """Get converted AMASS path from env var or config."""
    # 1. Try environment variable
    path = os.environ.get("MUSCLEMIMIC_CONVERTED_AMASS_PATH")
    if path:
        return Path(path)

    # 2. Try config file
    path = loco_mujoco.load_path_config().get("MUSCLEMIMIC_CONVERTED_AMASS_PATH")
    if path:
        return Path(path)

    raise RuntimeError(
        "MUSCLEMIMIC_CONVERTED_AMASS_PATH not found in environment or config. "
        "Run 'musclemimic-set-all-caches --path /your/path' to configure."
    )


def _get_base_and_gmr_dirs() -> tuple[Path, Path]:
    """Get BASE and GMR_DIR paths for MSK metrics comparison."""
    base = _get_converted_amass_path() / "MyoFullBody"
    gmr_dir = base / "gmr"
    return base, gmr_dir


def apply_spec_changes(spec: mj.MjSpec) -> mj.MjSpec:
    """Apply the same finger-disabling logic used by the retargeting pipeline."""

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


def joint_violation_pct(
    model,
    qpos_traj,
    threshold: float = 1e-5,
):
    """
    Percentage of joint-limit violations with tolerance.

    Matches original check_qpos semantics:
      val < low - threshold OR val > high + threshold
    """
    assert qpos_traj.ndim == 2

    num_frames = qpos_traj.shape[0]
    total_violations = 0
    total_checks = 0

    for j in range(model.njnt):
        jt = model.joint(j).type
        if jt not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            continue

        low, high = model.jnt_range[j]
        qpos_idx = model.jnt_qposadr[j]
        vals = qpos_traj[:, qpos_idx]

        violations = (vals < (low - threshold)) | (vals > (high + threshold))
        total_violations += int(np.sum(violations))
        total_checks += num_frames

    return 100.0 * total_violations / max(1, total_checks)


def ground_penetration_stats(
    model,
    qpos,
    floor: str = "floor",
    tol: float = 1e-3,
    return_info: bool = False,
):
    """
    Ground penetration over a trajectory using signed collision-surface distance.

    Returns:
      pen_pct: % frames with any penetration deeper than tol
      max_pen_m: maximum penetration depth (meters, positive)

    If return_info=True, also returns:
      (max_pen_m, max_pen_frame, max_pen_geom_name)
    """
    distance_evaluator = FloorSurfaceDistanceEvaluator(model, floor_name=floor)
    distances_m, closest_geom_ids = distance_evaluator.measure_trajectory(qpos)
    num_frames = len(qpos)
    violation_mask = distances_m < -tol
    violated_frames = int(np.count_nonzero(violation_mask))
    max_pen_frame = -1
    max_pen_geom_name = None
    if np.any(violation_mask):
        violating_distances_m = np.where(violation_mask, distances_m, np.inf)
        max_pen_frame = int(np.argmin(violating_distances_m))
        max_pen_m = float(-distances_m[max_pen_frame])
        max_pen_geom_name = model.geom(int(closest_geom_ids[max_pen_frame])).name
    else:
        max_pen_m = 0.0

    pen_pct = 100.0 * violated_frames / max(1, num_frames)

    if return_info:
        return pen_pct, max_pen_m, max_pen_frame, max_pen_geom_name
    return pen_pct, max_pen_m


def floating_height(model, qpos):
    """
    Return the minimum signed collision-surface distance to the floor in meters.
    """
    distance_evaluator = FloorSurfaceDistanceEvaluator(model)
    distances_m, _closest_geom_ids = distance_evaluator.measure_trajectory(qpos)
    return float(np.min(distances_m))


def tendon_jump_max(
    model,
    qpos,
    jump_factor=10.0,
    min_rel_jump=1e-3,
    ema_alpha=0.01,
    return_info=False,
):
    """
    Streaming tendon jump detector over a qpos trajectory.

    At each step, forward the model to obtain tendon lengths L and compute the
    per-tendon relative step change:

        dL_rel = |L[t] - L[t-1]| / max(L0, 1e-6)

    An exponential moving average (EMA) of dL_rel is maintained per tendon.
    A jump is flagged when:

        dL_rel > max(jump_factor * EMA(dL_rel), min_rel_jump)

    Returns the maximum dL_rel among all detected jumps (0.0 if none). If
    return_info=True, also returns the timestep index, tendon index, and
    tendon name corresponding to the maximum jump.
    """

    data = mj.MjData(model)
    nt = model.ntendon
    num_frames = qpos.shape[0]

    prev_len = None
    ema_abs_change = np.zeros(nt)

    max_jump = 0.0
    max_t = -1
    max_k = -1

    reference_lengths = model.tendon_length0.astype(np.float64)

    for t in range(num_frames):
        data.qpos[:] = qpos[t]
        mj.mj_forward(model, data)
        cur_len = data.ten_length.copy()

        if prev_len is not None:
            relative_length_change = np.abs(cur_len - prev_len) / np.maximum(reference_lengths, 1e-6)

            # Initialize EMA on first delta
            if t == 1:
                ema_abs_change[:] = relative_length_change
            else:
                ema_abs_change[:] = (1.0 - ema_alpha) * ema_abs_change + ema_alpha * relative_length_change

            threshold = np.maximum(jump_factor * ema_abs_change, min_rel_jump)

            mask = relative_length_change > threshold
            if np.any(mask):
                k_local = np.argmax(np.where(mask, relative_length_change, -np.inf))
                val = float(relative_length_change[k_local])

                if val > max_jump:
                    max_jump = val
                    max_t = t
                    max_k = int(k_local)

        prev_len = cur_len

    if return_info:
        tendon_name = model.tendon(max_k).name if max_k >= 0 else None
        return max_jump, max_t, max_k, tendon_name
    return max_jump


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


def compute_msk_metrics(
    motion_file: Path | str,
    compute: Iterable[str] = ("joint", "penetration", "float", "tendon", "speed"),
) -> dict[str, dict[str, float]]:

    motion_file = Path(motion_file)
    model = load_model()

    base, gmr_dir = _get_base_and_gmr_dirs()
    gmr = np.load(gmr_dir / motion_file, allow_pickle=True)
    mimic = np.load(base / motion_file, allow_pickle=True)

    out = {"gmr": {}, "mimic": {}}

    if "joint" in compute:
        out["gmr"]["joint_pct"] = joint_violation_pct(model, gmr["qpos"])
        out["mimic"]["joint_pct"] = joint_violation_pct(model, mimic["qpos"])

    if "penetration" in compute:
        pen_pct, max_pen_m = ground_penetration_stats(model, gmr["qpos"])
        out["gmr"]["pen_pct"] = pen_pct
        out["gmr"]["pen_max_depth_m"] = max_pen_m

        pen_pct, max_pen_m = ground_penetration_stats(model, mimic["qpos"])
        out["mimic"]["pen_pct"] = pen_pct
        out["mimic"]["pen_max_depth_m"] = max_pen_m

    if "float" in compute:
        out["gmr"]["float_height"] = floating_height(model, gmr["qpos"])
        out["mimic"]["float_height"] = floating_height(model, mimic["qpos"])

    if "tendon" in compute:
        max_jump, t, k, name = tendon_jump_max(model, gmr["qpos"], return_info=True)
        out["gmr"]["tendon_max_jump_%"] = max_jump
        out["gmr"]["tendon_max_jump_frame"] = float(t)
        out["gmr"]["tendon_max_jump_tendon_id"] = float(k)
        out["gmr"]["tendon_max_jump_tendon_name"] = name

        max_jump, t, k, name = tendon_jump_max(model, mimic["qpos"], return_info=True)
        out["mimic"]["tendon_max_jump_%"] = max_jump
        out["mimic"]["tendon_max_jump_frame"] = float(t)
        out["mimic"]["tendon_max_jump_tendon_id"] = float(k)
        out["mimic"]["tendon_max_jump_tendon_name"] = name

    if "speed" in compute:
        out["gmr"]["fps"], out["gmr"]["sec_per_frame"], out["gmr"]["pos_rmse"] = speed_stats(gmr_dir / motion_file)
        out["mimic"]["fps"], out["mimic"]["sec_per_frame"], out["mimic"]["pos_rmse"] = speed_stats(base / motion_file)

    return out


def _print(metrics):
    for k in ("gmr", "mimic"):
        print(f"\n========== {k.upper()} ==========")
        for n, v in metrics[k].items():
            print(f"{n:18s}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", required=True)
    args = parser.parse_args()

    _print(compute_msk_metrics(args.motion_file))
