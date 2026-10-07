import glob
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from musclemimic_models import get_xml_path
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

from loco_mujoco.smpl.contact_grounding import ground_contact_aware_trajectory
from loco_mujoco.smpl.gmr_cache_identity import GMR_CACHE_NAMESPACE, gmr_motion_relative_path
from loco_mujoco.smpl.motion_spec import AMASSMotionSpec, as_amass_motion_spec

try:
    import joblib
    import torch
    from torch.autograd import Variable

    from loco_mujoco.smpl import SMPLH_Parser

    _OPTIONAL_IMPORT_INSTALLED = True
except ImportError as e:
    _OPTIONAL_IMPORT_INSTALLED = False
    _OPTIONAL_IMPORT_EXCEPTION = e

# GMR imports
try:
    # Patch mink.solve_ik to disable safety_break (joint limit check)
    import mink
    from general_motion_retargeting import (
        IK_CONFIG_DICT,
        ROBOT_BASE_DICT,
        ROBOT_XML_DICT,
        VIEWER_CAM_DISTANCE_DICT,
        GeneralMotionRetargeting,
    )
    from general_motion_retargeting.utils.shape_fitting import (
        compute_alignment_offsets,
        fit_smpl_shape_to_robot,
        get_robot_tpose_targets,
        get_smpl_tpose_indices,
        save_fitted_shape,
    )
    from general_motion_retargeting.utils.smpl import (
        SMPLH_Parser as GMR_SMPLH_Parser,  # Renamed to avoid conflict with loco_mujoco.smpl.SMPLH_Parser
    )
    from general_motion_retargeting.utils.smpl import (
        get_smplh_data_offline_fast,
        load_smplh_file,
    )
    from mink.tasks.equality_constraint_task import EqualityConstraintTask

    # Patch mink.solve_ik to disable safety_break (joint limit check)
    _original_solve_ik = mink.solve_ik

    def _solve_ik_no_safety_break(
        configuration,
        tasks,
        dt,
        solver,
        damping=1e-12,
        safety_break=False,
        limits=None,
        constraints=None,
        **kwargs,
    ):
        if isinstance(safety_break, list):
            limits = safety_break
        return _original_solve_ik(
            configuration,
            tasks,
            dt,
            solver,
            damping=damping,
            safety_break=False,
            limits=limits,
            constraints=constraints,
            **kwargs,
        )

    mink.solve_ik = _solve_ik_no_safety_break

    _GMR_INSTALLED = True
    _GMR_SHAPE_FITTING_INSTALLED = True
except ImportError:
    _GMR_INSTALLED = False
    _GMR_SHAPE_FITTING_INSTALLED = False

import loco_mujoco
from loco_mujoco import PATH_TO_SMPL_ROBOT_CONF
from loco_mujoco.core.mujoco_base import Mujoco
from loco_mujoco.core.utils.ground import FloorSurfaceDistanceEvaluator
from loco_mujoco.core.utils.math import quat_scalarlast2scalarfirst
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from loco_mujoco.datasets.data_generation.utils import add_mocap_bodies
from loco_mujoco.smpl import SMPLH_BONE_ORDER_NAMES
from loco_mujoco.smpl.utils.smoothing import gaussian_filter_1d_batch
from loco_mujoco.trajectory import (
    LoadedTrajectorySet,
    Trajectory,
    TrajectoryCacheType,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
)
from loco_mujoco.utils import setup_logger
from musclemimic.utils import detect_headless_environment, setup_headless_rendering
from musclemimic.utils.gmr_cache import try_download_gmr_cache

OPTIMIZED_SHAPE_FILE_NAME = "shape_optimized.pkl"
BIMANUAL_ENV_NAME = "MyoBimanualArm"
# This cost keeps MyoFullBody's dependent joints coupled across GMR 0.2.0
# installations that construct equality tasks differently.
GMR_EQUALITY_CONSTRAINT_COST = 200.0
# Disabled finger joints remain welded in a neutral pose. Their distal
# collision geoms do not define the actor's trajectory-wide ground height.
MYOFULLBODY_GMR_FLOOR_EXCLUDED_BODY_SUBTREES = (
    "firstmc_l",
    "secondmc_l",
    "thirdmc_l",
    "fourthmc_l",
    "fifthmc_l",
    "firstmc_r",
    "secondmc_r",
    "thirdmc_r",
    "fourthmc_r",
    "fifthmc_r",
)
# Repeated solves remove the default-pose bias from the first motion frame.
GMR_INITIALIZATION_PASSES = 5
GMR_INITIALIZATION_MAX_PASSES = 20
# MuJoCo tangent-space distance between successive solves of the same frame.
# The mixed units are meters for root translation and radians for rotation and
# scalar joints. A 1e-3 limit bounds initialization-induced velocity to about
# 0.03 units/s at the native 30 Hz retargeting rate.
GMR_INITIALIZATION_QPOS_DELTA_TOLERANCE = 1e-3
GMR_INITIALIZATION_CONSECUTIVE_CONVERGED_PASSES = 2
# Prefix refinement starts from a sustained low-error frame and solves toward
# frame zero. This recovers motions whose first frame enters an incorrect IK
# branch before sequential warm starts reach the intended pose.
GMR_PREFIX_MEAN_ERROR_THRESHOLD_M = 0.10
GMR_RELIABLE_FRAME_MAX_ERROR_M = 0.10
GMR_PREFIX_STABLE_DURATION_S = 0.5
# Central qvel differentiation removes frame zero. An anchor must therefore be
# after frame one so backward refinement covers the first retained qpos.
GMR_PREFIX_MIN_ANCHOR_FRAME = 2
GMR_TRAJECTORY_MEAN_ERROR_THRESHOLD_M = 0.10
# The 11 cm threshold is the smallest 5 mm increment that retains every
# official KIT Kinesis train and test motion under the 10% frame-fraction gate.
GMR_FLOATING_CLEARANCE_THRESHOLD_M = 0.11
GMR_MAX_FLOATING_FRAME_FRACTION = 0.10
GMR_FLOOR_DISTANCE_NUMERICAL_TOLERANCE_M = 1e-9
# Cubic position interpolation can overshoot the native-grid no-penetration
# constraint. A 1 um output margin absorbs float32 serialization error after a
# single rigid world-z lift at the final trajectory frequency.
GMR_OUTPUT_FLOOR_CLEARANCE_MARGIN_M = 1e-6
# The 100 Hz gmr_v6 inventory contains one known initialization-induced root
# spin at 15.58 rad/s. Every other leading 0.1 s segment remains below 7.82
# rad/s. The linear limit is above the observed 4.83 m/s maximum.
GMR_LEADING_CONTINUITY_DURATION_S = 0.10
GMR_MAX_LEADING_ROOT_LINEAR_SPEED_M_S = 6.0
GMR_MAX_LEADING_ROOT_ANGULAR_SPEED_RAD_S = 10.0

# The unqualified gmr_v6 directory represents these exact source-specific
# defaults. Any behavior-affecting override receives a deterministic variant
# directory so trajectories and rejection markers cannot cross configurations.
_GMR_CACHE_CONFIG_DEFAULTS = {
    "target_fps": 30.0,
    "solver": "daqp",
    "damping": 0.5,
    "offset_to_ground": False,
    "use_velocity_limit": False,
    "shape_fitting_iterations": 500,
    "max_frames": None,
    "source_fps": None,
    "initialization_passes": GMR_INITIALIZATION_PASSES,
    "initialization_max_passes": GMR_INITIALIZATION_MAX_PASSES,
    "initialization_qpos_delta_tolerance": GMR_INITIALIZATION_QPOS_DELTA_TOLERANCE,
    "initialization_consecutive_converged_passes": GMR_INITIALIZATION_CONSECUTIVE_CONVERGED_PASSES,
    "refine_bad_prefix": True,
    "validate_tracking_quality": True,
    "record_tracking_positions": False,
    "stage2_position_cost_overrides": {},
    "floating_clearance_threshold_m": GMR_FLOATING_CLEARANCE_THRESHOLD_M,
    "max_floating_frame_fraction": GMR_MAX_FLOATING_FRAME_FRACTION,
    "soma_use_learned_offsets": False,
}
_GMR_CACHE_BOOLEAN_CONFIG_KEYS = {
    "offset_to_ground",
    "use_velocity_limit",
    "refine_bad_prefix",
    "validate_tracking_quality",
    "record_tracking_positions",
    "soma_use_learned_offsets",
    "use_fitted_shape",
}
_GMR_CACHE_INTEGER_CONFIG_KEYS = {
    "shape_fitting_iterations",
    "max_frames",
    "initialization_passes",
    "initialization_max_passes",
    "initialization_consecutive_converged_passes",
}
_GMR_CACHE_FLOAT_CONFIG_KEYS = {
    "target_fps",
    "damping",
    "source_fps",
    "initialization_qpos_delta_tolerance",
    "floating_clearance_threshold_m",
    "max_floating_frame_fraction",
}
_GMR_CACHE_NON_SEMANTIC_CONFIG_KEYS = {"verbose"}


def _normalize_gmr_cache_value(value):
    """Convert configuration values to deterministic JSON-compatible values."""
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _normalize_gmr_cache_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_normalize_gmr_cache_value(item) for item in value]
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.ndarray):
        return _normalize_gmr_cache_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_gmr_cache_identity(gmr_config: dict | DictConfig | None = None) -> tuple[str, str | None]:
    """Return the full configuration digest and optional noncanonical variant.

    The canonical fitted SMPL-H configuration retains the published ``gmr_v6``
    layout. Every behavior-affecting override uses ``gmr_v6/_variants/<sha256>``.
    """
    provided = dict(gmr_config or {})
    src_human = str(provided.get("src_human", "smplh"))
    use_fitted_shape = bool(provided.get("use_fitted_shape", True))
    payload = {
        "src_human": src_human,
        "use_fitted_shape": use_fitted_shape,
        **_GMR_CACHE_CONFIG_DEFAULTS,
    }
    for key, value in provided.items():
        if key in _GMR_CACHE_NON_SEMANTIC_CONFIG_KEYS:
            continue
        if key in _GMR_CACHE_BOOLEAN_CONFIG_KEYS:
            payload[key] = bool(value)
        elif key in _GMR_CACHE_INTEGER_CONFIG_KEYS:
            payload[key] = None if value is None else int(value)
        elif key in _GMR_CACHE_FLOAT_CONFIG_KEYS:
            payload[key] = None if value is None else float(value)
        elif key == "stage2_position_cost_overrides":
            payload[key] = {str(name): float(cost) for name, cost in sorted(dict(value or {}).items())}
        else:
            payload[key] = _normalize_gmr_cache_value(value)

    # The published gmr_v6 cache identities include this empty field.
    payload["soma_fitted_shape_sha256"] = ""

    canonical_payload = {
        "src_human": src_human,
        "use_fitted_shape": True,
        **_GMR_CACHE_CONFIG_DEFAULTS,
        "soma_fitted_shape_sha256": "",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return digest, None if payload == canonical_payload else digest


class GMRQualityGateError(RuntimeError):
    """Report a reproducible GMR quality-gate rejection."""

    def __init__(self, message: str, metrics: dict[str, float | int | bool]) -> None:
        super().__init__(message)
        self.metrics = metrics


def _resolve_gmr_motion_paths(dataset_name: str, gmr_config: dict | None = None) -> tuple[str, str]:
    """Return the normalized source path and a safe relative cache key."""
    source_root = Path(get_amass_dataset_path()).expanduser().resolve()
    source_suffix = ".npz"
    requested_path = Path(dataset_name).expanduser()
    if requested_path.suffix.lower() != source_suffix:
        requested_path = Path(f"{requested_path}{source_suffix}")

    source_path = (requested_path if requested_path.is_absolute() else source_root / requested_path).resolve()
    try:
        cache_key = source_path.relative_to(source_root).with_suffix("")
    except ValueError:
        # Keep external absolute paths inside the configured cache root and avoid
        # collisions between sources that happen to share the same basename.
        path_digest = hashlib.sha256(os.fsencode(source_path)).hexdigest()[:16]
        cache_key = Path("_external") / path_digest / source_path.stem

    return str(source_path), cache_key.as_posix()


def _fps_after_frame_skip(source_fps, skip: int) -> float:
    if skip <= 0:
        raise ValueError(f"skip must be positive, got {skip}")
    return float(np.asarray(source_fps).squeeze()) / float(skip)


def _resample_motion_frames(frames: list, source_fps: float, target_fps: float) -> tuple[list, float]:
    """Downsample frames at nearest source timestamps without inventing frames."""
    source_fps = float(source_fps)
    target_fps = float(target_fps)
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if target_fps >= source_fps or len(frames) < 2:
        return frames, source_fps

    target_times = np.arange(0.0, len(frames) / source_fps, 1.0 / target_fps)
    source_indices = np.rint(target_times * source_fps).astype(int)
    source_indices = np.clip(source_indices, 0, len(frames) - 1)
    return [frames[index] for index in source_indices], target_fps


def _compute_qvel_from_qpos(qpos: np.ndarray, fps: float, free_joint_name: str, model: mujoco.MjModel) -> np.ndarray:
    """
    Compute joint velocities from positions using MuJoCo's mj_differentiatePos.

    This uses MuJoCo's native differentiation which correctly handles quaternion
    differences for free joints, avoiding rotation vector singularities at π.

    Args:
        qpos: Joint positions array (T, nq)
        fps: Frequency of the trajectory in Hz
        free_joint_name: Name of the free joint (unused, kept for API compatibility)
        model: MuJoCo model

    Returns:
        Tuple of (qpos_trimmed, qvel) where first and last frames are removed due to central differencing
    """
    n_frames = len(qpos)
    dt = 1.0 / fps

    # Compute forward differences using MuJoCo's mj_differentiatePos
    qvel_fwd = np.zeros((n_frames - 1, model.nv), dtype=np.float64)
    for t in range(n_frames - 1):
        mujoco.mj_differentiatePos(model, qvel_fwd[t], dt, qpos[t], qpos[t + 1])

    # Central difference: average forward and backward velocities
    # qvel[t] = 0.5 * (v_fwd[t] + v_fwd[t-1]) for interior frames
    qvel = 0.5 * (qvel_fwd[1:] + qvel_fwd[:-1])

    # Trim qpos to match (remove first and last frames)
    qpos_trimmed = qpos[1:-1].copy()

    return qpos_trimmed, qvel


def _resolve_site_ids(model: mujoco.MjModel, site_names: list[str], *, context: str) -> list[int]:
    """Resolve MuJoCo site names and reject missing sites before indexing arrays."""
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name) for site_name in site_names]
    missing = [site_name for site_name, site_id in zip(site_names, site_ids, strict=True) if site_id < 0]
    if missing:
        raise ValueError(f"{context} references missing MuJoCo site(s): {', '.join(missing)}")
    return site_ids


def _resolve_smpl_joint_indices(sites_for_mimic: list[str], robot_conf: DictConfig, *, context: str) -> list[int]:
    """Map mimic sites to SMPL-H joint indices with clear configuration errors."""
    matches = robot_conf.site_joint_matches
    missing = [site_name for site_name in sites_for_mimic if site_name not in matches]
    if missing:
        raise ValueError(f"{context} is missing site_joint_matches entries for: {', '.join(missing)}")

    joint_indices = []
    invalid = []
    for site_name in sites_for_mimic:
        smpl_joint = matches[site_name].smpl_joint
        try:
            joint_indices.append(SMPLH_BONE_ORDER_NAMES.index(smpl_joint))
        except ValueError:
            invalid.append(f"{site_name}->{smpl_joint}")
    if invalid:
        raise ValueError(f"{context} references unknown SMPL-H joint(s): {', '.join(invalid)}")
    return joint_indices


def _record_site_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    site_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Forward final qpos frames and record selected site positions/orientations."""
    site_xpos = np.zeros((len(qpos), len(site_ids), 3), dtype=np.float32)
    site_xmat = np.zeros((len(qpos), len(site_ids), 9), dtype=np.float32)
    for frame_idx, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        site_xpos[frame_idx] = data.site_xpos[site_ids]
        site_xmat[frame_idx] = data.site_xmat[site_ids]
    return site_xpos, site_xmat


def ensure_pytorch_precision_consistency():
    """
    Ensure consistent PyTorch precision across RTX and A100 GPUs.

    This function disables TensorFloat-32 (TF32) which is enabled by default on A100 GPUs
    but not available on RTX GPUs, causing precision differences in retargeting results.
    """
    try:
        import torch

        # Disable TensorFloat-32 for full FP32 precision consistency across GPU types
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Enable deterministic algorithms for reproducible results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set manual seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    except ImportError:
        # Torch not available, skip precision settings
        print("[Warning] PyTorch not installed, skipping precision consistency settings.")


def check_optional_imports():
    if not _OPTIONAL_IMPORT_INSTALLED:
        raise ImportError(
            f"[LocoMuJoCo] Optional smpl dependencies not installed. "
            f"Checkout the README for installation instructions. {_OPTIONAL_IMPORT_EXCEPTION}"
        )


def _get_default_base_dir():
    """Get default base directory for MuscleMimic data"""
    return loco_mujoco.get_musclemimic_home()


def get_amass_dataset_path():
    """Get AMASS dataset path from env var, config, or default"""
    # 1. Try environment variable (support both old and new names)
    path = os.environ.get("AMASS_PATH") or os.environ.get("MUSCLEMIMIC_AMASS_PATH")
    if path:
        return path

    # 2. Try config file (support both keys)
    path_config = loco_mujoco.load_path_config()
    path = path_config.get("AMASS_PATH") or path_config.get("MUSCLEMIMIC_AMASS_PATH")
    if path:
        return path

    # 3. Use default
    default = str(_get_default_base_dir() / "AMASS")
    return default


def get_converted_amass_dataset_path():
    """
    Get converted AMASS dataset path from env var, config, or default.

    If the cache doesn't exist locally, will attempt to download demo motions
    from HuggingFace for quick testing.
    """
    # 1. Try environment variable (support both old and new names)
    path = os.environ.get("CONVERTED_AMASS_PATH") or os.environ.get("MUSCLEMIMIC_CONVERTED_AMASS_PATH")
    if path:
        return path

    # 2. Try config file (support both keys)
    path_config = loco_mujoco.load_path_config()
    path = path_config.get("CONVERTED_AMASS_PATH") or path_config.get("MUSCLEMIMIC_CONVERTED_AMASS_PATH")
    if path:
        return path

    # 3. Use default (will auto-download demos if needed)
    default = str(_get_default_base_dir() / "caches" / "AMASS")
    return default


def _get_demo_cache_path():
    """Get the default demo cache path (~/.musclemimic/caches/AMASS)."""
    return str(_get_default_base_dir() / "caches" / "AMASS")


def _resolve_cache_path(cache_path: str) -> str:
    """Resolve a cache path, falling back to the demo cache directory if not found."""
    if os.path.exists(cache_path):
        return cache_path

    demo_base = _get_demo_cache_path()
    configured_base = get_converted_amass_dataset_path()
    if configured_base != demo_base and cache_path.startswith(configured_base):
        demo_path = cache_path.replace(configured_base, demo_base, 1)
        if os.path.exists(demo_path):
            return demo_path
    return cache_path


def _motion_cache_path(cache_root: str, motion_spec: AMASSMotionSpec) -> str:
    """Return a cache path isolated by source motion and clip interval."""
    if motion_spec.is_full_motion:
        relative_path = f"{motion_spec.path}.npz"
    else:
        relative_path = os.path.join(
            "_clips",
            f"{motion_spec.path}__{motion_spec.cache_suffix}.npz",
        )
    return os.path.join(cache_root, relative_path)


def gmr_quality_rejection_path(cache_path: str | Path) -> Path:
    """Return the persistent quality-rejection record for one GMR cache."""
    path = Path(cache_path)
    return path.with_name(f"{path.stem}_quality_rejected.json")


def _clear_gmr_terminal_artifacts(cache_path: str | Path) -> None:
    """Remove one GMR cache, analysis sidecar, and rejection marker together."""
    path = Path(cache_path)
    analysis_path = path.with_name(f"{path.stem}_analysis.npz")
    for artifact_path in (path, analysis_path, gmr_quality_rejection_path(path)):
        artifact_path.unlink(missing_ok=True)


def get_gmr_motion_cache_path(
    env_name: str,
    motion_spec: AMASSMotionSpec,
    cache_root: str | Path | None = None,
    cache_namespace: str | None = None,
    *,
    gmr_config: dict | DictConfig | None = None,
) -> Path:
    """Return the configuration-isolated GMR trajectory path for one motion."""
    if cache_root is None:
        converted_root = Path(get_converted_amass_dataset_path())
    else:
        converted_root = Path(cache_root)
    _identity, cache_variant = get_gmr_cache_identity(gmr_config)
    return converted_root / gmr_motion_relative_path(
        env_name,
        motion_spec,
        cache_namespace or GMR_CACHE_NAMESPACE,
        cache_variant=cache_variant,
    )


def _write_gmr_quality_rejection(
    cache_path: str | Path,
    motion_spec: AMASSMotionSpec,
    error: GMRQualityGateError,
    *,
    gmr_config: dict | DictConfig | None = None,
    cache_namespace: str = GMR_CACHE_NAMESPACE,
) -> Path:
    """Persist one rejection so later dataset loads do not rerun bad motion."""
    rejection_path = gmr_quality_rejection_path(cache_path)
    rejection_path.parent.mkdir(parents=True, exist_ok=True)
    cache_identity, cache_variant = get_gmr_cache_identity(gmr_config)
    payload = {
        "schema_version": 1,
        "cache_namespace": cache_namespace,
        "cache_identity_sha256": cache_identity,
        "cache_variant": cache_variant,
        "cache_key": motion_spec.cache_key,
        "motion_path": motion_spec.path,
        "start_time_s": motion_spec.start_time_s,
        "end_time_s": motion_spec.end_time_s,
        "reason": str(error),
        "metrics": error.metrics,
    }
    rejection_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return rejection_path


def get_smpl_model_path():
    """Get SMPL model path from env var, config, or default"""
    # 1. Try environment variable (support both old and new names)
    path = os.environ.get("SMPL_MODEL_PATH") or os.environ.get("MUSCLEMIMIC_SMPL_MODEL_PATH")
    if path:
        return path

    # 2. Try config file (support both keys)
    path_config = loco_mujoco.load_path_config()
    path = path_config.get("SMPL_MODEL_PATH") or path_config.get("MUSCLEMIMIC_SMPL_MODEL_PATH")
    if path:
        return path

    # 3. Use default
    default = str(_get_default_base_dir() / "smpl")
    return default


def load_amass_data(
    data_path: str,
    motion_spec: AMASSMotionSpec | None = None,
) -> dict:
    """Load AMASS data from a file.

    Args:
        data_path: AMASS-root-relative path to the data file.
        motion_spec: Optional source-time interval to retain.


    Returns:
        dict: Parsed AMASS data including poses, translations, and other attributes.

    """
    path_to_amass_datasets = get_amass_dataset_path()

    # get paths to all amass files
    path_to_all_amass_files = os.path.join(path_to_amass_datasets, "**/*.npz")
    all_pkls = glob.glob(path_to_all_amass_files, recursive=True)

    # get full dataset path
    key_names = [os.path.relpath(data_path, path_to_amass_datasets).replace(".npz", "") for data_path in all_pkls]
    if data_path.startswith("/"):
        data_path = data_path[1:]
    data_path = data_path.replace(".npz", "")
    data_path = all_pkls[key_names.index(data_path)]

    # load data
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if "mocap_framerate" in entry_data:
        framerate = entry_data["mocap_framerate"]
    elif "mocap_frame_rate" in entry_data:
        framerate = entry_data["mocap_frame_rate"]
    else:
        raise ValueError("Framerate not found in the data file.")

    root_trans = entry_data["trans"]
    poses = entry_data["poses"]
    if motion_spec is not None:
        start_frame, end_frame = motion_spec.frame_bounds(framerate, len(root_trans))
        root_trans = root_trans[start_frame:end_frame]
        poses = poses[start_frame:end_frame]
    pose_aa = np.concatenate([poses[:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = entry_data["betas"]
    gender = entry_data["gender"]

    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


def load_robot_conf_file(env_name: str):
    """Load a robot configuration file."""
    if "Mjx" in env_name:
        conf_name = env_name.replace("Mjx", "")
    else:
        conf_name = env_name
    filename = f"{conf_name}.yaml"
    filepath = os.path.join(PATH_TO_SMPL_ROBOT_CONF, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML file '{filename}' not found in path: {PATH_TO_SMPL_ROBOT_CONF}")
    default_conf = OmegaConf.load(PATH_TO_SMPL_ROBOT_CONF / "defaults.yaml")
    robot_conf = OmegaConf.load(filepath)
    robot_conf = OmegaConf.merge(default_conf, robot_conf)
    return robot_conf


def to_t_pose(env, robot_conf):
    """
    Set the humanoid to a T-pose by modifying the Mujoco Data structure.

    Args:
        env: environment.
        robot_conf: robot configuration file including the joint positions for the T-pose.

    """
    data = env._data
    # apply init pose modifiers
    for modifier in robot_conf.robot_pose_modifier:
        name, val = next(iter(modifier.items()))
        if name != "root":
            # convert string to numpy value
            val = np.array(eval(val))
            qpos_id = mj_jntname2qposid(name, env._model)
            data.qpos[qpos_id] += val
        else:
            # convert string to numpy value
            val = sRot.from_euler("xyz", eval(val), degrees=False).as_quat()
            val = quat_scalarlast2scalarfirst(val)
            data.qpos[3:7] += val


def fit_smpl_motion(
    env_name: str,
    robot_conf: DictConfig,
    path_to_smpl_model: str,
    motion_data: str | dict,
    path_to_optimized_smpl_shape: str,
    logger: logging.Logger,
    skip_steps: bool = True,
    visualize: bool = False,
) -> tuple[Trajectory, dict]:
    """Fit SMPL motion data to a robot configuration.

    Args:
        env_name (str): Name of the environment.
        robot_conf (DictConfig): Configuration of the robot.
        path_to_smpl_model (str): Path to the SMPL model.
        motion_data (Dict): Dict containing the motion data to process.
        path_to_optimized_smpl_shape (str): Path to the optimized SMPL shape file for the robot.
        logger (logging.Logger): Logger for status updates.
        visualize (bool): Whether to visualize the optimization process.

    Returns:
        tuple[Trajectory, dict]: The fitted motion trajectory and analysis data
            containing pos_error, retarget_fps, and site_names.

    """

    def get_xpos_and_xquat(smpl_positions, smpl_rot_mats, s2m_pos, s2m_rot_mat):
        # get rotations of mimic sites
        new_smpl_rot_mats = np.einsum("bij,bjk->bik", smpl_rot_mats, s2m_rot_mat)
        new_smpl_quat = sRot.from_matrix(new_smpl_rot_mats).as_quat()
        new_smpl_quat = quat_scalarlast2scalarfirst(new_smpl_quat)
        pos_offset = np.einsum("bij,bj->bi", new_smpl_rot_mats, s2m_pos)
        new_smpl_pos = torch.squeeze(smpl_positions - pos_offset)

        return new_smpl_pos, new_smpl_quat

    check_optional_imports()

    # Normalize environment name for consistent retargeting
    if "Mjx" in env_name:
        env_name = env_name.replace("Mjx", "")

    # get environment
    env_cls = Mujoco.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})

    # add mocap bodies for all 'site_for_mimic' instances of an environment
    mjspec = env.mjspec
    sites_for_mimic = env.sites_for_mimic
    _resolve_site_ids(env._model, sites_for_mimic, context=f"{env_name} sites_for_mimic")
    smpl2mimic_site_idx = _resolve_smpl_joint_indices(
        sites_for_mimic,
        robot_conf,
        context=f"{env_name} retargeting config",
    )
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mjspec = add_mocap_bodies(mjspec, sites_for_mimic, target_mocap_bodies, robot_conf, add_equality_constraint=True)
    env.reload_mujoco(mjspec)
    key = jax.random.key(0)
    env.reset(key)

    smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral")

    shape_new, scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale = joblib.load(
        path_to_optimized_smpl_shape
    )

    skip = robot_conf.optimization_params.skip_frames if skip_steps else 1
    pose_aa = torch.from_numpy(motion_data["pose_aa"][::skip]).float()
    pose_aa = torch.cat([pose_aa, torch.zeros((pose_aa.shape[0], 156 - pose_aa.shape[1]))], axis=-1)
    len_traj = pose_aa.shape[0]

    total_z_offset = offset_z + robot_conf.optimization_params.z_offset_feet
    trans = torch.from_numpy(motion_data["trans"][::skip]) + torch.tensor([0.0, 0.0, total_z_offset])

    # apply height scaling while preserving init height
    trans[:, :2] *= height_scale  # scale x and y
    trans[:, 2] = (trans[:, 2] - trans[0, 2]) * height_scale + trans[0, 2]

    with torch.no_grad():
        transformations_matrices = smpl_parser_n.get_joint_transformations(
            pose_aa.reshape(len_traj, -1, 3), shape_new.repeat(len_traj, 1), trans
        )
        global_pos = transformations_matrices[..., :3, 3]
        global_rot_mats = transformations_matrices[..., :3, :3].detach().numpy()
        root_pos = global_pos[:, 0:1]
        global_pos = (global_pos - global_pos[:, 0:1]) * scale.detach() + root_pos

    # calculate initial qpos from initial mocap pos
    init_mocap_pos, init_mocap_quat = get_xpos_and_xquat(
        global_pos[0, smpl2mimic_site_idx], global_rot_mats[0, smpl2mimic_site_idx], smpl2robot_pos, smpl2robot_rot_mat
    )
    qpos_init = get_init_qpos_for_motion_retargeting(env, init_mocap_pos, init_mocap_quat, robot_conf)
    env._data.qpos = qpos_init

    site_ids = _resolve_site_ids(env._model, sites_for_mimic, context=f"{env_name} retarget sites")
    qpos = np.zeros((len_traj, env._model.nq))
    num_matched = len(smpl2mimic_site_idx)
    site_xpos = np.zeros((len_traj, len(site_ids), 3), dtype=np.float32)
    site_xmat = np.zeros((len_traj, len(site_ids), 9), dtype=np.float32)

    dist_array = np.zeros((len_traj, num_matched))

    # max_pen = 0
    # TODO: currently with per frame offset, test will overall offset if needed for ground penetrations.
    t_start = time.perf_counter()
    for i in tqdm(range(len_traj)):
        mocap_pos, mocap_quat = get_xpos_and_xquat(
            global_pos[i, smpl2mimic_site_idx],
            global_rot_mats[i, smpl2mimic_site_idx],
            smpl2robot_pos,
            smpl2robot_rot_mat,
        )
        env._data.mocap_pos = mocap_pos
        env._data.mocap_quat = mocap_quat

        mujoco.mj_step(env._model, env._data, robot_conf.optimization_params.motion_iterations)

        qpos[i] = env._data.qpos.copy()
        mujoco.mj_forward(env._model, env._data)
        pen, _geom_id, _floor_id = max_penetration_with_floor(env._model, env._data)
        if pen < 0:
            qpos[i][2] -= pen
            env._data.qpos = qpos[i]
            mujoco.mj_forward(env._model, env._data)

        # TODO: test with overall ground penetration
        # max_pen = min(max_pen, pen)

        for k, site_id in enumerate(site_ids):
            current_pos = env._data.site_xpos[site_id]  # real robot site position
            target_pos = mocap_pos[k].cpu().numpy()  # target from SMPL
            dist_array[i, k] = np.linalg.norm(current_pos - target_pos)
            site_xpos[i, k] = current_pos
            site_xmat[i, k] = env._data.site_xmat[site_id]

        if visualize:
            env.render()

    t_total = time.perf_counter() - t_start
    retarget_fps = float(len_traj / max(1e-12, t_total))

    logger.info(f"[OK] SMPL retargeting complete: {len_traj} frames in {t_total:.2f}s ({retarget_fps:.2f} FPS)")

    # TODO: test later for overall ground penetrations
    # if max_pen < 0:
    #    qpos[:, 2] -= max_pen

    fps = _fps_after_frame_skip(motion_data["fps"], skip)

    # Compute qvel from qpos using helper function
    qpos, qvel = _compute_qvel_from_qpos(qpos, fps, env.root_free_joint_xml_name, env._model)
    site_xpos = site_xpos[1:-1]
    site_xmat = site_xmat[1:-1]

    njnt = env._model.njnt
    jnt_type = env._model.jnt_type
    jnt_names = [mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]

    site_ids_array = jnp.array(site_ids)
    traj_model = TrajectoryModel(
        njnt=njnt,
        jnt_type=jnp.array(jnt_type),
        nbody=env._model.nbody,
        body_rootid=jnp.array(env._model.body_rootid),
        body_weldid=jnp.array(env._model.body_weldid),
        body_mocapid=jnp.array(env._model.body_mocapid),
        body_pos=jnp.array(env._model.body_pos),
        body_quat=jnp.array(env._model.body_quat),
        body_ipos=jnp.array(env._model.body_ipos),
        body_iquat=jnp.array(env._model.body_iquat),
        nsite=len(site_ids),
        site_bodyid=jnp.array(env._model.site_bodyid)[site_ids_array],
        site_pos=jnp.array(env._model.site_pos)[site_ids_array],
        site_quat=jnp.array(env._model.site_quat)[site_ids_array],
    )
    traj_info = TrajectoryInfo(
        jnt_names,
        model=traj_model,
        frequency=fps,
        site_names=sites_for_mimic,
    )

    traj_data = TrajectoryData(
        jnp.array(qpos),
        jnp.array(qvel),
        site_xpos=jnp.array(site_xpos),
        site_xmat=jnp.array(site_xmat),
        split_points=jnp.array([0, len(qpos)]),
    )

    analysis = {
        "pos_error": dist_array,
        "retarget_fps": retarget_fps,
        "site_names": sites_for_mimic,
    }

    return Trajectory(traj_info, traj_data), analysis


# ============================================================================
# GMR Fitted Shape Helpers
# ============================================================================


def get_gmr_fitted_shape_dir(env_name: str) -> str:
    """Get the GMR fitted shape directory under the AMASS cache path.

    Args:
        env_name: Environment name (e.g., "MyoFullBody")

    Returns:
        Path to converted gmr dataset
    """
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()
    # Normalize environment name for cache consistency
    cache_env_name = env_name.replace("Mjx", "") if "Mjx" in env_name else env_name
    fitted_dir = Path(path_to_converted_amass_datasets) / cache_env_name / GMR_CACHE_NAMESPACE
    fitted_dir.mkdir(parents=True, exist_ok=True)
    return str(fitted_dir)


def get_gmr_fitted_shape_path(env_name: str, gmr_robot: str) -> str:
    """Get the GMR fitted shape path for a robot.

    Args:
        env_name: Environment name (e.g., "MyoFullBody")
        gmr_robot: GMR robot name (e.g., "myofullbody")

    Returns:
        Path to the fitted shape .pkl file under the active GMR namespace
    """
    fitted_dir = get_gmr_fitted_shape_dir(env_name)
    return str(Path(fitted_dir) / f"{gmr_robot}_shape.pkl")


def ensure_gmr_fitted_shape(
    env_name: str,
    gmr_robot: str,
    robot_xml_path: str,
    ik_config_path: str,
    smpl_model_path: str,
    logger: logging.Logger,
    iterations: int = 500,
) -> str:
    """Ensure fitted shape exists for a robot, running shape fitting if needed.

    Args:
        env_name: Environment name (e.g., "MyoFullBody")
        gmr_robot: GMR robot name (e.g., "myofullbody")
        robot_xml_path: Path to robot XML file
        ik_config_path: Path to IK config JSON file
        smpl_model_path: Path to SMPL-H body models directory
        logger: Logger instance
        iterations: Number of optimization iterations if fitting is needed

    Returns:
        Path to the fitted shape .pkl file

    Raises:
        RuntimeError: If shape fitting fails or torch is not available
    """
    import json

    fitted_path = get_gmr_fitted_shape_path(env_name, gmr_robot)
    metadata_path = fitted_path.replace("_shape.pkl", "_shape_metadata.json")

    # Check if fitted shape already exists
    if os.path.exists(fitted_path) and os.path.exists(metadata_path):
        logger.info(f"Found existing fitted shape: {fitted_path}")
        return fitted_path

    # Need to run shape fitting
    logger.info(f"Fitted shape not found for '{gmr_robot}', running shape fitting...")

    # Load IK config to get joint mappings
    with open(ik_config_path) as f:
        ik_config = json.load(f)

    # Get robot T-pose targets (with rotations for computing alignment offsets)
    robot_targets, robot_joint_names, robot_rotations, human_joint_names = get_robot_tpose_targets(
        str(robot_xml_path), gmr_robot, ik_config, include_rotations=True
    )

    # Get SMPL joint indices from the SMPL joint names returned by get_robot_tpose_targets
    smpl_indices = get_smpl_tpose_indices(robot_joint_names)

    # Load SMPL parser (use GMR version for GMR shape fitting)
    smpl_parser = GMR_SMPLH_Parser(
        model_path=str(smpl_model_path),
        gender="neutral",
        use_pca=False,
    )

    # Run shape fitting (pass rotations for computing rotation offsets)
    logger.info(f"Running SMPL shape fitting ({iterations} iterations)...")
    (shape, scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale, metrics) = fit_smpl_shape_to_robot(
        smpl_parser=smpl_parser,
        target_positions=robot_targets,
        smpl_joint_indices=smpl_indices,
        target_rotations=robot_rotations,
        iterations=iterations,
        lr=0.001,
        device="cpu",
        robot_type=gmr_robot,
    )

    if not metrics["converged"]:
        logger.warning(f"Shape fitting did not converge (loss={float(metrics['final_loss']):.4f}m)")
    else:
        logger.info(f"Shape fitting converged (loss={float(metrics['final_loss']):.4f}m, scale={float(scale):.4f})")

    # Compute local offsets for IK (using rotations for better alignment)
    logger.info("Computing local frame offsets...")
    local_offsets = compute_alignment_offsets(
        smpl_parser=smpl_parser,
        shape=shape,
        scale=scale,
        smpl_joint_indices=smpl_indices,
        human_joint_names=human_joint_names,
        target_positions=robot_targets,
        target_rotations=robot_rotations,
        robot_type=gmr_robot,
    )

    # Save fitted shape
    save_fitted_shape(
        shape=shape,
        scale=scale,
        smpl2robot_pos=smpl2robot_pos,
        smpl2robot_rot_mat=smpl2robot_rot_mat,
        offset_z=offset_z,
        height_scale=height_scale,
        metrics=metrics,
        save_path=fitted_path,
        human_joint_names=human_joint_names,
        local_offsets=local_offsets,
    )

    logger.info(f"Saved fitted shape to: {fitted_path}")
    return fitted_path


def _configure_gmr_equality_tasks(retarget) -> None:
    """Install one strongly weighted task for the robot's joint equalities."""
    model = retarget.configuration.model
    equality_ids = [
        equality_id for equality_id in range(model.neq) if model.eq_type[equality_id] == mujoco.mjtEq.mjEQ_JOINT
    ]

    for task_list_name in ("tasks1", "tasks2"):
        tasks = getattr(retarget, task_list_name)
        frame_tasks = [task for task in tasks if not isinstance(task, EqualityConstraintTask)]
        if equality_ids:
            equality_task = EqualityConstraintTask(
                model,
                cost=GMR_EQUALITY_CONSTRAINT_COST,
                equalities=equality_ids,
            )
            tasks = [equality_task, *frame_tasks]
        else:
            tasks = frame_tasks
        setattr(retarget, task_list_name, tasks)


def _configure_gmr_stage2_position_costs(retarget, overrides: dict[str, float]) -> None:
    """Apply explicit diagnostic or experiment position costs to stage-two frame tasks."""
    if not overrides:
        return
    frame_tasks = {task.frame_name: task for task in retarget.tasks2 if isinstance(task, mink.FrameTask)}
    missing = sorted(set(overrides) - set(frame_tasks))
    if missing:
        raise ValueError(f"Unknown GMR stage-two frame task(s): {', '.join(missing)}")
    for frame_name, position_cost in overrides.items():
        position_cost = float(position_cost)
        if position_cost < 0.0:
            raise ValueError(f"GMR position cost for {frame_name} must be nonnegative, got {position_cost}")
        frame_tasks[frame_name].set_position_cost(position_cost)


def _solve_gmr_initial_frame(
    retarget,
    frame,
    *,
    offset_to_ground: bool,
    minimum_passes: int,
    maximum_passes: int,
    qpos_delta_tolerance: float,
    consecutive_converged_passes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Solve the first target repeatedly and report fixed-point convergence."""
    if minimum_passes < 1:
        raise ValueError(f"GMR minimum initialization passes must be at least 1, got {minimum_passes}")
    if maximum_passes < max(2, minimum_passes):
        raise ValueError(
            "GMR maximum initialization passes must be at least 2 and no smaller "
            f"than the minimum, got minimum={minimum_passes}, maximum={maximum_passes}"
        )
    if qpos_delta_tolerance <= 0.0:
        raise ValueError(f"GMR initialization qpos delta tolerance must be positive, got {qpos_delta_tolerance}")
    if consecutive_converged_passes < 1:
        raise ValueError(
            f"GMR consecutive converged initialization passes must be at least 1, got {consecutive_converged_passes}"
        )

    model = retarget.configuration.model
    previous_qpos = None
    consecutive_converged = 0
    maximum_qpos_delta = float("inf")
    qpos_frame = None
    distances_m = None

    for pass_index in range(maximum_passes):
        qpos_frame, distances_m = retarget.retarget(frame, offset_to_ground=offset_to_ground)
        qpos_frame = np.asarray(qpos_frame).copy()
        if previous_qpos is not None:
            qpos_delta = np.zeros(model.nv, dtype=np.float64)
            mujoco.mj_differentiatePos(model, qpos_delta, 1.0, previous_qpos, qpos_frame)
            maximum_qpos_delta = float(np.max(np.abs(qpos_delta)))
            if maximum_qpos_delta <= qpos_delta_tolerance:
                consecutive_converged += 1
            else:
                consecutive_converged = 0

        passes_used = pass_index + 1
        if passes_used >= minimum_passes and consecutive_converged >= consecutive_converged_passes:
            return (
                qpos_frame,
                np.asarray(distances_m),
                {
                    "initialization_converged": True,
                    "initialization_passes_used": passes_used,
                    "initialization_max_qpos_delta": maximum_qpos_delta,
                    "initialization_qpos_delta_tolerance": qpos_delta_tolerance,
                },
            )
        previous_qpos = qpos_frame

    metrics = {
        "initialization_converged": False,
        "initialization_passes_used": maximum_passes,
        "initialization_max_qpos_delta": maximum_qpos_delta,
        "initialization_qpos_delta_tolerance": qpos_delta_tolerance,
    }
    return qpos_frame, np.asarray(distances_m), metrics


def _refine_bad_gmr_prefix(
    retarget,
    smplh_frames,
    qpos: np.ndarray,
    distances_m: np.ndarray,
    *,
    aligned_fps: float,
    offset_to_ground: bool,
    logger: logging.Logger,
    force_refinement: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve an erroneous leading segment backward from a reliable anchor."""
    mean_errors_m = np.mean(distances_m, axis=1)
    maximum_errors_m = np.max(distances_m, axis=1)
    if (
        not force_refinement
        and mean_errors_m[0] <= GMR_PREFIX_MEAN_ERROR_THRESHOLD_M
        and maximum_errors_m[0] <= GMR_RELIABLE_FRAME_MAX_ERROR_M
    ):
        return qpos, distances_m, False

    stable_frames = max(1, round(GMR_PREFIX_STABLE_DURATION_S * aligned_fps))
    reliable = maximum_errors_m <= GMR_RELIABLE_FRAME_MAX_ERROR_M
    anchor = None
    for start in range(GMR_PREFIX_MIN_ANCHOR_FRAME, len(reliable) - stable_frames + 1):
        if np.all(reliable[start : start + stable_frames]):
            anchor = start
            break
    if anchor is None:
        logger.warning(
            "GMR leading segment requires recovery but no %.2f s reliable anchor exists",
            GMR_PREFIX_STABLE_DURATION_S,
        )
        return qpos, distances_m, False

    refined_qpos = qpos.copy()
    refined_distances_m = distances_m.copy()
    retarget.configuration.update(q=qpos[anchor])
    for frame_index in range(anchor - 1, -1, -1):
        qpos_frame, distance_m = retarget.retarget(
            smplh_frames[frame_index],
            offset_to_ground=offset_to_ground,
        )
        refined_qpos[frame_index] = qpos_frame
        refined_distances_m[frame_index] = distance_m

    original_prefix_error_m = float(np.mean(mean_errors_m[:anchor]))
    refined_prefix_error_m = float(np.mean(refined_distances_m[:anchor]))
    if refined_distances_m[0].mean() >= distances_m[0].mean() or refined_prefix_error_m >= original_prefix_error_m:
        logger.warning(
            "GMR backward prefix solve did not improve tracking error (original=%.4f m, refined=%.4f m)",
            original_prefix_error_m,
            refined_prefix_error_m,
        )
        return qpos, distances_m, False

    logger.info(
        "Refined %d leading GMR frames from anchor %.2f s (prefix error %.4f -> %.4f m)",
        anchor,
        anchor / aligned_fps,
        original_prefix_error_m,
        refined_prefix_error_m,
    )
    return refined_qpos, refined_distances_m, True


def _validate_gmr_tracking_quality(distances_m: np.ndarray) -> None:
    """Reject retained trajectories whose positional tracking remains invalid."""
    distances = np.asarray(distances_m, dtype=np.float64)
    if distances.ndim != 2 or len(distances) < 3 or not np.all(np.isfinite(distances)):
        raise ValueError("GMR tracking distances must have shape [num_frames >= 3, num_targets].")
    retained_distances_m = distances[1:-1]
    first_frame_mean_error_m = float(np.mean(retained_distances_m[0]))
    trajectory_mean_error_m = float(np.mean(retained_distances_m))
    metrics = {
        "first_retained_frame_mean_position_error_m": first_frame_mean_error_m,
        "retained_trajectory_mean_position_error_m": trajectory_mean_error_m,
        "max_first_retained_frame_mean_position_error_m": GMR_PREFIX_MEAN_ERROR_THRESHOLD_M,
        "max_retained_trajectory_mean_position_error_m": GMR_TRAJECTORY_MEAN_ERROR_THRESHOLD_M,
    }
    failed_metrics = []
    if first_frame_mean_error_m > GMR_PREFIX_MEAN_ERROR_THRESHOLD_M:
        failed_metrics.append(f"first retained-frame mean={first_frame_mean_error_m:.4f} m")
    if trajectory_mean_error_m > GMR_TRAJECTORY_MEAN_ERROR_THRESHOLD_M:
        failed_metrics.append(f"retained trajectory mean={trajectory_mean_error_m:.4f} m")
    if failed_metrics:
        thresholds = (
            f"first retained-frame mean<={GMR_PREFIX_MEAN_ERROR_THRESHOLD_M:.3f} m, "
            f"retained trajectory mean<={GMR_TRAJECTORY_MEAN_ERROR_THRESHOLD_M:.3f} m"
        )
        raise GMRQualityGateError(
            f"GMR tracking quality check failed: {', '.join(failed_metrics)}; required {thresholds}",
            metrics,
        )


def _validate_gmr_floor_contact_quality(
    surface_distances_m: np.ndarray,
    source_contact_frame_clearance_m: np.ndarray,
    *,
    source_contact_frame_fraction: float,
    floating_clearance_threshold_m: float,
    max_floating_frame_fraction: float,
) -> dict[str, float]:
    """Reject motions whose inferred source contacts remain visibly airborne.

    Args:
        surface_distances_m: Lowest robot collision-surface distances shaped
            ``[num_frames]`` [m] after root-z correction.
        source_contact_frame_clearance_m: Highest expected-support clearance
            per frame [m]. NaN marks frames without inferred source contact.
        source_contact_frame_fraction: Fraction of frames with source contact.
        floating_clearance_threshold_m: An expected source contact is missing
            when robot support clearance exceeds this value [m].
        max_floating_frame_fraction: Largest accepted mismatch fraction among
            source-contact frames.

    Returns:
        Scalar metrics recorded in accepted-cache analysis sidecars.

    Raises:
        GMRQualityGateError: If the floating-frame fraction exceeds the limit.
    """
    if floating_clearance_threshold_m < 0.0:
        raise ValueError("GMR floating clearance threshold must be nonnegative.")
    if not 0.0 <= max_floating_frame_fraction <= 1.0:
        raise ValueError("GMR maximum floating-frame fraction must be in [0, 1].")

    surface_distances = np.asarray(surface_distances_m, dtype=np.float64)
    contact_clearance = np.asarray(source_contact_frame_clearance_m, dtype=np.float64)
    if surface_distances.ndim != 1 or surface_distances.size == 0:
        raise ValueError("GMR floor surface distances must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(surface_distances)):
        raise ValueError("GMR floor surface distances must be finite.")
    if contact_clearance.shape != surface_distances.shape:
        raise ValueError("GMR source-contact clearance must match floor surface distances.")
    if np.any(np.isinf(contact_clearance)):
        raise ValueError("GMR source-contact clearance must not contain infinite values.")
    if not 0.0 <= source_contact_frame_fraction <= 1.0:
        raise ValueError("GMR source-contact frame fraction must be in [0, 1].")

    source_contact = np.isfinite(contact_clearance)
    contact_mismatch = contact_clearance[source_contact] > (
        floating_clearance_threshold_m + GMR_FLOOR_DISTANCE_NUMERICAL_TOLERANCE_M
    )
    floating_frame_fraction = float(np.mean(contact_mismatch)) if np.any(source_contact) else 0.0
    global_floating_frame_fraction = float(
        np.mean(surface_distances > floating_clearance_threshold_m + GMR_FLOOR_DISTANCE_NUMERICAL_TOLERANCE_M)
    )
    metrics = {
        "floating_clearance_threshold_m": float(floating_clearance_threshold_m),
        "max_floating_frame_fraction": float(max_floating_frame_fraction),
        "floating_frame_fraction": floating_frame_fraction,
        "source_contact_mismatch_frame_fraction": floating_frame_fraction,
        "source_contact_frame_fraction": float(source_contact_frame_fraction),
        "global_floating_frame_fraction": global_floating_frame_fraction,
        "maximum_surface_clearance_m": float(np.max(surface_distances)),
        "median_surface_clearance_m": float(np.median(surface_distances)),
        "maximum_source_contact_clearance_m": (
            float(np.max(contact_clearance[source_contact])) if np.any(source_contact) else 0.0
        ),
    }
    if floating_frame_fraction > max_floating_frame_fraction:
        raise GMRQualityGateError(
            "GMR floor contact quality check failed: "
            f"source-contact mismatch fraction={floating_frame_fraction:.6f} exceeds "
            f"{max_floating_frame_fraction:.6f}; expected robot support surface "
            f"must be <= {floating_clearance_threshold_m:.4f} m",
            metrics,
        )
    return metrics


def _validate_gmr_retained_continuity(
    qvel: np.ndarray,
    frequency_hz: float,
) -> dict[str, float | int]:
    """Reject initialization-induced root motion in the retained trajectory."""
    velocity = np.asarray(qvel, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] < 6 or len(velocity) == 0:
        raise ValueError("GMR qvel must have shape [num_frames, model.nv >= 6].")
    if not np.all(np.isfinite(velocity)):
        raise GMRQualityGateError(
            "GMR continuity quality check failed: qvel contains nonfinite values",
            {"qvel_is_finite": False},
        )
    if frequency_hz <= 0.0:
        raise ValueError("GMR trajectory frequency must be positive.")

    leading_frames = min(
        len(velocity),
        max(1, int(np.ceil(GMR_LEADING_CONTINUITY_DURATION_S * frequency_hz))),
    )
    leading_velocity = velocity[:leading_frames]
    root_linear_speed_m_s = np.linalg.norm(leading_velocity[:, :3], axis=1)
    root_angular_speed_rad_s = np.linalg.norm(leading_velocity[:, 3:6], axis=1)
    maximum_joint_speed_rad_s = float(np.max(np.abs(leading_velocity[:, 6:]))) if velocity.shape[1] > 6 else 0.0
    metrics: dict[str, float | int] = {
        "leading_continuity_duration_s": GMR_LEADING_CONTINUITY_DURATION_S,
        "leading_continuity_frames": leading_frames,
        "leading_maximum_root_linear_speed_m_s": float(np.max(root_linear_speed_m_s)),
        "leading_maximum_root_angular_speed_rad_s": float(np.max(root_angular_speed_rad_s)),
        "leading_maximum_joint_speed_rad_s": maximum_joint_speed_rad_s,
        "maximum_leading_root_linear_speed_m_s": GMR_MAX_LEADING_ROOT_LINEAR_SPEED_M_S,
        "maximum_leading_root_angular_speed_rad_s": GMR_MAX_LEADING_ROOT_ANGULAR_SPEED_RAD_S,
    }
    failed_metrics = []
    if metrics["leading_maximum_root_linear_speed_m_s"] > GMR_MAX_LEADING_ROOT_LINEAR_SPEED_M_S:
        failed_metrics.append(f"root linear speed={metrics['leading_maximum_root_linear_speed_m_s']:.4f} m/s")
    if metrics["leading_maximum_root_angular_speed_rad_s"] > GMR_MAX_LEADING_ROOT_ANGULAR_SPEED_RAD_S:
        failed_metrics.append(f"root angular speed={metrics['leading_maximum_root_angular_speed_rad_s']:.4f} rad/s")
    if failed_metrics:
        raise GMRQualityGateError(
            "GMR continuity quality check failed in the leading retained 0.1 s: " + ", ".join(failed_metrics),
            metrics,
        )
    return metrics


def _guard_gmr_resampled_floor_clearance(
    trajectory: Trajectory,
    model: mujoco.MjModel,
    root_free_joint_name: str,
    *,
    excluded_body_subtree_names: tuple[str, ...] = MYOFULLBODY_GMR_FLOOR_EXCLUDED_BODY_SUBTREES,
) -> tuple[Trajectory, dict[str, float]]:
    """Remove interpolation-induced penetration with one rigid world-z lift."""
    evaluator = FloorSurfaceDistanceEvaluator(
        model,
        excluded_body_subtree_names=excluded_body_subtree_names,
    )
    qpos = np.asarray(trajectory.data.qpos)
    surface_distances_m, _closest_geom_ids = evaluator.measure_trajectory(qpos)
    guarded_qpos, world_z_translation_m = evaluator.ensure_minimum_trajectory_clearance(
        qpos,
        root_free_joint_name,
        GMR_OUTPUT_FLOOR_CLEARANCE_MARGIN_M,
        surface_distances_m=surface_distances_m,
    )
    if isinstance(trajectory.data.qpos, jax.Array):
        stored_qpos = jnp.asarray(guarded_qpos, dtype=trajectory.data.qpos.dtype)
    else:
        stored_qpos = np.asarray(guarded_qpos, dtype=trajectory.data.qpos.dtype)
    guarded_data = trajectory.data.replace(qpos=stored_qpos)
    guarded_trajectory = Trajectory(
        info=trajectory.info,
        data=guarded_data,
        transitions=trajectory.transitions,
        obs_container=trajectory.obs_container,
    )
    minimum_before_m = float(np.min(surface_distances_m))
    return guarded_trajectory, {
        "output_floor_z_translation_m": world_z_translation_m,
        "output_minimum_surface_clearance_before_m": minimum_before_m,
        "output_minimum_surface_clearance_after_m": minimum_before_m + world_z_translation_m,
    }


def _update_gmr_analysis_for_output_floor_guard(
    analysis: dict,
    output_floor_metrics: dict[str, float],
) -> None:
    """Keep native-grid diagnostics consistent with the stored 100 Hz output."""
    translation_m = output_floor_metrics["output_floor_z_translation_m"]
    analysis["ground_alignment_method"] = "contact_aware_root_z_with_resampled_floor_guard"
    analysis["root_z_correction_m"] = np.asarray(analysis["root_z_correction_m"]) + translation_m
    for key in (
        "root_z_correction_minimum_m",
        "root_z_correction_median_m",
        "root_z_correction_maximum_m",
    ):
        analysis[key] = float(analysis[key]) + translation_m
    analysis["floor_surface_clearance_m"] = np.asarray(analysis["floor_surface_clearance_m"]) + translation_m
    contact_clearance_m = np.asarray(analysis["source_contact_frame_clearance_m"]).copy()
    finite_contact = np.isfinite(contact_clearance_m)
    contact_clearance_m[finite_contact] += translation_m
    analysis["source_contact_frame_clearance_m"] = contact_clearance_m
    for key in ("tracking_target_positions_m", "tracking_current_positions_m"):
        if key in analysis:
            positions_m = np.asarray(analysis[key]).copy()
            positions_m[..., 2] += translation_m
            analysis[key] = positions_m

    floor_contact_metrics = _validate_gmr_floor_contact_quality(
        analysis["floor_surface_clearance_m"],
        contact_clearance_m,
        source_contact_frame_fraction=float(analysis["source_contact_frame_fraction"]),
        floating_clearance_threshold_m=float(analysis["floating_clearance_threshold_m"]),
        max_floating_frame_fraction=float(analysis["max_floating_frame_fraction"]),
    )
    analysis.update(floor_contact_metrics)
    analysis.update(output_floor_metrics)


def fit_gmr_motion(
    env_name: str,
    robot_conf: DictConfig,
    motion_data: str | dict,
    logger: logging.Logger,
    gmr_config: dict | None = None,
    motion_spec: AMASSMotionSpec | None = None,
) -> tuple[Trajectory, dict]:
    """
    Fit SMPL-H motion data to a robot using General Motion Retargeting (GMR).

    An alternative to fit_smpl_motion() using GMR's IK-based retargeting.

    Args:
        env_name: Environment name
        robot_conf: Robot configuration
        motion_data: AMASS/SMPL-H file path
        logger: Logger instance
        gmr_config: GMR configuration dict (optional)
            offset_to_ground: If True, applies GMR's per-frame source-joint
                grounding. It defaults to False so vertical motion is preserved.
            initialization_passes: Minimum number of IK solves applied to the
                first frame before testing convergence.
            initialization_max_passes: Maximum first-frame IK solves before the
                forward pass and optional backward recovery continue.
            initialization_qpos_delta_tolerance: Maximum MuJoCo tangent-space
                change between consecutive first-frame solutions [m or rad].
            initialization_consecutive_converged_passes: Number of consecutive
                changes below the tolerance required for convergence.
            validate_tracking_quality: Reject motion caches that retain large
                positional errors after refinement. It defaults to True.
            record_tracking_positions: Include the GMR target and achieved body
                positions in the returned analysis payload. This is intended for
                diagnostic rendering and defaults to False.
            stage2_position_cost_overrides: Optional mapping from robot body name
                to stage-two position cost. This supports controlled retargeting
                ablations without editing the shared IK JSON.
            floating_clearance_threshold_m: A frame is considered floating when
                an inferred source support maps to a robot collision surface
                above this height [m].
            max_floating_frame_fraction: Reject a motion when its source-contact
                mismatch fraction is strictly greater than this value.
        motion_spec: Optional source-time interval to retain.

    Returns:
        tuple[Trajectory, dict]: Trajectory at GMR's native fps (~30Hz) and analysis data
            containing pos_error and retarget_fps. Use extend_motion() to get model-aligned source data.

    The retargeted output receives a smooth per-frame root-z correction. Source
    support contacts anchor the corresponding robot surfaces to the floor,
    temporal regularization preserves vertical motion, and signed collision
    distance imposes a no-penetration bound.
    """
    if not _GMR_INSTALLED:
        raise ImportError(
            "GMR (general_motion_retargeting) required. "
            "Please use uv sync --extra gmr to install the extra dependencies."
        )

    if not isinstance(motion_data, str):
        raise ValueError("GMR retargeting requires AMASS file path (str)")

    # GMR configuration
    gmr_config = gmr_config or {}
    target_fps = gmr_config.get("target_fps", 30)
    solver = gmr_config.get("solver", "daqp")
    damping = gmr_config.get("damping", 0.5)
    offset_to_ground = gmr_config.get("offset_to_ground", False)
    use_velocity_limit = gmr_config.get("use_velocity_limit", False)
    verbose = gmr_config.get("verbose", False)
    use_fitted_shape = gmr_config.get("use_fitted_shape", True)
    shape_fitting_iterations = gmr_config.get("shape_fitting_iterations", 500)

    initialization_passes = int(gmr_config.get("initialization_passes", GMR_INITIALIZATION_PASSES))
    initialization_max_passes = int(gmr_config.get("initialization_max_passes", GMR_INITIALIZATION_MAX_PASSES))
    initialization_qpos_delta_tolerance = float(
        gmr_config.get("initialization_qpos_delta_tolerance", GMR_INITIALIZATION_QPOS_DELTA_TOLERANCE)
    )
    initialization_consecutive_converged_passes = int(
        gmr_config.get(
            "initialization_consecutive_converged_passes",
            GMR_INITIALIZATION_CONSECUTIVE_CONVERGED_PASSES,
        )
    )
    refine_bad_prefix = bool(gmr_config.get("refine_bad_prefix", True))
    validate_tracking_quality = bool(gmr_config.get("validate_tracking_quality", True))
    record_tracking_positions = bool(gmr_config.get("record_tracking_positions", False))
    stage2_position_cost_overrides = dict(gmr_config.get("stage2_position_cost_overrides", {}))
    floating_clearance_threshold_m = float(
        gmr_config.get("floating_clearance_threshold_m", GMR_FLOATING_CLEARANCE_THRESHOLD_M)
    )
    max_floating_frame_fraction = float(gmr_config.get("max_floating_frame_fraction", GMR_MAX_FLOATING_FRAME_FRACTION))
    # Map environment to GMR robot name
    env_to_gmr_robot = {
        "MyoFullBody": "myofullbody",
        "MjxMyoFullBody": "myofullbody",
    }
    base_env_name = env_name.replace("Mjx", "") if "Mjx" in env_name else env_name
    if base_env_name not in env_to_gmr_robot:
        raise ValueError(f"GMR not configured for '{env_name}'. Supported: {list(env_to_gmr_robot.keys())}")
    gmr_robot = env_to_gmr_robot[base_env_name]

    # Get smpl model path
    smpl_model_path = get_smpl_model_path()
    if not os.path.exists(smpl_model_path):
        raise FileNotFoundError(
            f"SMPL-H models not found at {smpl_model_path}\nPlease download SMPL-H models to this location."
        )

    # Setup robot-specific paths and fitted shape BEFORE loading SMPL-H
    # (fitted_shape_path is used by load_smplh_file to apply fitted offsets)
    myofullbody_env = None
    fitted_shape_path = None

    if gmr_robot == "myofullbody":
        myofullbody_xml = get_xml_path("myofullbody")

        if not myofullbody_xml.exists():
            raise FileNotFoundError(f"MyoFullBody XML not found: {myofullbody_xml}")

        src_human_key = "smplh"
        myofullbody_ik_config = Path(__file__).parent / "gmr_configs" / "smplh_to_myofullbody.json"
        if not myofullbody_ik_config.exists():
            raise FileNotFoundError(f"MyoFullBody IK config not found: {myofullbody_ik_config}")

        # Inject into GMR's dictionaries
        ROBOT_XML_DICT["myofullbody"] = myofullbody_xml
        IK_CONFIG_DICT.setdefault(src_human_key, {})["myofullbody"] = myofullbody_ik_config
        ROBOT_BASE_DICT["myofullbody"] = "pelvis"
        VIEWER_CAM_DISTANCE_DICT["myofullbody"] = 3.0

        # Create MyoFullBody env to get model with fingers disabled
        from musclemimic.environments.humanoids.myofullbody import MyoFullBody

        myofullbody_env = MyoFullBody(**robot_conf.env_params)
        logger.info(f"Created MyoFullBody env (nq={myofullbody_env._model.nq}, fingers disabled)")

        # Auto-ensure fitted shape exists (if use_fitted_shape is enabled)
        if use_fitted_shape:
            fitted_shape_path = ensure_gmr_fitted_shape(
                env_name=env_name,
                gmr_robot=gmr_robot,
                robot_xml_path=str(myofullbody_xml),
                ik_config_path=str(myofullbody_ik_config),
                smpl_model_path=smpl_model_path,
                logger=logger,
                iterations=shape_fitting_iterations,
            )
    else:
        src_human_key = "smplh"
        myofullbody_ik_config = IK_CONFIG_DICT[src_human_key][gmr_robot]

    # Load source human motion (with fitted shape if available)
    logger.info(f"Loading {src_human_key.upper()} motion from {motion_data}")
    smplh_data, body_model, smplh_output, actual_human_height = load_smplh_file(
        motion_data, smpl_model_path, fitted_shape_path=fitted_shape_path
    )
    frame_extractor = get_smplh_data_offline_fast
    logger.info(f"Aligning to {target_fps} fps...")
    smplh_frames, aligned_fps = frame_extractor(smplh_data, body_model, smplh_output, tgt_fps=target_fps)
    source_frames = smplh_frames
    if motion_spec is not None:
        start_frame, end_frame = motion_spec.frame_bounds(aligned_fps, len(source_frames))
        source_frames = source_frames[start_frame:end_frame]
    logger.info(f"Aligned: {aligned_fps:.2f} fps, {len(source_frames)} frames")

    # Initialize GMR
    logger.info(f"GMR init: robot={gmr_robot}, solver={solver}, use_fitted_shape={use_fitted_shape}")
    retarget = GeneralMotionRetargeting(
        actual_human_height=actual_human_height,
        src_human=src_human_key,
        tgt_robot=gmr_robot,
        solver=solver,
        damping=damping,
        use_velocity_limit=use_velocity_limit,
        verbose=verbose,
        use_fitted_shape=use_fitted_shape,
        fitted_shape_path=fitted_shape_path,
        ik_config_path=str(myofullbody_ik_config),
    )
    _configure_gmr_equality_tasks(retarget)
    _configure_gmr_stage2_position_costs(retarget, stage2_position_cost_overrides)

    # Replace GMR's model with the finger-disabled version
    if gmr_robot == "myofullbody" and myofullbody_env is not None:
        import mink

        logger.info(f"Replacing GMR model (nq: {retarget.model.nq} -> {myofullbody_env._model.nq})")
        retarget.model = myofullbody_env._model
        # Recreate configuration and data with new model
        retarget.configuration = mink.Configuration(retarget.model)
        # Rebuild dof/body/motor name dictionaries
        retarget.robot_dof_names = {}
        for i in range(retarget.model.nv):
            dof_name = mujoco.mj_id2name(retarget.model, mujoco.mjtObj.mjOBJ_JOINT, retarget.model.dof_jntid[i])
            retarget.robot_dof_names[dof_name] = i
        retarget.robot_body_names = {}
        for i in range(retarget.model.nbody):
            body_name = mujoco.mj_id2name(retarget.model, mujoco.mjtObj.mjOBJ_BODY, i)
            retarget.robot_body_names[body_name] = i
        retarget.robot_motor_names = {}
        for i in range(retarget.model.nu):
            motor_name = mujoco.mj_id2name(retarget.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            retarget.robot_motor_names[motor_name] = i
        # Rebuild tasks and limits with new model
        retarget.setup_retarget_configuration()
        _configure_gmr_equality_tasks(retarget)
        _configure_gmr_stage2_position_costs(retarget, stage2_position_cost_overrides)
        retarget.ik_limits = [mink.ConfigurationLimit(retarget.model)]
        if use_velocity_limit:
            velocity_limits = dict.fromkeys(retarget.robot_motor_names.keys(), 3 * np.pi)
            retarget.ik_limits.append(mink.VelocityLimit(retarget.model, velocity_limits))
        logger.info("Model replaced and GMR reconfigured")

    if gmr_robot == "myofullbody":
        model = retarget.configuration.model

        # Initialize limited joints with invalid qpos0 to their range midpoints
        fixed_count = 0
        qpos = retarget.configuration.data.qpos.copy()
        for i in range(model.njnt):
            if model.jnt_limited[i]:
                qpos_adr = model.jnt_qposadr[i]
                jnt_min, jnt_max = model.jnt_range[i]
                if not (jnt_min <= qpos[qpos_adr] <= jnt_max):
                    mid_val = (jnt_min + jnt_max) / 2
                    qpos[qpos_adr] = mid_val
                    fixed_count += 1
        if fixed_count > 0:
            logger.info(f"Initialized {fixed_count} joints to range midpoints")
            # Update configuration with new qpos
            retarget.configuration.update(q=qpos)

    # Get environment for model structure
    if myofullbody_env is not None:
        # Reuse the MyoFullBody env we created earlier
        env = myofullbody_env
    elif env_name in ["MyoFullBody", "MjxMyoFullBody"]:
        # MuscleMimic environments - import directly
        if env_name == "MyoFullBody":
            from musclemimic.environments.humanoids.myofullbody import MyoFullBody

            env = MyoFullBody(**robot_conf.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})
        else:  # MjxMyoFullBody
            from musclemimic.environments.humanoids.myofullbody import MjxMyoFullBody

            env = MjxMyoFullBody(**robot_conf.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})
    else:
        # LocoMuJoCo environments
        env_cls = Mujoco.registered_envs[env_name]
        env = env_cls(**robot_conf.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})

    # Retarget frames
    logger.info("Running GMR retargeting...")

    qpos_list = []
    dist_list = []
    tracking_target_positions_m = []
    tracking_frame_names = []
    initialization_metrics = {}
    model = retarget.configuration.model
    data = retarget.configuration.data
    traj_site_names = getattr(env, "sites_for_mimic", None)
    traj_site_ids = None
    if traj_site_names is not None:
        traj_site_ids = _resolve_site_ids(model, traj_site_names, context=f"{env_name} GMR trajectory sites")

    t_start = time.perf_counter()

    for i, frame in enumerate(source_frames):
        if i % 30 == 0 and i > 0:
            logger.info(f"  {i}/{len(source_frames)} frames")
        if i == 0:
            qpos_frame, dist, initialization_metrics = _solve_gmr_initial_frame(
                retarget,
                frame,
                offset_to_ground=offset_to_ground,
                minimum_passes=initialization_passes,
                maximum_passes=initialization_max_passes,
                qpos_delta_tolerance=initialization_qpos_delta_tolerance,
                consecutive_converged_passes=initialization_consecutive_converged_passes,
            )
            if initialization_metrics["initialization_converged"]:
                logger.info(
                    "GMR first frame converged after %d passes (maximum qpos delta %.6g)",
                    initialization_metrics["initialization_passes_used"],
                    initialization_metrics["initialization_max_qpos_delta"],
                )
            else:
                logger.warning(
                    "GMR first frame did not converge after %d passes (maximum qpos delta %.6g); "
                    "continuing forward to find a reliable backward-recovery anchor",
                    initialization_metrics["initialization_passes_used"],
                    initialization_metrics["initialization_max_qpos_delta"],
                )
        else:
            qpos_frame, dist = retarget.retarget(frame, offset_to_ground=offset_to_ground)

        frame_tasks = [task for task in retarget.tasks2 if isinstance(task, mink.FrameTask)]
        if not tracking_frame_names:
            tracking_frame_names = [task.frame_name for task in frame_tasks]
        tracking_target_positions_m.append(
            np.stack([np.asarray(task.transform_target_to_world.translation()) for task in frame_tasks])
        )

        qpos_list.append(qpos_frame.copy())
        dist_list.append(dist.copy())

    t_total = time.perf_counter() - t_start
    retarget_fps = len(qpos_list) / t_total if t_total > 0 else float("inf")

    logger.info(
        f"[OK] GMR retargeting complete: "
        f"{len(qpos_list)} frames in {t_total:.2f}s "
        f"({retarget_fps:.2f} FPS, {1000 / retarget_fps:.2f} ms/frame)"
    )

    qpos = np.asarray(qpos_list)
    dist_array = np.asarray(dist_list)
    prefix_refined = False
    if refine_bad_prefix:
        qpos, dist_array, prefix_refined = _refine_bad_gmr_prefix(
            retarget,
            source_frames,
            qpos,
            dist_array,
            aligned_fps=aligned_fps,
            offset_to_ground=offset_to_ground,
            logger=logger,
            force_refinement=not bool(initialization_metrics["initialization_converged"]),
        )
    if not bool(initialization_metrics["initialization_converged"]) and not prefix_refined:
        logger.warning(
            "GMR first-frame fixed point remained unconverged and backward recovery did not improve "
            "tracking; retaining the forward candidate for output-quality validation"
        )
    if validate_tracking_quality:
        _validate_gmr_tracking_quality(dist_array)

    tracking_target_positions_array = np.asarray(tracking_target_positions_m, dtype=np.float64)
    grounding = ground_contact_aware_trajectory(
        model,
        qpos,
        env.root_free_joint_xml_name,
        tracking_target_positions_array,
        tracking_frame_names,
        aligned_fps,
        excluded_body_subtree_names=(
            MYOFULLBODY_GMR_FLOOR_EXCLUDED_BODY_SUBTREES if gmr_robot == "myofullbody" else ()
        ),
    )
    qpos = grounding.qpos
    logger.info(
        "Applied contact-aware root-z correction: min %.6f m, median %.6f m, max %.6f m",
        float(np.min(grounding.root_z_correction_m)),
        float(np.median(grounding.root_z_correction_m)),
        float(np.max(grounding.root_z_correction_m)),
    )
    floor_contact_metrics = _validate_gmr_floor_contact_quality(
        grounding.global_surface_clearance_m[1:-1],
        grounding.source_contact_frame_clearance_m[1:-1],
        source_contact_frame_fraction=float(np.mean(np.isfinite(grounding.source_contact_frame_clearance_m[1:-1]))),
        floating_clearance_threshold_m=floating_clearance_threshold_m,
        max_floating_frame_fraction=max_floating_frame_fraction,
    )
    logger.info(
        "GMR floor contact quality: source-contact mismatch fraction %.4f (limit %.4f at %.3f m)",
        floor_contact_metrics["floating_frame_fraction"],
        max_floating_frame_fraction,
        floating_clearance_threshold_m,
    )

    logger.info(f"Complete: {qpos.shape}")

    if traj_site_ids is not None:
        site_xpos_full, site_xmat_full = _record_site_kinematics(model, data, qpos, traj_site_ids)
    else:
        site_xpos_full = np.zeros((len(qpos), 0, 3), dtype=np.float32)
        site_xmat_full = np.zeros((len(qpos), 0, 9), dtype=np.float32)

    # Compute velocities
    qpos, qvel = _compute_qvel_from_qpos(qpos, aligned_fps, env.root_free_joint_xml_name, model)
    site_xpos = site_xpos_full[1:-1]
    site_xmat = site_xmat_full[1:-1]

    # Build Trajectory (minimal - extend_motion adds xpos/xquat/sites)
    njnt = model.njnt
    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]

    traj_site_names = getattr(env, "sites_for_mimic", None)
    if traj_site_names is not None:
        traj_site_ids = _resolve_site_ids(model, traj_site_names, context=f"{env_name} trajectory model sites")
        traj_site_ids_array = jnp.array(traj_site_ids)
        traj_model = TrajectoryModel(
            njnt=njnt,
            jnt_type=jnp.array(model.jnt_type),
            nbody=model.nbody,
            body_rootid=jnp.array(model.body_rootid),
            body_weldid=jnp.array(model.body_weldid),
            body_mocapid=jnp.array(model.body_mocapid),
            body_pos=jnp.array(model.body_pos),
            body_quat=jnp.array(model.body_quat),
            body_ipos=jnp.array(model.body_ipos),
            body_iquat=jnp.array(model.body_iquat),
            nsite=len(traj_site_ids),
            site_bodyid=jnp.array(model.site_bodyid)[traj_site_ids_array],
            site_pos=jnp.array(model.site_pos)[traj_site_ids_array],
            site_quat=jnp.array(model.site_quat)[traj_site_ids_array],
        )
    else:
        traj_model = TrajectoryModel(njnt, jnp.array(model.jnt_type))

    traj_info = TrajectoryInfo(
        jnt_names,
        model=traj_model,
        frequency=aligned_fps,
        site_names=traj_site_names,
    )

    traj_data = TrajectoryData(
        jnp.array(qpos),
        jnp.array(qvel),
        site_xpos=jnp.array(site_xpos),
        site_xmat=jnp.array(site_xmat),
        split_points=jnp.array([0, len(qpos)]),
    )

    analysis = {
        "gmr_cache_identity_sha256": get_gmr_cache_identity(gmr_config)[0],
        "pos_error": dist_array[1:-1],
        "retarget_fps": retarget_fps,
        "site_names": getattr(env, "sites_for_mimic", None),
        "prefix_refined": prefix_refined,
        "first_frame_mean_position_error_m": float(np.mean(dist_array[1])),
        "trajectory_mean_position_error_m": float(np.mean(dist_array[1:-1])),
        "ground_alignment_method": "contact_aware_root_z",
        "root_z_correction_m": grounding.root_z_correction_m[1:-1],
        "root_z_correction_minimum_m": float(np.min(grounding.root_z_correction_m)),
        "root_z_correction_median_m": float(np.median(grounding.root_z_correction_m)),
        "root_z_correction_maximum_m": float(np.max(grounding.root_z_correction_m)),
        "root_z_solver_cost": grounding.solver_cost,
        "root_z_solver_optimality": grounding.solver_optimality,
        "root_z_solver_iterations": grounding.solver_iterations,
        "floor_surface_clearance_m": grounding.global_surface_clearance_m[1:-1],
        "source_contact_frame_clearance_m": grounding.source_contact_frame_clearance_m[1:-1],
        **floor_contact_metrics,
        **initialization_metrics,
    }
    if record_tracking_positions:
        tracking_target_positions_array[..., 2] += grounding.root_z_correction_m[:, None]
        tracking_target_positions_array = tracking_target_positions_array[1:-1]
        tracking_body_ids = np.asarray(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in tracking_frame_names]
        )
        if np.any(tracking_body_ids < 0):
            missing_names = [
                name for name, body_id in zip(tracking_frame_names, tracking_body_ids, strict=True) if body_id < 0
            ]
            raise ValueError(f"GMR diagnostic tracking frames are missing from the model: {', '.join(missing_names)}")

        tracking_current_positions_m = np.zeros_like(tracking_target_positions_array)
        for frame_index, qpos_frame in enumerate(qpos):
            data.qpos[:] = qpos_frame
            mujoco.mj_forward(model, data)
            tracking_current_positions_m[frame_index] = data.xpos[tracking_body_ids]

        analysis.update(
            {
                "tracking_frame_names": np.asarray(tracking_frame_names),
                "tracking_target_positions_m": tracking_target_positions_array,
                "tracking_current_positions_m": tracking_current_positions_m,
            }
        )

    return Trajectory(traj_info, traj_data), analysis


def get_init_qpos_for_motion_retargeting(env, init_mocap_pos, init_mocap_quat, robot_conf):
    """
    Get the initial qpos for motion retargeting by temporarily disabling joint limits and collisions and
    running the simulation to solve for the initial qpos. This avoids problems that could arise from bad initialization
    from the default qpos (getting stuck in joint limits or collisions).

    Args:
        env: environment.
        init_mocap_pos: initial mocap positions.
        init_mocap_quat: initial mocap quaternions.

    Returns:
        np.ndarray: initial qpos.

    """

    old_mjspec = env.mjspec.copy()
    new_mjspec = env.mjspec

    # disable joint limits and collisions
    if robot_conf.optimization_params.disable_joint_limits_on_initialization:
        for j in new_mjspec.joints:
            j.limited = False
    if robot_conf.optimization_params.disable_collisions_on_initialization:
        for g in new_mjspec.geoms:
            g.contype = 0
            g.conaffinity = 0

    env.reload_mujoco(new_mjspec)

    env._data.mocap_pos = init_mocap_pos
    env._data.mocap_quat = init_mocap_quat
    mujoco.mj_step(env._model, env._data, robot_conf.optimization_params.init_motion_iterations)
    qpos = env._data.qpos.copy()

    # load old model to env
    env.reload_mujoco(old_mjspec)
    key = jax.random.key(0)
    env.reset(key)

    return qpos


def fit_smpl_shape(
    env_name: str,
    robot_conf: DictConfig,
    path_to_smpl_model: str,
    save_path_new_smpl_shape: str,
    logger: logging.Logger,
    visualize: bool = False,
) -> None:
    """Fit the SMPL shape to match the robot configuration.

    Args:
        env_name (str): Name of the environment.
        robot_conf (DictConfig): Configuration of the robot.
        path_to_smpl_model (str): Path to the SMPL model.
        save_path_new_smpl_shape (str): Path to save the optimized shape.
        logger (logging.Logger): Logger for status updates.
        visualize (bool): Whether to visualize the optimization process.

    """

    check_optional_imports()

    # Ensure consistent PyTorch precision across different devices
    ensure_pytorch_precision_consistency()

    z_offset_viz = 2.0  # for visualization only

    # get environment
    env_cls = Mujoco.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})

    # add mocap bodies for all 'site_for_mimic' instances of an environment
    mjspec = env.mjspec
    sites_for_mimic = env.sites_for_mimic
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mjspec = add_mocap_bodies(mjspec, sites_for_mimic, target_mocap_bodies, robot_conf, add_equality_constraint=False)
    env.reload_mujoco(mjspec)
    key = jax.random.key(0)
    env.reset(key)

    smpl2mimic_site_idx = []
    for s in sites_for_mimic:
        # find smpl name
        for site_name, conf in robot_conf.site_joint_matches.items():
            if site_name == s:
                smpl2mimic_site_idx.append(SMPLH_BONE_ORDER_NAMES.index(conf.smpl_joint))

    # set humanoid to T-pose
    to_t_pose(env, robot_conf)

    # save initial qpos
    qpos_init = env._data.qpos.copy()
    qpos_init[0:3] = 0
    qpos_init[2] = z_offset_viz

    # set initial qpos and forward
    env._data.qpos = qpos_init
    mujoco.mj_forward(env._model, env._data)

    # get joint names
    device = torch.device(robot_conf.optimization_params.torch_device)

    # get initial pose
    pose_aa_stand = np.zeros((1, 156)).reshape(-1, 52, 3)
    pose_aa_stand[:, SMPLH_BONE_ORDER_NAMES.index("Pelvis")] = sRot.from_euler(
        "xyz", [np.pi / 2, 0.0, np.pi / 2], degrees=False
    ).as_rotvec()
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 156)).requires_grad_(False)

    # setup parser
    smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral")

    # define optimization variables
    shape_new = Variable(torch.zeros([1, 16]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    trans = torch.zeros([1, 3]).requires_grad_(False)
    optimizer_shape = torch.optim.Adam([shape_new, scale], lr=robot_conf.optimization_params.shape_lr)

    # get target site positions
    site_ids = [mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, s) for s in sites_for_mimic]
    target_site_pos = torch.from_numpy(env._data.site_xpos[site_ids])[None]
    target_site_mat = torch.from_numpy(env._data.site_xmat[site_ids])

    # get z offset
    z_offset = torch.tensor([0.0, 0.0, z_offset_viz])[None]

    # get the transformation matrices
    transformations_matrices = smpl_parser_n.get_joint_transformations(pose_aa_stand, shape_new, trans)
    global_rot_mats = transformations_matrices.detach().numpy()[..., :3, :3]
    global_pos = transformations_matrices.detach().numpy()[..., :3, 3]
    global_rot_mats = sRot.from_matrix(global_rot_mats[0, smpl2mimic_site_idx])

    # get rotations of mimic sites
    target_site_mat = sRot.from_matrix(target_site_mat.detach().numpy().reshape(-1, 3, 3))

    # rel rotation smpl to robot
    smpl2robot_rot_mat = np.einsum("bij,bjk->bik", global_rot_mats.inv().as_matrix(), target_site_mat.as_matrix())

    # transform smpl rotations to match robot site rotations
    # (here done just for visualization, only used in motion_fit function)
    new_smpl_rot_mats = np.einsum("bij,bjk->bik", global_rot_mats.as_matrix(), smpl2robot_rot_mat)
    new_smpl_quats = quat_scalarlast2scalarfirst(sRot.from_matrix(new_smpl_rot_mats).as_quat())

    pbar = tqdm(range(robot_conf.optimization_params.shape_iterations))
    init_feet_z_pos = None
    init_head_z_pos = None
    for iteration in pbar:
        transformations_matrices = smpl_parser_n.get_joint_transformations(pose_aa_stand, shape_new, trans)
        global_pos = transformations_matrices[..., :3, 3]

        if init_feet_z_pos is None:
            init_feet_z_pos = np.minimum(
                global_pos[0, SMPLH_BONE_ORDER_NAMES.index("R_Ankle"), 2].detach().numpy(),
                global_pos[0, SMPLH_BONE_ORDER_NAMES.index("L_Ankle"), 2].detach().numpy(),
            )
            init_head_z_pos = global_pos[0, SMPLH_BONE_ORDER_NAMES.index("Head"), 2].detach().numpy()

        if iteration == robot_conf.optimization_params.shape_iterations - 1:
            final_feet_z_pos = np.minimum(
                global_pos[0, SMPLH_BONE_ORDER_NAMES.index("R_Ankle"), 2].detach().numpy(),
                global_pos[0, SMPLH_BONE_ORDER_NAMES.index("L_Ankle"), 2].detach().numpy(),
            )
            final_head_z_pos = global_pos[0, SMPLH_BONE_ORDER_NAMES.index("Head"), 2].detach().numpy()

        global_pos = (global_pos - global_pos[:, 0] + z_offset) * scale

        if visualize:
            env._data.qpos = qpos_init
            env._data.mocap_pos = torch.squeeze(global_pos[:, smpl2mimic_site_idx]).detach().numpy()
            env._data.mocap_quat = new_smpl_quats

            mujoco.mj_forward(env._model, env._data)
            env.render()

        # calculate loss
        diff = target_site_pos - global_pos[:, smpl2mimic_site_idx]
        loss = diff.norm(dim=-1).mean()

        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_shape.step()

    # save positions offset
    smpl_pos = global_pos[0, smpl2mimic_site_idx].detach().numpy()
    smpl2robot_pos = smpl_pos - target_site_pos.detach().numpy()
    smpl2robot_pos = np.squeeze(smpl2robot_pos)

    # save new z-offset
    offset_z = init_feet_z_pos - final_feet_z_pos
    height_scale = (final_head_z_pos - final_feet_z_pos) / (init_head_z_pos - init_feet_z_pos)

    # Extract the directory path from the save path
    directory = os.path.dirname(save_path_new_smpl_shape)

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # save
    joblib.dump(
        (shape_new.detach(), scale, smpl2robot_pos, smpl2robot_rot_mat, offset_z, height_scale),
        save_path_new_smpl_shape,
    )
    logger.info(f"Shape parameters saved at {save_path_new_smpl_shape}")


def max_penetration_with_floor(model, data, floor_name="floor", verbose=False):
    """
    Returns:
        (deepest_penetration, (geom_id, floor_id))
    where:
        deepest_penetration < 0  → penetration depth
        = 0 → no penetration
    """

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_name)
    if floor_id < 0:
        raise ValueError(f"No geom named '{floor_name}' found")

    max_pen = 0.0
    offending_geom = None

    for i in range(data.ncon):
        c = data.contact[i]

        # Check if either side is FLOOR
        if c.geom1 == floor_id or c.geom2 == floor_id:
            if c.dist < max_pen:
                max_pen = c.dist
                offending_geom = c.geom1 if c.geom2 == floor_id else c.geom2

    if offending_geom is not None and verbose:
        geom_name = model.geom(offending_geom).name
        logging.debug(f"Penetration {max_pen:.6f} m between geom {geom_name} and FLOOR ({floor_id})")

    return max_pen, offending_geom, floor_id


def motion_transfer_robot_to_robot(
    env_name_source: str,
    robot_conf_source: DictConfig,
    traj_source: Trajectory,
    path_source_robot_smpl_data: str,
    env_name_target: str,
    robot_conf_target: DictConfig,
    path_target_robot_smpl_data: str,
    path_to_smpl_model: str,
    logger: logging.Logger,
    path_to_fitted_motion_source: str | None = None,
    visualize: bool = False,
) -> Trajectory:
    def rotation_matrix_loss_geodesic(rot1, rot2):
        """
        Computes the geodesic distance loss between two rotation matrices with two batch dimensions.

        rot1, rot2: (B1, B2, 3, 3)
        Returns: Mean geodesic loss over (B1, B2)
        """
        rot_diff = torch.matmul(rot1.transpose(-2, -1), rot2)  # rot1^T * rot2, supports (B1, B2, 3, 3)
        trace = torch.einsum("...ii->...", rot_diff)  # Extract trace along last two dims (B1, B2)
        eps = 1e-6  # Small epsilon for numerical stability
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0 + eps, 1.0 - eps))
        return theta.mean()

    check_optional_imports()

    path_to_target_robot_smpl_shape = os.path.join(path_target_robot_smpl_data, OPTIMIZED_SHAPE_FILE_NAME)

    if path_to_fitted_motion_source is not None and not os.path.exists(path_to_fitted_motion_source):
        device = torch.device("cuda")

        # get the source env
        env_cls = Mujoco.registered_envs[env_name_source]
        env = env_cls(**robot_conf_source.env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})

        # add mocap bodies for all 'site_for_mimic' instances of an environment
        mjspec = env.mjspec
        sites_for_mimic = env.sites_for_mimic
        target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
        mjspec = add_mocap_bodies(
            mjspec, sites_for_mimic, target_mocap_bodies, robot_conf_source, add_equality_constraint=False
        )
        env.reload_mujoco(mjspec)
        key = jax.random.key(0)
        env.reset(key)

        # extend the trajectory to include more model-specific entities
        traj_source = extend_motion(env_name_source, robot_conf_source.env_params, traj_source, logger)

        # load the source trajectory
        env.load_trajectory(traj_source, warn=False)

        # convert traj to numpy
        env.th.to_numpy()

        # get the body_shape of the source robot
        path_to_source_robot_smpl_shape = os.path.join(path_source_robot_smpl_data, OPTIMIZED_SHAPE_FILE_NAME)
        if not os.path.exists(path_to_source_robot_smpl_shape):
            logger.info("Robot shape file not found, fitting new one ...")
            fit_smpl_shape(
                env_name_source, robot_conf_source, path_to_smpl_model, path_to_source_robot_smpl_shape, logger
            )
        else:
            logger.info(f"Found existing robot shape file at {path_to_source_robot_smpl_shape}")
        (
            shape_source,
            scale_source,
            smpl2robot_pos_source,
            smpl2robot_rot_mat_source,
            offset_z_source,
            height_scale_source,
        ) = joblib.load(path_to_source_robot_smpl_shape)

        # get the source site positions used as a target for optimization
        sites_for_mimic = env.sites_for_mimic
        site_ids = np.array([mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, s) for s in sites_for_mimic])
        target_site_pos = torch.from_numpy(env.th.traj.data.site_xpos[:, site_ids])
        target_site_mat = torch.from_numpy(env.th.traj.data.site_xmat[:, site_ids])
        len_dataset = env.th.traj.data.n_samples

        # define the optimization variables
        pose = np.zeros([len_dataset, 156]).reshape(-1, 52, 3)
        init_rot_mat = target_site_mat[:, sites_for_mimic.index("pelvis_mimic")].reshape(-1, 3, 3)
        init_rot_mat = np.einsum(
            "nij,jk->nik", init_rot_mat, np.linalg.inv(smpl2robot_rot_mat_source[sites_for_mimic.index("pelvis_mimic")])
        )
        pose[:, SMPLH_BONE_ORDER_NAMES.index("Pelvis")] = sRot.from_matrix(init_rot_mat).as_rotvec()
        pose = torch.from_numpy(pose.reshape(-1, 156))
        pose = Variable(pose.float().to(device), requires_grad=True)
        trans = target_site_pos[:, sites_for_mimic.index("pelvis_mimic")].clone().to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([pose, trans], lr=robot_conf_source.optimization_params.pose_lr)
        scale_source = torch.tensor(scale_source).float().to(device).detach()

        # setup parser
        smpl_parser_n = SMPLH_Parser(model_path=path_to_smpl_model, gender="neutral").to(device)

        smpl2mimic_site_idx = []
        for s in sites_for_mimic:
            # find smpl name
            for site_name, conf in robot_conf_source.site_joint_matches.items():
                if site_name == s:
                    smpl2mimic_site_idx.append(SMPLH_BONE_ORDER_NAMES.index(conf.smpl_joint))

        shape_source = shape_source.repeat(len_dataset, 1).detach().to(device)

        # convert target site poses from site frame to smpl frame
        robot2smpl_pos_source = -smpl2robot_pos_source
        robot2smpl_rot_mat_source = np.linalg.inv(smpl2robot_rot_mat_source)
        pos_offset = np.einsum("nbij,bj->nbi", target_site_mat.reshape(len_dataset, -1, 3, 3), robot2smpl_pos_source)
        target_site_mat = np.einsum(
            "nbij,bjk->nbik", target_site_mat.reshape(len_dataset, -1, 3, 3), robot2smpl_rot_mat_source
        )
        target_site_pos = target_site_pos - pos_offset

        # convert to torch
        target_site_pos = target_site_pos.float().to(device)
        target_site_mat = torch.from_numpy(target_site_mat).float().to(device)

        iterations = robot_conf_source.optimization_params.pose_iterations
        pos_loss_weight = robot_conf_source.optimization_params.pos_loss_weight
        rot_loss_weight = robot_conf_source.optimization_params.rot_loss_weight
        for _i in tqdm(range(iterations)):
            # sample random indices
            transformations_matrices = smpl_parser_n.get_joint_transformations(pose, shape_source, trans)

            # get the global positions and rotations
            global_pos = transformations_matrices[..., :3, 3]
            global_rot_mats = transformations_matrices[..., :3, :3]

            # scale
            global_pos = (global_pos - global_pos[:, 0:1]) * scale_source + global_pos[:, 0:1]

            # calculate the loss
            pos_loss = (target_site_pos - global_pos[:, smpl2mimic_site_idx]).norm(dim=-1).mean()
            mat_loss = rotation_matrix_loss_geodesic(
                global_rot_mats[:, smpl2mimic_site_idx], target_site_mat.reshape(len_dataset, -1, 3, 3)
            )
            root_consistency_loss = torch.norm(pose[:1, :3] - pose[1:, :3], p=2).mean()

            if torch.any(torch.isnan(mat_loss)):
                raise ValueError("NaN in rotation matrix loss.")

            loss = pos_loss_weight * pos_loss + rot_loss_weight * mat_loss + 0.0 * root_consistency_loss

            if visualize:
                index = 0

                # convert smpl frame to site frame for visualization
                new_global_pos = global_pos[index, smpl2mimic_site_idx].cpu().detach().numpy()
                new_smpl_rot_mats = np.einsum(
                    "bij,bjk->bik",
                    global_rot_mats[index, smpl2mimic_site_idx].cpu().detach().numpy(),
                    smpl2robot_rot_mat_source,
                )
                pos_offset = np.einsum("bij,bj->bi", new_smpl_rot_mats, smpl2robot_pos_source)
                new_global_pos = np.squeeze(new_global_pos - pos_offset)

                env._data.mocap_pos = new_global_pos
                env._data.mocap_quat = quat_scalarlast2scalarfirst(sRot.from_matrix(new_smpl_rot_mats).as_quat())
                env._data.qpos = env.th.traj.data.qpos[index]
                mujoco.mj_forward(env._model, env._data)
                env.render()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # apply smoothing
            kernel_size = robot_conf_target.optimization_params.smoothing_kernel_size
            sigma = robot_conf_target.optimization_params.smoothing_sigma
            if sigma > 0:
                pose_non_filtered = pose[:, 3:].reshape(len_dataset, -1, 3)
                pose_non_filtered = pose_non_filtered.permute(1, 2, 0)
                pose_filtered = gaussian_filter_1d_batch(pose_non_filtered, kernel_size, sigma, device)
                pose_filtered = pose_filtered.permute(2, 0, 1)
                pose.data[:, 3:] = pose_filtered.reshape(-1, 153)

        motion_file = {
            "pose_aa": pose.cpu().detach().numpy().reshape(-1, 156),
            "trans": trans.cpu().detach().numpy(),
            "fps": env.th.traj.info.frequency,
        }

        # account for scale
        motion_file["pose_aa"] /= scale_source.cpu().detach().numpy()
        motion_file["trans"][:, 2] -= offset_z_source

        # apply height scaling while preserving init height
        height_scale_source_inv = 1 / height_scale_source
        trans[:, :2].data *= height_scale_source_inv  # scale x and y
        trans[:, 2].data = (trans[:, 2] - trans[0, 2]) * height_scale_source_inv + trans[0, 2]

        if path_to_fitted_motion_source is not None:
            # create dir if it does not exist
            directory = os.path.dirname(path_to_fitted_motion_source)
            os.makedirs(directory, exist_ok=True)
            # if a file path is provided, save the fitted motion
            np.savez(path_to_fitted_motion_source, **motion_file)

    else:
        logger.info(f"Loading fitted motion from {path_to_fitted_motion_source}.")
        motion_file = np.load(path_to_fitted_motion_source)

    # generate the body_shape of the target robot if it does not exist
    if not os.path.exists(path_to_target_robot_smpl_shape):
        logger.info("Robot shape file not found, fitting new one ...")
        fit_smpl_shape(
            env_name_target, robot_conf_target, path_to_smpl_model, path_to_target_robot_smpl_shape, logger, visualize
        )
    else:
        logger.info(f"Found existing robot shape file at {path_to_target_robot_smpl_shape}")

    traj_target, _ = fit_smpl_motion(
        env_name_target,
        robot_conf_target,
        path_to_smpl_model,
        motion_file,
        path_to_target_robot_smpl_shape,
        logger,
        skip_steps=False,
        visualize=visualize,
    )

    return traj_target


def _extend_motion_with_environment(
    env_name: str, env_params: DictConfig, traj: Trajectory, logger: logging.Logger | None = None
) -> tuple[Trajectory, Mujoco]:
    """
    Extend a motion trajectory to include more model-specific entities
    like body xpos, site positions, etc. and to match the environment's frequency.

    Args:
        env_name (str): Name of the environment.
        env_params (DictConfig): Environment params.
        traj (Trajectory): The original trajectory data.
        logger (logging.Logger): Logger for status updates.

    Returns:
        Extended trajectory and the environment used to produce it.

    """
    # Special handling for MyoBimanualArm trajectories
    if "MyoBimanualArm" in env_name:
        if logger:
            logger.info("Applying MyoBimanualArm trajectory retargeting...")

        # First, retarget the trajectory to extract only upper body joints info: qpos, qvel
        traj = retarget_trajectory_for_bimanual(traj)

        # If env_params is empty (common case for MyoBimanualArm), create default params
        if not env_params or len(env_params) == 0:
            env_params = {}

    # Normalize environment name for consistent retargeting
    if "MjxMyoBimanualArm" in env_name:
        env_name = BIMANUAL_ENV_NAME
    elif "Mjx" in env_name:
        env_name = env_name.replace("Mjx", "")

    env_cls = Mujoco.registered_envs[env_name]
    env = env_cls(**env_params, th_params={"random_start": False, "fixed_start_conf": (0, 0)})

    original_frequency = traj.info.frequency
    target_frequency = 1.0 / env.dt

    env.load_trajectory(traj, warn=False, cache_type=TrajectoryCacheType.NONE)
    traj = env.th.traj

    if logger:
        logger.info(
            f"Aligned trajectory from {original_frequency:.3f} Hz to {target_frequency:.3f} Hz "
            f"({traj.data.n_samples} samples)"
        )

    return traj, env


def extend_motion(
    env_name: str, env_params: DictConfig, traj: Trajectory, logger: logging.Logger | None = None
) -> Trajectory:
    """Extend and resample a trajectory for the requested environment."""
    extended_trajectory, _environment = _extend_motion_with_environment(
        env_name,
        env_params,
        traj,
        logger,
    )
    return extended_trajectory


def create_multi_trajectory_hash(names: list[str]) -> str:
    """
    Generates a stable hash for a list of strings using SHA256 with incremental updates.

    Args:
        names (list[str]): The list of strings to hash.

    Returns:
        str: A hexadecimal hash string.
    """

    # Sort the list to ensure order invariance
    sorted_names = sorted(names)

    hash_obj = hashlib.sha256()
    for s in sorted_names:
        hash_obj.update(s.encode())
    return hash_obj.hexdigest()


def _prepare_gmr_cache_path(
    cache_path: str,
    env_name: str,
    dataset_name: str,
    cache_root: str,
    clear_cache: bool,
    logger,
    cache_namespace: str = GMR_CACHE_NAMESPACE,
    allow_remote_download: bool = True,
) -> str:
    """Resolve a GMR cache, checking remote storage only for published source namespaces."""
    resolved_cache_path = _resolve_cache_path(cache_path)
    if clear_cache or os.path.exists(resolved_cache_path) or gmr_quality_rejection_path(resolved_cache_path).exists():
        return resolved_cache_path

    # Alternate namespaces contain locally materialized retargeting variants.
    # Downloading the canonical GMR cache into one would silently change its
    # algorithmic meaning.
    if cache_namespace != GMR_CACHE_NAMESPACE or not allow_remote_download:
        return resolved_cache_path

    downloaded_path = try_download_gmr_cache(
        dataset_name=dataset_name,
        env_name=env_name,
        cache_dir=cache_root,
        logger_override=logger,
    )
    if downloaded_path is None:
        return resolved_cache_path

    return _resolve_cache_path(str(downloaded_path))


def _preprocess_trajectory_for_concatenation(traj: Trajectory) -> Trajectory:
    """Reduce a retargeted motion to the source data shared by all cache versions."""
    empty = np.empty(0, dtype=traj.data.qpos.dtype)
    data = traj.data.replace(
        xpos=empty,
        xquat=empty,
        cvel=empty,
        subtree_com=empty,
        cvel_parent=empty,
        subtree_com_root=empty,
        site_xpos=empty,
        site_xmat=empty,
    )
    info = TrajectoryInfo(
        joint_names=traj.info.joint_names,
        model=TrajectoryModel(
            njnt=traj.info.model.njnt,
            jnt_type=np.asarray(traj.info.model.jnt_type),
        ),
        frequency=traj.info.frequency,
        metadata=traj.info.metadata,
    )
    return Trajectory(info=info, data=data)


def _load_trajectories_individually(
    env_name: str,
    dataset_list: list[str | AMASSMotionSpec],
    robot_conf: DictConfig,
    logger,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    gmr_cache_namespace: str | None = None,
    clear_cache: bool = False,
) -> LoadedTrajectorySet:
    """Load motions independently and keep source names aligned."""
    path_to_converted_datasets = get_converted_amass_dataset_path()
    cache_env_name = env_name.replace("Mjx", "") if "Mjx" in env_name else env_name

    # Retargeting methods use separate cache namespaces.
    resolved_gmr_cache_namespace = gmr_cache_namespace or GMR_CACHE_NAMESPACE
    if retargeting_method == "gmr":
        path_robot_smpl_data = os.path.join(
            path_to_converted_datasets,
            cache_env_name,
            resolved_gmr_cache_namespace,
        )
    else:
        path_robot_smpl_data = os.path.join(path_to_converted_datasets, cache_env_name)
    os.makedirs(path_robot_smpl_data, exist_ok=True)

    loaded_sets = []

    for i, motion in enumerate(dataset_list):
        motion_spec = as_amass_motion_spec(motion)
        motion_name = motion_spec.path
        if retargeting_method == "gmr":
            _cache_identity, cache_variant = get_gmr_cache_identity(gmr_config)
            cache_path = str(
                get_gmr_motion_cache_path(
                    cache_env_name,
                    motion_spec,
                    path_to_converted_datasets,
                    cache_namespace=resolved_gmr_cache_namespace,
                    gmr_config=gmr_config,
                )
            )
        else:
            cache_variant = None
            cache_path = _motion_cache_path(path_robot_smpl_data, motion_spec)
        if retargeting_method == "gmr" and motion_spec.is_full_motion:
            cache_path = _prepare_gmr_cache_path(
                cache_path=cache_path,
                env_name=cache_env_name,
                dataset_name=motion_name,
                cache_root=path_to_converted_datasets,
                clear_cache=clear_cache,
                allow_remote_download=cache_variant is None,
                logger=logger,
                cache_namespace=resolved_gmr_cache_namespace,
            )
        else:
            cache_path = _resolve_cache_path(cache_path)

        rejected = retargeting_method == "gmr" and gmr_quality_rejection_path(cache_path).exists()
        if os.path.exists(cache_path) and not clear_cache and not rejected:
            logger.info(
                f"Dataset {i + 1}/{len(dataset_list)}: Found existing retargeted motion file at {cache_path}. Loading ..."
            )
            # Keep cached arrays on the host while combining motions.
            trajectory = Trajectory.load(cache_path, backend=np)
            loaded_sets.append(
                LoadedTrajectorySet(
                    _preprocess_trajectory_for_concatenation(trajectory),
                    (motion_name,),
                )
            )
            continue

        method_name = retargeting_method.upper() if retargeting_method else "SMPL"
        action = "Re-retargeting (clear_cache)" if clear_cache and os.path.exists(cache_path) else "Retargeting"
        logger.info(f"Dataset {i + 1}/{len(dataset_list)}: {action} AMASS motion file using {method_name} ...")

        try:
            motion_input = motion_name if motion_spec.is_full_motion else motion_spec
            loaded = load_retargeted_amass_trajectory_set(
                env_name,
                motion_input,
                robot_conf,
                retargeting_method=retargeting_method,
                gmr_config=gmr_config,
                gmr_cache_namespace=gmr_cache_namespace,
                clear_cache=clear_cache,
            )
            loaded_sets.append(
                LoadedTrajectorySet(
                    _preprocess_trajectory_for_concatenation(loaded.trajectory),
                    loaded.motion_names,
                )
            )
        except Exception as e:
            logger.error(f"Skipping dataset '{motion_spec.cache_key}' due to failure during retargeting: {e}")
            continue

    if not loaded_sets:
        raise RuntimeError(f"Retargeting produced no valid {env_name} trajectories.")
    if len(loaded_sets) == 1:
        return loaded_sets[0]

    logger.info(f"Concatenating {len(loaded_sets)} final {env_name} trajectories ...")
    loaded = LoadedTrajectorySet.concatenate(loaded_sets, backend=np)
    logger.info("Final concatenation successful!")
    return loaded


def load_retargeted_amass_trajectory(
    env_name: str,
    dataset_name: str | AMASSMotionSpec | list[str | AMASSMotionSpec] | tuple[str | AMASSMotionSpec, ...],
    robot_conf: DictConfig = None,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
    *,
    gmr_cache_namespace: str | None = None,
) -> Trajectory:
    """Load retargeted AMASS trajectory data."""
    return load_retargeted_amass_trajectory_set(
        env_name=env_name,
        dataset_name=dataset_name,
        robot_conf=robot_conf,
        retargeting_method=retargeting_method,
        gmr_config=gmr_config,
        gmr_cache_namespace=gmr_cache_namespace,
        clear_cache=clear_cache,
    ).trajectory


def load_retargeted_amass_trajectory_set(
    env_name: str,
    dataset_name: str | AMASSMotionSpec | list[str | AMASSMotionSpec] | tuple[str | AMASSMotionSpec, ...],
    robot_conf: DictConfig = None,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
    *,
    gmr_cache_namespace: str | None = None,
) -> LoadedTrajectorySet:
    """Load retargeted AMASS trajectories with their source names."""
    logger = setup_logger("amass", identifier="[MuscleMimic AMASS Retargeting Pipeline]")

    # if robot_conf is not provided, load default one it from the YAML file
    if robot_conf is None:
        robot_conf = load_robot_conf_file(env_name)

    # Check retargeting method
    if retargeting_method is None:
        retargeting_method = robot_conf.get("retargeting_method", "smpl")
    if gmr_config is None:
        gmr_config = robot_conf.get("gmr_config", None)
    if clear_cache and gmr_cache_namespace not in (None, GMR_CACHE_NAMESPACE):
        raise ValueError(f"clear_cache cannot be used with read-only GMR namespace {gmr_cache_namespace!r}")

    if isinstance(dataset_name, str | AMASSMotionSpec):
        motion_specs = [as_amass_motion_spec(dataset_name)]
    else:
        motion_specs = [as_amass_motion_spec(motion) for motion in dataset_name]

    # Process datasets individually to avoid trajectory concatenation with different traj info
    if len(motion_specs) > 1:
        return _load_trajectories_individually(
            env_name,
            motion_specs,
            robot_conf,
            logger,
            retargeting_method=retargeting_method,
            gmr_config=gmr_config,
            gmr_cache_namespace=gmr_cache_namespace,
            clear_cache=clear_cache,
        )

    path_to_converted_datasets = get_converted_amass_dataset_path()

    # Normalize environment name for cache consistency (MJX and non-MJX share same cache)
    cache_env_name = env_name.replace("Mjx", "") if "Mjx" in env_name else env_name

    # Use separate cache directories for different retargeting methods
    resolved_gmr_cache_namespace = gmr_cache_namespace or GMR_CACHE_NAMESPACE
    if retargeting_method == "gmr":
        path_robot_smpl_data = os.path.join(
            path_to_converted_datasets,
            cache_env_name,
            resolved_gmr_cache_namespace,
        )
    else:
        path_robot_smpl_data = os.path.join(path_to_converted_datasets, cache_env_name)

    if not os.path.exists(path_robot_smpl_data):
        os.makedirs(path_robot_smpl_data, exist_ok=True)

    loaded_sets = []
    for i, motion_spec in enumerate(motion_specs):
        motion_name = motion_spec.path
        if retargeting_method == "gmr":
            _cache_identity, cache_variant = get_gmr_cache_identity(gmr_config)
            d_path = str(
                get_gmr_motion_cache_path(
                    cache_env_name,
                    motion_spec,
                    path_to_converted_datasets,
                    cache_namespace=resolved_gmr_cache_namespace,
                    gmr_config=gmr_config,
                )
            )
        else:
            cache_variant = None
            d_path = _motion_cache_path(path_robot_smpl_data, motion_spec)
        if retargeting_method == "gmr" and motion_spec.is_full_motion:
            d_path = _prepare_gmr_cache_path(
                cache_path=d_path,
                env_name=cache_env_name,
                dataset_name=motion_name,
                cache_root=path_to_converted_datasets,
                clear_cache=clear_cache,
                allow_remote_download=cache_variant is None,
                logger=logger,
                cache_namespace=resolved_gmr_cache_namespace,
            )
        else:
            d_path = _resolve_cache_path(d_path)
        if retargeting_method == "gmr" and clear_cache:
            _clear_gmr_terminal_artifacts(d_path)
        quality_rejection_path = gmr_quality_rejection_path(d_path)
        if retargeting_method == "gmr" and quality_rejection_path.exists():
            rejection = json.loads(quality_rejection_path.read_text())
            raise GMRQualityGateError(
                f"Previously rejected GMR motion: {rejection['reason']}",
                dict(rejection.get("metrics", {})),
            )
        cache_exists = os.path.exists(d_path)
        if retargeting_method == "gmr" and resolved_gmr_cache_namespace != GMR_CACHE_NAMESPACE and not cache_exists:
            raise FileNotFoundError(
                "Precomputed GMR trajectory is missing from read-only namespace "
                f"{resolved_gmr_cache_namespace!r}: {d_path}"
            )
        if not cache_exists or clear_cache:
            # Only check imports when we actually need to run retargeting
            check_optional_imports()
            path_to_smpl_model = get_smpl_model_path()
            smpl_optimized_shape_cache_path = os.path.join(
                path_to_converted_datasets, cache_env_name, OPTIMIZED_SHAPE_FILE_NAME
            )

            # Cache validation for the SMPL-specific optimized shape file.
            if retargeting_method == "smpl":
                # Validate cached shape file compatibility
                smpl_shape_cache_valid = False
                if os.path.exists(smpl_optimized_shape_cache_path):
                    try:
                        # Load cached data and check site count compatibility
                        _shape_new, _scale, _smpl2robot_pos, smpl2robot_rot_mat, _offset_z, _height_scale = joblib.load(
                            smpl_optimized_shape_cache_path
                        )
                        cached_n_sites = smpl2robot_rot_mat.shape[0]

                        # Get expected site count from current environment
                        env_cls = Mujoco.registered_envs[env_name]
                        env = env_cls(
                            **robot_conf.env_params,
                            th_params={"random_start": False, "fixed_start_conf": (0, 0)},
                        )
                        expected_n_sites = len(env.sites_for_mimic)

                        if cached_n_sites == expected_n_sites:
                            smpl_shape_cache_valid = True
                            logger.info(f"Found valid SMPL optimized shape cache at {smpl_optimized_shape_cache_path}")
                        else:
                            logger.warning(
                                "SMPL optimized shape cache has "
                                f"{cached_n_sites} sites, but environment expects {expected_n_sites}. Regenerating..."
                            )
                            os.remove(smpl_optimized_shape_cache_path)
                    except Exception as e:
                        logger.warning(f"Failed to validate SMPL optimized shape cache: {e}. Regenerating...")
                        try:
                            os.remove(smpl_optimized_shape_cache_path)
                        except OSError:
                            print("Failed to remove invalid cache file.")

                if not smpl_shape_cache_valid:
                    logger.info("SMPL optimized shape cache not found or invalid, fitting a new one ...")
                    fit_smpl_shape(
                        env_name,
                        robot_conf,
                        path_to_smpl_model,
                        smpl_optimized_shape_cache_path,
                        logger,
                    )
            else:
                logger.info(
                    f"Using {retargeting_method.upper()} retargeting - skipping SMPL-only optimized shape cache setup"
                )

            # Choose retargeting method
            action = "Re-retargeting (clear_cache)" if clear_cache and cache_exists else "Retargeting"
            if retargeting_method == "gmr":
                logger.info(f"Dataset {i + 1}/{len(motion_specs)}: {action} AMASS motion file using GMR ...")
                amass_file_path, _cache_key = _resolve_gmr_motion_paths(motion_name, gmr_config)
                try:
                    trajectory, analysis = fit_gmr_motion(
                        env_name,
                        robot_conf,
                        amass_file_path,
                        logger,
                        gmr_config,
                        motion_spec=motion_spec,
                    )
                except GMRQualityGateError as error:
                    rejection_path = _write_gmr_quality_rejection(
                        d_path,
                        motion_spec,
                        error,
                        gmr_config=gmr_config,
                        cache_namespace=resolved_gmr_cache_namespace,
                    )
                    logger.warning("Saved GMR quality rejection to %s", rejection_path)
                    raise
            else:
                logger.info(
                    f"Dataset {i + 1}/{len(motion_specs)}: {action} AMASS motion file using optimized body shape ..."
                )
                motion_data = load_amass_data(motion_name, motion_spec=motion_spec)
                path_converted_shape = os.path.join(
                    path_to_converted_datasets, f"{cache_env_name}/{OPTIMIZED_SHAPE_FILE_NAME}"
                )
                trajectory, analysis = fit_smpl_motion(
                    env_name,
                    robot_conf,
                    path_to_smpl_model,
                    motion_data,
                    path_converted_shape,
                    logger,
                    # motion_name=d_name
                )

            logger.info("Using Mujoco to calculate other model-specific entities ...")
            if retargeting_method == "gmr":
                trajectory, extension_env = _extend_motion_with_environment(
                    env_name,
                    robot_conf.env_params,
                    trajectory,
                    logger,
                )
                try:
                    trajectory, output_floor_metrics = _guard_gmr_resampled_floor_clearance(
                        trajectory,
                        extension_env._model,
                        extension_env.root_free_joint_xml_name,
                    )
                    _update_gmr_analysis_for_output_floor_guard(
                        analysis,
                        output_floor_metrics,
                    )
                    logger.info(
                        "GMR final-frequency floor guard: minimum %.6f -> %.6f m with %.6f m world-z lift",
                        output_floor_metrics["output_minimum_surface_clearance_before_m"],
                        output_floor_metrics["output_minimum_surface_clearance_after_m"],
                        output_floor_metrics["output_floor_z_translation_m"],
                    )
                    continuity_metrics = _validate_gmr_retained_continuity(
                        np.asarray(trajectory.data.qvel),
                        float(trajectory.info.frequency),
                    )
                except GMRQualityGateError as error:
                    rejection_path = _write_gmr_quality_rejection(
                        d_path,
                        motion_spec,
                        error,
                        gmr_config=gmr_config,
                        cache_namespace=resolved_gmr_cache_namespace,
                    )
                    logger.warning("Saved GMR quality rejection to %s", rejection_path)
                    raise
                analysis.update(continuity_metrics)
            else:
                trajectory = extend_motion(env_name, robot_conf.env_params, trajectory, logger)
            trajectory.save(d_path)

            # Save analysis data to file
            analysis_path = d_path.replace(".npz", "_analysis.npz")
            np.savez(analysis_path, **analysis)
            logger.info(f"Saved analysis to {analysis_path}")
            loaded_sets.append(LoadedTrajectorySet(trajectory, (motion_name,)))
        else:
            logger.info(
                f"Dataset {i + 1}/{len(motion_specs)}: Found existing retargeted motion file at {d_path}. Loading ..."
            )
            trajectory = Trajectory.load(d_path, backend=np)
            loaded_sets.append(LoadedTrajectorySet(trajectory, (motion_name,)))

    if len(loaded_sets) == 1:
        loaded = loaded_sets[0]
    else:
        logger.info("Concatenating trajectories ...")
        loaded = LoadedTrajectorySet.concatenate(loaded_sets, backend=np)

    logger.info("Trajectory data loaded!")

    return loaded


def retarget_traj_from_robot_to_robot(
    env_name_source: str,
    traj_source: Trajectory,
    env_name_target: str,
    robot_conf_source: DictConfig = None,
    robot_conf_target: DictConfig = None,
    path_to_fitted_motion_source: str | None = None,
):
    check_optional_imports()

    logger = setup_logger("amass", identifier="[LocoMuJoCo's Robot2Robot Retargeting Pipeline]")

    path_to_smpl_model = get_smpl_model_path()
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()

    # if robot_conf is not provided, load default one it from the YAML file
    if robot_conf_source is None:
        robot_conf_source = load_robot_conf_file(env_name_source)
    if robot_conf_target is None:
        robot_conf_target = load_robot_conf_file(env_name_target)

    path_source_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, env_name_source)
    path_target_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, env_name_target)

    traj_target = motion_transfer_robot_to_robot(
        env_name_source,
        robot_conf_source,
        traj_source,
        path_source_robot_smpl_data,
        env_name_target,
        robot_conf_target,
        path_target_robot_smpl_data,
        path_to_smpl_model,
        logger,
        path_to_fitted_motion_source,
    )

    return traj_target


def retarget_trajectory_for_bimanual(traj: Trajectory) -> Trajectory:
    """
    Retarget a full-body trajectory for MyoBimanualArm by extracting only upper body joints.

    This function maps joints from full humanoid models to MyoBimanualArm's upper body structure,
    filtering out lower body joints (legs, pelvis) and mapping only relevant arm, shoulder, and hand joints.

    Args:
        traj (Trajectory): Full-body trajectory to retarget

    Returns:
        Trajectory: Retargeted trajectory with only upper body joints
    """

    # Define mapping from common full-body joint names to MyoBimanualArm joint names
    # Only include mappings that actually transform joint names
    joint_mapping = {
        # MyoFullBody joint mappings (primary - since we now use MyoFullBody as intermediate)
        # Right side (no suffix in MyoFullBody)
        "sternoclavicular_r2": "sternoclavicular_r2_r",
        "sternoclavicular_r3": "sternoclavicular_r3_r",
        "unrotscap_r3": "unrotscap_r3_r",
        "unrotscap_r2": "unrotscap_r2_r",
        "acromioclavicular_r2": "acromioclavicular_r2_r",
        "acromioclavicular_r3": "acromioclavicular_r3_r",
        "acromioclavicular_r1": "acromioclavicular_r1_r",
        "unrothum_r1": "unrothum_r1_r",
        "unrothum_r3": "unrothum_r3_r",
        "unrothum_r2": "unrothum_r2_r",
        "elv_angle": "elv_angle_r",
        "shoulder_elv": "shoulder_elv_r",
        "shoulder1_r2": "shoulder1_r2_r",
        "shoulder_rot": "shoulder_rot_r",
        "elbow_flexion": "elbow_flex_r",
        "pro_sup": "pro_sup_r",
        "deviation": "deviation_r",
        "flexion": "flexion_r",
        # Left side (L suffix in MyoFullBody)
        "sternoclavicular_r2L": "sternoclavicular_r2_l",
        "sternoclavicular_r3L": "sternoclavicular_r3_l",
        "unrotscap_r3L": "unrotscap_r3_l",
        "unrotscap_r2L": "unrotscap_r2_l",
        "acromioclavicular_r2L": "acromioclavicular_r2_l",
        "acromioclavicular_r3L": "acromioclavicular_r3_l",
        "acromioclavicular_r1L": "acromioclavicular_r1_l",
        "unrothum_r1L": "unrothum_r1_l",
        "unrothum_r3L": "unrothum_r3_l",
        "unrothum_r2L": "unrothum_r2_l",
        "elv_angleL": "elv_angle_l",
        "shoulder_elvL": "shoulder_elv_l",
        "shoulder1_r2L": "shoulder1_r2_l",
        "shoulder_rotL": "shoulder_rot_l",
        "elbow_flexionL": "elbow_flex_l",
        "pro_supL": "pro_sup_l",
        "deviationL": "deviation_l",
        "flexionL": "flexion_l",
    }

    # Define the complete set of MyoBimanualArm joint names
    bimanual_joint_names = [
        # Right arm joints
        "sternoclavicular_r2_r",
        "sternoclavicular_r3_r",
        "acromioclavicular_r1_r",
        "acromioclavicular_r2_r",
        "acromioclavicular_r3_r",
        "elv_angle_r",
        "shoulder_elv_r",
        "shoulder1_r2_r",
        "shoulder_rot_r",
        "elbow_flex_r",
        "pro_sup_r",
        "deviation_r",
        "flexion_r",
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
        "unrothum_r1_r",
        "unrothum_r2_r",
        "unrothum_r3_r",
        "unrotscap_r2_r",
        "unrotscap_r3_r",
        # Left arm joints
        "sternoclavicular_r2_l",
        "sternoclavicular_r3_l",
        "acromioclavicular_r1_l",
        "acromioclavicular_r2_l",
        "acromioclavicular_r3_l",
        "elv_angle_l",
        "shoulder_elv_l",
        "shoulder1_r2_l",
        "shoulder_rot_l",
        "elbow_flex_l",
        "pro_sup_l",
        "deviation_l",
        "flexion_l",
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
        "unrothum_r1_l",
        "unrothum_r2_l",
        "unrothum_r3_l",
        "unrotscap_r2_l",
        "unrotscap_r3_l",
    ]
    # Find which joints from the source trajectory can be mapped
    source_joint_names = traj.info.joint_names
    mapped_joints = []
    mapped_indices_qpos = []
    mapped_indices_qvel = []
    final_joint_names = []

    # Track qpos and qvel indices
    qpos_idx = 0
    qvel_idx = 0

    for i, joint_name in enumerate(source_joint_names):
        jnt_type = traj.info.model.jnt_type[i]

        # Calculate qpos and qvel dimensions for this joint
        if jnt_type == 0:  # free joint
            qpos_size = 7  # 3 pos + 4 quat
            qvel_size = 6  # 3 lin + 3 ang vel
        elif jnt_type == 1:  # ball joint
            qpos_size = 4  # quaternion
            qvel_size = 3  # 3 angular vel
        elif jnt_type == 2:  # slide joint
            qpos_size = 1
            qvel_size = 1
        elif jnt_type == 3:  # hinge joint
            qpos_size = 1
            qvel_size = 1
        else:
            qpos_size = 1
            qvel_size = 1

        # Check if this joint can be mapped to MyoBimanualArm
        target_name = joint_mapping.get(joint_name, joint_name)
        if target_name in bimanual_joint_names:
            mapped_joints.append(i)
            final_joint_names.append(target_name)

            # Store the qpos and qvel indices for this joint
            mapped_indices_qpos.extend(range(qpos_idx, qpos_idx + qpos_size))
            mapped_indices_qvel.extend(range(qvel_idx, qvel_idx + qvel_size))

        # Update indices for next joint
        qpos_idx += qpos_size
        qvel_idx += qvel_size

    if not mapped_joints:
        # If no joints can be mapped, create a minimal trajectory with zero positions
        print("Warning: No compatible joints found for MyoBimanualArm retargeting. Creating minimal trajectory.")

        n_joints = len(bimanual_joint_names)
        n_samples = traj.data.n_samples

        # Create zero trajectories
        qpos = jnp.zeros((n_samples, n_joints))
        qvel = jnp.zeros((n_samples, n_joints))
        jnt_type = jnp.full(n_joints, 3)  # All hinge joints

        # Create new trajectory info and data
        traj_model = TrajectoryModel(njnt=n_joints, jnt_type=jnt_type)
        traj_info = TrajectoryInfo(joint_names=bimanual_joint_names, model=traj_model, frequency=traj.info.frequency)
        traj_data = TrajectoryData(qpos=qpos, qvel=qvel, split_points=traj.data.split_points)

        return Trajectory(info=traj_info, data=traj_data)

    # Extract mapped joint data
    mapped_qpos = traj.data.qpos[:, mapped_indices_qpos] if mapped_indices_qpos else jnp.zeros((traj.data.n_samples, 0))
    mapped_qvel = traj.data.qvel[:, mapped_indices_qvel] if mapped_indices_qvel else jnp.zeros((traj.data.n_samples, 0))

    # Create joint type array (assuming all mapped joints are hinge joints for MyoBimanualArm)
    mapped_jnt_types = jnp.array([3] * len(final_joint_names))  # 3 = hinge joint

    # Create new trajectory info and data for MyoBimanualArm
    traj_model = TrajectoryModel(njnt=len(final_joint_names), jnt_type=mapped_jnt_types)
    traj_info = TrajectoryInfo(joint_names=final_joint_names, model=traj_model, frequency=traj.info.frequency)

    traj_data = TrajectoryData(
        qpos=mapped_qpos,
        qvel=mapped_qvel,
        split_points=traj.data.split_points,
    )

    return Trajectory(info=traj_info, data=traj_data)


def retarget_smpl_to_bimanual_via_intermediate(
    dataset_name: str | AMASSMotionSpec | list[str | AMASSMotionSpec] | tuple[str | AMASSMotionSpec, ...],
    robot_conf_skeleton: DictConfig = None,
    robot_conf_bimanual: DictConfig = None,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
) -> Trajectory:
    """Retarget AMASS trajectory data for MyoBimanualArm."""
    return retarget_smpl_to_bimanual_via_intermediate_set(
        dataset_name=dataset_name,
        robot_conf_skeleton=robot_conf_skeleton,
        robot_conf_bimanual=robot_conf_bimanual,
        retargeting_method=retargeting_method,
        gmr_config=gmr_config,
        clear_cache=clear_cache,
    ).trajectory


def retarget_smpl_to_bimanual_via_intermediate_set(
    dataset_name: str | AMASSMotionSpec | list[str | AMASSMotionSpec] | tuple[str | AMASSMotionSpec, ...],
    robot_conf_skeleton: DictConfig = None,
    robot_conf_bimanual: DictConfig = None,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
) -> LoadedTrajectorySet:
    """Retarget AMASS motions for MyoBimanualArm with their source names."""
    logger = setup_logger("bimanual_retargeting", identifier="[Three-Stage Retargeting Pipeline]")

    # Check for headless environment and setup EGL if needed
    is_headless = detect_headless_environment()
    if is_headless:
        setup_headless_rendering()

    # Load default configurations if not provided
    if robot_conf_skeleton is None:
        robot_conf_skeleton = load_robot_conf_file("MyoFullBody")
    if robot_conf_bimanual is None:
        # Create minimal config for MyoBimanualArm
        robot_conf_bimanual = DictConfig({"env_params": {}})

    # Normalize dataset list
    if isinstance(dataset_name, str | AMASSMotionSpec):
        dataset_list = [as_amass_motion_spec(dataset_name)]
    else:
        dataset_list = [as_amass_motion_spec(motion) for motion in dataset_name]

    # Determine cache directory for MyoBimanualArm trajectories
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()
    path_robot_smpl_data_bimanual = os.path.join(path_to_converted_amass_datasets, BIMANUAL_ENV_NAME)
    os.makedirs(path_robot_smpl_data_bimanual, exist_ok=True)

    # One cache per motion keeps names aligned with trajectory indices.
    loaded_sets = []
    skipped = 0
    if retargeting_method == "gmr":
        path_robot_smpl_data_bimanual = os.path.join(
            path_to_converted_amass_datasets, BIMANUAL_ENV_NAME, GMR_CACHE_NAMESPACE
        )
    else:
        path_robot_smpl_data_bimanual = os.path.join(path_to_converted_amass_datasets, BIMANUAL_ENV_NAME)

    for i, motion_spec in enumerate(tqdm(dataset_list, desc="Bimanual retargeting", unit="traj")):
        motion_name = motion_spec.path
        if retargeting_method == "gmr":
            _cache_identity, cache_variant = get_gmr_cache_identity(gmr_config)
            cache_path = str(
                get_gmr_motion_cache_path(
                    BIMANUAL_ENV_NAME,
                    motion_spec,
                    path_to_converted_amass_datasets,
                    gmr_config=gmr_config,
                )
            )
        else:
            cache_variant = None
            cache_path = _motion_cache_path(path_robot_smpl_data_bimanual, motion_spec)
        if retargeting_method == "gmr" and motion_spec.is_full_motion:
            cache_path = _prepare_gmr_cache_path(
                cache_path=cache_path,
                env_name=BIMANUAL_ENV_NAME,
                dataset_name=motion_name,
                cache_root=path_to_converted_amass_datasets,
                clear_cache=clear_cache,
                allow_remote_download=cache_variant is None,
                logger=logger,
            )
        else:
            cache_path = _resolve_cache_path(cache_path)

        if retargeting_method == "gmr":
            if clear_cache:
                _clear_gmr_terminal_artifacts(cache_path)
            elif gmr_quality_rejection_path(cache_path).exists():
                skipped += 1
                logger.warning("Skipping persistently rejected bimanual motion %s", motion_spec.cache_key)
                continue

        cache_exists = os.path.exists(cache_path)
        if cache_exists and not clear_cache:
            logger.info(
                f"Dataset {i + 1}/{len(dataset_list)}: Found existing cached MyoBimanualArm trajectory at {cache_path}. Loading ..."
            )
            trajectory = Trajectory.load(cache_path, backend=np)
            loaded_sets.append(LoadedTrajectorySet(trajectory, (motion_name,)))
            continue

        action = "Re-retargeting (clear_cache)" if clear_cache and cache_exists else "Retargeting"
        logger.info(f"Dataset {i + 1}/{len(dataset_list)}: {action} to MyoBimanualArm using three-stage pipeline ...")

        try:
            method_name = retargeting_method.upper() if retargeting_method else "SMPL"
            logger.info(f"Stage 1: SMPL → MyoFullBody retargeting (method: {method_name})")
            skeleton_traj = load_retargeted_amass_trajectory(
                env_name="MyoFullBody",
                dataset_name=motion_spec,
                robot_conf=robot_conf_skeleton,
                retargeting_method=retargeting_method,
                gmr_config=gmr_config,
                clear_cache=clear_cache,
            )

            logger.info("Stage 2: Extending and retargeting to MyoBimanualArm via environment forward kinematics")
            bimanual_traj = extend_motion(BIMANUAL_ENV_NAME, robot_conf_bimanual.env_params, skeleton_traj, logger)

            bimanual_traj.save(cache_path)
            logger.info(f"Saved MyoBimanualArm trajectory cache to {cache_path}")
            loaded_sets.append(LoadedTrajectorySet(bimanual_traj, (motion_name,)))
        except GMRQualityGateError as error:
            skipped += 1
            rejection_path = _write_gmr_quality_rejection(
                cache_path,
                motion_spec,
                error,
                gmr_config=gmr_config,
            )
            logger.warning("Saved bimanual GMR quality rejection to %s", rejection_path)
        except Exception as e:
            skipped += 1
            logger.error(f"Skipping dataset '{motion_spec.cache_key}' due to failure in three-stage retargeting: {e}")
            continue

    if len(loaded_sets) == 1:
        loaded = loaded_sets[0]
    else:
        if not loaded_sets:
            raise RuntimeError(f"Three-stage retargeting produced no valid trajectories. Skipped {skipped} dataset(s).")
        logger.info(f"Concatenating {len(loaded_sets)} final MyoBimanualArm trajectories (skipped {skipped}) ...")
        loaded = LoadedTrajectorySet.concatenate(loaded_sets, backend=np)
        logger.info("Final concatenation successful!")

    logger.info("Three-stage retargeting completed successfully")
    return loaded
