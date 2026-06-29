import glob
import hashlib
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
from loco_mujoco.core.utils.math import quat_scalarlast2scalarfirst
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from loco_mujoco.datasets.data_generation.utils import add_mocap_bodies
from loco_mujoco.smpl import SMPLH_BONE_ORDER_NAMES
from loco_mujoco.smpl.utils.smoothing import gaussian_filter_1d_batch
from loco_mujoco.trajectory import (
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


def _fps_after_frame_skip(source_fps, skip: int) -> float:
    if skip <= 0:
        raise ValueError(f"skip must be positive, got {skip}")
    return float(np.asarray(source_fps).squeeze()) / float(skip)


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


def load_amass_data(data_path: str) -> dict:
    """Load AMASS data from a file.

    Args:
        data_path (str): Path to the AMASS data file.


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
    pose_aa = np.concatenate([entry_data["poses"][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
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
    fitted_dir = Path(path_to_converted_amass_datasets) / cache_env_name / "gmr"
    fitted_dir.mkdir(parents=True, exist_ok=True)
    return str(fitted_dir)


def get_gmr_fitted_shape_path(env_name: str, gmr_robot: str) -> str:
    """Get the GMR fitted shape path for a robot.

    Args:
        env_name: Environment name (e.g., "MyoFullBody")
        gmr_robot: GMR robot name (e.g., "myofullbody")

    Returns:
        Path to the fitted shape .pkl file under gmr/ subfolder
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


def fit_gmr_motion(
    env_name: str,
    robot_conf: DictConfig,
    motion_data: str | dict,
    logger: logging.Logger,
    gmr_config: dict | None = None,
) -> tuple[Trajectory, dict]:
    """
    Fit SMPL-H motion data to a robot using General Motion Retargeting (GMR).

    An alternative to fit_smpl_motion() using GMR's IK-based retargeting.

    Args:
        env_name: Environment name
        robot_conf: Robot configuration
        motion_data: AMASS file path (str)
        logger: Logger instance
        gmr_config: GMR configuration dict (optional)
            offset_to_ground: If True, finds lowest foot and offsets body to ground.

    Returns:
        tuple[Trajectory, dict]: Trajectory at GMR's native fps (~30Hz) and analysis data
            containing pos_error and retarget_fps. Use extend_motion() to get model-aligned source data.

    Note:
        The GMR package's offset_human_data_to_ground method uses a hardcoded ground_offset.
        For feet-on-ground behavior matching LocoMuJoCo, modify the installed package:
        .venv/lib/.../general_motion_retargeting/motion_retarget.py line ~289:
        Change: we find lowest body parts and reset the frame during ground_offset calculations
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
    src_human = gmr_config.get("src_human", "smplh")
    target_fps = gmr_config.get("target_fps", 30)
    solver = gmr_config.get("solver", "daqp")
    damping = gmr_config.get("damping", 0.5)
    offset_to_ground = gmr_config.get("offset_to_ground", True)
    use_velocity_limit = gmr_config.get("use_velocity_limit", False)
    verbose = gmr_config.get("verbose", False)
    use_fitted_shape = gmr_config.get("use_fitted_shape", True)  # Default to True
    shape_fitting_iterations = gmr_config.get("shape_fitting_iterations", 500)

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

        # Local IK config path
        myofullbody_ik_config = Path(__file__).parent / "gmr_configs" / "smplh_to_myofullbody.json"
        if not myofullbody_ik_config.exists():
            raise FileNotFoundError(f"MyoFullBody IK config not found: {myofullbody_ik_config}")

        # Inject into GMR's dictionaries
        ROBOT_XML_DICT["myofullbody"] = myofullbody_xml
        IK_CONFIG_DICT.setdefault("smplh", {})["myofullbody"] = myofullbody_ik_config
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

    # Load SMPL-H motion (with fitted shape if available)
    logger.info(f"Loading SMPL-H motion from {motion_data}")
    smplh_data, body_model, smplh_output, actual_human_height = load_smplh_file(
        motion_data, smpl_model_path, fitted_shape_path=fitted_shape_path
    )

    # Align FPS
    logger.info(f"Aligning to {target_fps} fps...")
    smplh_frames, aligned_fps = get_smplh_data_offline_fast(smplh_data, body_model, smplh_output, tgt_fps=target_fps)
    logger.info(f"Aligned: {aligned_fps:.2f} fps, {len(smplh_frames)} frames")

    # Initialize GMR
    logger.info(f"GMR init: robot={gmr_robot}, solver={solver}, use_fitted_shape={use_fitted_shape}")
    retarget = GeneralMotionRetargeting(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=gmr_robot,
        solver=solver,
        damping=damping,
        use_velocity_limit=use_velocity_limit,
        verbose=verbose,
        use_fitted_shape=use_fitted_shape,
        fitted_shape_path=fitted_shape_path,
    )

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
    lowest_z_list = []
    model = retarget.configuration.model
    data = retarget.configuration.data
    traj_site_names = getattr(env, "sites_for_mimic", None)
    traj_site_ids = None
    if traj_site_names is not None:
        traj_site_ids = _resolve_site_ids(model, traj_site_names, context=f"{env_name} GMR trajectory sites")

    non_floor_geom_ids = [gid for gid in range(model.ngeom) if model.geom(gid).name != "floor"]
    use_ids = non_floor_geom_ids if non_floor_geom_ids else list(range(model.ngeom))

    t_start = time.perf_counter()

    for i, frame in enumerate(smplh_frames):
        if i % 30 == 0 and i > 0:
            logger.info(f"  {i}/{len(smplh_frames)} frames")
        qpos_frame, dist = retarget.retarget(frame, offset_to_ground=offset_to_ground)

        data.qpos[:] = qpos_frame
        mujoco.mj_forward(model, data)

        lowest_z = float(np.min(data.geom_xpos[use_ids, 2]))

        qpos_list.append(qpos_frame.copy())
        dist_list.append(dist.copy())
        lowest_z_list.append(lowest_z)

    t_total = time.perf_counter() - t_start
    retarget_fps = len(smplh_frames) / t_total if t_total > 0 else float("inf")

    logger.info(
        f"[OK] GMR retargeting complete: "
        f"{len(smplh_frames)} frames in {t_total:.2f}s "
        f"({retarget_fps:.2f} FPS, {1000 / retarget_fps:.2f} ms/frame)"
    )

    qpos = np.asarray(qpos_list)
    dist_array = np.asarray(dist_list)

    global_lowest_geom_z = float(min(lowest_z_list))

    if global_lowest_geom_z > 0.0:
        qpos[:, 2] -= global_lowest_geom_z

    for i in range(len(qpos)):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)

        pen, _geom_id, _floor_id = max_penetration_with_floor(model, data)

        if pen < 0.0:
            qpos[i, 2] -= pen

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
        "pos_error": dist_array,
        "retarget_fps": retarget_fps,
        "site_names": getattr(env, "sites_for_mimic", None),
    }

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


def extend_motion(
    env_name: str, env_params: DictConfig, traj: Trajectory, logger: logging.Logger | None = None
) -> Trajectory:
    """
    Extend a motion trajectory to include more model-specific entities
    like body xpos, site positions, etc. and to match the environment's frequency.

    Args:
        env_name (str): Name of the environment.
        env_params (DictConfig): Environment params.
        traj (Trajectory): The original trajectory data.
        logger (logging.Logger): Logger for status updates.

    Returns:
        Trajectory: The extended trajectory.

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

    return traj


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
) -> str:
    resolved_cache_path = _resolve_cache_path(cache_path)
    if clear_cache or os.path.exists(resolved_cache_path):
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


def _load_trajectories_individually(
    env_name: str,
    dataset_list: list[str],
    robot_conf: DictConfig,
    logger,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
) -> Trajectory:
    """
    Process each dataset individually and concatenate only final trajectories.
    This avoids dimensional incompatibilities during intermediate concatenation.
    """
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()
    # Normalize environment name for cache consistency (MJX and non-MJX share same cache)
    cache_env_name = env_name.replace("Mjx", "") if "Mjx" in env_name else env_name

    # Use separate cache directories for different retargeting methods
    if retargeting_method == "gmr":
        path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, cache_env_name, "gmr")
    else:
        path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, cache_env_name)
    os.makedirs(path_robot_smpl_data, exist_ok=True)

    final_trajectories = []

    for i, single_dataset in enumerate(dataset_list):
        cache_path = os.path.join(path_robot_smpl_data, f"{single_dataset}.npz")
        if retargeting_method == "gmr":
            cache_path = _prepare_gmr_cache_path(
                cache_path=cache_path,
                env_name=cache_env_name,
                dataset_name=single_dataset,
                cache_root=path_to_converted_amass_datasets,
                clear_cache=clear_cache,
                logger=logger,
            )
        else:
            cache_path = _resolve_cache_path(cache_path)

        if os.path.exists(cache_path) and not clear_cache:
            logger.info(
                f"Dataset {i + 1}/{len(dataset_list)}: Found existing retargeted motion file at {cache_path}. Loading ..."
            )
            # Use NumPy to avoid GPU OOM; to_jax() called later in instantiate_env()
            final_trajectories.append(Trajectory.load(cache_path, backend=np))
            continue

        method_name = retargeting_method.upper() if retargeting_method else "SMPL"
        action = "Re-retargeting (clear_cache)" if clear_cache and os.path.exists(cache_path) else "Retargeting"
        logger.info(f"Dataset {i + 1}/{len(dataset_list)}: {action} AMASS motion file using {method_name} ...")

        try:
            single_trajectory = load_retargeted_amass_trajectory(
                env_name,
                single_dataset,
                robot_conf,
                retargeting_method=retargeting_method,
                gmr_config=gmr_config,
                clear_cache=clear_cache,
            )
            final_trajectories.append(single_trajectory)
        except Exception as e:
            logger.error(f"Skipping dataset '{single_dataset}' due to failure during retargeting: {e}")
            continue

    # Concatenate final trajectories
    if len(final_trajectories) == 1:
        result_trajectory = final_trajectories[0]
    else:
        logger.info(f"Concatenating {len(final_trajectories)} final {env_name} trajectories ...")
        result_trajectory = Trajectory.concatenate(final_trajectories, backend=np)
        logger.info("Final concatenation successful!")

    return result_trajectory


def load_retargeted_amass_trajectory(
    env_name: str,
    dataset_name: str | list[str],
    robot_conf: DictConfig = None,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
) -> Trajectory:
    """
    Load a retargeted AMASS trajectory for a specific environment.

    Args:
        env_name: Name of the environment.
        dataset_name: Dataset name(s) to process.
        robot_conf: Robot configuration (optional).
        retargeting_method: "smpl" or "gmr" (optional, overrides robot_conf).
        gmr_config: GMR configuration dict (optional, overrides robot_conf).
        clear_cache: If True, overwrite existing cached files instead of loading them.

    Returns:
        Trajectory: The retargeted trajectories.

    """
    logger = setup_logger("amass", identifier="[MuscleMimic AMASS Retargeting Pipeline]")

    # if robot_conf is not provided, load default one it from the YAML file
    if robot_conf is None:
        robot_conf = load_robot_conf_file(env_name)

    # Check retargeting method
    if retargeting_method is None:
        retargeting_method = robot_conf.get("retargeting_method", "smpl")
    if gmr_config is None:
        gmr_config = robot_conf.get("gmr_config", None)

    # Process datasets individually to avoid trajectory concatenation with different traj info
    if isinstance(dataset_name, list | tuple) and len(dataset_name) > 1:
        return _load_trajectories_individually(
            env_name,
            dataset_name,
            robot_conf,
            logger,
            retargeting_method=retargeting_method,
            gmr_config=gmr_config,
            clear_cache=clear_cache,
        )

    path_to_converted_amass_datasets = get_converted_amass_dataset_path()

    # Normalize environment name for cache consistency (MJX and non-MJX share same cache)
    cache_env_name = env_name.replace("Mjx", "") if "Mjx" in env_name else env_name

    # Use separate cache directories for different retargeting methods
    if retargeting_method == "gmr":
        path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, cache_env_name, "gmr")
    else:
        path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, cache_env_name)

    if not os.path.exists(path_robot_smpl_data):
        os.makedirs(path_robot_smpl_data, exist_ok=True)

    # load trajectory file(s)
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    all_trajectories = []
    for i, d_name in enumerate(dataset_name):
        d_path = os.path.join(path_robot_smpl_data, f"{d_name}.npz")
        if retargeting_method == "gmr":
            d_path = _prepare_gmr_cache_path(
                cache_path=d_path,
                env_name=cache_env_name,
                dataset_name=d_name,
                cache_root=path_to_converted_amass_datasets,
                clear_cache=clear_cache,
                logger=logger,
            )
        else:
            d_path = _resolve_cache_path(d_path)
        cache_exists = os.path.exists(d_path)
        if not cache_exists or clear_cache:
            # Only check imports when we actually need to run retargeting
            check_optional_imports()
            path_to_smpl_model = get_smpl_model_path()
            smpl_optimized_shape_cache_path = os.path.join(
                path_to_converted_amass_datasets, cache_env_name, OPTIMIZED_SHAPE_FILE_NAME
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
                logger.info(f"Dataset {i + 1}/{len(dataset_name)}: {action} AMASS motion file using GMR ...")
                amass_file_path = os.path.join(get_amass_dataset_path(), f"{d_name}.npz")
                trajectory, analysis = fit_gmr_motion(env_name, robot_conf, amass_file_path, logger, gmr_config)
            else:
                logger.info(
                    f"Dataset {i + 1}/{len(dataset_name)}: {action} AMASS motion file using optimized body shape ..."
                )
                motion_data = load_amass_data(d_name)
                path_converted_shape = os.path.join(
                    path_to_converted_amass_datasets, f"{cache_env_name}/{OPTIMIZED_SHAPE_FILE_NAME}"
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
            trajectory = extend_motion(env_name, robot_conf.env_params, trajectory, logger)
            trajectory.save(d_path)

            # Save analysis data to file
            analysis_path = d_path.replace(".npz", "_analysis.npz")
            np.savez(analysis_path, **analysis)
            logger.info(f"Saved analysis to {analysis_path}")
            all_trajectories.append(trajectory)
        else:
            logger.info(
                f"Dataset {i + 1}/{len(dataset_name)}: Found existing retargeted motion file at {d_path}. Loading ..."
            )
            trajectory = Trajectory.load(d_path)
            all_trajectories.append(trajectory)

    if len(all_trajectories) == 1:
        trajectory = all_trajectories[0]
    else:
        logger.info("Concatenating trajectories ...")
        traj_datas = [t.data for t in all_trajectories]
        traj_infos = [t.info for t in all_trajectories]
        traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos, backend=np)
        trajectory = Trajectory(traj_info, traj_data)

    logger.info("Trajectory data loaded!")

    return trajectory


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
    dataset_name: str | list[str],
    robot_conf_skeleton: DictConfig = None,
    robot_conf_bimanual: DictConfig = None,
    retargeting_method: str | None = None,
    gmr_config: dict | None = None,
    clear_cache: bool = False,
) -> Trajectory:
    """
    Perform explicit three-stage retargeting: SMPL → MyoFullBody → MyoBimanualArm → SiteData.

    This function processes each motion individually through the complete pipeline,
    then concatenates the final MyoBimanualArm trajectories. This avoids dimensional
    incompatibilities that occur when trying to concatenate intermediate trajectories.

    Args:
        dataset_name: Name(s) of AMASS dataset(s) to retarget
        robot_conf_skeleton: Configuration for the intermediate full-body skeleton retargeting stage
        robot_conf_bimanual: Configuration for MyoBimanualArm (if None, loads default)
        retargeting_method: "smpl" or "gmr" (if None, defaults to SMPL optimization-based)
        gmr_config: GMR configuration dict (optional, only used if retargeting_method="gmr")
        clear_cache: If True, overwrite existing cached files instead of loading them.

    Returns:
        Trajectory: Retargeted trajectory for MyoBimanualArm
    """
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
    if isinstance(dataset_name, str):
        dataset_list = [dataset_name]
    else:
        dataset_list = list(dataset_name)

    # Determine cache directory for MyoBimanualArm trajectories
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()
    path_robot_smpl_data_bimanual = os.path.join(path_to_converted_amass_datasets, BIMANUAL_ENV_NAME)
    os.makedirs(path_robot_smpl_data_bimanual, exist_ok=True)

    # Process each motion individually through the complete three-stage pipeline,
    # caching a single file per dataset (consistent with other models)
    final_trajectories = []
    skipped = 0
    if retargeting_method == "gmr":
        path_robot_smpl_data_bimanual = os.path.join(path_to_converted_amass_datasets, BIMANUAL_ENV_NAME, "gmr")
    else:
        path_robot_smpl_data_bimanual = os.path.join(path_to_converted_amass_datasets, BIMANUAL_ENV_NAME)

    # Now use it inside the loop
    for i, single_dataset in enumerate(tqdm(dataset_list, desc="Bimanual retargeting", unit="traj")):
        cache_path = os.path.join(path_robot_smpl_data_bimanual, f"{single_dataset}.npz")
        if retargeting_method == "gmr":
            cache_path = _prepare_gmr_cache_path(
                cache_path=cache_path,
                env_name=BIMANUAL_ENV_NAME,
                dataset_name=single_dataset,
                cache_root=path_to_converted_amass_datasets,
                clear_cache=clear_cache,
                logger=logger,
            )
        else:
            cache_path = _resolve_cache_path(cache_path)

        cache_exists = os.path.exists(cache_path)
        if cache_exists and not clear_cache:
            logger.info(
                f"Dataset {i + 1}/{len(dataset_list)}: Found existing cached MyoBimanualArm trajectory at {cache_path}. Loading ..."
            )
            final_trajectories.append(Trajectory.load(cache_path))
            continue

        action = "Re-retargeting (clear_cache)" if clear_cache and cache_exists else "Retargeting"
        logger.info(f"Dataset {i + 1}/{len(dataset_list)}: {action} to MyoBimanualArm using three-stage pipeline ...")

        try:
            method_name = retargeting_method.upper() if retargeting_method else "SMPL"
            logger.info(f"Stage 1: SMPL → MyoFullBody retargeting (method: {method_name})")
            skeleton_traj = load_retargeted_amass_trajectory(
                env_name="MyoFullBody",
                dataset_name=single_dataset,
                robot_conf=robot_conf_skeleton,
                retargeting_method=retargeting_method,
                gmr_config=gmr_config,
            )

            logger.info("Stage 2: Extending and retargeting to MyoBimanualArm via environment forward kinematics")
            bimanual_traj = extend_motion(BIMANUAL_ENV_NAME, robot_conf_bimanual.env_params, skeleton_traj, logger)

            # Save per-dataset cache and collect
            bimanual_traj.save(cache_path)
            logger.info(f"Saved MyoBimanualArm trajectory cache to {cache_path}")
            final_trajectories.append(bimanual_traj)
        except Exception as e:
            skipped += 1
            logger.error(f"Skipping dataset '{single_dataset}' due to failure in three-stage retargeting: {e}")
            continue

    # Concatenate final trajectories in-memory for multi-dataset requests
    if len(final_trajectories) == 1:
        result_trajectory = final_trajectories[0]
    else:
        if len(final_trajectories) == 0:
            raise RuntimeError(f"Three-stage retargeting produced no valid trajectories. Skipped {skipped} dataset(s).")
        logger.info(
            f"Concatenating {len(final_trajectories)} final MyoBimanualArm trajectories (skipped {skipped}) ..."
        )
        result_trajectory = Trajectory.concatenate(final_trajectories, backend=np)
        logger.info("Final concatenation successful!")

    logger.info("Three-stage retargeting completed successfully")
    return result_trajectory
