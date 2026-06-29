from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path, PurePosixPath

import numpy as np

_SAFE_C3D_DATASET_PART_RE = re.compile(r"[^A-Za-z0-9._-]+")


def get_converted_c3d_dataset_path() -> Path:
    """Return the root for converted C3D-derived MuscleMimic trajectories."""
    import loco_mujoco

    path = os.environ.get("CONVERTED_C3D_PATH") or os.environ.get("MUSCLEMIMIC_CONVERTED_C3D_PATH")
    if path:
        return Path(path).expanduser()

    path_config = loco_mujoco.load_path_config()
    path = path_config.get("CONVERTED_C3D_PATH") or path_config.get("MUSCLEMIMIC_CONVERTED_C3D_PATH")
    if path:
        return Path(path).expanduser()

    return loco_mujoco.get_musclemimic_home() / "caches" / "C3D"


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def _safe_c3d_dataset_part(part: str) -> str:
    safe = _SAFE_C3D_DATASET_PART_RE.sub("_", part).strip("._")
    if not safe:
        raise ValueError(f"C3D dataset name contains an empty path component after sanitization: {part!r}")
    return safe


def normalize_c3d_dataset_name(name: str | Path) -> Path:
    raw = str(name).replace("\\", "/").strip()
    if not raw:
        raise ValueError("C3D dataset name cannot be empty.")

    rel = PurePosixPath(raw)
    if not rel.parts or rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"C3D dataset name must be a safe relative path, got {name!r}")

    if rel.suffix.lower() in {".npz", ".c3d"}:
        rel = rel.with_suffix("")

    parts = [_safe_c3d_dataset_part(part) for part in rel.parts]
    return Path(*parts)


_normalize_c3d_dataset_name = normalize_c3d_dataset_name


def _default_c3d_dataset_name(c3d_file: Path, *, cache_key: str) -> Path:
    resolved = c3d_file.expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        relative = Path(f"{resolved.stem}_{_short_hash(str(resolved))}")
    else:
        relative = relative.with_suffix("")

    dataset_name = _normalize_c3d_dataset_name(relative)
    return dataset_name.with_name(f"{dataset_name.name}_{_short_hash(cache_key)}")


def _robot_conf_env_params(robot_conf) -> dict:
    if isinstance(robot_conf, dict):
        return robot_conf.get("env_params", {})
    return getattr(robot_conf, "env_params", {})


def _save_analysis_npz(path: Path, analysis: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np_ready = {}
    for key, value in analysis.items():
        if value is None:
            np_ready[key] = np.array("")
        elif isinstance(value, Path):
            np_ready[key] = np.array(str(value))
        else:
            np_ready[key] = value
    np.savez(path, **np_ready)


def _coerce_analysis_npz_value(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _load_analysis_npz(path: Path, logger: logging.Logger | None = None) -> dict:
    if not path.exists():
        return {}

    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: _coerce_analysis_npz_value(data[key]) for key in data.files}
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not load converted C3D analysis cache at %s: %s", path, exc)
        return {}


def _load_converted_c3d_cache(
    *,
    motion_path: Path,
    analysis_path: Path,
    c3d_file: Path,
    dataset_name: str,
    retargeting_method: str,
    logger: logging.Logger | None = None,
):
    from loco_mujoco.trajectory import Trajectory

    try:
        trajectory = Trajectory.load(motion_path, backend=np)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load converted C3D trajectory cache at {motion_path}. Pass --clear-c3d-cache to rebuild it."
        ) from exc
    analysis = _load_analysis_npz(analysis_path, logger=logger)
    analysis.update(
        {
            "source": "c3d",
            "source_c3d_path": str(c3d_file.expanduser().resolve(strict=False)),
            "converted_c3d_path": str(motion_path),
            "converted_c3d_analysis_path": str(analysis_path),
            "converted_c3d_dataset": dataset_name,
            "retargeting_method": retargeting_method,
        }
    )
    return trajectory, analysis


def _converted_c3d_output_paths(
    c3d_file: Path,
    model_name: str,
    retargeting_method: str,
    *,
    converted_c3d_name: str | Path | None,
    cache_key: str,
) -> tuple[Path, Path, str]:
    converted_root = get_converted_c3d_dataset_path()
    cache_env = model_name.replace("Mjx", "") if "Mjx" in model_name else model_name
    dataset_name = (
        _normalize_c3d_dataset_name(converted_c3d_name)
        if converted_c3d_name is not None
        else _default_c3d_dataset_name(c3d_file, cache_key=cache_key)
    )
    motion_path = converted_root / cache_env / retargeting_method / dataset_name.with_suffix(".npz")
    analysis_path = motion_path.with_name(f"{motion_path.stem}_analysis.npz")
    return motion_path, analysis_path, dataset_name.as_posix()


def _c3d_cache_key(
    c3d_file: Path,
    model_name: str,
    retargeting_method: str,
    *,
    gmr_config: dict | None,
    optimize_toes: bool,
    pose_body_prior_path: str | None,
    head_marker_corr_path: str | None,
    surface_model_type: str,
    gender: str,
    c3d_fit_model_path: str,
    retarget_smpl_model_path: str,
    stage1_shape_solver: str,
    stage1_iters: int,
    stage2_iters: int,
) -> str:
    items = {
        "c3d_path": str(c3d_file.expanduser().resolve(strict=False)),
        "model_name": model_name,
        "retargeting_method": retargeting_method,
        "gmr_config": sorted((gmr_config or {}).items()),
        "optimize_toes": optimize_toes,
        "pose_body_prior_path": str(Path(pose_body_prior_path).expanduser().resolve(strict=False))
        if pose_body_prior_path
        else None,
        "head_marker_corr_path": str(Path(head_marker_corr_path).expanduser().resolve(strict=False))
        if head_marker_corr_path
        else None,
        "surface_model_type": surface_model_type,
        "gender": gender,
        "c3d_fit_model_path": str(Path(c3d_fit_model_path).expanduser().resolve(strict=False)),
        "retarget_smpl_model_path": str(Path(retarget_smpl_model_path).expanduser().resolve(strict=False)),
        "stage1_shape_solver": stage1_shape_solver,
        "stage1_iters": stage1_iters,
        "stage2_iters": stage2_iters,
    }
    return repr(items)


def retarget_c3d_to_trajectory(
    c3d_path: str,
    model_name: str,
    *,
    retargeting_method: str = "gmr",
    gmr_config: dict | None = None,
    logger: logging.Logger | None = None,
    optimize_toes: bool = False,
    pose_body_prior_path: str | None = None,
    head_marker_corr_path: str | None = None,
    surface_model_type: str = "smplx",
    gender: str = "male",
    c3d_fit_model_path: str | None = None,
    retarget_smpl_model_path: str | None = None,
    stage1_shape_solver: str = "joint_dogleg_jax",
    stage1_iters: int = 4,
    stage2_iters: int = 80,
    converted_c3d_name: str | Path | None = None,
    clear_cache: bool = False,
):
    """Fit C3D markers to a SMPL-family surface, then retarget to MuscleMimic."""
    from loco_mujoco.smpl.retargeting import get_smpl_model_path

    log = logger or logging.getLogger("c3d_pipeline")
    c3d_file = Path(c3d_path)
    if not c3d_file.exists():
        raise FileNotFoundError(f"C3D file not found: {c3d_path}")

    from .c3d.asset_paths import resolve_c3d_model_path, resolve_pose_body_prior_path
    from .c3d.markers import resolve_stagei_head_marker_corr_path

    configured_smpl_model_path = get_smpl_model_path()
    resolved_c3d_model_path = resolve_c3d_model_path(c3d_fit_model_path)
    if resolved_c3d_model_path is None and surface_model_type == "smplh":
        resolved_c3d_model_path = Path(configured_smpl_model_path).expanduser()
    if resolved_c3d_model_path is None:
        raise FileNotFoundError(
            "C3D fitting model path is not configured. Pass --c3d-model-path or run: "
            "musclemimic-set-c3d-model-path --path /path/to/c3d-fitting/models"
        )
    c3d_fit_model_path = str(resolved_c3d_model_path)
    retarget_smpl_model_path = retarget_smpl_model_path or configured_smpl_model_path

    pose_body_prior = resolve_pose_body_prior_path(pose_body_prior_path)
    pose_body_prior_path = str(pose_body_prior) if pose_body_prior is not None else None
    head_marker_corr_path = resolve_stagei_head_marker_corr_path(
        head_marker_corr_path,
        smpl_model_path=c3d_fit_model_path,
        surface_model_type=surface_model_type,
    )

    cache_key = _c3d_cache_key(
        c3d_file,
        model_name,
        retargeting_method,
        gmr_config=gmr_config,
        optimize_toes=optimize_toes,
        pose_body_prior_path=pose_body_prior_path,
        head_marker_corr_path=head_marker_corr_path,
        surface_model_type=surface_model_type,
        gender=gender,
        c3d_fit_model_path=c3d_fit_model_path,
        retarget_smpl_model_path=retarget_smpl_model_path,
        stage1_shape_solver=stage1_shape_solver,
        stage1_iters=stage1_iters,
        stage2_iters=stage2_iters,
    )
    motion_path, analysis_path, dataset_name = _converted_c3d_output_paths(
        c3d_file,
        model_name,
        retargeting_method,
        converted_c3d_name=converted_c3d_name,
        cache_key=cache_key,
    )
    if motion_path.exists() and not clear_cache:
        log.info("Loading converted C3D trajectory cache: %s", motion_path)
        return _load_converted_c3d_cache(
            motion_path=motion_path,
            analysis_path=analysis_path,
            c3d_file=c3d_file,
            dataset_name=dataset_name,
            retargeting_method=retargeting_method,
            logger=log,
        )
    if motion_path.exists() and clear_cache:
        log.info("Ignoring converted C3D trajectory cache because clear_cache=True: %s", motion_path)

    if not Path(c3d_fit_model_path).exists():
        raise FileNotFoundError(
            f"C3D fitting model not found at {c3d_fit_model_path}. "
            "Pass --c3d-model-path or run: musclemimic-set-c3d-model-path --path <path>"
        )
    if retargeting_method == "smpl":
        if not Path(retarget_smpl_model_path).exists():
            raise FileNotFoundError(
                f"Retargeting SMPL model not found at {retarget_smpl_model_path}. "
                "Pass --retarget-smpl-model-path or run: musclemimic-set-smpl-model-path <path>"
            )
        if (
            surface_model_type == "smplx"
            and Path(c3d_fit_model_path).resolve() == Path(retarget_smpl_model_path).resolve()
        ):
            raise ValueError(
                "retargeting_method='smpl' cannot reuse the SMPL-X C3D fitting model path for robot retargeting. "
                "Pass --retarget-smpl-model-path pointing to the SMPL-H/SMPL model directory expected by retargeting."
            )

    from loco_mujoco.smpl.retargeting import (
        OPTIMIZED_SHAPE_FILE_NAME,
        extend_motion,
        fit_gmr_motion,
        fit_smpl_motion,
        fit_smpl_shape,
        get_converted_amass_dataset_path,
        load_robot_conf_file,
    )

    from .c3d_to_smpl import fit_smpl_to_c3d_cached, save_motion_data_as_amass_smplh_npz

    log.info(
        "[1/4] C3D fit model: surface_model_type=%s, gender=%s, path=%s, stage1_solver=%s",
        surface_model_type,
        gender,
        c3d_fit_model_path,
        stage1_shape_solver,
    )

    log.info("[1/4] Resolving C3D -> SMPL motion: %s", c3d_path)
    motion_data = fit_smpl_to_c3d_cached(
        str(c3d_file),
        c3d_fit_model_path,
        optimize_toes=optimize_toes,
        pose_body_prior_path=pose_body_prior_path,
        head_marker_corr_path=head_marker_corr_path,
        surface_model_type=surface_model_type,
        gender=gender,
        stage1_shape_solver=stage1_shape_solver,
        stage1_iters=stage1_iters,
        stage2_iters=stage2_iters,
        clear_cache=clear_cache,
    )

    log.info("[2/4] Retargeting SMPL -> %s...", model_name)
    robot_conf = load_robot_conf_file(model_name)

    if retargeting_method == "gmr":
        gmr_motion_path = save_motion_data_as_amass_smplh_npz(
            motion_data,
            c3d_file.parent / ".smpl_cache" / f"{c3d_file.stem}_smplh_for_gmr.npz",
        )
        trajectory, analysis = fit_gmr_motion(
            model_name,
            robot_conf,
            str(gmr_motion_path),
            log,
            gmr_config or {},
        )
    else:
        log.info("[2/4] Retargeting SMPL model path: %s", retarget_smpl_model_path)
        cache_env = model_name.replace("Mjx", "") if "Mjx" in model_name else model_name
        shape_path = os.path.join(get_converted_amass_dataset_path(), cache_env, OPTIMIZED_SHAPE_FILE_NAME)
        if not os.path.exists(shape_path):
            log.info("Fitting SMPL shape to robot (one-time)...")
            os.makedirs(os.path.dirname(shape_path), exist_ok=True)
            fit_smpl_shape(model_name, robot_conf, retarget_smpl_model_path, shape_path, log)

        trajectory, analysis = fit_smpl_motion(
            model_name,
            robot_conf,
            retarget_smpl_model_path,
            motion_data,
            shape_path,
            log,
        )

    log.info("[3/4] Extending trajectory for training...")
    trajectory = extend_motion(model_name, _robot_conf_env_params(robot_conf), trajectory, log)

    log.info("[4/4] Saving converted C3D trajectory: %s", motion_path)
    trajectory.save(str(motion_path))

    analysis = dict(analysis)
    analysis.update(
        {
            "source": "c3d",
            "source_c3d_path": str(c3d_file.expanduser().resolve(strict=False)),
            "converted_c3d_path": str(motion_path),
            "converted_c3d_analysis_path": str(analysis_path),
            "converted_c3d_dataset": dataset_name,
            "retargeting_method": retargeting_method,
        }
    )
    _save_analysis_npz(analysis_path, analysis)
    log.info("Saved converted C3D analysis: %s", analysis_path)
    log.info("Retargeting complete. Use converted C3D dataset name: %s", dataset_name)
    return trajectory, analysis
