from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import numpy as np

_SAFE_C3D_DATASET_PART_RE = re.compile(r"[^A-Za-z0-9._-]+")
_JSON_ANALYSIS_PREFIX = "__musclemimic_json__:"


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_key_sha256(cache_key: str) -> str:
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()


def _cache_value(value):
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return tuple((str(key), _cache_value(item)) for key, item in items)
    if isinstance(value, (list, tuple)):
        return tuple(_cache_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


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
        elif isinstance(value, (dict, list, tuple)):
            serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            np_ready[key] = np.array(_JSON_ANALYSIS_PREFIX + serialized)
        else:
            np_ready[key] = value
    np.savez(path, **np_ready)


def _coerce_analysis_npz_value(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, str) and value.startswith(_JSON_ANALYSIS_PREFIX):
        return json.loads(value.removeprefix(_JSON_ANALYSIS_PREFIX))
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
    source_c3d_sha256: str,
    cache_key_sha256: str,
    dataset_name: str,
    retargeting_method: str,
    logger: logging.Logger | None = None,
):
    from loco_mujoco.trajectory import Trajectory

    analysis = _load_analysis_npz(analysis_path, logger=logger)
    cached_source_sha256 = analysis.get("source_c3d_sha256")
    if not isinstance(cached_source_sha256, str):
        raise RuntimeError(
            f"Converted C3D cache at {motion_path} has no source checksum. "
            "Pass --clear-c3d-cache to rebuild it."
        )
    if cached_source_sha256 != source_c3d_sha256:
        raise RuntimeError(
            f"Converted C3D cache at {motion_path} was built from different source contents. "
            "Pass --clear-c3d-cache to rebuild it."
        )
    cached_key_sha256 = analysis.get("c3d_cache_key_sha256")
    if not isinstance(cached_key_sha256, str):
        raise RuntimeError(
            f"Converted C3D cache at {motion_path} has no fitting-configuration checksum. "
            "Pass --clear-c3d-cache to rebuild it."
        )
    if cached_key_sha256 != cache_key_sha256:
        raise RuntimeError(
            f"Converted C3D cache at {motion_path} was built with different fitting or "
            "retargeting configuration. Pass --clear-c3d-cache to rebuild it."
        )
    try:
        trajectory = Trajectory.load(motion_path, backend=np)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load converted C3D trajectory cache at {motion_path}. Pass --clear-c3d-cache to rebuild it."
        ) from exc
    analysis.update(
        {
            "source": "c3d",
            "source_c3d_path": str(c3d_file.expanduser().resolve(strict=False)),
            "source_c3d_sha256": source_c3d_sha256,
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
    cache_root: str | Path | None = None,
) -> tuple[Path, Path, str]:
    converted_root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else get_converted_c3d_dataset_path()
    )
    cache_env = model_name.replace("Mjx", "") if "Mjx" in model_name else model_name
    dataset_name = (
        _normalize_c3d_dataset_name(converted_c3d_name)
        if converted_c3d_name is not None
        else _default_c3d_dataset_name(c3d_file, cache_key=cache_key)
    )
    motion_path = converted_root / cache_env / retargeting_method / dataset_name.with_suffix(".npz")
    analysis_path = motion_path.with_name(f"{motion_path.stem}_analysis.npz")
    return motion_path, analysis_path, dataset_name.as_posix()


def _c3d_fit_cache_dir(
    source_c3d_sha256: str,
    cache_root: str | Path | None,
) -> Path:
    """Keep fit caches content-addressed and outside the source directory."""

    managed_root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else get_converted_c3d_dataset_path()
    )
    return managed_root / ".c3d_smpl_cache" / source_c3d_sha256


def _c3d_cache_key(
    c3d_file: Path,
    source_c3d_sha256: str,
    model_name: str,
    retargeting_method: str,
    *,
    gmr_config: dict | None,
    terra_config: dict | None,
    fit_options: dict[str, object],
    c3d_fit_model_path: str,
    c3d_fit_model_file: Path | None,
    retarget_smpl_model_path: str,
) -> str:
    input_checksums = {
        name: _file_sha256(Path(value))
        for name in ("head_marker_corr_path", "pose_body_prior_path", "stage1_state_path")
        if (value := fit_options.get(name)) is not None and Path(value).is_file()
    }
    items = {
        "c3d_path": str(c3d_file.expanduser().resolve(strict=False)),
        "c3d_sha256": source_c3d_sha256,
        "model_name": model_name,
        "retargeting_method": retargeting_method,
        "gmr_config": _cache_value(gmr_config or {}),
        "terra_config": _cache_value(terra_config or {}),
        "fit_options": _cache_value(fit_options),
        "fit_input_sha256": _cache_value(input_checksums),
        "c3d_fit_model_path": str(Path(c3d_fit_model_path).expanduser().resolve(strict=False)),
        "c3d_fit_model_file": str(c3d_fit_model_file) if c3d_fit_model_file is not None else None,
        "c3d_fit_model_sha256": _file_sha256(c3d_fit_model_file) if c3d_fit_model_file is not None else None,
        "retarget_smpl_model_path": str(Path(retarget_smpl_model_path).expanduser().resolve(strict=False)),
    }
    return repr(items)


def _c3d_fit_summary(motion_data: Mapping[str, object]) -> dict[str, object] | None:
    debug = motion_data.get("debug")
    if not isinstance(debug, Mapping):
        return None
    marker_error = debug.get("marker_error")
    stage1 = debug.get("stagei")
    labels = debug.get("canonical_labels")
    summary: dict[str, object] = {
        "surface_model_type": debug.get("surface_model_type"),
        "canonical_labels": list(labels) if isinstance(labels, (list, tuple)) else None,
        "fit_pose_dim": debug.get("fit_pose_dim"),
        "pose_prior": debug.get("pose_prior"),
        "stage1_shape_solver": debug.get("stage1_shape_solver"),
        "stage2_solver": debug.get("stage2_solver"),
        "enforce_knee_hinge": debug.get("enforce_knee_hinge"),
        "stage2_marker_weight_overrides": debug.get("stage2_marker_weight_overrides"),
        "stage2_torso_frame_weight": debug.get("stage2_torso_frame_weight"),
    }
    if isinstance(marker_error, Mapping):
        summary["marker_error"] = {
            str(name): value.item() if isinstance(value, np.generic) else value
            for name, value in marker_error.items()
        }
    if isinstance(stage1, Mapping):
        summary["stage1"] = {
            "reused": bool(stage1.get("reused", False)),
            "state_path": stage1.get("state_path"),
        }
    return summary


def retarget_c3d_to_trajectory(
    c3d_path: str,
    model_name: str,
    *,
    retargeting_method: str = "gmr",
    gmr_config: dict | None = None,
    terra_config: dict | None = None,
    logger: logging.Logger | None = None,
    optimize_toes: bool = False,
    pose_body_prior_path: str | None = None,
    head_marker_corr_path: str | None = None,
    surface_model_type: str = "smplx",
    gender: str = "male",
    c3d_fit_model_path: str | None = None,
    retarget_smpl_model_path: str | None = None,
    stage1_shape_solver: str = "joint_dogleg_jax",
    stage1_iters: int = 320,
    stage2_iters: int = 80,
    n_ref_frames: int = 12,
    target_fps: float | None = None,
    device: str = "cpu",
    seed: int = 100,
    least_avail_markers: float = 1.0,
    strict_frame_picking: bool = True,
    wrist_markers_on_stick: bool = False,
    stage1_state_path: str | None = None,
    stage2_solver: str = "frame_lbfgs",
    enforce_knee_hinge: bool = False,
    stage2_marker_weight_overrides: dict[str, float] | None = None,
    stage2_torso_frame_weight: float = 0.0,
    converted_c3d_name: str | Path | None = None,
    cache_root: str | Path | None = None,
    clear_cache: bool = False,
):
    """Fit C3D markers to a SMPL-family surface, then retarget to MuscleMimic."""
    supported_methods = {"gmr", "smpl", "terra", "omniretarget"}
    if retargeting_method not in supported_methods:
        raise ValueError(
            f"Unsupported C3D retargeting method {retargeting_method!r}; "
            f"choose one of {sorted(supported_methods)}"
        )
    from musclemimic.environments import LocoEnv

    if model_name not in LocoEnv.registered_envs:
        supported = ", ".join(sorted(LocoEnv.registered_envs))
        raise ValueError(f"unknown MuscleMimic environment {model_name!r}; registered environments: {supported}")
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
    if stage1_state_path is not None:
        stage1_state = Path(stage1_state_path).expanduser().resolve()
        if not stage1_state.is_file():
            raise FileNotFoundError(f"C3D Stage-I state not found: {stage1_state}")
        stage1_state_path = str(stage1_state)

    fit_options: dict[str, object] = {
        "device": device,
        "enforce_knee_hinge": enforce_knee_hinge,
        "gender": gender,
        "head_marker_corr_path": head_marker_corr_path,
        "least_avail_markers": least_avail_markers,
        "n_ref_frames": n_ref_frames,
        "optimize_toes": optimize_toes,
        "pose_body_prior_path": pose_body_prior_path,
        "seed": seed,
        "stage1_iters": stage1_iters,
        "stage1_shape_solver": stage1_shape_solver,
        "stage1_state_path": stage1_state_path,
        "stage2_iters": stage2_iters,
        "stage2_marker_weight_overrides": stage2_marker_weight_overrides,
        "stage2_solver": stage2_solver,
        "stage2_torso_frame_weight": stage2_torso_frame_weight,
        "strict_frame_picking": strict_frame_picking,
        "surface_model_type": surface_model_type,
        "target_fps": target_fps,
        "wrist_markers_on_stick": wrist_markers_on_stick,
    }

    from .c3d_to_smpl import _surface_model_asset_path

    c3d_fit_model_file = _surface_model_asset_path(
        c3d_fit_model_path,
        surface_model_type,
        gender,
    )

    source_c3d_sha256 = _file_sha256(c3d_file)
    fit_cache_dir = _c3d_fit_cache_dir(source_c3d_sha256, cache_root)
    cache_key = _c3d_cache_key(
        c3d_file,
        source_c3d_sha256,
        model_name,
        retargeting_method,
        gmr_config=gmr_config,
        terra_config=terra_config,
        fit_options=fit_options,
        c3d_fit_model_path=c3d_fit_model_path,
        c3d_fit_model_file=c3d_fit_model_file,
        retarget_smpl_model_path=retarget_smpl_model_path,
    )
    cache_key_sha256 = _cache_key_sha256(cache_key)
    motion_path, analysis_path, dataset_name = _converted_c3d_output_paths(
        c3d_file,
        model_name,
        retargeting_method,
        converted_c3d_name=converted_c3d_name,
        cache_key=cache_key,
        cache_root=cache_root,
    )
    if motion_path.exists() and not clear_cache:
        log.info("Loading converted C3D trajectory cache: %s", motion_path)
        return _load_converted_c3d_cache(
            motion_path=motion_path,
            analysis_path=analysis_path,
            c3d_file=c3d_file,
            source_c3d_sha256=source_c3d_sha256,
            cache_key_sha256=cache_key_sha256,
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
    if retargeting_method != "gmr":
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
        cache_dir=str(fit_cache_dir),
        clear_cache=clear_cache,
        **fit_options,
    )

    log.info("[2/4] Retargeting SMPL -> %s...", model_name)
    robot_conf = load_robot_conf_file(model_name)

    shape_path = None
    if retargeting_method != "gmr":
        log.info("[2/4] Retargeting SMPL model path: %s", retarget_smpl_model_path)
        cache_env = model_name.replace("Mjx", "") if "Mjx" in model_name else model_name
        shape_root = (
            Path(cache_root).expanduser().resolve()
            if cache_root is not None
            else Path(get_converted_amass_dataset_path())
        )
        shape_path = str(shape_root / cache_env / OPTIMIZED_SHAPE_FILE_NAME)
        if not os.path.exists(shape_path):
            log.info("Fitting SMPL shape to robot (one-time)...")
            os.makedirs(os.path.dirname(shape_path), exist_ok=True)
            fit_smpl_shape(model_name, robot_conf, retarget_smpl_model_path, shape_path, log)

    if retargeting_method == "gmr":
        gmr_motion_path = save_motion_data_as_amass_smplh_npz(
            motion_data,
            fit_cache_dir / f"{c3d_file.stem}_smplh_for_gmr.npz",
        )
        trajectory, analysis = fit_gmr_motion(
            model_name,
            robot_conf,
            str(gmr_motion_path),
            log,
            gmr_config or {},
        )
    elif retargeting_method in {"terra", "omniretarget"}:
        import terra

        resolved_config = terra.resolve_method_profile(retargeting_method, terra_config)
        trajectory, analysis = terra.fit_terra_motion(
            model_name,
            robot_conf,
            motion_data,
            log,
            resolved_config,
            smpl_model_path=retarget_smpl_model_path,
            fitted_shape_path=shape_path,
        )
    else:
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
    fit_summary = _c3d_fit_summary(motion_data)
    analysis.update(
        {
            "source": "c3d",
            "source_c3d_path": str(c3d_file.expanduser().resolve(strict=False)),
            "source_c3d_sha256": source_c3d_sha256,
            "converted_c3d_path": str(motion_path),
            "converted_c3d_analysis_path": str(analysis_path),
            "converted_c3d_dataset": dataset_name,
            "retargeting_method": retargeting_method,
            "c3d_fit_model_path": str(
                Path(c3d_fit_model_path).expanduser().resolve(strict=False)
            ),
            "c3d_fit_model_file": str(c3d_fit_model_file) if c3d_fit_model_file is not None else None,
            "c3d_fit_model_sha256": (
                _file_sha256(c3d_fit_model_file) if c3d_fit_model_file is not None else None
            ),
            "c3d_surface_model_type": surface_model_type,
            "c3d_gender": gender,
            "c3d_cache_key_sha256": cache_key_sha256,
            "c3d_fit_cache_dir": str(fit_cache_dir),
            "c3d_fit_options": fit_options,
            "smpl_model_path": str(
                Path(retarget_smpl_model_path).expanduser().resolve(strict=False)
            ),
        }
    )
    if fit_summary is not None:
        analysis["c3d_fit"] = fit_summary
    if shape_path is not None:
        analysis.update(
            fitted_shape_path=str(Path(shape_path).expanduser().resolve(strict=False)),
            fitted_shape_sha256=_file_sha256(Path(shape_path)),
        )
    _save_analysis_npz(analysis_path, analysis)
    log.info("Saved converted C3D analysis: %s", analysis_path)
    log.info("Retargeting complete. Use converted C3D dataset name: %s", dataset_name)
    return trajectory, analysis
