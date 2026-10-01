#!/usr/bin/env python
"""Compare the current C3D Stage-I fit against a MoSh++ Stage-I pickle.

This is development tooling: it intentionally imports private fitter helpers so
production code does not need to carry MoSh++ pickle loading paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from dataclasses import dataclass
from inspect import signature
from pathlib import Path

import numpy as np
import yaml

from musclemimic.web_viewer.c3d.asset_paths import resolve_c3d_model_path, resolve_pose_body_prior_path
from musclemimic.web_viewer.c3d.markers import (
    PreparedMarkerObservations,
    build_marker_layout,
    compute_marker_availability_mask,
    load_c3d_markers,
    load_stagei_head_marker_correlation,
    pick_stagei_frames,
    prepare_marker_observations,
    resolve_stagei_head_marker_corr_path,
)
from musclemimic.web_viewer.c3d.pose_prior import MoshGmmPosePrior
from musclemimic.web_viewer.c3d.smpl_models import _make_surface_model
from musclemimic.web_viewer.c3d.surface_markers import SurfaceMarkerModel
from musclemimic.web_viewer.c3d_to_smpl import _fit_stagei

LOG = logging.getLogger("validate_stagei_against_moshpp")


@dataclass(frozen=True)
class ConfigValue:
    value: object
    source: str


@dataclass(frozen=True)
class StageIReference:
    path: Path
    cfg: object
    labels: list[str]
    betas: np.ndarray
    markers_latent: np.ndarray
    marker_vids: dict[str, int]
    ref_indices: np.ndarray | None
    surface_model_type: str
    gender: str
    c3d_path_hint: Path | None


@dataclass(frozen=True)
class StageIResult:
    labels: list[str]
    marker_vids: dict[str, int]
    ref_indices: np.ndarray
    betas: np.ndarray
    markers_latent: np.ndarray
    terms: dict[str, float]


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin-1")


def _load_config_file(path: Path | None) -> dict:
    if path is None:
        return {}
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix in {".pkl", ".pickle"}:
        data = _load_pickle(path)
    else:
        raise ValueError(f"Unsupported config file type: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _nested_get(obj, keys: tuple[str, ...], default=None):
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict) or hasattr(cur, "get"):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
        if cur is default:
            return default
    return cur


def _first_config_value(configs: list[tuple[str, dict]], paths: tuple[tuple[str, ...], ...]) -> ConfigValue | None:
    for source, cfg in configs:
        for path in paths:
            value = _nested_get(cfg, path)
            if value is not None and value != "":
                return ConfigValue(value=value, source=f"{source}:{'.'.join(path)}")
    return None


def _production_c3d_pipeline_default(name: str):
    from musclemimic.web_viewer.c3d_pipeline import retarget_c3d_to_trajectory

    return signature(retarget_c3d_to_trajectory).parameters[name].default


def _existing_path(value: object, *, base_dir: Path | None = None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    candidates = [path]
    if base_dir is not None and not path.is_absolute():
        candidates.insert(0, base_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_existing_path_value(
    value: ConfigValue | None,
    *,
    name: str,
    config_path: Path | None,
    required: bool,
    log_missing: bool = True,
) -> Path | None:
    if value is None:
        if required:
            raise SystemExit(f"Could not resolve {name}; pass it on the command line or in --config.")
        return None

    base_dir = config_path.expanduser().resolve(strict=False).parent if config_path is not None else None
    exact = _existing_path(value.value, base_dir=base_dir)
    if exact is not None:
        LOG.info("%s: %s (%s)", name, exact, value.source)
        return exact

    if required:
        raise SystemExit(f"Could not resolve {name} from {value.source}: {value.value}")
    if log_missing:
        LOG.info("%s: none (%s does not exist locally)", name, value.source)
    return None


def _coerce_marker_vids(value) -> dict[str, int]:
    marker_vids: dict[str, int] = {}
    if not isinstance(value, dict):
        return marker_vids
    for label, vid in value.items():
        arr = np.asarray(vid).reshape(-1)
        if arr.size:
            marker_vids[str(label)] = int(arr[0])
    return marker_vids


def _parse_stagei_ref_indices(stagei_data: dict) -> np.ndarray | None:
    fnames = stagei_data.get("stagei_debug_details", {}).get("stagei_fnames")
    if fnames is None:
        return None
    indices: list[int] = []
    for fname in fnames:
        match = re.search(r"_(\d+)$", str(fname))
        if match is None:
            return None
        indices.append(int(match.group(1)))
    return np.asarray(indices, dtype=np.int64)


def _local_c3d_from_stagei(stagei_path: Path, c3d_name: str | None) -> Path | None:
    trial_dir = stagei_path.parents[3] if len(stagei_path.parents) >= 4 else stagei_path.parent
    candidates = []
    if c3d_name:
        candidates.append(trial_dir / f"{Path(c3d_name).stem}.c3d")
    candidates.extend(sorted(trial_dir.glob("*.c3d")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_stagei_reference(stagei_path: str | Path) -> StageIReference:
    path = Path(stagei_path)
    data = _load_pickle(path)
    labels = [str(label) for label in data.get("latent_labels", [])]
    if not labels:
        raise ValueError(f"MoSh++ Stage-I pickle has no latent_labels: {path}")

    betas = np.asarray(data.get("betas"), dtype=np.float64).reshape(-1)
    markers_latent = np.asarray(data.get("markers_latent"), dtype=np.float64)
    if betas.size < 16:
        raise ValueError(f"MoSh++ Stage-I pickle has fewer than 16 betas: {path}")
    if markers_latent.shape != (len(labels), 3):
        raise ValueError(
            f"MoSh++ markers_latent shape {markers_latent.shape} does not match {len(labels)} labels: {path}"
        )

    marker_meta = data.get("marker_meta", {})
    marker_vids = _coerce_marker_vids(marker_meta.get("marker_vids") if isinstance(marker_meta, dict) else None)
    if not marker_vids:
        marker_vids = _coerce_marker_vids(data.get("markers_latent_vids"))

    cfg = data.get("stagei_debug_details", {}).get("cfg", {})
    surface_model_type = _nested_get(cfg, ("surface_model", "type"))
    if surface_model_type is None and isinstance(marker_meta, dict):
        surface_model_type = marker_meta.get("surface_model_type")
    gender = _nested_get(cfg, ("surface_model", "gender"), "male")
    c3d_hint = _existing_path(_nested_get(cfg, ("mocap", "fname")))
    if c3d_hint is None:
        c3d_hint = _local_c3d_from_stagei(path, _nested_get(cfg, ("mocap", "basename")))

    return StageIReference(
        path=path,
        cfg=cfg or {},
        labels=labels,
        betas=betas[:16],
        markers_latent=markers_latent,
        marker_vids=marker_vids,
        ref_indices=_parse_stagei_ref_indices(data),
        surface_model_type=str(surface_model_type or "smplx"),
        gender=str(gender or "male"),
        c3d_path_hint=c3d_hint,
    )


def run_stagei(
    *,
    c3d_path: Path,
    smpl_model_path: str,
    surface_model_type: str,
    gender: str,
    stage1_shape_solver: str,
    stage1_iters: int,
    n_ref_frames: int,
    target_fps: float | None,
    seed: int,
    least_avail_markers: float,
    strict_frame_picking: bool,
    wrist_markers_on_stick: bool,
    optimize_toes: bool,
    pose_body_prior_path: str | None,
    head_marker_corr_path: str | None,
    device: str,
) -> StageIResult:
    import torch

    raw_positions, raw_labels, fps = load_c3d_markers(str(c3d_path))
    prepared = prepare_marker_observations(raw_positions, raw_labels, surface_model_type=surface_model_type)
    if target_fps is not None and fps > float(target_fps) * 1.5:
        skip = max(1, round(fps / target_fps))
        prepared = PreparedMarkerObservations(
            positions=prepared.positions[::skip],
            labels=prepared.labels,
            unknown_labels=prepared.unknown_labels,
        )

    marker_layout = build_marker_layout(
        prepared.labels,
        wrist_markers_on_stick=wrist_markers_on_stick,
        surface_model_type=surface_model_type,
    )
    label_to_index = {label: idx for idx, label in enumerate(prepared.labels)}
    labels = marker_layout.labels
    observed = np.stack([prepared.positions[:, label_to_index[label]] for label in labels], axis=1).astype(np.float32)
    availability = compute_marker_availability_mask(observed)
    ref_indices = pick_stagei_frames(
        observed,
        num_frames=min(n_ref_frames, observed.shape[0]),
        seed=seed,
        least_avail_markers=least_avail_markers,
        strict=strict_frame_picking,
    )

    smpl, pose_dim = _make_surface_model(
        surface_model_type=surface_model_type,
        smpl_model_path=smpl_model_path,
        gender=gender,
        device=device,
    )
    marker_model = SurfaceMarkerModel.from_layout(smpl, marker_layout, pose_dim=pose_dim, device=device)
    pose_prior = MoshGmmPosePrior(pose_body_prior_path, device=device) if pose_body_prior_path else None
    resolved_head_corr_path = resolve_stagei_head_marker_corr_path(
        head_marker_corr_path,
        smpl_model_path=smpl_model_path,
        surface_model_type=surface_model_type,
    )
    head_marker_corr = load_stagei_head_marker_correlation(resolved_head_corr_path, labels)

    betas, markers_latent, debug = _fit_stagei(
        smpl=smpl,
        marker_model=marker_model,
        observed_t=torch.as_tensor(observed, dtype=torch.float32, device=device),
        availability=availability,
        ref_indices=ref_indices,
        pose_dim=pose_dim,
        body_end=66,
        stage1_iters=stage1_iters,
        device=device,
        optimize_toes=optimize_toes,
        pose_prior=pose_prior,
        pose_prior_name="moshpp_gmm" if pose_prior is not None else "l2",
        head_marker_corr=head_marker_corr,
        stage1_shape_solver=stage1_shape_solver,
    )
    return StageIResult(
        labels=list(labels),
        marker_vids={label: int(marker_layout.marker_vids[label]) for label in labels},
        ref_indices=ref_indices,
        betas=betas.detach().cpu().numpy().reshape(-1).astype(np.float64),
        markers_latent=markers_latent.detach().cpu().numpy().astype(np.float64),
        terms=dict(debug["term_history"][-1]),
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _bool_from_config(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _value_from_cli_or_config(
    cli_value: object,
    cli_source: str,
    configs: list[tuple[str, dict]],
    paths: tuple[tuple[str, ...], ...],
) -> ConfigValue | None:
    if cli_value is not None and cli_value != "":
        return ConfigValue(value=cli_value, source=cli_source)
    return _first_config_value(configs, paths)


def _resolve_stagei_settings(args: argparse.Namespace, reference: StageIReference) -> dict[str, object]:
    config_path = args.config.expanduser() if args.config is not None else None
    external_config = _load_config_file(config_path)
    path_configs = [(str(config_path), external_config)] if external_config else []
    configs = [*path_configs]
    if reference.cfg:
        configs.append(("stagei-pkl", reference.cfg))

    c3d_value = _value_from_cli_or_config(
        args.c3d_file,
        "--c3d-file",
        configs,
        (
            ("c3d_file",),
            ("c3d_path",),
            ("mocap", "fname"),
        ),
    )
    c3d_path = _resolve_existing_path_value(
        c3d_value,
        name="C3D",
        config_path=config_path,
        required=False,
        log_missing=False,
    )
    c3d_path = c3d_path or reference.c3d_path_hint
    if c3d_path is None:
        raise SystemExit("Could not infer C3D path from config or Stage-I pickle. Pass --c3d-file.")
    if not c3d_path.exists():
        raise SystemExit(f"C3D file not found: {c3d_path}")

    model_value = _value_from_cli_or_config(
        args.c3d_model_path,
        "--c3d-model-path",
        path_configs,
        (
            ("c3d_model_path",),
            ("c3d_fit_model_path",),
            ("smpl_model_path",),
        ),
    )
    c3d_model_path = resolve_c3d_model_path(model_value.value if model_value else None)
    if c3d_model_path is None:
        raise SystemExit(
            "C3D fit model path is not configured. Pass --c3d-model-path or set MUSCLEMIMIC_C3D_MODEL_PATH."
        )
    if not c3d_model_path.exists():
        raise SystemExit(f"C3D fit model path not found: {c3d_model_path}")

    pose_prior_value = _value_from_cli_or_config(
        args.pose_body_prior_path,
        "--pose-body-prior-path",
        path_configs,
        (
            ("pose_body_prior_path",),
            ("moshpp", "pose_body_prior_fname"),
        ),
    )
    pose_body_prior_path = resolve_pose_body_prior_path(pose_prior_value.value if pose_prior_value else None)

    head_corr_value = _value_from_cli_or_config(
        args.head_marker_corr_path,
        "--head-marker-corr-path",
        path_configs,
        (
            ("head_marker_corr_path",),
            ("moshpp", "head_marker_corr_fname"),
        ),
    )

    surface_model_value = _value_from_cli_or_config(
        args.surface_model,
        "--surface-model",
        configs,
        (
            ("surface_model_type",),
            ("c3d_surface_model",),
            ("surface_model", "type"),
        ),
    )
    gender_value = _value_from_cli_or_config(
        args.gender,
        "--gender",
        configs,
        (
            ("gender",),
            ("c3d_gender",),
            ("surface_model", "gender"),
        ),
    )
    stage1_iters_value = _value_from_cli_or_config(
        args.stage1_iters,
        "--stage1-iters",
        configs,
        (("stage1_iters",),),
    )
    stage1_shape_solver_value = _value_from_cli_or_config(
        args.stage1_shape_solver,
        "--stage1-shape-solver",
        configs,
        (("stage1_shape_solver",),),
    )
    n_ref_frames_value = _value_from_cli_or_config(
        args.n_ref_frames,
        "--n-ref-frames",
        configs,
        (
            ("n_ref_frames",),
            ("moshpp", "stagei_frame_picker", "num_frames"),
        ),
    )
    seed_value = _value_from_cli_or_config(
        args.seed,
        "--seed",
        configs,
        (
            ("seed",),
            ("moshpp", "stagei_frame_picker", "seed"),
        ),
    )
    least_avail_value = _value_from_cli_or_config(
        args.least_avail_markers,
        "--least-avail-markers",
        configs,
        (
            ("least_avail_markers",),
            ("moshpp", "stagei_frame_picker", "least_avail_markers"),
        ),
    )
    optimize_toes_value = _value_from_cli_or_config(
        args.optimize_toes,
        "--optimize-toes",
        configs,
        (
            ("optimize_toes",),
            ("moshpp", "optimize_toes"),
        ),
    )
    wrist_markers_value = _value_from_cli_or_config(
        args.wrist_markers_on_stick,
        "--wrist-markers-on-stick",
        configs,
        (
            ("wrist_markers_on_stick",),
            ("moshpp", "wrist_markers_on_stick"),
        ),
    )

    production_stage1_iters = _production_c3d_pipeline_default("stage1_iters")
    production_stage1_solver = _production_c3d_pipeline_default("stage1_shape_solver")
    surface_model_type = str(surface_model_value.value if surface_model_value else reference.surface_model_type)
    head_marker_corr_path = resolve_stagei_head_marker_corr_path(
        head_corr_value.value if head_corr_value else None,
        smpl_model_path=c3d_model_path,
        surface_model_type=surface_model_type,
    )

    return {
        "c3d_path": c3d_path,
        "c3d_model_path": str(c3d_model_path),
        "pose_body_prior_path": str(pose_body_prior_path) if pose_body_prior_path is not None else None,
        "head_marker_corr_path": str(head_marker_corr_path) if head_marker_corr_path is not None else None,
        "surface_model_type": surface_model_type,
        "gender": str(gender_value.value if gender_value else reference.gender),
        "stage1_iters": int(stage1_iters_value.value if stage1_iters_value else production_stage1_iters),
        "stage1_shape_solver": str(
            stage1_shape_solver_value.value if stage1_shape_solver_value else production_stage1_solver
        ),
        "n_ref_frames": int(n_ref_frames_value.value if n_ref_frames_value else 12),
        "seed": int(seed_value.value if seed_value else 100),
        "least_avail_markers": float(least_avail_value.value if least_avail_value else 1.0),
        "optimize_toes": _bool_from_config(optimize_toes_value.value if optimize_toes_value else None, default=False),
        "wrist_markers_on_stick": _bool_from_config(
            wrist_markers_value.value if wrist_markers_value else None, default=False
        ),
    }


def compare_stagei(current: StageIResult, reference: StageIReference) -> dict[str, object]:
    missing = [label for label in current.labels if label not in reference.labels]
    if missing:
        raise ValueError(f"MoSh++ Stage-I pickle is missing labels used by current fit: {missing}")

    ref_label_to_idx = {label: idx for idx, label in enumerate(reference.labels)}
    ref_order = [ref_label_to_idx[label] for label in current.labels]
    marker_delta = current.markers_latent - reference.markers_latent[ref_order]
    marker_error_m = np.linalg.norm(marker_delta, axis=-1)
    beta_abs = np.abs(current.betas[:16] - reference.betas[:16])

    marker_vid_mismatches = {
        label: (current.marker_vids[label], reference.marker_vids.get(label))
        for label in current.labels
        if reference.marker_vids.get(label) is not None and current.marker_vids[label] != reference.marker_vids[label]
    }
    frames_match = None
    if reference.ref_indices is not None:
        frames_match = bool(np.array_equal(current.ref_indices, reference.ref_indices))

    return {
        "num_labels": len(current.labels),
        "missing_labels": missing,
        "marker_vid_mismatches": marker_vid_mismatches,
        "frames_match": frames_match,
        "current_ref_indices": current.ref_indices,
        "reference_ref_indices": reference.ref_indices,
        "beta_abs": _summary(beta_abs),
        "marker_error_mm": _summary(marker_error_m * 1000.0),
        "worst_markers": [
            (label, float(error * 1000.0))
            for label, error in sorted(
                zip(current.labels, marker_error_m, strict=True), key=lambda item: item[1], reverse=True
            )[:5]
        ],
        "current_terms": current.terms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stagei-pkl", required=True, type=Path, help="MoSh++ *_stagei.pkl to compare against.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional YAML/JSON/Pickle config. CLI flags override this config; "
            "without it, the script uses the config embedded in --stagei-pkl."
        ),
    )
    parser.add_argument(
        "--c3d-file", type=Path, default=None, help="C3D file. Defaults to a sibling C3D inferred from the pickle."
    )
    parser.add_argument(
        "--c3d-model-path",
        default=None,
        help=("SMPL-X/SMPL-H model path for C3D marker fitting. Defaults to config, then MUSCLEMIMIC_C3D_MODEL_PATH."),
    )
    parser.add_argument("--surface-model", choices=["smplx", "smplh"], default=None)
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default=None)
    parser.add_argument("--pose-body-prior-path", default=None)
    parser.add_argument("--head-marker-corr-path", default=None)
    parser.add_argument("--stage1-shape-solver", choices=["joint_dogleg", "joint_dogleg_jax"], default=None)
    parser.add_argument(
        "--stage1-iters",
        type=int,
        default=None,
        help="Defaults to the production C3D pipeline default, not the lower-level fitter default.",
    )
    parser.add_argument("--n-ref-frames", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--least-avail-markers", type=float, default=None)
    parser.add_argument("--nonstrict-frame-picking", action="store_true")
    parser.add_argument("--wrist-markers-on-stick", action="store_true", default=None)
    parser.add_argument("--optimize-toes", action="store_true", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-marker-mm", type=float, default=None, help="Fail if max latent-marker difference exceeds this."
    )
    parser.add_argument("--max-beta-abs", type=float, default=None, help="Fail if max beta difference exceeds this.")
    parser.add_argument("--allow-frame-mismatch", action="store_true")
    parser.add_argument("--allow-marker-vid-mismatch", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    reference = load_stagei_reference(args.stagei_pkl)
    settings = _resolve_stagei_settings(args, reference)
    c3d_path = settings["c3d_path"]
    surface_model_type = settings["surface_model_type"]
    gender = settings["gender"]

    LOG.info("Reference: %s", reference.path)
    LOG.info("C3D:       %s", c3d_path)
    LOG.info("Surface:   %s %s", surface_model_type, gender)
    LOG.info("Stage I:   solver=%s iters=%s", settings["stage1_shape_solver"], settings["stage1_iters"])

    current = run_stagei(
        c3d_path=c3d_path,
        smpl_model_path=settings["c3d_model_path"],
        surface_model_type=surface_model_type,
        gender=gender,
        stage1_shape_solver=settings["stage1_shape_solver"],
        stage1_iters=settings["stage1_iters"],
        n_ref_frames=settings["n_ref_frames"],
        target_fps=args.target_fps,
        seed=settings["seed"],
        least_avail_markers=settings["least_avail_markers"],
        strict_frame_picking=not args.nonstrict_frame_picking,
        wrist_markers_on_stick=settings["wrist_markers_on_stick"],
        optimize_toes=settings["optimize_toes"],
        pose_body_prior_path=settings["pose_body_prior_path"],
        head_marker_corr_path=settings["head_marker_corr_path"],
        device=args.device,
    )
    comparison = compare_stagei(current, reference)

    beta = comparison["beta_abs"]
    marker = comparison["marker_error_mm"]
    LOG.info("Labels compared: %d", comparison["num_labels"])
    LOG.info("Reference frames match: %s", comparison["frames_match"])
    LOG.info("Beta abs diff:    mean=%.6g p95=%.6g max=%.6g", beta["mean"], beta["p95"], beta["max"])
    LOG.info(
        "Latent marker diff: mean=%.3fmm p95=%.3fmm max=%.3fmm",
        marker["mean"],
        marker["p95"],
        marker["max"],
    )
    LOG.info(
        "Worst latent markers: %s", ", ".join(f"{label}={error:.2f}mm" for label, error in comparison["worst_markers"])
    )
    LOG.info("Current final Stage-I terms: %s", comparison["current_terms"])

    failures: list[str] = []
    if comparison["frames_match"] is False and not args.allow_frame_mismatch:
        failures.append(
            f"Stage-I frame picks differ: current={comparison['current_ref_indices'].tolist()} "
            f"reference={comparison['reference_ref_indices'].tolist()}"
        )
    if comparison["marker_vid_mismatches"] and not args.allow_marker_vid_mismatch:
        failures.append(f"Marker vertex IDs differ: {comparison['marker_vid_mismatches']}")
    if args.max_marker_mm is not None and marker["max"] > args.max_marker_mm:
        failures.append(f"max marker diff {marker['max']:.3f}mm > {args.max_marker_mm:.3f}mm")
    if args.max_beta_abs is not None and beta["max"] > args.max_beta_abs:
        failures.append(f"max beta diff {beta['max']:.6g} > {args.max_beta_abs:.6g}")

    if failures:
        for failure in failures:
            LOG.error("FAIL: %s", failure)
        return 1
    LOG.info("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
