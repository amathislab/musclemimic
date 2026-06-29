"""Convert C3D marker data to SMPL parameters with a MoSh++-style pipeline.

This module does not embed the original chumpy-based `moshpp` implementation.
Instead, it mirrors the parts that are practical in this repo's existing torch
stack:

1. Canonical marker labels and MoSh++ marker vertices for SMPL-H/SMPL-X.
2. Marker-type-dependent distance-to-skin handling.
3. Stage-I frame picking based on marker availability.
4. Stage-I optimization over shared betas, per-frame rigid pose, and
   canonical-space latent marker positions.
5. Stage-II warm-started per-frame pose fitting with missing-marker-dependent
   weight annealing and temporal smoothing.

Outputs an AMASS-compatible dict that feeds directly into the existing
retargeting pipeline (fit_smpl_motion / fit_gmr_motion).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from .c3d.asset_paths import resolve_pose_body_prior_path
from .c3d.markers import (
    MOSHPP_STAGEI_WT_ANNEALING,
    MOSHPP_STAGEI_WT_INIT,
    PreparedMarkerObservations,
    StageIHeadMarkerCorrelation,
    StageIIWeights,
    StageIWeights,
    build_marker_layout,
    compute_marker_availability_mask,
    compute_stagei_weights,
    compute_stageii_weights,
    load_c3d_markers,
    load_stagei_head_marker_correlation,
    pick_stagei_frames,
    prepare_marker_observations,
    resolve_stagei_head_marker_corr_path,
)
from .c3d.optim import _minimize_dogleg_dense
from .c3d.pose_prior import MoshGmmPosePrior
from .c3d.smpl_models import _make_surface_model, _resolve_smplx_model_file
from .c3d.surface_markers import (
    SurfaceMarkerModel,
    _point_to_triangle_distance_torch,
    _procrustes_align,
    _surface_distances_to_mesh,
)

logger = logging.getLogger(__name__)

C3D_TO_SMPL_CACHE_VERSION = 1
DEFAULT_STAGE1_SHAPE_SOLVER = "joint_dogleg_jax"


# moshpp_conf.yaml defaults `optimize_toes: false`. In SMPL bone order joints 10
# (L_Foot) and 11 (R_Foot) occupy axis-angle dims 30:36; moshpp keeps them
# frozen at zero in both Stage I and Stage II.
_TOES_POSE_AA_SLICE = slice(30, 36)


def _make_pose_freeze_hook(body_end: int, *, optimize_toes: bool):
    """Backward hook matching MoSh++ default pose free variables.

    MoSh++ defaults optimize only root and body pose; hand/face pose dims are
    frozen unless explicit finger/face options are enabled. This fitter has no
    finger/face optimizer yet, so dims after `body_end` stay fixed at zero.
    """

    def hook(grad):
        new_grad = grad.clone()
        if not optimize_toes:
            new_grad[..., _TOES_POSE_AA_SLICE] = 0.0
        new_grad[..., body_end:] = 0.0
        return new_grad

    return hook


def _pose_to_full_pose_numpy(smpl, pose_np: np.ndarray, *, device: str) -> np.ndarray:
    converter = getattr(smpl, "to_full_pose", None)
    if converter is None:
        return pose_np

    import torch

    with torch.no_grad():
        pose_t = torch.as_tensor(pose_np, dtype=torch.float32, device=device)
        return converter(pose_t).detach().cpu().numpy().astype(np.float32)


def fit_smpl_to_c3d(
    c3d_path: str,
    smpl_model_path: str,
    stage1_iters: int = 320,
    stage2_iters: int = 80,
    n_ref_frames: int = 12,
    target_fps: float | None = None,
    device: str = "cpu",
    seed: int = 100,
    least_avail_markers: float = 1.0,
    strict_frame_picking: bool = True,
    wrist_markers_on_stick: bool = False,
    optimize_toes: bool = False,
    pose_body_prior_path: str | None = None,
    head_marker_corr_path: str | None = None,
    surface_model_type: str = "smplx",
    gender: str = "male",
    stage1_shape_solver: str = DEFAULT_STAGE1_SHAPE_SOLVER,
) -> dict:
    # `optimize_toes` matches moshpp_conf.yaml's flag (default false). When false,
    # SMPL joints L_Foot (idx 10) and R_Foot (idx 11) are frozen at zero. For gait
    # data with explicit LTOE/RTOE markers, set this True or the ankle will
    # over-rotate to compensate (visible "tiptoe" walking).
    #
    # Defaults match moshpp's `random_strict` Stage-I frame picking. For noisy
    # C3Ds with occlusions, callers can lower least_avail_markers and set
    # strict_frame_picking=False.
    import torch

    body_end = 66

    raw_positions, raw_labels, fps = load_c3d_markers(c3d_path)
    prepared = prepare_marker_observations(raw_positions, raw_labels, surface_model_type=surface_model_type)

    # MoSh++ fits at native mocap rate. Keep that as the default; callers can
    # still opt into downsampling for quick diagnostics by passing target_fps.
    if target_fps is not None and fps > float(target_fps) * 1.5:
        skip = max(1, round(fps / target_fps))
        prepared = PreparedMarkerObservations(
            positions=prepared.positions[::skip],
            labels=prepared.labels,
            unknown_labels=prepared.unknown_labels,
        )
        effective_fps = fps / skip
        logger.info(
            "Downsampled %.0f Hz → %.0f Hz (skip=%d, %d → %d frames)",
            fps,
            effective_fps,
            skip,
            raw_positions.shape[0],
            prepared.positions.shape[0],
        )
    else:
        effective_fps = fps
    if prepared.positions.shape[1] < 3:
        raise ValueError(
            f"Need at least 3 usable markers after canonicalization, got {prepared.positions.shape[1]} "
            f"from labels {prepared.labels}"
        )

    marker_layout = build_marker_layout(
        prepared.labels,
        wrist_markers_on_stick=wrist_markers_on_stick,
        surface_model_type=surface_model_type,
    )
    label_to_index = {label: idx for idx, label in enumerate(prepared.labels)}
    ordered_labels = marker_layout.labels
    observed = np.stack([prepared.positions[:, label_to_index[label]] for label in ordered_labels], axis=1).astype(
        np.float32
    )
    availability = compute_marker_availability_mask(observed)

    logger.info(
        "Loaded %s: %d frames, %d/%d markers matched to the moshpp layout @ %.1f Hz",
        c3d_path,
        observed.shape[0],
        len(ordered_labels),
        len(raw_labels),
        fps,
    )
    logger.info(
        "C3D fit configuration: surface_model_type=%s, gender=%s, model_path=%s",
        surface_model_type,
        gender,
        smpl_model_path,
    )
    logger.info(
        "Planned optimization work: Stage I %d ref frames x %d iters, Stage II %d frames x %d iters/frame",
        min(n_ref_frames, observed.shape[0]),
        stage1_iters,
        observed.shape[0],
        stage2_iters,
    )
    if prepared.unknown_labels:
        logger.info("Ignored unknown markers after canonicalization: %s", ", ".join(prepared.unknown_labels))

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

    pose_body_prior_path = resolve_pose_body_prior_path(pose_body_prior_path)
    pose_prior = MoshGmmPosePrior(pose_body_prior_path, device=device) if pose_body_prior_path else None
    pose_prior_name = "moshpp_gmm" if pose_prior is not None else "l2"
    resolved_head_marker_corr_path = resolve_stagei_head_marker_corr_path(
        head_marker_corr_path,
        smpl_model_path=smpl_model_path,
        surface_model_type=surface_model_type,
    )
    head_marker_corr = load_stagei_head_marker_correlation(resolved_head_marker_corr_path, ordered_labels)
    if head_marker_corr is not None:
        logger.info(
            "Using MoSh++ head marker correlation: %s (%s)",
            head_marker_corr.path,
            ", ".join(head_marker_corr.labels),
        )

    observed_t = torch.as_tensor(observed, dtype=torch.float32, device=device)
    betas_opt, markers_latent_opt, stage1_debug = _fit_stagei(
        smpl=smpl,
        marker_model=marker_model,
        observed_t=observed_t,
        availability=availability,
        ref_indices=ref_indices,
        pose_dim=pose_dim,
        body_end=body_end,
        stage1_iters=stage1_iters,
        device=device,
        optimize_toes=optimize_toes,
        pose_prior=pose_prior,
        pose_prior_name=pose_prior_name,
        head_marker_corr=head_marker_corr,
        stage1_shape_solver=stage1_shape_solver,
    )

    fit_pose, trans, stage2_debug = _fit_stageii(
        smpl=smpl,
        marker_model=marker_model,
        observed_t=observed_t,
        availability=availability,
        betas_opt=betas_opt,
        markers_latent_opt=markers_latent_opt,
        pose_dim=pose_dim,
        body_end=body_end,
        stage2_iters=stage2_iters,
        device=device,
        optimize_toes=optimize_toes,
        pose_prior=pose_prior,
        pose_prior_name=pose_prior_name,
    )
    fullpose = _pose_to_full_pose_numpy(smpl, fit_pose, device=device)
    stageii_frame_vids_t = None
    if isinstance(stage2_debug, dict) and stage2_debug.get("frame_vids") is not None:
        import torch as _torch  # local alias

        stageii_frame_vids_t = _torch.as_tensor(stage2_debug["frame_vids"], dtype=_torch.long, device=device)
    marker_error = _compute_fitted_marker_error(
        smpl=smpl,
        marker_model=marker_model,
        observed_t=observed_t,
        availability=availability,
        betas_opt=betas_opt,
        markers_latent_opt=markers_latent_opt,
        pose_aa_156=fit_pose,
        trans=trans,
        pose_dim=pose_dim,
        device=device,
        frame_vids_override=stageii_frame_vids_t,
    )
    logger.info(
        "Final marker fit: mean=%.2fmm p95=%.2fmm max=%.2fmm",
        marker_error["mean_mm"],
        marker_error["p95_mm"],
        marker_error["max_mm"],
    )

    pose_aa = np.concatenate([fullpose[:, :body_end], np.zeros((fullpose.shape[0], 6), dtype=np.float32)], axis=-1)
    return {
        "pose_aa": pose_aa,
        "fullpose": fullpose,
        "trans": trans,
        "betas": betas_opt.detach().cpu().numpy().reshape(-1),
        "fps": float(effective_fps),
        "gender": gender,
        "debug": {
            "surface_model_type": surface_model_type,
            "canonical_labels": ordered_labels,
            "stagei_ref_indices": ref_indices,
            "stagei": stage1_debug,
            "stageii": stage2_debug,
            "marker_error": marker_error,
            "fit_pose_dim": int(fit_pose.shape[1]),
            "pose_prior": pose_prior_name,
            "head_marker_corr_path": None if head_marker_corr is None else head_marker_corr.path,
            "stage1_shape_solver": stage1_shape_solver,
        },
    }


def fit_smpl_to_c3d_cached(
    c3d_path: str,
    smpl_model_path: str,
    cache_dir: str | None = None,
    clear_cache: bool = False,
    **kwargs,
) -> dict:
    """Fit SMPL to C3D with npz caching."""
    kwargs = dict(kwargs)
    pose_body_prior_path = resolve_pose_body_prior_path(kwargs.get("pose_body_prior_path"))
    if pose_body_prior_path is not None or "pose_body_prior_path" in kwargs:
        kwargs["pose_body_prior_path"] = str(pose_body_prior_path) if pose_body_prior_path is not None else None
    head_marker_corr_path = resolve_stagei_head_marker_corr_path(
        kwargs.get("head_marker_corr_path"),
        smpl_model_path=smpl_model_path,
        surface_model_type=kwargs.get("surface_model_type", "smplx"),
    )
    if head_marker_corr_path is not None or "head_marker_corr_path" in kwargs:
        kwargs["head_marker_corr_path"] = head_marker_corr_path

    c3d_name = Path(c3d_path).stem
    if cache_dir is None:
        cache_dir = str(Path(c3d_path).parent / ".smpl_cache")

    suffix = _cache_suffix_for_fit(smpl_model_path, kwargs, c3d_path=c3d_path)
    cache_path = Path(cache_dir) / f"{c3d_name}_smpl_v{C3D_TO_SMPL_CACHE_VERSION}{suffix}.npz"
    if cache_path.exists() and not clear_cache:
        logger.info("Loading cached SMPL fit from %s", cache_path)
        data = np.load(cache_path, allow_pickle=False)
        result = {
            "pose_aa": data["pose_aa"],
            "trans": data["trans"],
            "betas": data["betas"],
            "fps": float(data["fps"]),
            "gender": str(data["gender"]),
        }
        if "fullpose" in data.files:
            result["fullpose"] = data["fullpose"]
        return result
    if cache_path.exists() and clear_cache:
        logger.info("Ignoring cached SMPL fit because clear_cache=True: %s", cache_path)

    result = fit_smpl_to_c3d(c3d_path, smpl_model_path, **kwargs)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        pose_aa=result["pose_aa"],
        fullpose=result.get("fullpose", result["pose_aa"]),
        trans=result["trans"],
        betas=result["betas"],
        fps=np.array(result["fps"], dtype=np.float32),
        gender=np.array(str(result["gender"])),
        cache_version=np.array(C3D_TO_SMPL_CACHE_VERSION, dtype=np.int64),
    )
    logger.info("Cached SMPL fit to %s", cache_path)
    return result


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def _path_hash(path: str | Path | None) -> str:
    if path is None:
        return "none"
    resolved = Path(path).expanduser().resolve(strict=False)
    return _short_hash(str(resolved))


def _cache_suffix_for_fit(smpl_model_path: str, kwargs: dict, *, c3d_path: str | Path | None = None) -> str:
    """Build a compact cache key from every argument that changes the fit."""

    suffix = f"_model{_path_hash(smpl_model_path)}"

    defaults = {
        "stage1_iters": 320,
        "stage2_iters": 80,
        "n_ref_frames": 12,
        "target_fps": None,
        "seed": 100,
        "least_avail_markers": 1.0,
        "strict_frame_picking": True,
        "wrist_markers_on_stick": False,
        "optimize_toes": False,
        "pose_body_prior_path": None,
        "head_marker_corr_path": None,
        "surface_model_type": "smplx",
        "gender": "male",
        "stage1_shape_solver": DEFAULT_STAGE1_SHAPE_SOLVER,
    }

    def get(name: str):
        return kwargs.get(name, defaults[name])

    surface_model_type = get("surface_model_type")
    gender = get("gender")
    suffix += f"_{surface_model_type}_{gender}"

    if get("optimize_toes"):
        suffix += "_toes"
    if get("wrist_markers_on_stick"):
        suffix += "_wriststick"
    suffix += f"_s1shape{get('stage1_shape_solver')}"

    if get("pose_body_prior_path") is not None:
        suffix += f"_gmm{_path_hash(get('pose_body_prior_path'))}"
    resolved_head_corr_path = resolve_stagei_head_marker_corr_path(
        get("head_marker_corr_path"),
        smpl_model_path=smpl_model_path,
        surface_model_type=surface_model_type,
    )
    if resolved_head_corr_path is not None:
        suffix += f"_headcorr{_path_hash(resolved_head_corr_path)}"

    numeric_fields = (
        ("stage1_iters", "s1"),
        ("stage2_iters", "s2"),
        ("n_ref_frames", "ref"),
        ("target_fps", "fps"),
        ("seed", "seed"),
        ("least_avail_markers", "avail"),
    )
    for name, label in numeric_fields:
        value = get(name)
        default = defaults[name]
        if value != default:
            if isinstance(value, float):
                suffix += f"_{label}{value:g}"
            else:
                suffix += f"_{label}{value}"

    if not get("strict_frame_picking"):
        suffix += "_nonstrict"

    return suffix


def save_motion_data_as_amass_smplh_npz(
    motion_data: dict,
    output_path: str | Path,
) -> Path:
    """Write fitted motion data to an AMASS/GMR-compatible SMPL-H npz file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pose_aa = np.asarray(motion_data["pose_aa"])
    if not np.issubdtype(pose_aa.dtype, np.floating):
        pose_aa = pose_aa.astype(np.float32)
    if pose_aa.ndim != 2:
        raise ValueError(f"Expected pose_aa with shape (T, D), got {pose_aa.shape}")

    if pose_aa.shape[1] == 156:
        poses = pose_aa
    elif pose_aa.shape[1] <= 156:
        poses = np.concatenate(
            [pose_aa, np.zeros((pose_aa.shape[0], 156 - pose_aa.shape[1]), dtype=pose_aa.dtype)],
            axis=-1,
        )
    else:
        raise ValueError(f"pose_aa has too many columns for SMPL-H export: {pose_aa.shape[1]}")

    betas = np.asarray(motion_data["betas"])
    trans = np.asarray(motion_data["trans"])
    if not np.issubdtype(betas.dtype, np.floating):
        betas = betas.astype(np.float32)
    if not np.issubdtype(trans.dtype, np.floating):
        trans = trans.astype(np.float32)
    fps = float(motion_data["fps"])
    gender = str(motion_data.get("gender", "neutral"))

    np.savez(
        output_path,
        poses=poses,
        trans=trans,
        betas=betas,
        gender=np.array(gender),
        mocap_framerate=np.array(fps, dtype=np.float32),
    )
    return output_path


def _fit_stagei(
    *,
    smpl,
    marker_model: SurfaceMarkerModel,
    observed_t,
    availability: np.ndarray,
    ref_indices: np.ndarray,
    pose_dim: int,
    body_end: int,
    stage1_iters: int,
    device: str,
    optimize_toes: bool = False,
    pose_prior=None,
    pose_prior_name: str = "l2",
    head_marker_corr: StageIHeadMarkerCorrelation | None = None,
    stage1_shape_solver: str = DEFAULT_STAGE1_SHAPE_SOLVER,
):
    import torch

    if stage1_shape_solver not in {"joint_dogleg", "joint_dogleg_jax"}:
        raise ValueError("stage1_shape_solver must be 'joint_dogleg' or 'joint_dogleg_jax'.")

    active_stage1_shape_solver = stage1_shape_solver
    if active_stage1_shape_solver == "joint_dogleg_jax":
        try:
            from .stagei_jax_solver import _extract_lbs_arrays

            _extract_lbs_arrays(smpl)
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "JAX Stage-I solver unavailable for this surface model (%s); falling back to joint_dogleg.",
                exc,
            )
            active_stage1_shape_solver = "joint_dogleg"

    logger.info(
        "Stage I (%s residual blocks): fitting shape across %d reference frames, %d iters with %s body-pose prior (%s solver)",
        "joint",
        len(ref_indices),
        stage1_iters,
        pose_prior_name,
        active_stage1_shape_solver,
    )

    ref_observed = observed_t[ref_indices]
    ref_availability = torch.as_tensor(availability[ref_indices], dtype=torch.bool, device=device)

    n_ref = len(ref_indices)
    n_markers = len(marker_model.vids)
    betas_np = np.zeros((1, 16), dtype=np.float32)
    markers_latent_np = np.asarray(marker_model.initial_latents, dtype=np.float32).copy()
    ref_poses_np = np.zeros((n_ref, pose_dim), dtype=np.float32)
    ref_trans_np = np.zeros((n_ref, 3), dtype=np.float32)

    initial_latents_t = torch.as_tensor(marker_model.initial_latents, dtype=torch.float32, device=device)
    for local_idx, frame_idx in enumerate(ref_indices):
        pose_init, trans_init = _initialize_pose_and_trans(
            smpl=smpl,
            marker_model=marker_model,
            betas=torch.zeros(1, 16, dtype=torch.float32, device=device),
            markers_latent=initial_latents_t,
            observed_frame=ref_observed[local_idx],
            availability_mask=ref_availability[local_idx],
            pose_dim=pose_dim,
            device=device,
        )
        ref_poses_np[local_idx] = pose_init.detach().cpu().numpy().reshape(-1)
        ref_trans_np[local_idx] = trans_init.detach().cpu().numpy().reshape(-1)
        logger.debug(
            "Stage I init frame %d -> global orient norm %.4f",
            int(frame_idx),
            pose_init[:, :3].norm().item(),
        )

    init_coeffs_t = torch.as_tensor(marker_model.initial_coeffs, dtype=torch.float32, device=device)
    init_frame_vids_t = torch.as_tensor(marker_model.frame_vids, dtype=torch.long, device=device)
    coeff_surface_target = torch.as_tensor(marker_model.desired_distances, dtype=torch.float32, device=device)
    head_marker_indices_t = None
    head_marker_non_head_mask_t = None
    head_marker_corr_t = None
    if head_marker_corr is not None:
        head_marker_indices_t = torch.as_tensor(head_marker_corr.marker_indices, dtype=torch.long, device=device)
        non_head_mask = np.ones(n_markers, dtype=bool)
        non_head_mask[head_marker_corr.marker_indices] = False
        head_marker_non_head_mask_t = torch.as_tensor(non_head_mask, dtype=torch.bool, device=device)
        head_marker_corr_t = torch.as_tensor(head_marker_corr.corr, dtype=torch.float32, device=device)
    zero_pose = torch.zeros(1, pose_dim, dtype=torch.float32, device=device)
    marker_type_masks_t = {
        marker_kind: torch.as_tensor(
            [marker_type == marker_kind for marker_type in marker_model.marker_types],
            dtype=torch.bool,
            device=device,
        )
        for marker_kind in sorted(set(marker_model.marker_types))
    }

    stage1_losses: list[float] = []
    stage1_term_history: list[dict[str, float]] = []
    stage1_phase_results: list[dict[str, object]] = []
    iters_per_phase = max(25, stage1_iters // len(MOSHPP_STAGEI_WT_ANNEALING))

    def init_residual_terms(init_loss, coeff_type_weights, weights: StageIWeights):
        if head_marker_corr_t is None:
            return {
                f"init_{marker_kind}": (init_loss[marker_mask] * weights.init_by_type[marker_kind]).reshape(-1)
                for marker_kind, marker_mask in marker_type_masks_t.items()
                if torch.any(marker_mask)
            }

        terms = {}
        for marker_kind, marker_mask in marker_type_masks_t.items():
            non_head_marker_mask = marker_mask & head_marker_non_head_mask_t
            if torch.any(non_head_marker_mask):
                terms[f"init_{marker_kind}"] = (
                    init_loss[non_head_marker_mask] * weights.init_by_type[marker_kind]
                ).reshape(-1)

        head_loss = init_loss[head_marker_indices_t]
        head_weight = weights.init_by_type.get("body", MOSHPP_STAGEI_WT_INIT * weights.anneal_factor)
        terms["init_head_corr"] = (head_marker_corr_t @ head_loss * head_weight).reshape(-1)
        return terms

    free_pose_np = _stagei_free_pose_indices(body_end, optimize_toes=optimize_toes)
    free_pose_t = torch.as_tensor(free_pose_np, dtype=torch.long, device=device)
    n_free_pose = int(free_pose_np.shape[0])

    def pack_joint_absolute() -> np.ndarray:
        return np.concatenate(
            [
                ref_trans_np.reshape(-1),
                markers_latent_np.reshape(-1),
                ref_poses_np[:, free_pose_np].reshape(-1),
                betas_np.reshape(-1),
            ]
        ).astype(np.float64)

    def unpack_joint_absolute_numpy(x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cursor = 0
        ref_trans = x_np[cursor : cursor + n_ref * 3].reshape(n_ref, 3).astype(np.float32)
        cursor += n_ref * 3
        markers_latent = x_np[cursor : cursor + n_markers * 3].reshape(n_markers, 3).astype(np.float32)
        cursor += n_markers * 3
        ref_poses = ref_poses_np.copy()
        ref_poses[:, free_pose_np] = (
            x_np[cursor : cursor + n_ref * n_free_pose].reshape(n_ref, n_free_pose).astype(np.float32)
        )
        cursor += n_ref * n_free_pose
        betas = x_np[cursor : cursor + 16].reshape(1, 16).astype(np.float32)
        return betas, markers_latent, ref_poses, ref_trans

    def make_joint_absolute_residual_functions(weights: StageIWeights):
        base_ref_poses_t = torch.as_tensor(ref_poses_np, dtype=torch.float32, device=device)

        def unpack_torch(x_t):
            cursor = 0
            ref_trans = x_t[cursor : cursor + n_ref * 3].reshape(n_ref, 3)
            cursor += n_ref * 3
            markers_latent = x_t[cursor : cursor + n_markers * 3].reshape(n_markers, 3)
            cursor += n_markers * 3
            ref_poses = base_ref_poses_t.clone()
            ref_poses[:, free_pose_t] = x_t[cursor : cursor + n_ref * n_free_pose].reshape(n_ref, n_free_pose)
            cursor += n_ref * n_free_pose
            betas = x_t[cursor : cursor + 16].reshape(1, 16)
            return betas, markers_latent, ref_poses, ref_trans

        def residual_torch(x_t):
            betas, markers_latent, ref_poses_t, ref_trans_t = unpack_torch(x_t)
            canonical_verts, _ = smpl.get_joints_verts(zero_pose, th_betas=betas)
            with torch.no_grad():
                frame_vids_np = SurfaceMarkerModel.compute_dynamic_frame_vids(
                    canonical_verts[0].detach().cpu().numpy(),
                    markers_latent.detach().cpu().numpy(),
                )
            frame_vids_t = torch.as_tensor(frame_vids_np, dtype=torch.long, device=device)
            marker_coeffs = marker_model.latents_to_coeffs(
                canonical_verts[0], markers_latent, frame_vids_override=frame_vids_t
            )

            data_blocks = []
            for ref_idx in range(n_ref):
                verts, _ = smpl.get_joints_verts(
                    ref_poses_t[ref_idx : ref_idx + 1],
                    th_betas=betas,
                    th_trans=ref_trans_t[ref_idx : ref_idx + 1],
                )
                pred_markers = marker_model.reconstruct(
                    verts[0], coeffs=marker_coeffs, frame_vids_override=frame_vids_t
                )
                mask = ref_availability[ref_idx]
                if torch.any(mask):
                    data_blocks.append((pred_markers[mask] - ref_observed[ref_idx][mask]).reshape(-1) * weights.data)

            if not data_blocks:
                raise ValueError("No valid Stage-I reference frames contain usable markers.")

            body_pose = ref_poses_t[:, 3:body_end]
            if pose_prior is not None and hasattr(pose_prior, "residuals"):
                pose_res = pose_prior.residuals(body_pose) * weights.pose_body
            elif pose_prior is not None:
                pose_res = torch.sqrt(pose_prior(body_pose).clamp_min(0.0) + 1e-12).reshape(1) * weights.pose_body
            else:
                pose_res = body_pose.reshape(-1) * weights.pose_body

            init_markers_latent = marker_model.reconstruct(
                canonical_verts[0],
                coeffs=init_coeffs_t,
                frame_vids_override=init_frame_vids_t,
            )
            surface_distances = _surface_distances_to_mesh(
                marker_model,
                canonical_verts[0],
                markers_latent,
                nearest_vids=frame_vids_t,
                fallback_coeffs=marker_coeffs,
            )
            init_terms = init_residual_terms(markers_latent - init_markers_latent, None, weights)
            return {
                "data": torch.cat(data_blocks),
                "pose": pose_res,
                "beta": betas.reshape(-1) * weights.betas,
                **init_terms,
                "surf": (surface_distances - coeff_surface_target) * weights.surf,
            }

        def residual_np(x_np: np.ndarray) -> np.ndarray:
            x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                blocks = residual_torch(x_t)
            return torch.cat(list(blocks.values())).detach().cpu().numpy().astype(np.float64)

        def residual_and_jacobian_np(x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device).requires_grad_(True)

            def flat_residual(y_t):
                blocks = residual_torch(y_t)
                return torch.cat(list(blocks.values()))

            residual = flat_residual(x_t)
            try:
                jacobian = torch.autograd.functional.jacobian(
                    flat_residual,
                    x_t,
                    vectorize=True,
                    strategy="forward-mode",
                )
            except (TypeError, RuntimeError):
                jacobian = torch.autograd.functional.jacobian(flat_residual, x_t, vectorize=True)
            return (
                residual.detach().cpu().numpy().astype(np.float64),
                jacobian.detach().cpu().numpy().astype(np.float64),
            )

        def terms_np(x_np: np.ndarray) -> dict[str, float]:
            x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                blocks = residual_torch(x_t)
            return {name: float(block.pow(2).sum().cpu().item()) for name, block in blocks.items()}

        return residual_np, residual_and_jacobian_np, terms_np

    for phase_idx, anneal_factor in enumerate(MOSHPP_STAGEI_WT_ANNEALING):
        weights = compute_stagei_weights(n_markers, marker_model.marker_types, anneal_factor)
        coeff_type_weights = torch.as_tensor(
            [weights.init_by_type[marker_type] for marker_type in marker_model.marker_types],
            dtype=torch.float32,
            device=device,
        )
        last_terms: dict[str, float] = {}

        if active_stage1_shape_solver == "joint_dogleg":
            logger.info(
                "  Stage I phase %d/%d joint residual pass (anneal=%.3f, max %d dogleg iters)",
                phase_idx + 1,
                len(MOSHPP_STAGEI_WT_ANNEALING),
                anneal_factor,
                iters_per_phase,
            )
            residual_np, residual_and_jacobian_np, terms_np = make_joint_absolute_residual_functions(weights)
            result = _minimize_dogleg_dense(
                residual_np=residual_np,
                residual_and_jacobian_np=residual_and_jacobian_np,
                x0=pack_joint_absolute(),
                maxiter=iters_per_phase,
                e_3=1e-3,
                delta_0=0.5,
            )
            betas_np, markers_latent_np, ref_poses_np, ref_trans_np = unpack_joint_absolute_numpy(result.x)
            last_terms = terms_np(result.x)
            nit = result.nit
            nfev = result.nfev
            success = result.success
            message = result.message
        elif active_stage1_shape_solver == "joint_dogleg_jax":
            logger.info(
                "  Stage I phase %d/%d joint residual pass (anneal=%.3f, max %d JAX dogleg iters)",
                phase_idx + 1,
                len(MOSHPP_STAGEI_WT_ANNEALING),
                anneal_factor,
                iters_per_phase,
            )
            from .stagei_jax_solver import fit_stagei_joint_dogleg

            jax_result = fit_stagei_joint_dogleg(
                smpl=smpl,
                marker_model=marker_model,
                ref_observed=ref_observed.detach().cpu().numpy(),
                ref_availability=availability[ref_indices],
                ref_poses=ref_poses_np,
                ref_trans=ref_trans_np,
                base_betas=betas_np,
                base_markers_latent=markers_latent_np,
                weights=weights,
                coeff_type_weights=coeff_type_weights.detach().cpu().numpy(),
                free_pose_indices=free_pose_np,
                body_end=body_end,
                pose_prior=pose_prior,
                head_marker_corr=head_marker_corr,
                maxiter=iters_per_phase,
            )
            betas_np = jax_result.betas
            markers_latent_np = jax_result.markers_latent
            ref_poses_np = jax_result.ref_poses
            ref_trans_np = jax_result.ref_trans
            last_terms = jax_result.terms
            nit = jax_result.nit
            nfev = jax_result.nfev
            success = jax_result.success
            message = jax_result.message
        else:  # pragma: no cover - guarded above.
            raise AssertionError(active_stage1_shape_solver)

        last_loss = float(sum(last_terms.values()))
        init_loss = float(sum(value for name, value in last_terms.items() if name.startswith("init")))
        stage1_losses.append(last_loss)
        stage1_term_history.append(last_terms)
        logger.info(
            "    phase %d/%d done, loss=%.3f (data=%.3f pose=%.3f beta=%.3f init=%.3f surf=%.3f, nit=%d, nfev=%d, success=%s)",
            phase_idx + 1,
            len(MOSHPP_STAGEI_WT_ANNEALING),
            last_loss,
            last_terms.get("data", float("nan")),
            last_terms.get("pose", float("nan")),
            last_terms.get("beta", float("nan")),
            init_loss,
            last_terms.get("surf", float("nan")),
            nit,
            nfev,
            success,
        )
        stage1_phase_results.append(
            {
                "phase": int(phase_idx + 1),
                "anneal_factor": float(anneal_factor),
                "nit": int(nit),
                "nfev": int(nfev),
                "success": bool(success),
                "message": message,
                "terms": dict(last_terms),
            }
        )

    betas_t = torch.as_tensor(betas_np, dtype=torch.float32, device=device)
    markers_latent_t = torch.as_tensor(markers_latent_np, dtype=torch.float32, device=device)
    debug = {
        "loss_curve": np.asarray(stage1_losses, dtype=np.float32),
        "term_history": stage1_term_history,
        "phase_results": stage1_phase_results,
        "ref_indices": ref_indices.copy(),
        "ref_pose": ref_poses_np.copy(),
        "ref_trans": ref_trans_np.copy(),
        "solver": f"joint_{active_stage1_shape_solver}_residual",
    }
    return betas_t.detach(), markers_latent_t.detach(), debug


def _fit_stageii(
    *,
    smpl,
    marker_model: SurfaceMarkerModel,
    observed_t,
    availability: np.ndarray,
    betas_opt,
    markers_latent_opt,
    pose_dim: int,
    body_end: int,
    optimize_toes: bool = False,
    stage2_iters: int,
    device: str,
    pose_prior=None,
    pose_prior_name: str = "l2",
):
    import torch

    n_frames = observed_t.shape[0]
    logger.info(
        "Stage II: fitting %d frames (%d iters/frame) with %s body-pose prior...",
        n_frames,
        stage2_iters,
        pose_prior_name,
    )
    pose_out = np.zeros((n_frames, pose_dim), dtype=np.float32)
    trans_out = np.zeros((n_frames, 3), dtype=np.float32)
    stage2_losses = np.full(n_frames, np.nan, dtype=np.float32)
    missing_markers = np.zeros(n_frames, dtype=np.int64)

    markers_latent_opt = markers_latent_opt.to(device)
    availability_t = torch.as_tensor(availability, dtype=torch.bool, device=device)

    # Pixel-perfect parity with MoSh++: with betas + markers_latent frozen at
    # Stage I output, recompute the closest 3 vertices (kNN) to anchor each
    # marker's local triangle frame in the *current* canonical mesh. These
    # frame_vids are then re-used across all frames during Stage II — what
    # MoSh stores as `markers_latent_vids` in its output pkl.
    with torch.no_grad():
        canonical_verts_s2, _ = smpl.get_joints_verts(
            torch.zeros(1, pose_dim, dtype=torch.float32, device=device),
            th_betas=betas_opt,
        )
        frame_vids_np = SurfaceMarkerModel.compute_dynamic_frame_vids(
            canonical_verts_s2[0].detach().cpu().numpy(),
            markers_latent_opt.detach().cpu().numpy(),
        )
    stageii_frame_vids = torch.as_tensor(frame_vids_np, dtype=torch.long, device=device)

    prev_pose = None
    prev_prev_pose = None
    prev_trans = None

    for frame_idx in range(n_frames):
        obs_frame = observed_t[frame_idx]
        mask = availability_t[frame_idx]
        missing_markers[frame_idx] = int((~mask).sum().item())

        if not torch.any(mask):
            if prev_pose is None or prev_trans is None:
                raise ValueError(f"Frame {frame_idx} has no visible markers and no previous solution to copy.")
            pose_out[frame_idx] = prev_pose.detach().cpu().numpy().reshape(-1)
            trans_out[frame_idx] = prev_trans.detach().cpu().numpy().reshape(-1)
            continue

        if prev_pose is None or prev_trans is None:
            pose, trans = _initialize_pose_and_trans(
                smpl=smpl,
                marker_model=marker_model,
                betas=betas_opt,
                markers_latent=markers_latent_opt,
                observed_frame=obs_frame,
                availability_mask=mask,
                pose_dim=pose_dim,
                device=device,
                frame_vids_override=stageii_frame_vids,
            )
            pose, trans, _ = _optimize_frame(
                smpl=smpl,
                marker_model=marker_model,
                betas_opt=betas_opt,
                markers_latent_opt=markers_latent_opt,
                obs_frame=obs_frame,
                mask=mask,
                init_pose=pose,
                init_trans=trans,
                body_end=body_end,
                stage2_iters=max(10, stage2_iters // 4),
                weights=compute_stageii_weights(int(mask.sum().item()), len(marker_model.vids)),
                prev_pose=None,
                prev_prev_pose=None,
                body_pose_weight_scale=10.0,
                optimize_toes=optimize_toes,
                pose_prior=pose_prior,
                frame_vids_override=stageii_frame_vids,
            )
            pose, trans, _ = _optimize_frame(
                smpl=smpl,
                marker_model=marker_model,
                betas_opt=betas_opt,
                markers_latent_opt=markers_latent_opt,
                obs_frame=obs_frame,
                mask=mask,
                init_pose=pose.detach(),
                init_trans=trans.detach(),
                body_end=body_end,
                stage2_iters=max(10, stage2_iters // 4),
                weights=compute_stageii_weights(int(mask.sum().item()), len(marker_model.vids)),
                prev_pose=None,
                prev_prev_pose=None,
                body_pose_weight_scale=5.0,
                optimize_toes=optimize_toes,
                pose_prior=pose_prior,
                frame_vids_override=stageii_frame_vids,
            )
        else:
            pose = prev_pose.detach().clone()
            trans = prev_trans.detach().clone()

        weights = compute_stageii_weights(int(mask.sum().item()), len(marker_model.vids))
        pose, trans, loss = _optimize_frame(
            smpl=smpl,
            marker_model=marker_model,
            betas_opt=betas_opt,
            markers_latent_opt=markers_latent_opt,
            obs_frame=obs_frame,
            mask=mask,
            init_pose=pose.detach(),
            init_trans=trans.detach(),
            body_end=body_end,
            stage2_iters=stage2_iters,
            weights=weights,
            prev_pose=prev_pose,
            prev_prev_pose=prev_prev_pose,
            body_pose_weight_scale=1.0,
            optimize_toes=optimize_toes,
            pose_prior=pose_prior,
            frame_vids_override=stageii_frame_vids,
        )

        pose_out[frame_idx] = pose.detach().cpu().numpy().reshape(-1)
        trans_out[frame_idx] = trans.detach().cpu().numpy().reshape(-1)
        stage2_losses[frame_idx] = loss
        prev_prev_pose = None if prev_pose is None else prev_pose.detach()
        prev_pose = pose.detach()
        prev_trans = trans.detach()

        if (frame_idx + 1) % 50 == 0 or frame_idx == n_frames - 1:
            logger.info(
                "  Frame %d/%d  loss=%.5f  missing=%d", frame_idx + 1, n_frames, loss, int(missing_markers[frame_idx])
            )

    return (
        pose_out,
        trans_out,
        {
            "frame_loss": stage2_losses,
            "missing_markers": missing_markers,
            "frame_vids": stageii_frame_vids.detach().cpu().numpy(),
        },
    )


def _compute_fitted_marker_error(
    *,
    smpl,
    marker_model: SurfaceMarkerModel,
    observed_t,
    availability: np.ndarray,
    betas_opt,
    markers_latent_opt,
    pose_aa_156: np.ndarray,
    trans: np.ndarray,
    pose_dim: int,
    device: str,
    frame_vids_override=None,
) -> dict:
    import torch

    pose_t = torch.as_tensor(pose_aa_156, dtype=torch.float32, device=device)
    trans_t = torch.as_tensor(trans, dtype=torch.float32, device=device)
    availability_t = torch.as_tensor(availability, dtype=torch.bool, device=device)

    with torch.no_grad():
        zero_pose = torch.zeros(1, pose_dim, dtype=torch.float32, device=device)
        canonical_verts, _ = smpl.get_joints_verts(zero_pose, th_betas=betas_opt)
        # Recompute frame_vids on the optimized canonical mesh + latents for
        # parity with MoSh++; fall back to override if the caller already
        # computed it (avoids redundant kNN).
        if frame_vids_override is None:
            frame_vids_np = SurfaceMarkerModel.compute_dynamic_frame_vids(
                canonical_verts[0].detach().cpu().numpy(),
                markers_latent_opt.detach().cpu().numpy(),
            )
            frame_vids_override = torch.as_tensor(frame_vids_np, dtype=torch.long, device=device)
        marker_coeffs = marker_model.latents_to_coeffs(
            canonical_verts[0], markers_latent_opt.to(device), frame_vids_override=frame_vids_override
        )

        frame_errors = []
        for frame_idx in range(pose_t.shape[0]):
            mask = availability_t[frame_idx]
            if not torch.any(mask):
                continue
            verts, _ = smpl.get_joints_verts(
                pose_t[frame_idx : frame_idx + 1],
                th_betas=betas_opt,
                th_trans=trans_t[frame_idx : frame_idx + 1],
            )
            pred_markers = marker_model.reconstruct(
                verts[0], coeffs=marker_coeffs, frame_vids_override=frame_vids_override
            )
            err = torch.linalg.norm(pred_markers[mask] - observed_t[frame_idx][mask], dim=-1)
            frame_errors.append(err.detach().cpu().numpy())

    if frame_errors:
        flat = np.concatenate(frame_errors)
        mean = float(np.mean(flat))
        median = float(np.median(flat))
        p95 = float(np.percentile(flat, 95))
        max_err = float(np.max(flat))
    else:
        mean = median = p95 = max_err = np.nan

    return {
        "num_frames": int(pose_t.shape[0]),
        "num_observations": int(0 if not frame_errors else sum(len(x) for x in frame_errors)),
        "mean_m": mean,
        "median_m": median,
        "p95_m": p95,
        "max_m": max_err,
        "mean_mm": mean * 1000.0,
        "median_mm": median * 1000.0,
        "p95_mm": p95 * 1000.0,
        "max_mm": max_err * 1000.0,
    }


def _stagei_free_pose_indices(body_end: int, *, optimize_toes: bool) -> np.ndarray:
    free = np.ones(body_end, dtype=bool)
    if not optimize_toes:
        free[_TOES_POSE_AA_SLICE] = False
    return np.flatnonzero(free).astype(np.int64)


def _optimize_frame(
    *,
    smpl,
    marker_model: SurfaceMarkerModel,
    betas_opt,
    markers_latent_opt,
    obs_frame,
    mask,
    init_pose,
    init_trans,
    body_end: int,
    stage2_iters: int,
    weights: StageIIWeights,
    prev_pose,
    prev_prev_pose,
    body_pose_weight_scale: float,
    optimize_toes: bool = False,
    pose_prior=None,
    frame_vids_override=None,
):
    """Per-frame Stage II solver.

    Uses LBFGS with strong-Wolfe line search on the (sum-based) loss. The
    per-frame problem has only 69 params (66 pose + 3 trans), is smooth, and
    is well-conditioned with warm start — exactly LBFGS's sweet spot.
    Adam's first-order moments aren't aggressive enough to navigate the
    narrow data/pose-prior equilibrium valley in 80 iterations.

    `frame_vids_override` (computed once at Stage II start from the optimized
    markers_latent) anchors the local triangle frame at the correct verts.
    """
    import torch

    pose = init_pose.clone().detach().requires_grad_(True)
    trans = init_trans.clone().detach().requires_grad_(True)
    pose.register_hook(_make_pose_freeze_hook(body_end, optimize_toes=optimize_toes))

    optimizer = torch.optim.LBFGS(
        [pose, trans],
        lr=1.0,
        max_iter=stage2_iters,
        history_size=20,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    # Pre-compute marker coeffs on the canonical mesh (shared across all
    # iterations — depends only on betas + markers_latent which are frozen).
    with torch.no_grad():
        canonical_verts, _ = smpl.get_joints_verts(
            torch.zeros_like(init_pose),
            th_betas=betas_opt,
        )
        marker_coeffs = marker_model.latents_to_coeffs(
            canonical_verts[0], markers_latent_opt, frame_vids_override=frame_vids_override
        )

    final_loss = [float("nan")]

    def closure():
        optimizer.zero_grad()
        verts, _ = smpl.get_joints_verts(pose, th_betas=betas_opt, th_trans=trans)
        pred_markers = marker_model.reconstruct(verts[0], coeffs=marker_coeffs, frame_vids_override=frame_vids_override)

        # Pixel-perfect parity with MoSh++: configured weights multiply residuals,
        # then the solver minimizes their sum of squares.
        residual = pred_markers[mask] - obs_frame[mask]
        data_loss = residual.pow(2).sum() * (weights.data**2)
        if pose_prior is not None:
            pose_loss = pose_prior(pose[:, 3:body_end]) * ((weights.pose_body * body_pose_weight_scale) ** 2)
        else:
            pose_loss = pose[:, 3:body_end].pow(2).sum() * ((weights.pose_body * body_pose_weight_scale) ** 2)

        velo_loss = torch.zeros((), dtype=torch.float32, device=pose.device)
        if prev_pose is not None and prev_prev_pose is not None:
            expected = 2.0 * prev_pose - prev_prev_pose
            velo_loss = (pose[:, :body_end] - expected[:, :body_end]).pow(2).sum() * (weights.velo**2)
        elif prev_pose is not None:
            velo_loss = (pose[:, :body_end] - prev_pose[:, :body_end]).pow(2).sum() * ((0.5 * weights.velo) ** 2)

        total = data_loss + pose_loss + velo_loss
        total.backward()
        final_loss[0] = float(total.detach().cpu().item())
        return total

    optimizer.step(closure)
    return pose.detach(), trans.detach(), final_loss[0]


def _initialize_pose_and_trans(
    *,
    smpl,
    marker_model: SurfaceMarkerModel,
    betas,
    markers_latent,
    observed_frame,
    availability_mask,
    pose_dim: int,
    device: str,
    frame_vids_override=None,
):
    import torch

    with torch.no_grad():
        verts, _ = smpl.get_joints_verts(
            torch.zeros(1, pose_dim, dtype=torch.float32, device=device),
            th_betas=betas,
        )
        template_markers = (
            marker_model.reconstruct(
                verts[0],
                markers_latent=markers_latent,
                canonical_verts=verts[0],
                frame_vids_override=frame_vids_override,
            )
            .detach()
            .cpu()
            .numpy()
        )

    obs_np = observed_frame.detach().cpu().numpy()
    mask_np = availability_mask.detach().cpu().numpy().astype(bool)
    if mask_np.sum() < 3:
        raise ValueError("At least 3 visible markers are required for rigid initialization.")

    rot, trans = _procrustes_align(template_markers[mask_np], obs_np[mask_np])
    pose_init = np.zeros((1, pose_dim), dtype=np.float32)
    pose_init[0, :3] = sRot.from_matrix(rot).as_rotvec().astype(np.float32)
    trans_init = trans.reshape(1, 3).astype(np.float32)

    return (
        torch.as_tensor(pose_init, dtype=torch.float32, device=device),
        torch.as_tensor(trans_init, dtype=torch.float32, device=device),
    )


def _build_local_frame_np(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1 = v1 - v0
    e2 = v2 - v0
    t1 = e1 / (np.linalg.norm(e1) + 1e-8)
    n = np.cross(e1, e2)
    n = n / (np.linalg.norm(n) + 1e-8)
    t2 = np.cross(n, t1)
    t2 = t2 / (np.linalg.norm(t2) + 1e-8)
    return t1.astype(np.float32), t2.astype(np.float32), n.astype(np.float32)
