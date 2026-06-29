from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .asset_paths import resolve_head_marker_corr_path

logger = logging.getLogger(__name__)

try:
    import ezc3d
except ImportError:  # pragma: no cover - exercised in environments without the c3d extra.
    ezc3d = None


# Marker-type-dependent defaults copied from the public moshpp config.
MOSHPP_STAGEI_WT_ANNEALING = (1.0, 0.5, 0.25, 0.125)
MOSHPP_STAGEI_WT_POSE_BODY = 3.0
MOSHPP_STAGEI_WT_BETAS = 10.0
MOSHPP_STAGEI_WT_INIT = 300.0
MOSHPP_STAGEI_WT_DATA = 75.0
MOSHPP_STAGEI_WT_SURF = 10000.0
MOSHPP_STAGEII_WT_DATA = 400.0
MOSHPP_STAGEII_WT_POSE_BODY = 1.6
MOSHPP_STAGEII_WT_VELO = 2.5
MOSHPP_STAGEII_WT_ANNEALING = 2.5
MOSHPP_NUM_TRAIN_MARKERS = 46


MARKER_TYPE_DISTANCES = {
    "body": 0.0095,
    "wrist": 0.0390,
}

WRIST_MARKER_LABELS = {"LIWR", "LOWR", "RIWR", "ROWR"}
_SMPLX_EYEBALLS_PATH = Path("references/moshpp/support_data/smplx_eyeballs.npz")
_SMPLX_NO_EYEBALL_VIDS: np.ndarray | None = None

# Subset of moshpp marker-label aliases that matter for standard Plug-in Gait /
# gait-lab body marker sets.
MOSHPP_LABEL_MAP: dict[str, str] = {
    "HEAD_TOP": "ARIEL",
    "TOPBACK": "C7",
    "NECK_BASE": "C7",
    "CHEST": "CLAV",
    "UPTHRX": "CLAV",
    "STERNUM": "STRN",
    "SETRNUM": "STRN",
    "LOTHRX": "STRN",
    "LOBACK": "T10",
    "MIDBACK": "T8",
    "MID_BACK": "T8",
    "RBAC": "RBAK",
    "RBAC-1": "RBAK",
    "LASI": "LFWT",
    "LPSI": "LBWT",
    "RASI": "RFWT",
    "RPSI": "RBWT",
    "LWRA": "LIWR",
    "LWRB": "LOWR",
    "RWRA": "RIWR",
    "RWRB": "ROWR",
    "LMELB": "LELBIN",
    "RMELB": "RELBIN",
    "LMKNE": "LKNI",
    "RMKNE": "RKNI",
    "LMANK": "LHEEI",
    "RMANK": "RHEEI",
    "LANKIN": "LHEEI",
    "RANKIN": "RHEEI",
    "L_INWRIST": "LIWR",
    "R_INWRIST": "RIWR",
    "L_OUTWRIST": "LOWR",
    "R_OUTWRIST": "ROWR",
    "L_INKNEE": "LKNI",
    "R_INKNEE": "RKNI",
    "L_OUTKNEE": "LKNE",
    "R_OUTKNEE": "RKNE",
    "L_ANKLE": "LANK",
    "R_ANKLE": "RANK",
    "L_KNEE": "LKNE",
    "R_KNEE": "RKNE",
    "L_SHIN": "LSHN",
    "R_SHIN": "RSHN",
    "LSHIN": "LSHN",
    "RSHIN": "RSHN",
    "L_HEEL": "LHEE",
    "R_HEEL": "RHEE",
    "L_PINKYTOE": "LMT5",
    "R_PINKYTOE": "RMT5",
    "L_TOETIP": "LTOE",
    "R_TOETIP": "RTOE",
    "LTOE1": "LTOE",
    "RTOE1": "RTOE",
}

# SMPL-H marker vertices used by MoSh++ after label canonicalization.
MOSHPP_SMPLH_MARKER_VIDS: dict[str, int] = {
    "C7": 3470,
    "CLAV": 3171,
    "LBHD": 182,
    "LFHD": 0,
    "RBHD": 3694,
    "RFHD": 3512,
    "STRN": 3506,
    "T10": 3016,
    "RBAK": 5273,
    "LBWT": 3122,
    "LFWT": 857,
    "RBWT": 6544,
    "RFWT": 4343,
    "LSHO": 1861,
    "RSHO": 5322,
    "LUPA": 1443,
    "RUPA": 4918,
    "LELB": 1666,
    "LELBIN": 1725,
    "RELB": 5135,
    "RELBIN": 5194,
    "LFRM": 1568,
    "RFRM": 5037,
    "LIWR": 2112,
    "LOWR": 2108,
    "RIWR": 5573,
    "ROWR": 5568,
    "LFIN": 2174,
    "RFIN": 5635,
    "LTHI": 1454,
    "RTHI": 4927,
    "LKNE": 1053,
    "LKNI": 1058,
    "RKNE": 4538,
    "RKNI": 4544,
    "LSHN": 1082,
    "RSHN": 4568,
    "LTIB": 1112,
    "RTIB": 4598,
    "LANK": 3327,
    "RANK": 6728,
    "LHEEI": 3432,
    "RHEEI": 6832,
    "LHEE": 3387,
    "RHEE": 6786,
    "LMT5": 3346,
    "RMT5": 6747,
    "LTOE": 3233,
    "RTOE": 6633,
}

MOSHPP_SMPLX_MARKER_VIDS: dict[str, int] = {
    "C7": 3832,
    "CLAV": 5533,
    "LBHD": 2026,
    "LFHD": 707,
    "RBHD": 3066,
    "RFHD": 2198,
    "STRN": 5531,
    "T10": 5623,
    "RBAK": 6127,
    "LBWT": 5697,
    "LFWT": 3486,
    "RBWT": 8391,
    "RFWT": 6248,
    "LSHO": 4481,
    "RSHO": 6627,
    "LUPA": 4030,
    "RUPA": 6777,
    "LELB": 4302,
    "LELBIN": 4363,
    "RELB": 7040,
    "RELBIN": 7099,
    "LFRM": 4198,
    "RFRM": 6942,
    "LIWR": 4726,
    "LOWR": 4722,
    "RIWR": 7462,
    "ROWR": 7458,
    "LFIN": 4788,
    "RFIN": 7524,
    "LTHI": 4088,
    "RTHI": 6832,
    "LKNE": 3682,
    "LKNI": 3688,
    "RKNE": 6443,
    "RKNI": 6449,
    "LSHN": 3712,
    "RSHN": 6473,
    "LTIB": 3745,
    "RTIB": 6503,
    "LANK": 5882,
    "RANK": 8576,
    "LHEEI": 8892,
    "RHEEI": 8680,
    "LHEE": 8846,
    "RHEE": 8634,
    "LMT5": 5901,
    "RMT5": 8595,
    "LTOE": 5787,
    "RTOE": 8481,
}


def _marker_vids_for_surface(surface_model_type: str) -> dict[str, int]:
    if surface_model_type == "smplh":
        return MOSHPP_SMPLH_MARKER_VIDS
    if surface_model_type == "smplx":
        return MOSHPP_SMPLX_MARKER_VIDS
    raise ValueError(f"Unsupported surface_model_type={surface_model_type!r}; expected 'smplh' or 'smplx'.")


def _smplx_no_eyeball_vids(n_verts: int) -> np.ndarray | None:
    """Return MoSh++'s SMPL-X kNN search vertex subset.

    `TransformedCoeffs` excludes the eyeballs for SMPL-X meshes and, in the
    reference code, builds the base range with `np.arange(10474)`. Keep that
    off-by-one behavior for parity.
    """

    global _SMPLX_NO_EYEBALL_VIDS
    if n_verts != 10475 or not _SMPLX_EYEBALLS_PATH.exists():
        return None
    if _SMPLX_NO_EYEBALL_VIDS is None:
        eyeballs = set(np.load(_SMPLX_EYEBALLS_PATH)["eyeballs"].astype(np.int64).tolist())
        _SMPLX_NO_EYEBALL_VIDS = np.asarray(
            [vid for vid in range(10474) if vid not in eyeballs],
            dtype=np.int64,
        )
    return _SMPLX_NO_EYEBALL_VIDS


@dataclass(frozen=True)
class MarkerLayout:
    marker_vids: dict[str, int]
    marker_type: dict[str, str]
    marker_type_mask: dict[str, np.ndarray]
    m2b_distance: dict[str, float]
    surface_model_type: str = "smplx"

    @property
    def labels(self) -> list[str]:
        return list(self.marker_vids.keys())


@dataclass(frozen=True)
class StageIWeights:
    anneal_factor: float
    data: float
    pose_body: float
    betas: float
    init_by_type: dict[str, float]
    surf: float


@dataclass(frozen=True)
class StageIIWeights:
    anneal_factor: float
    data: float
    pose_body: float
    velo: float


@dataclass(frozen=True)
class StageIHeadMarkerCorrelation:
    path: str
    labels: tuple[str, ...]
    marker_indices: np.ndarray
    corr: np.ndarray


@dataclass(frozen=True)
class PreparedMarkerObservations:
    positions: np.ndarray
    labels: list[str]
    unknown_labels: list[str]


def canonicalize_marker_label(label: str, labels_map: dict[str, str] | None = None) -> str:
    normalized = label.replace(" ", "")
    normalized = normalized.split(":")[-1]
    normalized = normalized.upper()
    if labels_map is None:
        labels_map = MOSHPP_LABEL_MAP
    return labels_map.get(normalized, normalized)


def compute_marker_availability_mask(markers: np.ndarray) -> np.ndarray:
    """True where a marker is available in a frame."""
    markers = np.asarray(markers, dtype=np.float32)
    return ~(np.isnan(markers).any(axis=-1) | np.all(np.isclose(markers, 0.0), axis=-1))


def compute_stagei_weights(
    num_markers: int,
    marker_types: Iterable[str],
    anneal_factor: float,
) -> StageIWeights:
    marker_types = tuple(dict.fromkeys(marker_types))
    init_by_type = dict.fromkeys(marker_types, MOSHPP_STAGEI_WT_INIT * anneal_factor)
    return StageIWeights(
        anneal_factor=anneal_factor,
        data=(MOSHPP_STAGEI_WT_DATA / anneal_factor) * (MOSHPP_NUM_TRAIN_MARKERS / max(1, num_markers)),
        pose_body=MOSHPP_STAGEI_WT_POSE_BODY * anneal_factor,
        betas=MOSHPP_STAGEI_WT_BETAS * anneal_factor,
        init_by_type=init_by_type,
        surf=MOSHPP_STAGEI_WT_SURF,
    )


def compute_stageii_weights(num_observed_markers: int, num_total_markers: int) -> StageIIWeights:
    missing_ratio = 0.0 if num_total_markers == 0 else (num_total_markers - num_observed_markers) / num_total_markers
    anneal_factor = 1.0 + missing_ratio * MOSHPP_STAGEII_WT_ANNEALING
    return StageIIWeights(
        anneal_factor=anneal_factor,
        data=MOSHPP_STAGEII_WT_DATA * (MOSHPP_NUM_TRAIN_MARKERS / max(1, num_observed_markers)),
        pose_body=MOSHPP_STAGEII_WT_POSE_BODY * anneal_factor,
        velo=MOSHPP_STAGEII_WT_VELO,
    )


def load_stagei_head_marker_correlation(
    corr_path: str | Path | None,
    labels: Iterable[str],
) -> StageIHeadMarkerCorrelation | None:
    """Load MoSh++ Stage-I head marker covariance residual metadata.

    MoSh++ activates this block only when every `mrk_labels` entry in the npz is
    present in the marker layout. Missing labels disable the block instead of
    failing the fit, matching `chmosh.py`.
    """

    if corr_path is None:
        return None

    path = Path(corr_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Head marker correlation file not found: {path}")

    data = np.load(path, allow_pickle=True)
    if "mrk_labels" not in data or "corr" not in data:
        raise ValueError(f"Expected {path} to contain 'mrk_labels' and 'corr'.")

    corr_labels = tuple(canonicalize_marker_label(str(label)) for label in data["mrk_labels"])
    corr = np.asarray(data["corr"], dtype=np.float32)
    if corr.ndim != 2:
        raise ValueError(f"Expected head marker corr matrix with shape (K, H), got {corr.shape}.")
    if corr.shape[1] != len(corr_labels):
        raise ValueError(f"Head marker corr columns ({corr.shape[1]}) do not match mrk_labels ({len(corr_labels)}).")

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    missing = [label for label in corr_labels if label not in label_to_index]
    if missing:
        logger.info(
            "Head marker correlation inactive; missing markers in layout: %s",
            ", ".join(missing),
        )
        return None

    return StageIHeadMarkerCorrelation(
        path=str(path),
        labels=corr_labels,
        marker_indices=np.asarray([label_to_index[label] for label in corr_labels], dtype=np.int64),
        corr=corr,
    )


def resolve_stagei_head_marker_corr_path(
    corr_path: str | Path | None,
    *,
    smpl_model_path: str | Path | None = None,
    surface_model_type: str = "smplx",
) -> str | None:
    if surface_model_type != "smplx":
        return None
    configured = resolve_head_marker_corr_path(corr_path)
    if configured is not None:
        return str(configured)

    candidates: list[Path] = []
    if smpl_model_path is not None:
        model_path = Path(smpl_model_path).expanduser()
        if model_path.is_file():
            candidates.extend(
                [
                    model_path.parent / "ssm_head_marker_corr.npz",
                    model_path.parent.parent / "ssm_head_marker_corr.npz",
                ]
            )
        else:
            candidates.extend(
                [
                    model_path / "ssm_head_marker_corr.npz",
                    model_path.parent / "ssm_head_marker_corr.npz",
                ]
            )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def build_marker_layout(
    labels: Iterable[str],
    *,
    labels_map: dict[str, str] | None = None,
    wrist_markers_on_stick: bool = False,
    surface_model_type: str = "smplx",
) -> MarkerLayout:
    marker_vid_db = _marker_vids_for_surface(surface_model_type)
    canonical_labels = [canonicalize_marker_label(label, labels_map=labels_map) for label in labels]
    available_labels = sorted({label for label in canonical_labels if label in marker_vid_db})

    marker_type = {}
    for label in available_labels:
        marker_type[label] = "wrist" if wrist_markers_on_stick and label in WRIST_MARKER_LABELS else "body"

    marker_vids = {label: marker_vid_db[label] for label in available_labels}
    marker_type_mask = {}
    for marker_kind in sorted(set(marker_type.values())):
        marker_type_mask[marker_kind] = np.array(
            [marker_type[label] == marker_kind for label in available_labels],
            dtype=bool,
        )

    return MarkerLayout(
        marker_vids=marker_vids,
        marker_type=marker_type,
        marker_type_mask=marker_type_mask,
        m2b_distance={marker_kind: MARKER_TYPE_DISTANCES[marker_kind] for marker_kind in marker_type_mask},
        surface_model_type=surface_model_type,
    )


def prepare_marker_observations(
    positions: np.ndarray,
    labels: Iterable[str],
    *,
    labels_map: dict[str, str] | None = None,
    surface_model_type: str = "smplx",
) -> PreparedMarkerObservations:
    marker_vid_db = _marker_vids_for_surface(surface_model_type)
    groups: dict[str, list[int]] = {}
    unknown_labels: list[str] = []

    for idx, label in enumerate(labels):
        canonical = canonicalize_marker_label(label, labels_map=labels_map)
        if canonical == "NAN":
            continue
        if canonical not in marker_vid_db:
            unknown_labels.append(canonical)
            continue
        groups.setdefault(canonical, []).append(idx)

    ordered_labels = sorted(groups)
    collapsed = np.zeros((positions.shape[0], len(ordered_labels), 3), dtype=np.float32)
    for out_idx, canonical in enumerate(ordered_labels):
        collapsed[:, out_idx] = _nanmean_with_all_nan(positions[:, groups[canonical], :], axis=1)

    return PreparedMarkerObservations(
        positions=collapsed,
        labels=ordered_labels,
        unknown_labels=sorted(set(unknown_labels)),
    )


def pick_stagei_frames(
    markers: np.ndarray,
    *,
    num_frames: int,
    seed: int = 100,
    least_avail_markers: float = 1.0,
    strict: bool = True,
) -> np.ndarray:
    if markers.ndim != 3:
        raise ValueError(f"Expected markers with shape (T, N, 3), got {markers.shape}")

    availability = compute_marker_availability_mask(markers).sum(axis=-1) / max(1, markers.shape[1])
    threshold = float(least_avail_markers)

    while True:
        eligible = np.flatnonzero(availability >= threshold)
        if len(eligible) >= num_frames:
            break
        if strict:
            raise ValueError(
                f"Not enough frames have at least {threshold * 100:.1f}% of the markers: "
                f"requested {num_frames}, found {len(eligible)}"
            )
        threshold -= 0.01
        if threshold < 0.01:
            raise ValueError("Unable to find enough stage-I frames with sufficient marker coverage.")

    # Match MoSh++ random_strict for a single mocap sequence: shuffle all frame
    # ids with the legacy RandomState seed, keep the first num_frames eligible
    # frames, then apply the second random choice MoSh++ uses before returning.
    rng = np.random.RandomState(seed)
    candidates: list[int] = []
    for frame_idx in rng.choice(markers.shape[0], markers.shape[0], replace=False):
        frame_idx = int(frame_idx)
        if availability[frame_idx] >= threshold:
            candidates.append(frame_idx)
        if len(candidates) >= num_frames:
            break
    if len(candidates) < num_frames:
        raise ValueError(
            f"Not enough frames have at least {threshold * 100:.1f}% of the markers: "
            f"requested {num_frames}, found {len(candidates)}"
        )
    ids = rng.choice(len(candidates), num_frames, replace=False)
    return np.asarray(candidates, dtype=np.int64)[ids]


def _read_point_labels(point_params: dict, marker_count: int) -> list[str]:
    """Read C3D POINT:LABELS chunks and align label count with point count."""

    def label_key_order(key: str) -> int:
        if key == "LABELS":
            return 0
        suffix = key.removeprefix("LABELS")
        return int(suffix) if suffix.isdigit() else 10_000

    label_keys = [key for key in point_params if key == "LABELS" or (key.startswith("LABELS") and key[6:].isdigit())]
    labels: list[str] = []
    for key in sorted(label_keys, key=label_key_order):
        labels.extend(str(label) for label in point_params[key].get("value", []))

    if len(labels) < marker_count:
        labels.extend(f"UNLABELED_{idx + 1}" for idx in range(len(labels), marker_count))
    elif len(labels) > marker_count:
        labels = labels[:marker_count]

    return labels


def load_c3d_markers(path: str) -> tuple[np.ndarray, list[str], float]:
    """Load C3D markers as (frames, markers, xyz) in meters, preserving Z-up world axes."""
    if ezc3d is None:
        raise ImportError("ezc3d is required for C3D support. Install the optional `c3d` extra.")

    c3d = ezc3d.c3d(path)
    fps = float(c3d["header"]["points"]["frame_rate"])

    points = np.asarray(c3d["data"]["points"][:3], dtype=np.float32).transpose(2, 1, 0)
    labels = _read_point_labels(c3d["parameters"]["POINT"], points.shape[1])
    units = c3d["parameters"]["POINT"].get("UNITS", {}).get("value", ["mm"])
    unit = str(units[0]).strip().lower() if units else "mm"
    scale = {
        "millimeter": 1000.0,
        "millimeters": 1000.0,
        "mm": 1000.0,
        "centimeter": 100.0,
        "centimeters": 100.0,
        "cm": 100.0,
        "meter": 1.0,
        "meters": 1.0,
        "m": 1.0,
    }.get(unit, 1000.0)
    points /= scale

    residuals = c3d["data"].get("meta_points", {}).get("residuals")
    if residuals is not None:
        residuals = np.asarray(residuals, dtype=np.float32)
        if residuals.ndim == 3:
            residuals = residuals.transpose(2, 1, 0)[..., 0]
        elif residuals.ndim == 2:
            residuals = residuals.T
        else:
            residuals = None
    elif c3d["data"]["points"].shape[0] > 3:
        residuals = np.asarray(c3d["data"]["points"][3], dtype=np.float32).T

    if residuals is not None:
        points[residuals < 0] = np.nan

    zero_mask = np.all(np.isclose(points, 0.0), axis=-1)
    points[zero_mask] = np.nan

    return points, labels, fps


def _nanmean_with_all_nan(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = ~np.isnan(values)
    counts = valid.sum(axis=axis)
    safe_values = np.where(valid, values, 0.0)
    sums = safe_values.sum(axis=axis)
    mean = sums / np.maximum(counts, 1)
    mean[counts == 0] = np.nan
    return mean.astype(np.float32)
