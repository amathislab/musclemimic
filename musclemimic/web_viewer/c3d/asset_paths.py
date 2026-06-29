from __future__ import annotations

import os
from pathlib import Path

import loco_mujoco

C3D_MODEL_PATH_KEY = "MUSCLEMIMIC_C3D_MODEL_PATH"
MOSHPP_ASSETS_PATH_KEY = "MUSCLEMIMIC_MOSHPP_ASSETS_PATH"
POSE_BODY_PRIOR_PATH_KEY = "MUSCLEMIMIC_MOSHPP_POSE_BODY_PRIOR_PATH"
POSE_HAND_PRIOR_PATH_KEY = "MUSCLEMIMIC_MOSHPP_POSE_HAND_PRIOR_PATH"
HEAD_MARKER_CORR_PATH_KEY = "MUSCLEMIMIC_MOSHPP_HEAD_MARKER_CORR_PATH"


def _configured_path(key: str) -> Path | None:
    value = os.environ.get(key)
    if value:
        return Path(value).expanduser()

    value = loco_mujoco.load_path_config().get(key)
    return Path(value).expanduser() if value else None


def _configured_asset_path(path_key: str, filename: str) -> Path | None:
    path = _configured_path(path_key)
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Configured {path_key} does not exist: {path}")
        return path

    root = _configured_path(MOSHPP_ASSETS_PATH_KEY)
    if root is None:
        return None
    if not root.exists():
        raise FileNotFoundError(f"Configured {MOSHPP_ASSETS_PATH_KEY} does not exist: {root}")

    path = root / filename
    return path if path.exists() else None


def _explicit_path(path: str | Path | None, *, description: str) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def resolve_c3d_model_path(path: str | Path | None = None) -> Path | None:
    return Path(path).expanduser() if path is not None else _configured_path(C3D_MODEL_PATH_KEY)


def resolve_pose_body_prior_path(path: str | Path | None = None) -> Path | None:
    return _explicit_path(path, description="MoSh++ pose body prior") or _configured_asset_path(
        POSE_BODY_PRIOR_PATH_KEY,
        "pose_body_prior.pkl",
    )


def resolve_pose_hand_prior_path(path: str | Path | None = None) -> Path | None:
    return _explicit_path(path, description="MoSh++ pose hand prior") or _configured_asset_path(
        POSE_HAND_PRIOR_PATH_KEY,
        "pose_hand_prior.npz",
    )


def resolve_head_marker_corr_path(path: str | Path | None = None) -> Path | None:
    return _explicit_path(path, description="MoSh++ head marker correlation") or _configured_asset_path(
        HEAD_MARKER_CORR_PATH_KEY,
        "ssm_head_marker_corr.npz",
    )
