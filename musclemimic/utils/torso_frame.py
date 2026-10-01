"""Anatomical torso frames built only from landmark positions."""

from __future__ import annotations

import numpy as np


def torso_frame(
    pelvis: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
) -> np.ndarray:
    """Return right-handed ``[forward, left, up]`` frames with shape ``(..., 3, 3)``."""
    pelvis = np.asarray(pelvis, dtype=float)
    left_shoulder = np.asarray(left_shoulder, dtype=float)
    right_shoulder = np.asarray(right_shoulder, dtype=float)
    up = 0.5 * (left_shoulder + right_shoulder) - pelvis
    up /= _norm(up)
    left = left_shoulder - right_shoulder
    left -= np.sum(left * up, axis=-1, keepdims=True) * up
    left /= _norm(left)
    forward = np.cross(left, up)
    forward /= _norm(forward)
    return np.stack((forward, left, up), axis=-1)


def torso_frame_torch(pelvis, left_shoulder, right_shoulder):
    """Torch equivalent of :func:`torso_frame`, preserving autograd."""
    import torch

    up = 0.5 * (left_shoulder + right_shoulder) - pelvis
    up = up / torch.linalg.vector_norm(up, dim=-1, keepdim=True).clamp_min(1e-8)
    left = left_shoulder - right_shoulder
    left = left - torch.sum(left * up, dim=-1, keepdim=True) * up
    left = left / torch.linalg.vector_norm(left, dim=-1, keepdim=True).clamp_min(1e-8)
    forward = torch.linalg.cross(left, up, dim=-1)
    forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack((forward, left, up), dim=-1)


def relative_torso_frame_torch(
    left_pelvis,
    right_pelvis,
    left_shoulder,
    right_shoulder,
):
    """Return the shoulder frame relative to a position-derived pelvis frame."""
    import torch

    pelvis = 0.5 * (left_pelvis + right_pelvis)
    up = 0.5 * (left_shoulder + right_shoulder) - pelvis
    up = up / torch.linalg.vector_norm(up, dim=-1, keepdim=True).clamp_min(1e-8)

    def frame(left_axis):
        left_axis = left_axis - torch.sum(left_axis * up, dim=-1, keepdim=True) * up
        left_axis = left_axis / torch.linalg.vector_norm(
            left_axis, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        forward = torch.linalg.cross(left_axis, up, dim=-1)
        return torch.stack((forward, left_axis, up), dim=-1)

    pelvis_frame = frame(left_pelvis - right_pelvis)
    shoulder_frame = frame(left_shoulder - right_shoulder)
    return pelvis_frame.transpose(-1, -2) @ shoulder_frame


def _norm(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("pelvis and shoulder landmarks do not define a torso frame")
    return norm
