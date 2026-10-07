"""AMASS motion selection with optional source-time bounds."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class AMASSMotionSpec:
    """Select one AMASS motion clip.

    Args:
        path: AMASS-root-relative ``.npz`` path without the extension.
        start_time_s: Inclusive clip start time [s].
        end_time_s: Exclusive clip end time [s]. ``None`` selects to the end.
    """

    path: str
    start_time_s: float = 0.0
    end_time_s: float | None = None

    def __post_init__(self) -> None:
        normalized_path = self.path.removesuffix(".npz")
        path_parts = normalized_path.split("/")
        if (
            not normalized_path
            or normalized_path.startswith("/")
            or "\\" in normalized_path
            or any(part in {"", ".", ".."} for part in path_parts)
        ):
            raise ValueError(f"AMASS motion path must be a normalized relative POSIX path, got {self.path!r}.")
        if normalized_path != self.path:
            object.__setattr__(self, "path", normalized_path)

        start_time_s = float(self.start_time_s)
        end_time_s = None if self.end_time_s is None else float(self.end_time_s)
        if not math.isfinite(start_time_s) or start_time_s < 0.0:
            raise ValueError(f"start_time_s must be finite and nonnegative, got {start_time_s}.")
        if end_time_s is not None and (not math.isfinite(end_time_s) or end_time_s <= start_time_s):
            raise ValueError(f"end_time_s must be finite and greater than start_time_s, got {end_time_s}.")
        object.__setattr__(self, "start_time_s", start_time_s)
        object.__setattr__(self, "end_time_s", end_time_s)

    @property
    def is_full_motion(self) -> bool:
        """Return whether the specification selects the complete source file."""
        return self.start_time_s == 0.0 and self.end_time_s is None

    @property
    def cache_key(self) -> str:
        """Return a stable identity that includes the clip bounds."""
        if self.is_full_motion:
            return self.path
        end = "end" if self.end_time_s is None else float(self.end_time_s).hex()
        return f"{self.path}#t={float(self.start_time_s).hex()}:{end}"

    @property
    def cache_suffix(self) -> str:
        """Return a compact hash for a bounded-motion cache filename."""
        return hashlib.sha256(self.cache_key.encode()).hexdigest()[:16]

    def frame_bounds(self, frequency_hz: float, num_frames: int) -> tuple[int, int]:
        """Convert the time bounds to an inclusive-exclusive frame interval.

        Args:
            frequency_hz: Motion sampling frequency [Hz].
            num_frames: Number of available frames.

        Returns:
            ``(start_frame, end_frame)`` with ``end_frame`` excluded.
        """
        frequency_hz = float(frequency_hz)
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError(f"frequency_hz must be finite and positive, got {frequency_hz}.")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}.")

        start_frame = round(self.start_time_s * frequency_hz)
        requested_end = num_frames if self.end_time_s is None else round(self.end_time_s * frequency_hz)
        end_frame = min(requested_end, num_frames)
        if start_frame < 0 or start_frame >= end_frame:
            raise ValueError(
                f"Motion clip {self.cache_key!r} selects no frames at {frequency_hz} Hz "
                f"from a {num_frames}-frame source."
            )
        return start_frame, end_frame


def as_amass_motion_spec(value: str | AMASSMotionSpec) -> AMASSMotionSpec:
    """Normalize a legacy path or an existing motion specification."""
    if isinstance(value, AMASSMotionSpec):
        return value
    if isinstance(value, str):
        return AMASSMotionSpec(path=value)
    raise TypeError(f"AMASS motion must be a path or AMASSMotionSpec, got {type(value).__name__}.")
