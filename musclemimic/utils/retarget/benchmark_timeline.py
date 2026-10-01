"""Timebase primitives shared by baseline retargeting and cache-only evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MotionTimeline:
    source_start_s: float
    source_end_s: float
    source_fps: float
    source_frame_count: int
    output_start_s: float
    output_end_s: float
    output_fps: float
    output_frame_count: int
    native_fps: float
    native_frame_count: int
    trim_start_frames: int = 1
    trim_end_frames: int = 1

    def validate(self) -> None:
        numeric = asdict(self)
        if not all(np.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("timeline contains a non-finite value")
        if min(self.source_fps, self.output_fps, self.native_fps) <= 0:
            raise ValueError("timeline frequencies must be positive")
        if min(self.source_frame_count, self.output_frame_count, self.native_frame_count) < 1:
            raise ValueError("timeline frame counts must be positive")
        if self.source_end_s < self.source_start_s or self.output_end_s < self.output_start_s:
            raise ValueError("timeline ends before it starts")

    def output_times(self) -> np.ndarray:
        """Exact endpoint-aligned timestamps used by trajectory materialization."""
        self.validate()
        if self.output_frame_count == 1:
            return np.array([self.output_start_s])
        return np.linspace(self.output_start_s, self.output_end_s, self.output_frame_count, endpoint=True)

    def to_analysis(self) -> dict[str, float | int]:
        return asdict(self)


def central_difference_timeline(
    *,
    source_frame_count: int,
    source_fps: float,
    native_frame_count: int,
    native_fps: float,
    output_frame_count: int,
    output_fps: float,
) -> MotionTimeline:
    """Timeline for a trajectory trimmed once at each end then endpoint-resampled.

    ``native_frame_count`` is the count *after* central-difference trimming.  This also
    derives metadata for immutable TERRA caches: use ``source_frame_count - 2`` and the
    source rate, regardless of whether the stored trajectory stayed at 100 Hz or was
    materialized from (for example) 120 to 100 Hz.
    """
    if source_frame_count < 3:
        raise ValueError("central-difference trajectories need at least three source frames")
    if native_frame_count < 1:
        raise ValueError("native_frame_count must be positive")
    timeline = MotionTimeline(
        source_start_s=0.0,
        source_end_s=(source_frame_count - 1) / float(source_fps),
        source_fps=float(source_fps),
        source_frame_count=int(source_frame_count),
        output_start_s=1.0 / float(native_fps),
        output_end_s=(native_frame_count) / float(native_fps),
        output_fps=float(output_fps),
        output_frame_count=int(output_frame_count),
        native_fps=float(native_fps),
        native_frame_count=int(native_frame_count),
    )
    timeline.validate()
    return timeline


def common_interval(source_start_s: float, source_end_s: float, timelines) -> tuple[float, float]:
    """Intersection of the source and every method output interval."""
    timelines = list(timelines)
    if not timelines:
        raise ValueError("at least one method timeline is required")
    start = max(float(source_start_s), *(t.output_start_s for t in timelines))
    end = min(float(source_end_s), *(t.output_end_s for t in timelines))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError(f"empty common source-time intersection [{start}, {end}]")
    return start, end


def sample_durations(times: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Voronoi-cell duration represented by each sample, clipped to an interval."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) < 0):
        raise ValueError("times must be a non-empty sorted vector")
    if end_s <= start_s:
        raise ValueError("duration interval must be non-empty")
    if len(times) == 1:
        edges = np.array([start_s, end_s])
    else:
        mid = 0.5 * (times[:-1] + times[1:])
        edges = np.concatenate(([-np.inf], mid, [np.inf]))
    left = np.maximum(edges[:-1], start_s)
    right = np.minimum(edges[1:], end_s)
    return np.maximum(right - left, 0.0)


def time_weighted_fraction(mask: np.ndarray, times: np.ndarray, start_s: float, end_s: float) -> float:
    """Fraction of real time covered by a Boolean sample series."""
    mask = np.asarray(mask, dtype=bool)
    weights = sample_durations(times, start_s, end_s)
    if mask.shape != weights.shape:
        raise ValueError("mask and times must have matching lengths")
    denominator = float(weights.sum())
    return float(np.dot(mask, weights) / denominator) if denominator > 0 else float("nan")


def map_boolean_intervals(source_mask: np.ndarray, source_fps: float, target_times: np.ndarray) -> np.ndarray:
    """Map sample-held Boolean source intervals without interpolating their edges away."""
    source_mask = np.asarray(source_mask, dtype=bool)
    target_times = np.asarray(target_times, dtype=float)
    indices = np.floor(target_times * float(source_fps) + 1e-12).astype(int)
    valid = (indices >= 0) & (indices < len(source_mask))
    out = np.zeros(len(target_times), dtype=bool)
    out[valid] = source_mask[indices[valid]]
    return out


def interpolate_samples(values: np.ndarray, source_fps: float, target_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate continuous source values at fixed, declared timestamps."""
    values = np.asarray(values, dtype=float)
    target_times = np.asarray(target_times, dtype=float)
    source_times = np.arange(len(values), dtype=float) / float(source_fps)
    flat = values.reshape(len(values), -1)
    out = np.empty((len(target_times), flat.shape[1]), dtype=float)
    for column in range(flat.shape[1]):
        out[:, column] = np.interp(target_times, source_times, flat[:, column])
    return out.reshape((len(target_times), *values.shape[1:]))
