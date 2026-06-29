from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


class ViserTendonLineRenderer:
    """Render MuJoCo tendon visualization geoms as one Viser line-segment handle."""

    _SCENE_MIN_GEOMS = 50000
    _GEOMS_PER_TENDON = 100
    _SCENE_MAX_RETRIES = 3
    _MIN_SEGMENT_LENGTH = 1e-7

    def __init__(
        self,
        model: mujoco.MjModel,
        server: Any,
        *,
        path: str = "/tendons",
        line_width: int = 3,
        visible: bool = True,
    ) -> None:
        self._model = model
        self._server = server
        self._path = path
        self._line_width = line_width
        self._visible = visible

        self._scene = None
        self._option = None
        self._camera = None
        self._handle = None
        self._capacity = 0
        self._scene_maxgeom = 0

        if model.ntendon == 0:
            return

        self._scene_maxgeom = max(
            self._SCENE_MIN_GEOMS,
            model.ngeom + model.ntendon * self._GEOMS_PER_TENDON,
        )
        self._scene = mujoco.MjvScene(model, maxgeom=self._scene_maxgeom)
        self._option = mujoco.MjvOption()
        self._option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1
        self._camera = mujoco.MjvCamera()

    @property
    def available(self) -> bool:
        return self._scene is not None

    @property
    def visible(self) -> bool:
        return self._visible

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._handle is not None:
            self._handle.visible = visible

    def update(self, data: mujoco.MjData) -> None:
        if not self.available:
            return
        if not self._visible:
            self.set_visible(False)
            return

        tendon_indices, geoms = self._collect_tendon_geoms(data)
        points, colors = self._build_segments(tendon_indices, geoms)
        if points is None or colors is None:
            # Degenerate wrap geometry can leave a frame with no valid segments.
            # Preserve the previous buffer instead of showing all tendons absent.
            if self._handle is not None:
                self._handle.visible = True
            return

        points, colors = self._pad_segments(points, colors)
        if self._handle is None:
            self._handle = self._server.scene.add_line_segments(
                self._path,
                points=points,
                colors=colors,
                line_width=self._line_width,
                visible=True,
            )
            return

        self._handle.points = points
        self._handle.colors = colors
        self._handle.visible = True

    def _collect_tendon_geoms(self, data: mujoco.MjData):
        if self._scene is None or self._option is None or self._camera is None:
            return np.empty(0, dtype=np.int64), None

        for attempt in range(self._SCENE_MAX_RETRIES):
            mujoco.mjv_updateScene(
                self._model,
                data,
                self._option,
                None,
                self._camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self._scene,
            )
            if self._scene.ngeom < self._scene_maxgeom:
                break
            if attempt == self._SCENE_MAX_RETRIES - 1:
                break
            self._scene_maxgeom *= 2
            self._scene = mujoco.MjvScene(self._model, maxgeom=self._scene_maxgeom)

        ngeom = self._scene.ngeom
        if ngeom == 0:
            return np.empty(0, dtype=np.int64), None

        geoms = self._scene.geoms[:ngeom]
        objtype_arr = np.array([g.objtype for g in geoms], dtype=np.int32)
        tendon_indices = np.where(objtype_arr == int(mujoco.mjtObj.mjOBJ_TENDON))[0]
        return tendon_indices, geoms

    def _build_segments(
        self, tendon_indices: np.ndarray, geoms
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        count = 0 if tendon_indices is None else len(tendon_indices)
        if count == 0 or geoms is None:
            return None, None

        positions = np.empty((count, 3), dtype=np.float32)
        axes = np.empty((count, 3), dtype=np.float32)
        halves = np.empty(count, dtype=np.float32)
        rgbas = np.empty((count, 4), dtype=np.float32)
        for i, geom_idx in enumerate(tendon_indices):
            geom = geoms[geom_idx]
            positions[i] = geom.pos
            axes[i] = geom.mat[:, 2]
            halves[i] = geom.size[2]
            rgbas[i] = geom.rgba

        offsets = axes * halves[:, np.newaxis]
        p0 = positions - offsets
        p1 = positions + offsets
        points = np.stack([p0, p1], axis=1).astype(np.float32)

        rgbas = np.clip(rgbas, 0.0, 1.0)
        valid = self._valid_segment_mask(points, rgbas)
        if not np.any(valid):
            return None, None

        points = points[valid]
        rgbas = rgbas[valid]
        colors = self._colors_from_rgba(rgbas)
        return points, colors

    def _valid_segment_mask(self, points: np.ndarray, rgbas: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(points[:, 1] - points[:, 0], axis=1)
        extent = max(1.0, float(self._model.stat.extent))
        coord_limit = max(1000.0, extent * 1000.0)
        length_limit = max(100.0, extent * 100.0)
        return (
            np.isfinite(points).all(axis=(1, 2))
            & np.isfinite(rgbas).all(axis=1)
            & (lengths > self._MIN_SEGMENT_LENGTH)
            & (lengths < length_limit)
            & (np.max(np.abs(points), axis=(1, 2)) < coord_limit)
        )

    @staticmethod
    def _colors_from_rgba(rgbas: np.ndarray) -> np.ndarray:
        rgb = rgbas[:, :3].copy()
        dark = np.linalg.norm(rgb, axis=1) < 0.05
        if np.any(dark):
            rgb[dark] = np.array([0.95, 0.32, 0.08], dtype=np.float32)
        alpha = np.maximum(rgbas[:, 3:4], 0.45)
        colors_uint8 = (rgb * 0.85 * alpha * 255.0).astype(np.uint8)
        return np.stack([colors_uint8, colors_uint8], axis=1)

    def _pad_segments(self, points: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = points.shape[0]
        self._capacity = max(self._capacity, count)
        if count == self._capacity:
            return points, colors

        padded_points = np.empty((self._capacity, 2, 3), dtype=np.float32)
        padded_colors = np.zeros((self._capacity, 2, 3), dtype=np.uint8)
        padded_points[:count] = points
        padded_colors[:count] = colors
        padded_points[count:] = points[0, 0]
        return padded_points, padded_colors
