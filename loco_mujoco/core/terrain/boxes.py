"""Parametric box terrain: the representation shared by retargeting and simulation.

A terrain is a list of axis-aligned-in-its-own-frame boxes, each with a yaw about z and a
pitch about its own lateral axis, on top of the flat floor at z=0. That is deliberately
restrictive. Two properties fall out of it that an arbitrary mesh would not give:

* **It is augmentable.** OmniRetarget's terrain augmentation is a uniform ``scale``
  substitution over a box assembly; a mesh makes that meaningless, which would forfeit
  the main reason the method was adopted.
* **It is the same object everywhere.** The boxes that the retargeting solver constrains
  against, the geoms the policy walks on, and the heights the height-map observation
  reads are all generated from one :class:`TerrainSpec`. They cannot drift apart because
  there is only one description.

A pitched box is how a ramp is represented: its top face is a plane sloping along the
box's own major axis, and the box is thick enough that its underside stays below z=0, so
the geometry is a solid wedge standing on the floor rather than a slab hanging over a gap.
Every query that used to return "the box top" therefore takes an (x, y) - see
:meth:`BoxSpec.top_at`, which is the one to reach for; the scalar :attr:`BoxSpec.top` is
the *highest* point of the top face and is only a bound.

:class:`TerrainSpec` is the pure geometry (serialisable, no MuJoCo, no env);
:class:`BoxTerrain` is the env-side :class:`~loco_mujoco.core.terrain.Terrain` plugin
that realises it in a model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Union

import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.terrain import Terrain
from loco_mujoco.core.utils.backend import assert_backend_is_supported

#: Geom-name prefix marking a box as a seat rather than ground. It extends
#: :attr:`BoxTerrain.GEOM_PREFIX`, which is how `scene_geom_ids` finds the static
#: environment, so a seat collides like any other terrain while staying identifiable as
#: one - which is what :attr:`TerrainSpec.walkable` needs to tell support from solidity.
#: The standalone ``terra.terrain`` package builds the names.
SEAT_GEOM_PREFIX = "terrain_box_seat"

#: Greatest pitch a box may carry, radians. Beyond this the top face is closer to a wall
#: than to a surface, and the ``margin / cos(pitch)`` in :meth:`BoxSpec.vertical_span` -
#: which maps a horizontal margin onto the sloped face - runs away. Real walking ramps are
#: well inside it; the steepest in AMASS is ~27 deg.
MAX_PITCH = np.deg2rad(60.0)


@dataclass(frozen=True)
class BoxSpec:
    """One box: centre position, half-extents in its own frame, yaw about z, pitch about u.

    The rotation is ``Rz(yaw) @ Ry(pitch)``, so `yaw` orients the box's major axis `u` in
    the ground plane and `pitch` then tilts that axis out of it. A positive pitch tips the
    top face *down* along +u, which is the sign convention MuJoCo's own ``Ry`` uses; a ramp
    climbing along +u therefore carries a negative pitch.
    """

    pos: tuple[float, float, float]
    size: tuple[float, float, float]  # half-extents
    yaw: float = 0.0
    #: Tilt of the box's major axis out of the ground plane, radians. 0 for every box that
    #: is not a ramp, which keeps a flat terrain's serialisation unchanged.
    pitch: float = 0.0
    name: str = ""

    def __post_init__(self):
        object.__setattr__(self, "pos", tuple(float(v) for v in self.pos))
        object.__setattr__(self, "size", tuple(float(v) for v in self.size))
        object.__setattr__(self, "yaw", float(self.yaw))
        object.__setattr__(self, "pitch", float(self.pitch))
        if any(s <= 0 for s in self.size):
            raise ValueError(f"Box half-extents must be positive, got {self.size}")
        if abs(self.pitch) > MAX_PITCH:
            raise ValueError(
                f"Box pitch {np.degrees(self.pitch):.1f} deg exceeds the "
                f"{np.degrees(MAX_PITCH):.0f} deg limit; a surface that steep is a wall"
            )

    @property
    def is_sloped(self) -> bool:
        return self.pitch != 0.0

    @property
    def rotation(self) -> np.ndarray:
        """The 3x3 world-from-box rotation, ``Rz(yaw) @ Ry(pitch)``.

        Its columns are the box's own axes in world coordinates: `u` (major), `v`
        (lateral, always horizontal - ``Ry`` leaves it alone) and the top-face normal.
        """
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        return np.array([
            [cy * cp, -sy, cy * sp],
            [sy * cp, cy, sy * sp],
            [-sp, 0.0, cp],
        ])

    @property
    def top(self) -> float:
        """World z of the *highest* point of the top face.

        Equal to the top face's height everywhere only when the box is flat. Anything
        asking what surface is under a foot wants :meth:`top_at`.
        """
        return self.pos[2] + self.size[0] * abs(np.sin(self.pitch)) + self.size[2] * np.cos(self.pitch)

    @property
    def bottom(self) -> float:
        """World z of the lowest point of the box."""
        return self.pos[2] - self.size[0] * abs(np.sin(self.pitch)) - self.size[2] * np.cos(self.pitch)

    @property
    def quat(self) -> tuple[float, float, float, float]:
        """MuJoCo ``[w, x, y, z]`` quaternion for ``Rz(yaw) @ Ry(pitch)``."""
        cy, sy = np.cos(self.yaw / 2), np.sin(self.yaw / 2)
        cp, sp = np.cos(self.pitch / 2), np.sin(self.pitch / 2)
        return (float(cy * cp), float(-sy * sp), float(cy * sp), float(sy * cp))

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """World points -> box frame (centred on the box, aligned with its axes)."""
        p = np.asarray(points, dtype=float).reshape(-1, 3) - np.asarray(self.pos)
        return p @ self.rotation  # R^T p, written as a right-multiply

    def to_world(self, local: np.ndarray) -> np.ndarray:
        """Box-frame points -> world."""
        p = np.asarray(local, dtype=float).reshape(-1, 3)
        return p @ self.rotation.T + np.asarray(self.pos)

    def vertical_span(self, x, y, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Where a vertical line through each (x, y) enters and leaves the box.

        The one primitive behind both :meth:`contains_xy` and :meth:`top_at`, so the two
        can never disagree about where the box is. A ray-slab test in the box's own frame:
        the line stays a line under rotation, and each half-extent gives an interval of the
        parameter - here the world z itself - for which that axis is satisfied.

        Doing it this way rather than intersecting the top *face* matters once a box is
        pitched. The vertical silhouette of a tilted slab is longer than its top face by
        the projection of its end walls, and a query that only knows the top face reports
        no surface over that strip - a foot standing against the raised end of a ramp reads
        the floor. Pinned against MuJoCo's own ray cast by
        `test_height_at_matches_a_mujoco_ray_cast`.

        Args:
            x, y: World coordinates, broadcast against each other.
            margin: Horizontal growth of the box before the test, metres. Applied along the
                surface, so a slope is grown by `margin` measured on the ground rather than
                along the face.

        Returns:
            ``(enter_z, exit_z)``. The line misses the box wherever ``enter_z > exit_z``.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        dx, dy = x - self.pos[0], y - self.pos[1]
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        sp, cp = np.sin(self.pitch), np.cos(self.pitch)
        q = cy * dx + sy * dy  # horizontal distance along the fall line

        # The line z -> (x, y, z) in box coordinates is `a + z * w`.
        a = (cp * q + sp * self.pos[2], -sy * dx + cy * dy, sp * q - cp * self.pos[2])
        w = (-sp, 0.0, cp)
        half = (self.size[0] + margin / cp, self.size[1] + margin, self.size[2])

        lo = np.full(np.broadcast(x, y).shape, -np.inf)
        hi = np.full(lo.shape, np.inf)
        for a_i, w_i, s_i in zip(a, w, half, strict=True):
            if abs(w_i) < 1e-12:
                # This axis does not vary along the line: it either admits every z or none.
                miss = np.abs(a_i) > s_i
                lo, hi = np.where(miss, np.inf, lo), np.where(miss, -np.inf, hi)
                continue
            e0, e1 = (-s_i - a_i) / w_i, (s_i - a_i) / w_i
            lo = np.maximum(lo, np.minimum(e0, e1))
            hi = np.minimum(hi, np.maximum(e0, e1))
        return lo, hi

    def top_at(self, x, y, margin: float = 0.0) -> np.ndarray:
        """Highest world z of the box above each (x, y); 0 where the box is not over it.

        The walkable surface this box provides. Use it wherever a scalar box top used to
        do - :attr:`top` is only the highest point anywhere on the box.
        """
        lo, hi = self.vertical_span(x, y, margin)
        return np.where(lo <= hi, hi, 0.0)

    def top_plane_at(self, x, y) -> np.ndarray:
        """Height of the top face's *plane* at each (x, y), extended past the box's edges.

        What a rigid foot resting on this box would put a landmark at, even a landmark
        overhanging the edge. Distinct from :meth:`top_at`, which stops at the box: use
        that to ask what surface is at a point, and this to ask where a body resting on
        *this* surface would be at a point.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        q = np.cos(self.yaw) * (x - self.pos[0]) + np.sin(self.yaw) * (y - self.pos[1])
        return self.pos[2] + (self.size[2] - np.sin(self.pitch) * q) / np.cos(self.pitch)

    def penetration(self, points: np.ndarray) -> np.ndarray:
        """Depth by which each point lies inside the box, 0 outside.

        Depth is the distance to the *nearest* face, so a point resting exactly on the
        top face reads 0. That is what makes this usable as a free-space test on a motion
        whose feet are supposed to be touching the surface.
        """
        local = np.abs(self.to_local(points))
        inside_per_axis = np.asarray(self.size) - local  # >0 on an axis => inside on it
        inside = np.all(inside_per_axis > 0, axis=1)
        return np.where(inside, inside_per_axis.min(axis=1), 0.0)

    def contains_xy(self, x: np.ndarray, y: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Whether each (x, y) lies within the box's footprint, grown by `margin`.

        Args:
            x, y: World coordinates, broadcast against each other.
            margin: How far outside the footprint still counts, metres. A foot approaching
                a step is *beside* the box rather than over it, so anything asking "what is
                the foot about to meet" has to test a grown footprint.
        """
        lo, hi = self.vertical_span(x, y, margin)
        return lo <= hi

    def top_face_points(self, spacing: float) -> np.ndarray:
        """Grid of world points on the top face, at most ``spacing`` apart."""
        pts = []
        for half in (self.size[0], self.size[1]):
            n = max(int(np.ceil(2 * half / max(spacing, 1e-6))) + 1, 2)
            pts.append(np.linspace(-half, half, n))
        gu, gv = np.meshgrid(pts[0], pts[1], indexing="ij")
        local = np.stack(
            [gu.ravel(), gv.ravel(), np.full(gu.size, self.size[2])], axis=1
        )
        return self.to_world(local)



@dataclass(frozen=True)
class TerrainSpec:
    """A flat floor at z=0 plus a set of boxes standing on it."""

    boxes: tuple[BoxSpec, ...] = ()
    #: Free-text record of how this terrain was produced. Carried through save/load so a
    #: terrain file is self-describing; never read by any code path.
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "boxes",
            tuple(b if isinstance(b, BoxSpec) else BoxSpec(**b) for b in self.boxes),
        )

    def __len__(self) -> int:
        return len(self.boxes)

    @property
    def is_flat(self) -> bool:
        return len(self.boxes) == 0

    @property
    def walkable(self) -> TerrainSpec:
        """This terrain without its seats: the surfaces a foot may come to rest on.

        Every box here is solid and every box is collided with, but they do not all answer
        the same question. "What is the ground under this foot" has one right answer and a
        seat is never it - a chair is walked *past*, not onto - so each of the foot-facing
        queries (:meth:`height_at` for stance detection and the foot-orientation datum,
        :meth:`height_near` for swing clearance, and the quality scorer's support datum)
        reads this rather than the whole terrain.

        Non-penetration deliberately does not: the robot still may not pass through a
        chair. The split is between *support* and *solidity*, which for a walking surface
        coincide and for a seat do not.

        Inert on every terrain with no seat in it, where it returns an equal spec.
        """
        kept = tuple(b for b in self.boxes if not b.name.startswith(SEAT_GEOM_PREFIX))
        if len(kept) == len(self.boxes):
            return self
        return TerrainSpec(boxes=kept, provenance={**self.provenance, "walkable": True})

    # -- serialisation ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "boxes": [asdict(b) for b in self.boxes],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TerrainSpec:
        return cls(
            boxes=tuple(BoxSpec(**b) for b in d.get("boxes", ())),
            provenance=dict(d.get("provenance", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> TerrainSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_env_params(self) -> dict:
        """``terrain_params`` for :class:`BoxTerrain`."""
        return {"boxes": [asdict(b) for b in self.boxes]}

    def scale_height(self, factor: float) -> TerrainSpec:
        """Return a terrain whose elevations are scaled about the ``z=0`` floor.

        Horizontal placement, top-face footprint, primitive order and names are preserved.
        An upright box is the direct world-z scale: both its centre height and vertical
        half-extent are multiplied by ``factor``.  A pitched box needs a reconstruction,
        because scaling only its local z half-extent changes neither the rise nor the top
        plane.  Its horizontal run is held fixed and its slope becomes::

            tan(pitch') = factor * tan(pitch)

        The new local half-extents and centre make every point on the *top plane* equal to
        the old point with its world z multiplied by ``factor``.  The vertical projection
        of the buried thickness is scaled as well, so a ramp that extended below the floor
        remains a solid support rather than becoming a hanging slab.

        ``factor == 1`` returns this object.  That explicit identity path is important to
        the nominal-regression gate: it avoids even round-off from reconstructing ramps.

        Args:
            factor: Positive finite world-z scale.

        Returns:
            The height-morphed terrain.

        Raises:
            ValueError: If ``factor`` is not positive and finite, or makes a ramp steeper
                than :data:`MAX_PITCH`.
        """
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"terrain height scale must be positive and finite, got {factor!r}")
        if factor == 1.0:
            return self

        scaled = []
        for box in self.boxes:
            if not box.is_sloped:
                scaled.append(BoxSpec(
                    pos=(box.pos[0], box.pos[1], factor * box.pos[2]),
                    size=(box.size[0], box.size[1], factor * box.size[2]),
                    yaw=box.yaw,
                    pitch=box.pitch,
                    name=box.name,
                ))
                continue

            pitch = float(np.arctan(factor * np.tan(box.pitch)))
            if abs(pitch) > MAX_PITCH:
                raise ValueError(
                    f"terrain height scale {factor:g} makes box {box.name!r} pitch "
                    f"{np.degrees(pitch):.1f} deg, beyond the "
                    f"{np.degrees(MAX_PITCH):.0f} deg limit"
                )

            cp0, cp1 = float(np.cos(box.pitch)), float(np.cos(pitch))
            u_half = box.size[0] * cp0 / cp1
            z_half = factor * box.size[2] * cp0 / cp1

            # Keep the top-face centre fixed in xy and scale it in world z.  Together with
            # the half-extents above this maps the complete top plane by (x, y, z) ->
            # (x, y, factor*z), not merely its high point.
            top_centre = box.to_world(np.array([[0.0, 0.0, box.size[2]]]))[0]
            top_centre[2] *= factor
            cy, sy = np.cos(box.yaw), np.sin(box.yaw)
            sp, cp = np.sin(pitch), np.cos(pitch)
            normal = np.array([cy * sp, sy * sp, cp])
            pos = top_centre - z_half * normal
            scaled.append(BoxSpec(
                pos=tuple(pos),
                size=(u_half, box.size[1], z_half),
                yaw=box.yaw,
                pitch=pitch,
                name=box.name,
            ))

        return TerrainSpec(
            boxes=tuple(scaled),
            provenance={
                **self.provenance,
                "augmentation": {"transform": "height_scale", "factor": factor},
            },
        )

    # -- geometry -----------------------------------------------------------------------

    def height_at(self, x, y) -> np.ndarray:
        """Support height at each (x, y): the highest box surface covering it, else 0."""
        return self.height_near(x, y, 0.0)

    def height_near(self, x, y, reach: float) -> np.ndarray:
        """Highest box surface within `reach` in xy of each point, else 0 (the floor).

        The same query as :meth:`height_at` with `reach` = 0, and the two must stay one
        function. A foot swinging towards a step has to clear that step's top before it
        arrives over it, so a swing requirement reads the surface with a reach of about a
        foot length - and anything the requirement is *compared against* has to be read the
        same way. Measuring the source's own clearance with `height_at` and enforcing it
        with `height_near` asks the foot to stand a whole riser above the tread it is on,
        because the tread above is within reach of it.

        Args:
            x, y: World coordinates, broadcast against each other.
            reach: How far ahead in xy to look, metres.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        h = np.zeros(np.broadcast(x, y).shape, dtype=float)
        for b in self.boxes:
            h = np.maximum(h, b.top_at(x, y, reach))
        return h

    def boxes_under(self, foot_xy) -> list[BoxSpec]:
        """Every box any of `foot_xy` is over that stands above the floor, highest first.

        A box whose surface under this foot is at or below z=0 is left out: it offers the
        foot nothing the floor does not, and the low end of a fitted ramp meets the floor
        exactly, so including it would let that ramp's plane be *extrapolated below zero*
        for a foot standing at its foot.

        Args:
            foot_xy: (n, 2) landmark positions of the one foot, in world xy.
        """
        foot_xy = np.asarray(foot_xy, dtype=float).reshape(-1, 2)
        found = []
        for b in self.boxes:
            over = b.contains_xy(foot_xy[:, 0], foot_xy[:, 1])
            if not over.any():
                continue
            top = float(np.max(b.top_at(foot_xy[over, 0], foot_xy[over, 1])))
            if top > 0.0:
                found.append((top, b))
        return [b for _, b in sorted(found, key=lambda t: -t[0])]

    def supporting_box(self, foot_xy) -> BoxSpec | None:
        """The box a foot standing over `foot_xy` is resting on, or None for the floor.

        The highest one wins, because boxes are solid from z=0: a foot with any landmark
        over a taller box is resting on that box, not on what is under the others.

        Shared by :meth:`support_plane_at` and :meth:`support_normal_at` so that the height
        and the tilt of "the surface this foot is on" can never be read off two different
        surfaces.

        Args:
            foot_xy: (n, 2) landmark positions of the one foot, in world xy.
        """
        under = self.boxes_under(foot_xy)
        return under[0] if under else None

    def support_plane_at(self, foot_xy, x, y) -> float:
        """Height at (x, y) of the surface a foot standing over `foot_xy` is resting on.

        The datum for "is this landmark where a foot on this terrain would put it". Not the
        same question as :meth:`height_at`, and the difference is a step: a foot resting on
        one has all of its landmarks at the tread's height whether or not each one is over
        the tread, because the sole is one rigid plane and the overhanging part of it is in
        free space.

        **That premise holds only while the surfaces under the foot share a tilt**, and it
        is the surfaces that decide, not the foot. Where they do not - a sole spanning the
        top of a ramp, part on the incline and part on the landing - there is no single
        plane to extend, and extending either one puts the far end of the foot centimetres
        inside the ground or above it. Each landmark is then read against the surface
        actually beneath it, which for a continuous bend is what it is resting on; a
        landmark overhanging every box still falls back to the plane, which is the case the
        rule exists for.

        Args:
            foot_xy: (n, 2) landmark positions of the one foot, in world xy.
            x, y: Where to evaluate the resulting plane.

        Returns:
            The height, or 0 where the foot is over nothing but the floor.
        """
        under = self.boxes_under(foot_xy)
        if not under:
            return 0.0
        best = under[0]
        up = best.rotation[:, 2]
        if all(float(b.rotation[:, 2] @ up) > 1.0 - 1e-9 for b in under):
            return float(best.top_plane_at(x, y))
        beneath = [b for b in under if bool(b.contains_xy(np.asarray(x), np.asarray(y)))]
        return (float(max(b.top_at(x, y) for b in beneath)) if beneath
                else float(best.top_plane_at(x, y)))

    def support_normal_at(self, foot_xy) -> np.ndarray:
        """Upward unit normal of the surface a foot standing over `foot_xy` is resting on.

        The orientation half of :meth:`support_plane_at`, and needed for the same reason:
        "flat on the surface" is a statement about the surface, and on a slope that is not
        world-vertical. Anything calibrated against ``+z`` instead absorbs the slope angle
        into what it takes to be a constant.

        Returns:
            (3,) unit vector; ``[0, 0, 1]`` for the floor and for every unpitched box.
        """
        best = self.supporting_box(foot_xy)
        return np.array([0.0, 0.0, 1.0]) if best is None else best.rotation[:, 2].copy()

    def penetration(self, points: np.ndarray) -> np.ndarray:
        """Deepest intrusion of each point into any box (0 if outside all of them).

        The floor is not included: floor penetration is z < 0 and is checked separately,
        because a motion is normalised to put its lowest contact at z=0 and so grazes it
        by construction.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if not self.boxes:
            return np.zeros(len(points))
        return np.max([b.penetration(points) for b in self.boxes], axis=0)

    def surface_points(self, spacing: float = 0.15, ground_range=(-3.0, 3.0), ground_size: int = 8) -> np.ndarray:
        """Point samples of the walkable surface, for the interaction mesh.

        Box tops are sampled at ``spacing``; the floor is sampled on a coarse grid with
        the box footprints removed, so a point never sits *underneath* a box where no
        foot could reach it.
        """
        parts = [b.top_face_points(spacing) for b in self.boxes]

        g = np.linspace(ground_range[0], ground_range[1], ground_size)
        gx, gy = np.meshgrid(g, g, indexing="ij")
        ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        if self.boxes:
            covered = np.zeros(len(ground), dtype=bool)
            for b in self.boxes:
                covered |= b.contains_xy(ground[:, 0], ground[:, 1])
            ground = ground[~covered]
        parts.append(ground)

        return np.concatenate(parts, axis=0) if parts else ground


# --------------------------------------------------------------------------------------
# Env-side plugin
# --------------------------------------------------------------------------------------


class BoxTerrain(Terrain):
    """Static terrain made of boxes on the flat floor.

    Realises a :class:`TerrainSpec` as real MuJoCo geoms, so the policy walks on the same
    geometry the retargeter solved against. Construct it through the usual env arguments::

        env = MyoFullBody(terrain_type="BoxTerrain", terrain_params=spec.to_env_params())
    """

    GEOM_PREFIX = "terrain_box"

    def __init__(self, env: Any, boxes: Sequence[dict] = (), rgba=(0.55, 0.45, 0.35, 1.0), **kwargs: Any):
        """
        Args:
            env: The environment instance.
            boxes: Box descriptions, as produced by ``TerrainSpec.to_env_params()``.
            rgba: Colour of the terrain geoms.
        """
        super().__init__(env, **kwargs)
        self.rgba = tuple(rgba)
        self._pack(TerrainSpec(boxes=tuple(boxes)))

    def _pack(self, spec: TerrainSpec) -> None:
        """Adopt `spec` and flatten it into the arrays height sampling reads.

        Its own method because :meth:`sample_heights_at_points` runs under jit and cannot
        walk a tuple of dataclasses. Anything constructing a `BoxTerrain` without going
        through ``__init__`` has to call this, or the sampler reads a stale field set -
        which is how adding `pitch` broke two tests that packed the arrays themselves.
        """
        self.spec_ = spec
        n = len(spec)
        self._pos = np.array([b.pos for b in spec.boxes], dtype=float).reshape(n, 3)
        self._size = np.array([b.size for b in spec.boxes], dtype=float).reshape(n, 3)
        self._yaw = np.array([b.yaw for b in spec.boxes], dtype=float).reshape(n)
        self._pitch = np.array([b.pitch for b in spec.boxes], dtype=float).reshape(n)

    # -- Terrain interface ---------------------------------------------------------------

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        """Add one box geom per box to the worldbody.

        Raises:
            ValueError: If a box is named without :attr:`GEOM_PREFIX`. Downstream code finds
                the static environment *by that prefix* - `scene_geom_ids` hands the set to
                the retargeting solver's non-penetration term and to the quality metric - so
                a differently-named geom is built, drawn, and then silently collided with by
                nothing. It cost a full 40-motion ramp run to notice: the boxes were there in
                the video, the fit gate passed, and every clip solved with no terrain
                constraint at all while penetration read exactly 0.0 mm.
        """
        wb = spec.worldbody
        for i, b in enumerate(self.spec_.boxes):
            name = b.name or f"{self.GEOM_PREFIX}_{i}"
            if not name.startswith(self.GEOM_PREFIX):
                raise ValueError(
                    f"Terrain box {i} is named {name!r}, which does not start with "
                    f"{self.GEOM_PREFIX!r}. `scene_geom_ids` selects the static environment "
                    f"by that prefix, so this box would be invisible to non-penetration and "
                    f"to every terrain metric."
                )
            wb.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=b.pos,
                quat=b.quat,
                size=b.size,
                rgba=self.rgba,
                # Match the floor so a foot behaves the same on either surface.
                friction=(1.0, 0.1, 0.1),
                condim=3,
                contype=1,
                conaffinity=1,
                group=2,
            )
        return spec

    def reset(self, env, model, data, carry, backend: ModuleType):
        assert_backend_is_supported(backend)
        return data, carry

    def update(self, env, model, data, carry, backend: ModuleType):
        assert_backend_is_supported(backend)
        return model, data, carry

    @property
    def is_dynamic(self) -> bool:
        return False

    def sample_heights_at_points(
        self,
        x: np.ndarray | jnp.ndarray,
        y: np.ndarray | jnp.ndarray,
        model: MjModel | Model,
        carry: Any,
        backend: ModuleType,
    ) -> np.ndarray | jnp.ndarray:
        """Height of the walkable surface at each (x, y): highest covering box, else 0.

        The backend twin of :meth:`TerrainSpec.height_at`, and it has to stay one with it -
        a policy reading a height map that disagrees with the surface the retargeter solved
        against is the drift this module exists to prevent. Pinned by
        `test_env_height_sampling_matches_the_spec`.

        Vectorised over boxes so it stays jit-compatible under the jnp backend.
        """
        assert_backend_is_supported(backend)

        h = backend.zeros_like(x)
        if len(self._pitch) == 0:
            return h

        pos = backend.asarray(self._pos)
        size = backend.asarray(self._size)
        yaw = backend.asarray(self._yaw)
        pitch = backend.asarray(self._pitch)

        # (n_points, n_boxes). The same ray-slab test as `BoxSpec.vertical_span`, written
        # branchlessly: the lateral axis is the one whose direction component is zero, so
        # it becomes a plain containment test rather than an interval.
        dx = x[..., None] - pos[:, 0]
        dy = y[..., None] - pos[:, 1]
        cy, sy = backend.cos(yaw), backend.sin(yaw)
        cp, sp = backend.cos(pitch), backend.sin(pitch)
        q = cy * dx + sy * dy

        lo = backend.full(dx.shape, -backend.inf)
        hi = backend.full(dx.shape, backend.inf)
        for a_i, w_i, s_i in (
            (cp * q + sp * pos[:, 2], -sp, size[:, 0]),
            (sp * q - cp * pos[:, 2], cp, size[:, 2]),
        ):
            safe = backend.where(backend.abs(w_i) < 1e-12, 1.0, w_i)
            e0, e1 = (-s_i - a_i) / safe, (s_i - a_i) / safe
            flat = backend.abs(w_i) < 1e-12  # axis constant along the line
            miss = flat & (backend.abs(a_i) > s_i)
            lo = backend.where(flat, backend.where(miss, backend.inf, lo),
                               backend.maximum(lo, backend.minimum(e0, e1)))
            hi = backend.where(flat, backend.where(miss, -backend.inf, hi),
                               backend.minimum(hi, backend.maximum(e0, e1)))
        lateral = backend.abs(-sy * dx + cy * dy) <= size[:, 1]

        heights = backend.where(lateral & (lo <= hi), hi, 0.0)
        return backend.max(heights, axis=-1)


BoxTerrain.register()
