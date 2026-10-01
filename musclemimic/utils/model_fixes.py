"""Small, versioned corrections applied to bundled biomechanical model specifications."""

from __future__ import annotations

from mujoco import MjSpec


# musclemimic-models 1.0.5 places this side selector only about 2 mm outside the 20 mm
# BIClong wrap sphere. Near ordinary shoulder poses MuJoCo can then alternate between wrap
# sides on adjacent frames, changing tendon length by 35--40 mm while every joint moves
# smoothly. A 20 mm lateral coordinate leaves the selector well outside the sphere and was
# checked across the non-flat 100: it removes both observed BIClong topology switches and
# creates none. This changes only MuJoCo's wrap-side choice; it is not a collision geom or a
# tracked kinematic site.
MYOFULLBODY_BICLONG_SIDESITE_Y = 0.020


# The bundled gastrocnemius wrap cylinders sit on a topology boundary at their 55 mm
# radius. On ordinary bent-knee poses MuJoCo can drop the cylinder from the spatial path
# for a handful of frames (four wrap points become two), changing GasLat length by about
# 22.7 mm although the joint trajectory is smooth. A 52 mm radius keeps the path on the
# wrapped branch. It was checked against every trajectory in nonflat100, ramp40, chair40,
# and flat: neither side produced a threshold event (worst relative event 0.0355). The
# correction is symmetric and changes a non-colliding muscle wrap geom, not body geometry.
MYOFULLBODY_GASLAT_WRAP_RADIUS = 0.052


def apply_myofullbody_model_fixes(spec: MjSpec) -> MjSpec:
    """Apply local corrections needed on every compiled MyoFullBody model.

    The helper is deliberately tolerant of reduced test specs and future model variants
    that omit these sites. Keeping the edit at the spec level makes CPU MuJoCo, MJX,
    retargeting, policy environments, and offline metrics all compile the same muscle path.
    """
    for name in (
        "BIClong_ellipsoid_BIClong_2_sidesite",
        "BIClong_ellipsoid_BIClong_2_sidesite_left",
    ):
        site = spec.site(name)
        if site is not None:
            site.pos[1] = MYOFULLBODY_BICLONG_SIDESITE_Y
    for name in ("GasLat_at_shank_r_wrap", "GasLat_at_shank_l_wrap"):
        geom = spec.geom(name)
        if geom is not None:
            geom.size[0] = MYOFULLBODY_GASLAT_WRAP_RADIUS
    return spec
