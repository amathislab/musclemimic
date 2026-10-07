"""Versioned GMR cache identities shared by retargeting and downloads."""

from pathlib import Path

from .motion_spec import AMASSMotionSpec, as_amass_motion_spec

GMR_CACHE_NAMESPACE = "gmr_v6"


def gmr_motion_relative_path(
    env_name: str,
    motion: str | AMASSMotionSpec,
    namespace: str = GMR_CACHE_NAMESPACE,
    *,
    cache_variant: str | None = None,
) -> Path:
    """Return a cache-root-relative path preserving source-time bounds [s]."""
    if not namespace or namespace in {".", ".."} or "/" in namespace or "\\" in namespace:
        raise ValueError("GMR cache namespace must be one nonempty path component.")
    if cache_variant is not None and (
        not cache_variant or cache_variant in {".", ".."} or "/" in cache_variant or "\\" in cache_variant
    ):
        raise ValueError("GMR cache variant must be one nonempty path component.")
    spec = as_amass_motion_spec(motion)
    root = Path(env_name.removeprefix("Mjx")) / namespace
    if cache_variant is not None:
        root = root / "_variants" / cache_variant
    if spec.is_full_motion:
        return root / f"{spec.path}.npz"
    return root / "_clips" / f"{spec.path}__{spec.cache_suffix}.npz"
