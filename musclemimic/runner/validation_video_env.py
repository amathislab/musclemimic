"""Environment setup shared by validation video recorder and worker."""

from __future__ import annotations

import os
from collections.abc import MutableMapping


def configure_validation_video_worker_env(
    env: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Force validation video workers onto CPU JAX with headless MuJoCo."""
    target = os.environ if env is None else env
    target.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    target["JAX_PLATFORMS"] = "cpu"
    target["JAX_PLATFORM_NAME"] = "cpu"
    target["MUJOCO_GL"] = "egl"
    return target
