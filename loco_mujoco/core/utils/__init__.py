from .backend import *
from .env import MDPInfo, Box
from .mujoco import *
from .decorators import info_property

_REWARD_EXPORTS = {
    "NoReward",
    "TargetXVelocityReward",
    "TargetVelocityGoalReward",
    "LocomotionReward",
}

__all__ = sorted({name for name in globals() if not name.startswith("_")} | _REWARD_EXPORTS)


def __getattr__(name):
    if name in _REWARD_EXPORTS:
        from ..reward import default as reward_default

        value = getattr(reward_default, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
