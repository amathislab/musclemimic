from types import ModuleType
from typing import Any

from loco_mujoco.core.reward_termination.base import RewardTerminationHandler


class NoRewardTerminationHandler(RewardTerminationHandler):
    """Reward history never terminates the episode."""

    def update(self, reward: Any, carry: Any, backend: ModuleType):
        return carry, False
