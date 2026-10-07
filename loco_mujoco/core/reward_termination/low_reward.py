from types import ModuleType
from typing import Any

from flax import struct

from loco_mujoco.core.reward_termination.base import RewardTerminationHandler


@struct.dataclass
class LowRewardTerminationState:
    low_reward_steps: Any


class LowRewardTerminationHandler(RewardTerminationHandler):
    """Terminate after reward stays below a threshold for consecutive steps."""

    def __init__(self, env: Any, threshold: float, patience: int = 100, **handler_config: Any):
        super().__init__(env, **handler_config)
        self.threshold = float(threshold)
        self.patience = int(patience)
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")

    def init_state(self, env: Any, key: Any, model: Any, data: Any, backend: ModuleType):
        return LowRewardTerminationState(low_reward_steps=backend.zeros((), dtype=int))

    def reset(self, env: Any, model: Any, data: Any, carry: Any, backend: ModuleType):
        return data, carry.replace(
            reward_termination_state=self.init_state(env, None, model, data, backend),
        )

    def update(self, reward: Any, carry: Any, backend: ModuleType):
        state = carry.reward_termination_state
        low_reward_steps = backend.where(
            reward < self.threshold,
            state.low_reward_steps + 1,
            0,
        )
        carry = carry.replace(
            reward_termination_state=state.replace(low_reward_steps=low_reward_steps),
        )
        return carry, low_reward_steps >= self.patience
