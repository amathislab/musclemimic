from types import ModuleType
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from loco_mujoco.core.stateful_object import StatefulObject


class RewardTerminationHandler(StatefulObject):
    """Base interface for episode termination driven by reward history."""

    registered: Dict[str, type] = {}

    def __init__(self, env: Any, **handler_config: Any):
        self._handler_config = handler_config

    def reset(self, env: Any, model: Any, data: Any, carry: Any, backend: ModuleType):
        return data, carry

    def update(self, reward: float | np.ndarray | jnp.ndarray, carry: Any, backend: ModuleType):
        raise NotImplementedError

    @classmethod
    def register(cls) -> None:
        name = cls.__name__
        if name not in RewardTerminationHandler.registered:
            RewardTerminationHandler.registered[name] = cls
