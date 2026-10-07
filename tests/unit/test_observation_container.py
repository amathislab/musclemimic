"""Tests for stateful observation container state."""

from loco_mujoco.core.observations.base import (
    ObservationContainer,
    StatefulObservation,
)


def test_observation_state_type_is_stable_after_lock():
    container = ObservationContainer()
    container["history"] = StatefulObservation("history")
    container.lock()

    first = container.init_state(None, None, None, None, None)
    second = container.init_state(None, None, None, None, None)

    assert type(first) is type(second)
