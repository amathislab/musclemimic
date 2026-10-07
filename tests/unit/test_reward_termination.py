import os
from types import ModuleType
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

import loco_mujoco.core.control_functions  # noqa: F401
import loco_mujoco.core.domain_randomizer  # noqa: F401
import loco_mujoco.core.initial_state_handler  # noqa: F401
import loco_mujoco.core.reward  # noqa: F401
import loco_mujoco.core.reward_termination  # noqa: F401
import loco_mujoco.core.terminal_state_handler  # noqa: F401
import loco_mujoco.core.terrain  # noqa: F401
from loco_mujoco.core.mujoco_base import Mujoco
from loco_mujoco.core.observations import ObservationType
from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.reward_termination.low_reward import LowRewardTerminationHandler
from musclemimic.core.mujoco_mjx import Mjx


class _AlwaysLowReward(Reward):
    def __call__(
        self,
        state: np.ndarray | jnp.ndarray,
        action: np.ndarray | jnp.ndarray,
        next_state: np.ndarray | jnp.ndarray,
        absorbing: bool,
        info: dict[str, Any],
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ):
        return 0.0, carry


_AlwaysLowReward.register()


class _NonFiniteReward(Reward):
    def __call__(
        self,
        state: np.ndarray | jnp.ndarray,
        action: np.ndarray | jnp.ndarray,
        next_state: np.ndarray | jnp.ndarray,
        absorbing: bool,
        info: dict[str, Any],
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ):
        return backend.asarray(backend.inf), carry


_NonFiniteReward.register()


def _write_minimal_xml(tmp_path, *, actuated=False):
    xml = """\
<mujoco model="reward_termination">
  <option timestep="0.01"/>
  <worldbody>
    <body name="root" pos="0 0 0">
      <joint name="free" type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
    if actuated:
        actuator = (
            '<actuator><motor name="root_force" joint="free" gear="1 0 0 0 0 0" '
            'ctrllimited="true" ctrlrange="-1 1"/></actuator>'
        )
        xml = xml.replace("</mujoco>", actuator + "</mujoco>")
    path = tmp_path / "model.xml"
    path.write_text(xml)
    return str(path)


def _make_env(env_cls, xml_path, horizon=100, reward_type=None, actuation_spec=None):
    return env_cls(
        spec=xml_path,
        actuation_spec=actuation_spec or [],
        observation_spec=[ObservationType.FreeJointPos("q_free", xml_name="free")],
        reward_type=reward_type or _AlwaysLowReward.get_name(),
        reward_termination_type="LowRewardTerminationHandler",
        reward_termination_params={"threshold": 0.5, "patience": 2},
        terminal_state_type="NoTerminalStateHandler",
        domain_randomization_type="NoDomainRandomization",
        terrain_type="StaticTerrain",
        init_state_type="DefaultInitialStateHandler",
        control_type="DefaultControl",
        horizon=horizon,
    )


def test_low_reward_handler_counts_consecutive_numpy(tmp_path):
    env = _make_env(Mujoco, _write_minimal_xml(tmp_path))
    carry = env._init_additional_carry(jax.random.PRNGKey(0), env._model, env._data, np)
    handler = LowRewardTerminationHandler(env, threshold=0.5, patience=3)

    carry, done = handler.update(0.4, carry, np)
    assert not bool(done)
    assert int(carry.reward_termination_state.low_reward_steps) == 1

    carry, done = handler.update(0.4, carry, np)
    assert not bool(done)
    assert int(carry.reward_termination_state.low_reward_steps) == 2

    carry, done = handler.update(0.4, carry, np)
    assert bool(done)
    assert int(carry.reward_termination_state.low_reward_steps) == 3

    carry, done = handler.update(0.8, carry, np)
    assert not bool(done)
    assert int(carry.reward_termination_state.low_reward_steps) == 0


def test_low_reward_handler_counts_consecutive_jax(tmp_path):
    env = _make_env(Mjx, _write_minimal_xml(tmp_path))
    carry = env._init_additional_carry(jax.random.PRNGKey(0), env._model, env._data, jnp)
    handler = LowRewardTerminationHandler(env, threshold=0.5, patience=2)

    carry, done = handler.update(jnp.array(0.4), carry, jnp)
    assert not bool(np.asarray(done))
    assert int(np.asarray(carry.reward_termination_state.low_reward_steps)) == 1

    carry, done = handler.update(jnp.array(0.4), carry, jnp)
    assert bool(np.asarray(done))
    assert int(np.asarray(carry.reward_termination_state.low_reward_steps)) == 2

    carry, done = handler.update(jnp.array(0.8), carry, jnp)
    assert not bool(np.asarray(done))
    assert int(np.asarray(carry.reward_termination_state.low_reward_steps)) == 0


def test_numpy_step_uses_reward_termination_done(tmp_path):
    env = _make_env(Mujoco, _write_minimal_xml(tmp_path))
    env.reset(jax.random.PRNGKey(0))

    _, _, _, done, info = env.step(np.zeros(0, dtype=np.float32))
    assert not done
    assert not info["done_horizon"]
    assert not info["done_reward"]

    _, _, _, done, info = env.step(np.zeros(0, dtype=np.float32))
    assert done
    assert not info["done_horizon"]
    assert info["done_reward"]


def test_mjx_step_uses_reward_termination_done(tmp_path):
    env = _make_env(Mjx, _write_minimal_xml(tmp_path))
    state = env.mjx_reset(jax.random.PRNGKey(0))

    state = env.mjx_step(state, jnp.zeros(0, dtype=jnp.float32))
    assert not bool(np.asarray(state.done))
    assert not bool(np.asarray(state.info["done_horizon"]))
    assert not bool(np.asarray(state.info["done_reward"]))

    state = env.mjx_step(state, jnp.zeros(0, dtype=jnp.float32))
    assert bool(np.asarray(state.done))
    assert not bool(np.asarray(state.info["done_horizon"]))
    assert bool(np.asarray(state.info["done_reward"]))


def test_numpy_step_marks_horizon_done(tmp_path):
    env = _make_env(Mujoco, _write_minimal_xml(tmp_path), horizon=1)
    env.reset(jax.random.PRNGKey(0))

    _, _, _, done, info = env.step(np.zeros(0, dtype=np.float32))

    assert done
    assert info["done_horizon"]
    assert not info["done_reward"]


def test_mjx_step_marks_horizon_done(tmp_path):
    env = _make_env(Mjx, _write_minimal_xml(tmp_path), horizon=1)
    state = env.mjx_reset(jax.random.PRNGKey(0))

    state = env.mjx_step(state, jnp.zeros(0, dtype=jnp.float32))

    assert bool(np.asarray(state.done))
    assert bool(np.asarray(state.info["done_horizon"]))
    assert not bool(np.asarray(state.info["done_reward"]))


def test_nonfinite_reward_is_an_explicit_sanitized_terminal_on_cpu(tmp_path):
    env = _make_env(
        Mujoco,
        _write_minimal_xml(tmp_path),
        reward_type=_NonFiniteReward.get_name(),
    )
    env.reset(jax.random.PRNGKey(0))

    observation, reward, absorbing, done, info = env.step(np.zeros(0, dtype=np.float32))

    assert done
    assert absorbing
    assert info["done_numerical"]
    assert np.isfinite(observation).all()
    assert np.isfinite(reward)


def test_nonfinite_reward_is_an_explicit_sanitized_terminal_under_jit(tmp_path):
    env = _make_env(
        Mjx,
        _write_minimal_xml(tmp_path),
        reward_type=_NonFiniteReward.get_name(),
    )
    state = env.mjx_reset(jax.random.PRNGKey(0))

    state = jax.jit(env.mjx_step)(state, jnp.zeros(0, dtype=jnp.float32))

    assert bool(np.asarray(state.done))
    assert bool(np.asarray(state.absorbing))
    assert bool(np.asarray(state.info["done_numerical"]))
    assert np.isfinite(np.asarray(state.observation)).all()
    assert np.isfinite(np.asarray(state.reward))


def test_nonfinite_observation_is_an_explicit_sanitized_terminal_under_jit(tmp_path):
    env = _make_env(Mjx, _write_minimal_xml(tmp_path))
    state = env.mjx_reset(jax.random.PRNGKey(0))
    create_observation = env._mjx_create_observation

    def create_nonfinite_observation(model, data, carry):
        observation, carry = create_observation(model, data, carry)
        return observation.at[0].set(jnp.inf), carry

    env._mjx_create_observation = create_nonfinite_observation
    state = jax.jit(env.mjx_step)(state, jnp.zeros(0, dtype=jnp.float32))

    assert bool(np.asarray(state.done))
    assert bool(np.asarray(state.absorbing))
    assert bool(np.asarray(state.info["done_numerical"]))
    assert np.isfinite(np.asarray(state.observation)).all()


@pytest.mark.parametrize("env_cls", [Mujoco, Mjx])
@pytest.mark.parametrize("invalid_action", [np.nan, np.inf])
def test_nonfinite_action_terminates_before_sanitized_control_can_hide_it(tmp_path, env_cls, invalid_action):
    env = _make_env(env_cls, _write_minimal_xml(tmp_path, actuated=True), actuation_spec=["root_force"])
    action = np.asarray([invalid_action], dtype=np.float32)
    if env_cls is Mujoco:
        env.reset(jax.random.key(0))
        observation, reward, absorbing, done, info = env.step(action)
    else:
        state = env.mjx_reset(jax.random.key(0))
        state = jax.jit(env.mjx_step)(state, jnp.asarray(action))
        observation, reward, absorbing, done, info = (
            state.observation,
            state.reward,
            state.absorbing,
            state.done,
            state.info,
        )

    assert bool(done)
    assert bool(absorbing)
    assert bool(info["done_numerical"])
    assert np.isfinite(np.asarray(observation)).all()
    assert np.isfinite(np.asarray(reward)).all()
