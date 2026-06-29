import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx

jax.config.update("jax_platform_name", "cpu")

# Ensure default registry entries are available for this minimal test env.
import loco_mujoco.core.reward  # noqa: F401
import loco_mujoco.core.terminal_state_handler  # noqa: F401

from loco_mujoco.core.observations import ObservationType
from loco_mujoco.core.domain_randomizer.default import DefaultRandomizer
from loco_mujoco.core.mujoco_base import Mujoco
from loco_mujoco.core.terrain import RoughTerrain
from loco_mujoco.core.terrain.base import Terrain
from loco_mujoco.core.utils.env import Box
from loco_mujoco.trajectory.dataclasses import SingleData
from musclemimic.core.mujoco_mjx import Mjx, MjxState
from musclemimic.core.wrappers.mjx import AutoResetWrapper, LogWrapper, NStepWrapper, NormalizeVecReward, VecEnv
from musclemimic.environments.base import LocoEnv
from musclemimic.environments.humanoids.myofullbody import MjxMyoFullBody, MyoFullBody


@pytest.fixture
def minimal_freejoint_xml(tmp_path):
    xml = """\
<mujoco model="mjx_reset_parity">
  <option timestep="0.01"/>
  <worldbody>
    <body name="root" pos="0 0 0">
      <joint name="free" type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
    path = tmp_path / "model.xml"
    path.write_text(xml)
    return str(path)


class _ResetMutatesQposEnv(Mjx):
    """Test-only env to ensure reset_carry and pre_step affect data on reset."""

    def _mjx_reset_carry(self, model, data, carry):
        data, carry = super()._mjx_reset_carry(model, data, carry)
        # Move the root in +x after the initial forward() call in mjx_reset().
        data = data.replace(qpos=data.qpos.at[0].set(1.0))
        return data, carry

    def _mjx_simulation_pre_step(self, model, data, carry):
        model, data, carry = super()._mjx_simulation_pre_step(model, data, carry)
        # Apply an extra deterministic offset to prove pre_step runs during reset().
        data = data.replace(qpos=data.qpos.at[0].add(1.0))
        return model, data, carry


class _ModelMassObservationEnv(Mjx):
    """Test-only env to ensure the model returned from pre_step is used for observations."""

    UPDATED_MASS = 5.0
    TARGET_BODY_ID = 1  # skip world body at index 0

    def _mjx_simulation_pre_step(self, model, data, carry):
        model, data, carry = super()._mjx_simulation_pre_step(model, data, carry)
        model = model.replace(body_mass=model.body_mass.at[self.TARGET_BODY_ID].set(self.UPDATED_MASS))
        return model, data, carry

    def _mjx_create_observation(self, model, data, carry):
        obs = jnp.zeros(self.info.observation_space.shape, dtype=jnp.float32)
        obs = obs.at[0].set(model.body_mass[self.TARGET_BODY_ID])
        return obs, carry


class _DummyLocoEnv(LocoEnv):
    @property
    def root_free_joint_xml_name(self) -> str:
        return "free"


@struct.dataclass
class _DummyTrajState:
    traj_no: int
    subtraj_step_no_init: int


@struct.dataclass
class _DummyTrajCarry:
    traj_state: _DummyTrajState


class _DummyTrajHandler:
    def __init__(self, init_qpos: np.ndarray, init_qvel: np.ndarray):
        self._init_qpos = init_qpos
        self._init_qvel = init_qvel
        self.called = False

    def get_init_traj_data(self, _carry, backend):
        self.called = True
        return SingleData(
            qpos=backend.array(self._init_qpos, dtype=backend.float32),
            qvel=backend.array(self._init_qvel, dtype=backend.float32),
        )


class _DeterministicRoughTerrain(RoughTerrain):
    """Rough terrain variant with deterministic heightfield for parity checks."""

    def _np_random_uniform_terrain(self):
        return np.full((self.hfield_length, self.hfield_length), 0.03, dtype=np.float32)

    def _jnp_random_uniform_terrain(self, key):
        return jnp.full((self.hfield_length, self.hfield_length), 0.03, dtype=jnp.float32)


# Register deterministic terrain under a unique name if not already present.
if "TestRoughTerrain" not in Terrain.registered:
    _DeterministicRoughTerrain.register()
    Terrain.registered["TestRoughTerrain"] = _DeterministicRoughTerrain


class _FullbodyTerrainParityEnv(MyoFullBody):
    """Fullbody non-MJX env that optionally applies deterministic rough terrain adjustment."""

    def __init__(self, use_rough: bool, **kwargs):
        self.use_rough = use_rough
        super().__init__(
            spec=None,
            actuation_spec=[],
            observation_spec=[ObservationType.FreeJointPos("root_free", xml_name="root")],
            horizon=2,
            reward_type="NoReward",
            terminal_state_type="NoTerminalStateHandler",
            domain_randomization_type="NoDomainRandomization",
            terrain_type="TestRoughTerrain" if use_rough else "StaticTerrain",
            init_state_type="DefaultInitialStateHandler",
            control_type="DefaultControl",
            **kwargs,
        )

    def _simulation_pre_step(self, model, data, carry):
        if self.use_rough:
            hfield = np.full_like(model.hfield_data, 0.03)
            model.hfield_data[:] = hfield
            carry = carry.replace(terrain_state=carry.terrain_state.replace(height_field_raw=hfield))
            data.qpos[2] += 0.08
            return model, data, carry
        return super()._simulation_pre_step(model, data, carry)


class _MjxFullbodyTerrainParityEnv(MjxMyoFullBody):
    """MJX fullbody env with deterministic rough terrain for parity checks."""

    def __init__(self, use_rough: bool, **kwargs):
        self.use_rough = use_rough
        self._last_sys = None
        super().__init__(
            actuation_spec=[],
            observation_spec=[ObservationType.FreeJointPos("root_free", xml_name="root")],
            horizon=2,
            reward_type="NoReward",
            terminal_state_type="NoTerminalStateHandler",
            domain_randomization_type="NoDomainRandomization",
            terrain_type="TestRoughTerrain" if use_rough else "StaticTerrain",
            init_state_type="DefaultInitialStateHandler",
            control_type="DefaultControl",
            **kwargs,
        )

    def _mjx_simulation_pre_step(self, model, data, carry):
        if self.use_rough:
            hfield = jnp.full_like(model.hfield_data, 0.03)
            model = model.replace(hfield_data=hfield)
            carry = carry.replace(terrain_state=carry.terrain_state.replace(height_field_raw=hfield))
            data = data.replace(qpos=data.qpos.at[2].add(0.08))
            self._last_sys = model
            return model, data, carry
        model, data, carry = super()._mjx_simulation_pre_step(model, data, carry)
        self._last_sys = model
        return model, data, carry

    def mjx_reset(self, key: jax.random.PRNGKey) -> MjxState:
        """Lightweight reset to keep fullbody MJX parity test fast."""
        data = mjx.put_data(self._model, self._data, impl=self._backend_impl)
        if self.use_rough:
            data = data.replace(qpos=data.qpos.at[2].add(0.08))
        carry = self._init_additional_carry(key, self._model, data, jnp)
        obs = jnp.asarray(data.qpos[:7])
        return MjxState(
            data=data,
            observation=obs,
            reward=0.0,
            absorbing=jnp.array(False),
            done=jnp.array(False),
            info={},
            additional_carry=carry,
        )


class _RandomizingNumpyEnv(Mujoco):
    """Non-MJX env to compare reset parity against MJX."""

    TARGET_BODY_ID = 1  # skip world body at index 0
    RAND_TRANSLATION = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    RAND_QUAT = np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float64)  # 45 deg around z
    RAND_BODY_SHIFT = 0.25

    def __init__(self, apply_randomization: bool, **kwargs):
        self.apply_randomization = apply_randomization
        super().__init__(**kwargs)

    def _simulation_pre_step(self, model, data, carry):
        model, data, carry = super()._simulation_pre_step(model, data, carry)
        if self.apply_randomization:
            data.qpos[:3] = self.RAND_TRANSLATION
            data.qpos[3:7] = self.RAND_QUAT
            model.body_pos[self.TARGET_BODY_ID, 0] = self.RAND_BODY_SHIFT
        return model, data, carry


class _RandomizingMjxEnv(Mjx):
    """MJX env that applies deterministic terrain/domain perturbation in pre_step."""

    TARGET_BODY_ID = 1  # skip world body at index 0
    RAND_TRANSLATION = jnp.array([0.5, 0.0, 0.0], dtype=jnp.float32)
    RAND_QUAT = jnp.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=jnp.float32)  # 45 deg around z
    RAND_BODY_SHIFT = 0.25

    def __init__(self, apply_randomization: bool, **kwargs):
        self.apply_randomization = apply_randomization
        self._last_randomized_sys = None
        super().__init__(**kwargs)

    def _mjx_simulation_pre_step(self, model, data, carry):
        model, data, carry = super()._mjx_simulation_pre_step(model, data, carry)
        if self.apply_randomization:
            data = data.replace(qpos=data.qpos.at[:3].set(self.RAND_TRANSLATION))
            data = data.replace(qpos=data.qpos.at[3:7].set(self.RAND_QUAT))
            model = model.replace(body_pos=model.body_pos.at[self.TARGET_BODY_ID, 0].set(self.RAND_BODY_SHIFT))
        self._last_randomized_sys = model
        return model, data, carry


def test_mjx_reset_runs_pre_step_and_forward_after_reset_carry(minimal_freejoint_xml):
    env = _ResetMutatesQposEnv(
        spec=minimal_freejoint_xml,
        actuation_spec=[],
        observation_spec=[ObservationType.FreeJointPos("q_free", xml_name="free")],
        horizon=2,
        reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
        domain_randomization_type="NoDomainRandomization",
        terrain_type="StaticTerrain",
        init_state_type="DefaultInitialStateHandler",
        control_type="DefaultControl",
    )

    state = env.mjx_reset(jax.random.PRNGKey(0))

    # reset_carry set qpos[0]=1.0 and pre_step adds +1.0 -> 2.0
    assert float(state.data.qpos[0]) == pytest.approx(2.0)

    # Data returned by reset should already be forward()'d for its qpos and model; forward again is idempotent.
    forwarded = mjx.forward(env.sys, state.data)
    np.testing.assert_allclose(np.asarray(state.data.xpos), np.asarray(forwarded.xpos), atol=1e-6, rtol=1e-6)

    # Observation should reflect the pre_step-updated data.
    assert float(state.observation[0]) == pytest.approx(2.0)


def test_mjx_reset_in_step_applies_pre_step_and_forward(minimal_freejoint_xml):
    env = _ResetMutatesQposEnv(
        spec=minimal_freejoint_xml,
        actuation_spec=[],
        observation_spec=[ObservationType.FreeJointPos("q_free", xml_name="free")],
        horizon=2,
        reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
        domain_randomization_type="NoDomainRandomization",
        terrain_type="StaticTerrain",
        init_state_type="DefaultInitialStateHandler",
        control_type="DefaultControl",
    )

    first_state = env.mjx_reset(jax.random.PRNGKey(0))
    reset_state = env._mjx_reset_in_step(first_state)

    # reset_carry set qpos[0]=1.0 and pre_step adds +1.0 -> 2.0
    assert float(reset_state.data.qpos[0]) == pytest.approx(2.0)

    # Data returned by reset_in_step should also be forward()'d.
    forwarded = mjx.forward(env.sys, reset_state.data)
    np.testing.assert_allclose(np.asarray(reset_state.data.xpos), np.asarray(forwarded.xpos), atol=1e-6, rtol=1e-6)

    # Observation should reflect the pre_step-updated data.
    assert float(reset_state.observation[0]) == pytest.approx(2.0)


def test_mjx_reset_uses_pre_step_model_for_observation(minimal_freejoint_xml):
    env = _ModelMassObservationEnv(
        spec=minimal_freejoint_xml,
        actuation_spec=[],
        observation_spec=[ObservationType.FreeJointPos("q_free", xml_name="free")],
        horizon=2,
        reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
        domain_randomization_type="NoDomainRandomization",
        terrain_type="StaticTerrain",
        init_state_type="DefaultInitialStateHandler",
        control_type="DefaultControl",
    )

    state = env.mjx_reset(jax.random.PRNGKey(0))
    assert float(state.observation[0]) == pytest.approx(env.UPDATED_MASS)

    # The in-step reset path should also propagate the randomized model to observation creation.
    reset_state = env._mjx_reset_in_step(state)
    assert float(reset_state.observation[0]) == pytest.approx(env.UPDATED_MASS)


def test_mjx_set_sim_state_from_traj_data_uses_init_offset(minimal_freejoint_xml):
    model = mujoco.MjModel.from_xml_path(minimal_freejoint_xml)
    data = mujoco.MjData(model)
    mjx_data = mjx.put_data(model, data)

    env = _DummyLocoEnv.__new__(_DummyLocoEnv)
    env._model = model

    init_qpos = np.array([1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    init_qvel = np.zeros((model.nv,), dtype=np.float32)
    env.th = _DummyTrajHandler(init_qpos, init_qvel)

    traj_qpos = jnp.array([10.0, 20.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    traj_data = SingleData(qpos=traj_qpos, qvel=jnp.zeros((model.nv,), dtype=jnp.float32))
    carry = _DummyTrajCarry(traj_state=_DummyTrajState(traj_no=0, subtraj_step_no_init=0))

    updated = LocoEnv.mjx_set_sim_state_from_traj_data(env, mjx_data, traj_data, carry)

    assert env.th.called
    expected_qpos = np.array(traj_qpos)
    expected_qpos[0] -= init_qpos[0]
    expected_qpos[1] -= init_qpos[1]
    np.testing.assert_allclose(np.asarray(updated.qpos), expected_qpos, atol=0.0, rtol=0.0)


def test_mjx_and_numpy_reset_randomization_parity(minimal_freejoint_xml):
    def make_obs_spec():
        return [
            ObservationType.BodyPos("root_pos", xml_name="root"),
            ObservationType.BodyRot("root_rot", xml_name="root"),
        ]

    np_base_env = _RandomizingNumpyEnv(
        apply_randomization=False,
        spec=minimal_freejoint_xml,
        actuation_spec=[],
        observation_spec=make_obs_spec(),
        horizon=2,
        reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
        domain_randomization_type="NoDomainRandomization",
        terrain_type="StaticTerrain",
        init_state_type="DefaultInitialStateHandler",
        control_type="DefaultControl",
    )
    np_rand_env = _RandomizingNumpyEnv(
        apply_randomization=True,
        spec=minimal_freejoint_xml,
        actuation_spec=[],
        observation_spec=make_obs_spec(),
        horizon=2,
        reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
        domain_randomization_type="NoDomainRandomization",
        terrain_type="StaticTerrain",
        init_state_type="DefaultInitialStateHandler",
        control_type="DefaultControl",
    )

    base_obs_np = np_base_env.reset(jax.random.PRNGKey(0))
    rand_obs_np = np_rand_env.reset(jax.random.PRNGKey(1))
    assert not np.allclose(base_obs_np, rand_obs_np)

    expected_np = np.concatenate(
        [np_rand_env._data.xpos[np_rand_env.TARGET_BODY_ID], np_rand_env._data.xquat[np_rand_env.TARGET_BODY_ID]]
    )
    np.testing.assert_allclose(rand_obs_np[:7], expected_np, atol=1e-6, rtol=1e-6)

    mjx_base_env = _RandomizingMjxEnv(
        apply_randomization=False,
        spec=minimal_freejoint_xml,
        actuation_spec=[],
        observation_spec=make_obs_spec(),
        horizon=2,
        reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
        domain_randomization_type="NoDomainRandomization",
        terrain_type="StaticTerrain",
        init_state_type="DefaultInitialStateHandler",
        control_type="DefaultControl",
    )
    mjx_rand_env = _RandomizingMjxEnv(
        apply_randomization=True,
        spec=minimal_freejoint_xml,
        actuation_spec=[],
        observation_spec=make_obs_spec(),
        horizon=2,
        reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
        domain_randomization_type="NoDomainRandomization",
        terrain_type="StaticTerrain",
        init_state_type="DefaultInitialStateHandler",
        control_type="DefaultControl",
    )

    base_state = mjx_base_env.mjx_reset(jax.random.PRNGKey(0))
    rand_state = mjx_rand_env.mjx_reset(jax.random.PRNGKey(1))

    assert not np.allclose(np.asarray(base_state.observation), np.asarray(rand_state.observation))

    # Observation should match forward kinematics using the randomized sys.
    forward_rand = mjx.forward(mjx_rand_env._last_randomized_sys, rand_state.data)
    np.testing.assert_allclose(
        np.asarray(rand_state.data.xpos), np.asarray(forward_rand.xpos), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(rand_state.data.xquat), np.asarray(forward_rand.xquat), atol=1e-6, rtol=1e-6
    )
    expected_mjx = np.concatenate(
        [np.asarray(rand_state.data.xpos)[mjx_rand_env.TARGET_BODY_ID], np.asarray(rand_state.data.xquat)[1]]
    )
    np.testing.assert_allclose(np.asarray(rand_state.observation[:7]), expected_mjx, atol=1e-6, rtol=1e-6)

    # Randomized MJX observation should align with numpy version on the shared pose signal.
    np.testing.assert_allclose(np.asarray(rand_state.observation[:7]), rand_obs_np[:7], atol=1e-5, rtol=1e-5)


def test_fullbody_rough_terrain_reset_parity():
    # Non-MJX parity: Static vs deterministic rough terrain should change the observation.
    np_static_env = _FullbodyTerrainParityEnv(use_rough=False)
    np_rough_env = _FullbodyTerrainParityEnv(use_rough=True)

    base_obs_np = np_static_env.reset(jax.random.PRNGKey(0))
    rough_obs_np = np_rough_env.reset(jax.random.PRNGKey(0))
    assert not np.allclose(base_obs_np, rough_obs_np)
    np.testing.assert_allclose(rough_obs_np[:7], np_rough_env._data.qpos[:7], atol=1e-6, rtol=1e-6)

    # MJX parity: Rough terrain perturbation changes observation and matches forward kinematics.
    mjx_static_env = _MjxFullbodyTerrainParityEnv(use_rough=False)
    mjx_rough_env = _MjxFullbodyTerrainParityEnv(use_rough=True)

    base_state = mjx_static_env.mjx_reset(jax.random.PRNGKey(1))
    rough_state = mjx_rough_env.mjx_reset(jax.random.PRNGKey(1))

    assert not np.allclose(np.asarray(base_state.observation), np.asarray(rough_state.observation))

    np.testing.assert_allclose(np.asarray(rough_state.observation), np.asarray(rough_state.data.qpos[:7]), atol=1e-6, rtol=1e-6)

    # MJX rough pose matches numpy rough pose on the shared free-joint observation.
    np.testing.assert_allclose(np.asarray(rough_state.observation), rough_obs_np[:7], atol=1e-5, rtol=1e-5)


@struct.dataclass
class _LinkMassCarry:
    key: jax.Array


def test_default_randomizer_link_mass_multiplier_shapes(minimal_freejoint_xml):
    model = mujoco.MjModel.from_xml_path(minimal_freejoint_xml)
    mjx_model = mjx.put_model(model)

    rand_conf_base = {
        "randomize_link_mass": False,
        "randomize_base_mass": False,
        "link_mass_multiplier_range": {"root_body": (0.5, 0.5), "other_bodies": (0.5, 0.5)},
    }

    dummy_env = _RandomizingMjxEnv(
        apply_randomization=False,
        spec=minimal_freejoint_xml,
        actuation_spec=[],
        observation_spec=[ObservationType.FreeJointPos("q_free", xml_name="free")],
        horizon=2,
        reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
        domain_randomization_type="NoDomainRandomization",
        terrain_type="StaticTerrain",
        init_state_type="DefaultInitialStateHandler",
        control_type="DefaultControl",
    )

    randomizer_off = DefaultRandomizer.__new__(DefaultRandomizer)
    randomizer_off.rand_conf = rand_conf_base
    multipliers_off, _ = randomizer_off._sample_link_mass_multipliers(
        mjx_model, _LinkMassCarry(key=jax.random.PRNGKey(0)), jnp
    )
    assert multipliers_off.shape == (1,)
    np.testing.assert_allclose(np.asarray(multipliers_off), np.ones((1,)), atol=0.0, rtol=0.0)

    rand_conf_on = dict(rand_conf_base, randomize_link_mass=True)
    randomizer_on = DefaultRandomizer.__new__(DefaultRandomizer)
    randomizer_on.rand_conf = rand_conf_on
    multipliers_on, _ = randomizer_on._sample_link_mass_multipliers(
        mjx_model, _LinkMassCarry(key=jax.random.PRNGKey(1)), jnp
    )
    assert multipliers_on.shape == (1,)
    assert multipliers_on.ndim == 1


@struct.dataclass
class _AutoResetCarry:
    autoreset_rng: jax.Array | None
    key: jax.Array
    cur_step_in_episode: jax.Array
    last_action: jax.Array
    domain_randomizer_state: jax.Array
    terrain_state: jax.Array


@struct.dataclass
class _AutoResetState:
    data: jax.Array
    observation: jax.Array
    reward: jax.Array
    absorbing: jax.Array
    done: jax.Array
    additional_carry: _AutoResetCarry
    info: dict


def _encode_keys_to_terrain(keys: jax.Array) -> jax.Array:
    return (jnp.sum(keys.astype(jnp.uint32), axis=-1) % 1024).astype(jnp.float32).reshape(keys.shape[0], 1)


def _encode_keys_with_offset(keys: jax.Array, offset: float) -> jax.Array:
    return _encode_keys_to_terrain(keys) + offset


def _assert_done_swaps(actual, reset_value, stepped_value, done_mask):
    done_mask = np.asarray(done_mask, dtype=bool)
    done_idx = np.where(done_mask)[0]
    not_done_idx = np.where(~done_mask)[0]
    np.testing.assert_allclose(np.asarray(actual)[done_idx], np.asarray(reset_value)[done_idx], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(actual)[not_done_idx], np.asarray(stepped_value)[not_done_idx], atol=0.0, rtol=0.0
    )


@struct.dataclass
class _AutoResetMinimalCarry:
    autoreset_rng: jax.Array | None


@struct.dataclass
class _AutoResetMinimalState:
    data: jax.Array
    observation: jax.Array
    reward: jax.Array
    absorbing: jax.Array
    done: jax.Array
    additional_carry: _AutoResetMinimalCarry
    info: dict


@struct.dataclass
class _AutoResetObsStates:
    hist: jax.Array
    vel: jax.Array


@struct.dataclass
class _AutoResetRichTrajState:
    traj_no: jax.Array
    phase: jax.Array
    scale: jax.Array


@struct.dataclass
class _AutoResetRichCarry:
    autoreset_rng: jax.Array | None
    key: jax.Array
    cur_step_in_episode: jax.Array
    last_action: jax.Array
    domain_randomizer_state: jax.Array
    terrain_state: jax.Array
    traj_state: _AutoResetRichTrajState
    observation_states: _AutoResetObsStates


@struct.dataclass
class _AutoResetRichState:
    data: dict
    observation: jax.Array
    reward: jax.Array
    absorbing: jax.Array
    done: jax.Array
    additional_carry: _AutoResetRichCarry
    info: dict


class _AutoResetSmokeEnv:
    """Minimal env to exercise AutoResetWrapper with domain/terrain state swaps."""

    def reset(self, rng_keys):
        batch = int(rng_keys.shape[0])
        obs = jnp.zeros((batch, 1), dtype=jnp.float32)
        encoded = jnp.sum(rng_keys.astype(jnp.uint32), axis=-1, keepdims=True).astype(jnp.float32)
        carry = _AutoResetCarry(
            autoreset_rng=None,
            key=rng_keys,
            cur_step_in_episode=jnp.ones((batch,), dtype=jnp.int32),
            last_action=jnp.zeros((batch, 1), dtype=jnp.float32),
            domain_randomizer_state=encoded,
            terrain_state=encoded + 100.0,
        )
        state = _AutoResetState(
            data=encoded,
            observation=obs,
            reward=jnp.zeros((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=jnp.zeros((batch,), dtype=bool),
            additional_carry=carry,
            info={},
        )
        return obs, state

    def step(self, state, action):
        batch = int(state.observation.shape[0])
        done = jnp.array([True] + [False] * (batch - 1), dtype=bool)

        stepped_dr = state.additional_carry.domain_randomizer_state + 10.0
        stepped_terrain = state.additional_carry.terrain_state + 5.0
        carry = state.additional_carry.replace(
            cur_step_in_episode=state.additional_carry.cur_step_in_episode + 1,
            domain_randomizer_state=stepped_dr,
            terrain_state=stepped_terrain,
        )
        next_state = _AutoResetState(
            data=stepped_terrain,
            observation=state.observation + 1.0,
            reward=jnp.ones((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=done,
            additional_carry=carry,
            info={},
        )
        return next_state.observation, next_state.reward, next_state.absorbing, next_state.done, next_state.info, next_state


class _AutoResetTerrainSwapEnv:
    """Minimal env to exercise terrain-only swapping on AutoResetWrapper."""

    def reset(self, rng_keys):
        batch = int(rng_keys.shape[0])
        obs = jnp.zeros((batch, 1), dtype=jnp.float32)

        # Encode the key into terrain_state so we can check per-episode swapping deterministically.
        terrain_state = (jnp.sum(rng_keys.astype(jnp.uint32), axis=-1) % 1024).astype(jnp.float32).reshape(batch, 1)

        carry = _AutoResetCarry(
            autoreset_rng=None,
            key=rng_keys,
            cur_step_in_episode=jnp.ones((batch,), dtype=jnp.int32),
            last_action=jnp.zeros((batch, 1), dtype=jnp.float32),
            domain_randomizer_state=jnp.zeros((batch, 1), dtype=jnp.float32),
            terrain_state=terrain_state,
        )
        state = _AutoResetState(
            data=terrain_state,
            observation=obs,
            reward=jnp.zeros((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=jnp.zeros((batch,), dtype=bool),
            additional_carry=carry,
            info={},
        )
        return obs, state

    def step(self, state, action):
        batch = int(state.observation.shape[0])
        done = jnp.array([True] + [False] * (batch - 1), dtype=bool)

        # Deterministically change terrain_state during the step so it must be swapped on reset.
        stepped_terrain_state = state.additional_carry.terrain_state + 1000.0
        next_carry = state.additional_carry.replace(
            cur_step_in_episode=state.additional_carry.cur_step_in_episode + 1,
            terrain_state=stepped_terrain_state,
        )
        next_state = state.replace(
            data=stepped_terrain_state,
            observation=state.observation + 1.0,
            reward=jnp.ones((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=done,
            additional_carry=next_carry,
            info={},
        )
        return next_state.observation, next_state.reward, next_state.absorbing, next_state.done, next_state.info, next_state


class _AutoResetRewardAbsorbEnv:
    """Minimal env to validate done_count and reward/absorbing pass-through."""

    def reset(self, rng_keys):
        batch = int(rng_keys.shape[0])
        obs = jnp.zeros((batch, 1), dtype=jnp.float32)
        carry = _AutoResetMinimalCarry(autoreset_rng=None)
        state = _AutoResetMinimalState(
            data=jnp.zeros((batch, 1), dtype=jnp.float32),
            observation=obs,
            reward=jnp.full((batch,), -1.0, dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=jnp.zeros((batch,), dtype=bool),
            additional_carry=carry,
            info={"marker": jnp.arange(batch, dtype=jnp.int32)},
        )
        return obs, state

    def step(self, state, action):
        batch = int(state.observation.shape[0])
        idx = jnp.arange(batch)
        done = (idx % 2 == 0)
        reward = idx.astype(jnp.float32) + 1.0
        absorbing = (idx % 2 == 0)
        next_state = state.replace(
            data=state.data + 10.0,
            observation=state.observation + 5.0,
            reward=reward,
            absorbing=absorbing,
            done=done,
            info={"step_marker": jnp.arange(batch, dtype=jnp.int32) + 10},
        )
        return next_state.observation, reward, absorbing, done, next_state.info, next_state


class _NStepInfo:
    def __init__(self, observation_space):
        self.observation_space = observation_space


@struct.dataclass
class _NStepDummyState:
    step: jax.Array
    observation: jax.Array


class _NStepDummyEnv:
    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim
        low = -np.arange(1, obs_dim + 1, dtype=np.float32)
        high = np.arange(1, obs_dim + 1, dtype=np.float32)
        self.info = _NStepInfo(Box(low, high))

    def _initial_obs(self):
        return jnp.arange(self.obs_dim, dtype=jnp.float32) + 1.0

    def reset(self, rng_key):
        obs = self._initial_obs()
        state = _NStepDummyState(step=jnp.array(0, dtype=jnp.int32), observation=obs)
        return obs, state

    def step(self, state, action):
        step = state.step + 1
        step_f = step.astype(jnp.float32)
        obs = jnp.full((self.obs_dim,), step_f)
        reward = step_f + 0.5
        absorbing = (step % 2 == 0)
        done = (step % 3 == 0)
        info = {"step": step, "tag": jnp.array(7, dtype=jnp.int32)}
        next_state = state.replace(step=step, observation=obs)
        return obs, reward, absorbing, done, info, next_state


class _AutoResetMaskEnv:
    """Env that returns a fixed done mask and tracks reset calls."""

    def __init__(self, done_mask):
        self.done_mask = done_mask
        self.reset_calls = 0

    def reset(self, rng_keys):
        self.reset_calls += 1
        batch = int(rng_keys.shape[0])
        obs = jnp.zeros((batch, 1), dtype=jnp.float32)
        terrain_state = _encode_keys_to_terrain(rng_keys)
        carry = _AutoResetCarry(
            autoreset_rng=None,
            key=rng_keys,
            cur_step_in_episode=jnp.ones((batch,), dtype=jnp.int32),
            last_action=jnp.zeros((batch, 1), dtype=jnp.float32),
            domain_randomizer_state=jnp.zeros((batch, 1), dtype=jnp.float32),
            terrain_state=terrain_state,
        )
        state = _AutoResetState(
            data=terrain_state,
            observation=obs,
            reward=jnp.zeros((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=jnp.zeros((batch,), dtype=bool),
            additional_carry=carry,
            info={},
        )
        return obs, state

    def _resolve_done(self, batch: int) -> jax.Array:
        done = jnp.asarray(self.done_mask, dtype=bool)
        if done.shape == ():
            return jnp.full((batch,), done, dtype=bool)
        if done.shape[0] != batch:
            raise ValueError("done_mask shape does not match batch")
        return done

    def step(self, state, action):
        batch = int(state.observation.shape[0])
        done = self._resolve_done(batch)
        stepped_terrain_state = state.additional_carry.terrain_state + 5.0
        next_carry = state.additional_carry.replace(
            cur_step_in_episode=state.additional_carry.cur_step_in_episode + 1,
            terrain_state=stepped_terrain_state,
        )
        next_state = state.replace(
            data=stepped_terrain_state,
            observation=state.observation + 1.0,
            reward=jnp.ones((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=done,
            additional_carry=next_carry,
            info={},
        )
        return next_state.observation, next_state.reward, next_state.absorbing, next_state.done, next_state.info, next_state


class _AutoResetSingleKeyEnv:
    """Env that accepts a single key and splits internally to a fixed batch size."""

    def __init__(self, batch: int):
        self._batch = batch

    def reset(self, rng_keys):
        if hasattr(rng_keys, "shape") and rng_keys.shape == (2,):
            rng_keys = jax.random.split(rng_keys, self._batch)
        batch = int(rng_keys.shape[0])
        obs = jnp.zeros((batch, 1), dtype=jnp.float32)
        carry = _AutoResetMinimalCarry(autoreset_rng=None)
        state = _AutoResetMinimalState(
            data=jnp.zeros((batch, 1), dtype=jnp.float32),
            observation=obs,
            reward=jnp.zeros((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=jnp.zeros((batch,), dtype=bool),
            additional_carry=carry,
            info={},
        )
        return obs, state

    def step(self, state, action):
        batch = int(state.observation.shape[0])
        done = jnp.zeros((batch,), dtype=bool)
        next_state = state.replace(done=done)
        return next_state.observation, next_state.reward, next_state.absorbing, next_state.done, next_state.info, next_state


class _AutoResetKeyArrayEnv:
    """Env that does not operate on keys (for KeyArray caching coverage)."""

    def reset(self, rng_keys):
        batch = int(rng_keys.shape[0])
        obs = jnp.zeros((batch, 1), dtype=jnp.float32)
        carry = _AutoResetMinimalCarry(autoreset_rng=None)
        state = _AutoResetMinimalState(
            data=jnp.zeros((batch, 1), dtype=jnp.float32),
            observation=obs,
            reward=jnp.zeros((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=jnp.zeros((batch,), dtype=bool),
            additional_carry=carry,
            info={},
        )
        return obs, state

    def step(self, state, action):
        batch = int(state.observation.shape[0])
        done = jnp.zeros((batch,), dtype=bool)
        next_state = state.replace(done=done)
        return next_state.observation, next_state.reward, next_state.absorbing, next_state.done, next_state.info, next_state


class _AutoResetStressEnv:
    """Env with action-driven done masks to stress AutoResetWrapper swaps."""

    def reset(self, rng_keys):
        batch = int(rng_keys.shape[0])
        encoded = _encode_keys_to_terrain(rng_keys)
        obs = encoded + 0.5
        carry = _AutoResetCarry(
            autoreset_rng=None,
            key=rng_keys,
            cur_step_in_episode=jnp.ones((batch,), dtype=jnp.int32),
            last_action=jnp.zeros((batch, 1), dtype=jnp.float32),
            domain_randomizer_state=encoded + 10.0,
            terrain_state=encoded + 20.0,
        )
        state = _AutoResetState(
            data=encoded,
            observation=obs,
            reward=jnp.zeros((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=jnp.zeros((batch,), dtype=bool),
            additional_carry=carry,
            info={},
        )
        return obs, state

    def step(self, state, action):
        batch = int(state.observation.shape[0])
        done = action[:, 0] > 0
        stepped_key = jax.vmap(jax.random.split)(state.additional_carry.key)[:, 0]
        step_encoded = _encode_keys_to_terrain(stepped_key)
        reward = step_encoded[:, 0] + 1.0
        absorbing = (step_encoded[:, 0].astype(jnp.int32) % 2 == 0)
        next_carry = state.additional_carry.replace(
            key=stepped_key,
            cur_step_in_episode=state.additional_carry.cur_step_in_episode + 1,
            last_action=action,
            domain_randomizer_state=step_encoded + 13.0,
            terrain_state=step_encoded + 23.0,
        )
        next_state = state.replace(
            data=step_encoded + 3.0,
            observation=step_encoded + 3.5,
            reward=reward,
            absorbing=absorbing,
            done=done,
            additional_carry=next_carry,
            info={},
        )
        return next_state.observation, reward, absorbing, done, next_state.info, next_state


class _AutoResetRichEnv:
    """Env that exercises optional carry swaps and unbatched leaf protection."""

    def reset(self, rng_keys):
        batch = int(rng_keys.shape[0])
        encoded = _encode_keys_to_terrain(rng_keys)
        obs = encoded + 0.5
        data = {
            "batched": encoded + 70.0,
            "matrix": encoded + jnp.arange(batch, dtype=jnp.float32)[None, :],
            "unbatched": jnp.array([5.0, 6.0], dtype=jnp.float32),
        }
        traj_state = _AutoResetRichTrajState(
            traj_no=jnp.arange(batch, dtype=jnp.int32),
            phase=encoded + 10.0,
            scale=encoded + 20.0,
        )
        obs_states = _AutoResetObsStates(hist=encoded + 50.0, vel=encoded + 60.0)
        carry = _AutoResetRichCarry(
            autoreset_rng=None,
            key=rng_keys,
            cur_step_in_episode=jnp.ones((batch,), dtype=jnp.int32),
            last_action=encoded + 3.0,
            domain_randomizer_state=encoded + 11.0,
            terrain_state=encoded + 13.0,
            traj_state=traj_state,
            observation_states=obs_states,
        )
        state = _AutoResetRichState(
            data=data,
            observation=obs,
            reward=jnp.zeros((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=jnp.zeros((batch,), dtype=bool),
            additional_carry=carry,
            info={},
        )
        return obs, state

    def step(self, state, action):
        batch = int(state.observation.shape[0])
        done = (jnp.arange(batch) % 2 == 0)
        stepped_key = jax.vmap(jax.random.split)(state.additional_carry.key)[:, 0]
        next_state = state.replace(
            data={
                "batched": state.data["batched"] + 1000.0,
                "matrix": state.data["matrix"] + 1000.0,
                "unbatched": jnp.array([9.0, 10.0], dtype=jnp.float32),
            },
            observation=state.observation + 1.0,
            reward=jnp.zeros((batch,), dtype=jnp.float32),
            absorbing=jnp.zeros((batch,), dtype=bool),
            done=done,
            additional_carry=state.additional_carry.replace(
                key=stepped_key,
                cur_step_in_episode=state.additional_carry.cur_step_in_episode + 1,
                last_action=state.additional_carry.last_action + 100.0,
                domain_randomizer_state=state.additional_carry.domain_randomizer_state + 100.0,
                terrain_state=state.additional_carry.terrain_state + 100.0,
                traj_state=state.additional_carry.traj_state.replace(
                    phase=state.additional_carry.traj_state.phase + 100.0,
                    scale=state.additional_carry.traj_state.scale + 100.0,
                ),
                observation_states=state.additional_carry.observation_states.replace(
                    hist=state.additional_carry.observation_states.hist + 100.0,
                    vel=state.additional_carry.observation_states.vel + 100.0,
                ),
            ),
        )
        return next_state.observation, next_state.reward, next_state.absorbing, next_state.done, next_state.info, next_state


class _VecEnvBaseEnv:
    """Single-env base to validate VecEnv + AutoResetWrapper integration."""

    def __init__(self):
        low = np.array([-1.0], dtype=np.float32)
        high = np.array([1.0], dtype=np.float32)
        self.info = _NStepInfo(Box(low, high))

    def reset(self, rng_key):
        obs = jnp.zeros((1,), dtype=jnp.float32)
        encoded = jnp.sum(rng_key.astype(jnp.uint32), keepdims=True).astype(jnp.float32)
        carry = _AutoResetCarry(
            autoreset_rng=None,
            key=rng_key,
            cur_step_in_episode=jnp.array(1, dtype=jnp.int32),
            last_action=jnp.zeros((1,), dtype=jnp.float32),
            domain_randomizer_state=encoded,
            terrain_state=encoded + 100.0,
        )
        state = _AutoResetState(
            data=encoded,
            observation=obs,
            reward=jnp.array(0.0, dtype=jnp.float32),
            absorbing=jnp.array(False, dtype=bool),
            done=jnp.array(False, dtype=bool),
            additional_carry=carry,
            info={},
        )
        return obs, state

    def reset_to(self, rng_key, _traj_idx):
        return self.reset(rng_key)

    def step(self, state, action):
        key_sum = jnp.sum(state.additional_carry.key.astype(jnp.uint32))
        done = (key_sum % 2 == 0)
        stepped_domain = state.additional_carry.domain_randomizer_state + 10.0
        stepped_terrain = state.additional_carry.terrain_state + 5.0
        carry = state.additional_carry.replace(
            cur_step_in_episode=state.additional_carry.cur_step_in_episode + 1,
            domain_randomizer_state=stepped_domain,
            terrain_state=stepped_terrain,
        )
        next_state = state.replace(
            data=stepped_terrain,
            observation=state.observation + 1.0,
            reward=jnp.array(1.0, dtype=jnp.float32),
            absorbing=jnp.array(False, dtype=bool),
            done=done,
            additional_carry=carry,
            info={},
        )
        return next_state.observation, next_state.reward, next_state.absorbing, next_state.done, next_state.info, next_state


@struct.dataclass
class _AutoResetWrappedState:
    env_state: _AutoResetState
    tag: jax.Array


class _AutoResetWrappedEnv:
    """Env that wraps state in an env_state field to exercise unwrap/rewrap paths."""

    def __init__(self):
        self._base = _AutoResetTerrainSwapEnv()

    def reset(self, rng_keys):
        obs, state = self._base.reset(rng_keys)
        tag = jnp.zeros((state.observation.shape[0],), dtype=jnp.int32)
        return obs, _AutoResetWrappedState(env_state=state, tag=tag)

    def step(self, state, action):
        obs, reward, absorbing, done, info, next_state = self._base.step(state.env_state, action)
        tag = state.tag + 1
        return obs, reward, absorbing, done, info, _AutoResetWrappedState(env_state=next_state, tag=tag)


def test_autoreset_wrapper_swaps_terrain_state_where_done():
    base_env = _AutoResetTerrainSwapEnv()
    env = AutoResetWrapper(base_env)

    batch = 32
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)

    _, state = env.reset(rng_keys)
    _, _, _, cleared_done, _, next_state = env.step(state, action=jnp.zeros((batch, 1), dtype=jnp.float32))

    # Compute expected reset_keys used by AutoResetWrapper.
    split = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split[:, 1]
    expected_reset_terrain = _encode_keys_to_terrain(reset_keys)

    initial_terrain = _encode_keys_to_terrain(rng_keys)
    expected_stepped_terrain = initial_terrain + 1000.0

    # Env 0 is done -> should take reset terrain_state. Env 1 continues -> should keep stepped terrain_state.
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.terrain_state[0]),
        np.asarray(expected_reset_terrain[0]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.terrain_state[1]),
        np.asarray(expected_stepped_terrain[1]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.data[0]),
        np.asarray(expected_reset_terrain[0]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.data[1]),
        np.asarray(expected_stepped_terrain[1]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(np.asarray(next_state.observation[0]), np.asarray(jnp.zeros((1,), dtype=jnp.float32)))
    np.testing.assert_allclose(np.asarray(next_state.observation[1]), np.asarray(jnp.ones((1,), dtype=jnp.float32)))

    # done_out mirrors the inner env's done (env 0 done, rest not).
    assert np.asarray(cleared_done).tolist() == [True] + [False] * (batch - 1)


def test_autoreset_wrapper_rollout_swaps_domain_and_terrain_states():
    base_env = _AutoResetSmokeEnv()
    env = AutoResetWrapper(base_env)

    batch = 2
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)

    _, state = env.reset(rng_keys)
    _, _, _, done_after, _, next_state = env.step(state, action=jnp.zeros((batch, 1), dtype=jnp.float32))

    split_keys = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split_keys[:, 1]
    expected_domain = jnp.sum(reset_keys.astype(jnp.uint32), axis=-1, keepdims=True).astype(jnp.float32)
    expected_terrain = expected_domain + 100.0

    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.domain_randomizer_state[0]),
        np.asarray(expected_domain[0]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.terrain_state[0]),
        np.asarray(expected_terrain[0]),
        atol=0.0,
        rtol=0.0,
    )


def test_autoreset_wrapper_info_has_adaptive_keys_on_reset():
    base_env = _AutoResetSmokeEnv()
    env = AutoResetWrapper(base_env)

    batch = 3
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)

    _, state = env.reset(rng_keys)
    info = state.info

    assert "final_traj_no" in info
    assert "imitation_error_total" in info
    np.testing.assert_array_equal(np.asarray(info["final_traj_no"]), np.zeros((batch,), dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(info["imitation_error_total"]), np.zeros((batch,), dtype=np.float32)
    )

    _, _, _, done_after, _, next_state = env.step(state, action=jnp.zeros((batch, 1), dtype=jnp.float32))

    initial_domain = jnp.sum(rng_keys.astype(jnp.uint32), axis=-1, keepdims=True).astype(jnp.float32)
    initial_terrain = initial_domain + 100.0
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.domain_randomizer_state[1]),
        np.asarray(initial_domain[1] + 10.0),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.terrain_state[1]),
        np.asarray(initial_terrain[1] + 5.0),
        atol=0.0,
        rtol=0.0,
    )

    # done_out mirrors the inner env's done (env 0 done, rest not).
    assert np.asarray(done_after).tolist() == [True] + [False] * (batch - 1)

    # Second step is a smoke test for repeated RNG splits and swapping.
    _, _, _, done_after_2, _, _ = env.step(next_state, action=jnp.zeros((batch, 1), dtype=jnp.float32))
    assert np.asarray(done_after_2).shape == (batch,)


def test_nstep_wrapper_reset_buffer_structure():
    base_env = _NStepDummyEnv(obs_dim=2)
    n_steps = 4
    env = NStepWrapper(base_env, n_steps)

    obs, state = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (n_steps * base_env.obs_dim,)

    expected = np.zeros((n_steps, base_env.obs_dim), dtype=np.float32)
    expected[-1] = np.asarray(state.env_state.observation)

    np.testing.assert_allclose(np.asarray(obs).reshape(n_steps, base_env.obs_dim), expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(state.observation_buffer), expected, atol=0.0, rtol=0.0)


def test_nstep_wrapper_rolling_buffer_accumulates():
    base_env = _NStepDummyEnv(obs_dim=2)
    n_steps = 4
    env = NStepWrapper(base_env, n_steps)

    _, state = env.reset(jax.random.PRNGKey(0))
    obs0 = np.asarray(state.env_state.observation)
    obs1 = np.full((base_env.obs_dim,), 1.0, dtype=np.float32)
    obs2 = np.full((base_env.obs_dim,), 2.0, dtype=np.float32)
    obs3 = np.full((base_env.obs_dim,), 3.0, dtype=np.float32)
    obs4 = np.full((base_env.obs_dim,), 4.0, dtype=np.float32)

    action = jnp.zeros((1,), dtype=jnp.float32)

    obs, _, _, _, _, state = env.step(state, action)
    expected1 = np.zeros((n_steps, base_env.obs_dim), dtype=np.float32)
    expected1[2] = obs0
    expected1[3] = obs1
    np.testing.assert_allclose(np.asarray(obs).reshape(n_steps, base_env.obs_dim), expected1, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(state.observation_buffer), expected1, atol=0.0, rtol=0.0)

    obs, _, _, _, _, state = env.step(state, action)
    expected2 = np.zeros((n_steps, base_env.obs_dim), dtype=np.float32)
    expected2[1] = obs0
    expected2[2] = obs1
    expected2[3] = obs2
    np.testing.assert_allclose(np.asarray(obs).reshape(n_steps, base_env.obs_dim), expected2, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(state.observation_buffer), expected2, atol=0.0, rtol=0.0)

    obs, _, _, _, _, state = env.step(state, action)
    expected3 = np.stack([obs0, obs1, obs2, obs3], axis=0)
    np.testing.assert_allclose(np.asarray(obs).reshape(n_steps, base_env.obs_dim), expected3, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(state.observation_buffer), expected3, atol=0.0, rtol=0.0)

    obs, _, _, _, _, state = env.step(state, action)
    expected4 = np.stack([obs1, obs2, obs3, obs4], axis=0)
    np.testing.assert_allclose(np.asarray(obs).reshape(n_steps, base_env.obs_dim), expected4, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(state.observation_buffer), expected4, atol=0.0, rtol=0.0)


def test_nstep_wrapper_observation_space_tiling():
    base_env = _NStepDummyEnv(obs_dim=2)
    n_steps = 3
    env = NStepWrapper(base_env, n_steps)

    obs_space = env.info.observation_space
    expected_low = np.tile(base_env.info.observation_space.low, n_steps)
    expected_high = np.tile(base_env.info.observation_space.high, n_steps)

    np.testing.assert_allclose(obs_space.low, expected_low, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(obs_space.high, expected_high, atol=0.0, rtol=0.0)
    assert obs_space.shape == expected_low.shape


def test_nstep_wrapper_n_steps_one_passthrough():
    base_env = _NStepDummyEnv(obs_dim=3)
    env = NStepWrapper(base_env, n_steps=1)

    obs, state = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (base_env.obs_dim,)
    np.testing.assert_allclose(np.asarray(obs), np.asarray(state.env_state.observation), atol=0.0, rtol=0.0)

    action = jnp.zeros((1,), dtype=jnp.float32)

    next_obs, reward, absorbing, done, info, state = env.step(state, action)
    expected_obs1 = np.full((base_env.obs_dim,), 1.0, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(next_obs), expected_obs1, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(state.observation_buffer).reshape(base_env.obs_dim),
        expected_obs1,
        atol=0.0,
        rtol=0.0,
    )
    assert np.asarray(reward).item() == pytest.approx(1.5)
    assert np.asarray(absorbing).item() is False
    assert np.asarray(done).item() is False
    np.testing.assert_array_equal(np.asarray(info["step"]), np.asarray(1))
    np.testing.assert_array_equal(np.asarray(info["tag"]), np.asarray(7))

    next_obs_2, reward_2, absorbing_2, done_2, _, state = env.step(state, action)
    expected_obs2 = np.full((base_env.obs_dim,), 2.0, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(next_obs_2), expected_obs2, atol=0.0, rtol=0.0)
    assert np.asarray(reward_2).item() == pytest.approx(2.5)
    assert np.asarray(absorbing_2).item() is True
    assert np.asarray(done_2).item() is False

    _, reward_3, absorbing_3, done_3, _, _ = env.step(state, action)
    assert np.asarray(reward_3).item() == pytest.approx(3.5)
    assert np.asarray(absorbing_3).item() is False
    assert np.asarray(done_3).item() is True


def test_autoreset_wrapper_resets_nstep_history_on_done():
    base_env = _VecEnvBaseEnv()
    n_steps = 3
    env = AutoResetWrapper(LogWrapper(VecEnv(NStepWrapper(base_env, n_steps))))

    batch = 2
    rng_keys = jax.random.split(jax.random.PRNGKey(1), batch)
    obs, state = env.reset(rng_keys)
    assert obs.shape == (batch, n_steps)

    action = jnp.zeros((batch, 1), dtype=jnp.float32)
    next_obs, _, _, cleared_done, _, next_state = env.step(state, action=action)

    expected_done = np.zeros((n_steps,), dtype=np.float32)
    expected_keep = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    done_mask = (np.sum(np.asarray(rng_keys, dtype=np.uint32), axis=-1) % 2 == 0)
    # done_out mirrors the inner env's per-env done mask.
    assert np.asarray(cleared_done).tolist() == done_mask.tolist()
    assert np.any(done_mask)
    assert np.any(~done_mask)

    expected_done_batch = np.tile(expected_done, (batch, 1))
    expected_keep_batch = np.tile(expected_keep, (batch, 1))

    _assert_done_swaps(next_obs, expected_done_batch, expected_keep_batch, done_mask)

    obs_buffer = np.asarray(next_state.env_state.observation_buffer)
    _assert_done_swaps(obs_buffer.squeeze(-1), expected_done_batch, expected_keep_batch, done_mask)

    inner_obs = np.asarray(next_state.env_state.env_state.observation)
    reset_inner = np.zeros((batch, 1), dtype=np.float32)
    keep_inner = np.ones((batch, 1), dtype=np.float32)
    _assert_done_swaps(inner_obs, reset_inner, keep_inner, done_mask)


def test_autoreset_wrapper_done_count_and_reward_absorbing_passthrough():
    base_env = _AutoResetRewardAbsorbEnv()
    env = AutoResetWrapper(base_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 1), dtype=jnp.float32)
    _, reward, absorbing, cleared_done, info, next_state = env.step(state, action=action)

    expected_done = (np.arange(batch) % 2 == 0)
    expected_reward = np.arange(1, batch + 1, dtype=np.float32)

    np.testing.assert_allclose(np.asarray(reward), expected_reward, atol=0.0, rtol=0.0)
    assert np.asarray(absorbing).tolist() == expected_done.tolist()
    # done_out mirrors inner env's done (even indices done).
    assert np.asarray(cleared_done).tolist() == expected_done.tolist()
    np.testing.assert_allclose(np.asarray(next_state.reward), expected_reward, atol=0.0, rtol=0.0)
    assert np.asarray(next_state.absorbing).tolist() == expected_done.tolist()

    expected_done_count = expected_done.astype(np.int32)
    np.testing.assert_allclose(
        np.asarray(info["AutoResetWrapper_done_count"]),
        expected_done_count,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.info["AutoResetWrapper_done_count"]),
        expected_done_count,
        atol=0.0,
        rtol=0.0,
    )

    _, reward_2, absorbing_2, cleared_done_2, info_2, _ = env.step(next_state, action=action)
    np.testing.assert_allclose(np.asarray(reward_2), expected_reward, atol=0.0, rtol=0.0)
    assert np.asarray(absorbing_2).tolist() == expected_done.tolist()
    # done_out mirrors inner env's done (same pattern on every step).
    assert np.asarray(cleared_done_2).tolist() == expected_done.tolist()
    np.testing.assert_allclose(
        np.asarray(info_2["AutoResetWrapper_done_count"]),
        expected_done_count * 2,
        atol=0.0,
        rtol=0.0,
    )


def test_autoreset_wrapper_rng_chain_uses_reset_keys():
    base_env = _AutoResetTerrainSwapEnv()
    env = AutoResetWrapper(base_env)

    batch = 2
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 1), dtype=jnp.float32)
    _, _, _, _, _, next_state = env.step(state, action=action)

    split = jax.vmap(jax.random.split)(rng_keys)
    expected_rng_next = split[:, 0]
    expected_reset_keys = split[:, 1]
    expected_reset_terrain = _encode_keys_to_terrain(expected_reset_keys)

    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.autoreset_rng),
        np.asarray(expected_rng_next),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.terrain_state[0]),
        np.asarray(expected_reset_terrain[0]),
        atol=0.0,
        rtol=0.0,
    )
    assert not np.array_equal(np.asarray(expected_rng_next[0]), np.asarray(expected_reset_keys[0]))

    _, _, _, _, _, next_state_2 = env.step(next_state, action=action)
    split_2 = jax.vmap(jax.random.split)(expected_rng_next)
    expected_rng_next_2 = split_2[:, 0]
    expected_reset_keys_2 = split_2[:, 1]
    expected_reset_terrain_2 = _encode_keys_to_terrain(expected_reset_keys_2)

    np.testing.assert_allclose(
        np.asarray(next_state_2.additional_carry.autoreset_rng),
        np.asarray(expected_rng_next_2),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state_2.additional_carry.terrain_state[0]),
        np.asarray(expected_reset_terrain_2[0]),
        atol=0.0,
        rtol=0.0,
    )
    assert not np.array_equal(np.asarray(expected_reset_keys_2[0]), np.asarray(expected_reset_keys[0]))


def test_autoreset_wrapper_missing_optional_fields_are_safe():
    base_env = _AutoResetRewardAbsorbEnv()
    env = AutoResetWrapper(base_env)

    batch = 2
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)
    _, _, _, _, _, next_state = env.step(state, action=jnp.zeros((batch, 1), dtype=jnp.float32))

    assert not hasattr(next_state.additional_carry, "terrain_state")
    assert not hasattr(next_state.additional_carry, "traj_state")
    assert not hasattr(next_state.additional_carry, "observation_states")


def test_autoreset_wrapper_all_envs_done():
    base_env = _AutoResetMaskEnv(done_mask=True)
    env = AutoResetWrapper(base_env)

    batch = 3
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    _, _, _, cleared_done, info, next_state = env.step(state, action=jnp.zeros((batch, 1), dtype=jnp.float32))

    split = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split[:, 1]
    expected_reset = _encode_keys_to_terrain(reset_keys)

    np.testing.assert_allclose(np.asarray(next_state.data), np.asarray(expected_reset), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.terrain_state),
        np.asarray(expected_reset),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.observation),
        np.asarray(jnp.zeros((batch, 1), dtype=jnp.float32)),
        atol=0.0,
        rtol=0.0,
    )

    expected_done_count = np.ones((batch,), dtype=np.int32)
    np.testing.assert_allclose(np.asarray(info["AutoResetWrapper_done_count"]), expected_done_count, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(next_state.info["AutoResetWrapper_done_count"]), expected_done_count, atol=0.0, rtol=0.0
    )
    # done_out mirrors inner env's done (all envs done).
    assert np.asarray(cleared_done).tolist() == [True] * batch


def test_autoreset_wrapper_no_envs_done():
    base_env = _AutoResetMaskEnv(done_mask=False)
    env = AutoResetWrapper(base_env)

    batch = 3
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    _, _, _, cleared_done, info, next_state = env.step(state, action=jnp.zeros((batch, 1), dtype=jnp.float32))

    expected_step = _encode_keys_to_terrain(rng_keys) + 5.0
    np.testing.assert_allclose(np.asarray(next_state.data), np.asarray(expected_step), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.terrain_state),
        np.asarray(expected_step),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.observation),
        np.asarray(jnp.ones((batch, 1), dtype=jnp.float32)),
        atol=0.0,
        rtol=0.0,
    )

    expected_done_count = np.zeros((batch,), dtype=np.int32)
    np.testing.assert_allclose(np.asarray(info["AutoResetWrapper_done_count"]), expected_done_count, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(next_state.info["AutoResetWrapper_done_count"]), expected_done_count, atol=0.0, rtol=0.0
    )
    assert np.asarray(cleared_done).tolist() == [False] * batch


def test_autoreset_wrapper_batched_keyarray_caches_rng():
    base_env = _AutoResetKeyArrayEnv()
    env = AutoResetWrapper(base_env)

    batch = 3
    keys = jax.vmap(jax.random.key)(jnp.arange(batch))
    _, state = env.reset(keys)

    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(state.additional_carry.autoreset_rng)),
        np.asarray(jax.random.key_data(keys)),
    )


def test_autoreset_wrapper_jit_compatible():
    base_env = _AutoResetTerrainSwapEnv()
    env = AutoResetWrapper(base_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)
    action = jnp.zeros((batch, 1), dtype=jnp.float32)

    ref_obs, ref_reward, ref_absorbing, ref_done, ref_info, ref_state = env.step(state, action)

    step_fn = jax.jit(lambda s, a: env.step(s, a))
    jit_obs, jit_reward, jit_absorbing, jit_done, jit_info, jit_state = step_fn(state, action)

    np.testing.assert_allclose(np.asarray(jit_obs), np.asarray(ref_obs), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(jit_reward), np.asarray(ref_reward), atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(np.asarray(jit_absorbing), np.asarray(ref_absorbing))
    np.testing.assert_array_equal(np.asarray(jit_done), np.asarray(ref_done))
    np.testing.assert_allclose(
        np.asarray(jit_info["AutoResetWrapper_done_count"]),
        np.asarray(ref_info["AutoResetWrapper_done_count"]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(jit_state.additional_carry.terrain_state),
        np.asarray(ref_state.additional_carry.terrain_state),
        atol=0.0,
        rtol=0.0,
    )


def test_autoreset_wrapper_step_with_transition_keeps_prereset_state():
    base_env = _AutoResetTerrainSwapEnv()
    env = AutoResetWrapper(base_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    next_obs, _, _, done_out, _, next_state, transition_state, transition_obs = env.step_with_transition(
        state, action=jnp.zeros((batch, 1), dtype=jnp.float32)
    )

    done_mask = np.array([True] + [False] * (batch - 1))
    split = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split[:, 1]
    expected_reset = _encode_keys_to_terrain(reset_keys)
    expected_stepped = _encode_keys_to_terrain(rng_keys) + 1000.0

    _assert_done_swaps(next_state.data, expected_reset, expected_stepped, done_mask)
    _assert_done_swaps(next_state.additional_carry.terrain_state, expected_reset, expected_stepped, done_mask)
    np.testing.assert_allclose(np.asarray(transition_state.data), np.asarray(expected_stepped), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(transition_state.additional_carry.terrain_state),
        np.asarray(expected_stepped),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(transition_state.observation),
        np.asarray(jnp.ones((batch, 1), dtype=jnp.float32)),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(transition_obs),
        np.asarray(jnp.ones((batch, 1), dtype=jnp.float32)),
        atol=0.0,
        rtol=0.0,
    )
    assert not np.allclose(np.asarray(next_obs[0]), np.asarray(transition_obs[0]))
    np.testing.assert_array_equal(np.asarray(done_out), done_mask)
    np.testing.assert_array_equal(np.asarray(transition_state.done), done_mask)
    np.testing.assert_array_equal(np.asarray(next_state.done), np.zeros((batch,), dtype=bool))


def test_normalize_vec_reward_step_with_transition_preserves_inner_prereset_state():
    base_env = _AutoResetTerrainSwapEnv()
    env = NormalizeVecReward(AutoResetWrapper(base_env), gamma=0.99)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    _, reward, _, done_out, _, next_state, transition_state, _ = env.step_with_transition(
        state, action=jnp.zeros((batch, 1), dtype=jnp.float32)
    )

    done_mask = np.array([True] + [False] * (batch - 1))
    split = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split[:, 1]
    expected_reset = _encode_keys_to_terrain(reset_keys)
    expected_stepped = _encode_keys_to_terrain(rng_keys) + 1000.0

    _assert_done_swaps(next_state.env_state.data, expected_reset, expected_stepped, done_mask)
    _assert_done_swaps(next_state.env_state.additional_carry.terrain_state, expected_reset, expected_stepped, done_mask)
    np.testing.assert_allclose(
        np.asarray(transition_state.env_state.data), np.asarray(expected_stepped), atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(transition_state.env_state.additional_carry.terrain_state),
        np.asarray(expected_stepped),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(np.asarray(next_state.mean), np.asarray(transition_state.mean), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(next_state.var), np.asarray(transition_state.var), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(next_state.count), np.asarray(transition_state.count), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(next_state.return_val),
        np.asarray(transition_state.return_val),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(done_out), done_mask)
    assert np.all(np.isfinite(np.asarray(reward)))


def test_autoreset_wrapper_vecenv_integration():
    base_env = _VecEnvBaseEnv()
    env = AutoResetWrapper(VecEnv(base_env))

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)
    _, _, _, cleared_done, info, next_state = env.step(state, action=jnp.zeros((batch, 1), dtype=jnp.float32))

    done_mask = (np.sum(np.asarray(rng_keys, dtype=np.uint32), axis=-1) % 2 == 0)
    split = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split[:, 1]
    expected_domain_initial = jnp.sum(rng_keys.astype(jnp.uint32), axis=-1, keepdims=True).astype(jnp.float32)
    expected_domain_reset = jnp.sum(reset_keys.astype(jnp.uint32), axis=-1, keepdims=True).astype(jnp.float32)
    expected_domain_step = expected_domain_initial + 10.0
    expected_terrain_reset = expected_domain_reset + 100.0
    expected_terrain_step = expected_domain_initial + 100.0 + 5.0

    _assert_done_swaps(
        next_state.additional_carry.domain_randomizer_state,
        expected_domain_reset,
        expected_domain_step,
        done_mask,
    )
    _assert_done_swaps(
        next_state.additional_carry.terrain_state,
        expected_terrain_reset,
        expected_terrain_step,
        done_mask,
    )

    expected_done_count = done_mask.astype(np.int32)
    np.testing.assert_allclose(
        np.asarray(info["AutoResetWrapper_done_count"]),
        expected_done_count,
        atol=0.0,
        rtol=0.0,
    )
    # done_out mirrors the inner env's per-env done mask.
    assert np.asarray(cleared_done).tolist() == done_mask.tolist()


def test_autoreset_wrapper_long_rollout_rng_independence():
    base_env = _AutoResetMaskEnv(done_mask=True)
    env = AutoResetWrapper(base_env)

    batch = 1
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)
    action = jnp.zeros((batch, 1), dtype=jnp.float32)

    prev_rng = None
    for _ in range(12):
        _, _, _, _, _, state = env.step(state, action=action)
        current_rng = np.asarray(state.additional_carry.autoreset_rng[0])
        if prev_rng is not None:
            assert not np.array_equal(current_rng, prev_rng)
        prev_rng = current_rng


def test_autoreset_wrapper_stress_random_done_masks():
    base_env = _AutoResetStressEnv()
    env = AutoResetWrapper(base_env)

    batch = 6
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action_key = jax.random.PRNGKey(1)
    expected_done_count = np.zeros((batch,), dtype=np.int32)

    for _ in range(50):
        action_key, subkey = jax.random.split(action_key)
        done_mask = jax.random.bernoulli(subkey, 0.5, (batch,))
        done_mask = done_mask.at[0].set(True)
        done_mask = done_mask.at[1].set(False)
        action = jnp.where(done_mask, 1.0, -1.0).reshape(batch, 1).astype(jnp.float32)

        split = jax.vmap(jax.random.split)(state.additional_carry.autoreset_rng)
        expected_rng_next = split[:, 0]
        reset_keys = split[:, 1]

        stepped_key = jax.vmap(jax.random.split)(state.additional_carry.key)[:, 0]

        expected_reset = _encode_keys_to_terrain(reset_keys)
        expected_step_base = _encode_keys_to_terrain(stepped_key)

        expected_reset_obs = expected_reset + 0.5
        expected_step_obs = expected_step_base + 3.5
        expected_step_data = expected_step_base + 3.0
        expected_step_domain = expected_step_base + 13.0
        expected_step_terrain = expected_step_base + 23.0
        expected_reset_domain = expected_reset + 10.0
        expected_reset_terrain = expected_reset + 20.0
        expected_reward = np.asarray(expected_step_base[:, 0] + 1.0)
        expected_absorbing = np.asarray((expected_step_base[:, 0].astype(jnp.int32) % 2 == 0))

        obs, reward, absorbing, cleared_done, info, next_state = env.step(state, action)

        _assert_done_swaps(next_state.data, expected_reset, expected_step_data, done_mask)
        _assert_done_swaps(obs, expected_reset_obs, expected_step_obs, done_mask)
        _assert_done_swaps(
            next_state.additional_carry.domain_randomizer_state,
            expected_reset_domain,
            expected_step_domain,
            done_mask,
        )
        _assert_done_swaps(
            next_state.additional_carry.terrain_state,
            expected_reset_terrain,
            expected_step_terrain,
            done_mask,
        )
        _assert_done_swaps(
            next_state.additional_carry.last_action,
            jnp.zeros_like(action),
            action,
            done_mask,
        )

        done_idx = np.where(np.asarray(done_mask))[0]
        not_done_idx = np.where(~np.asarray(done_mask))[0]
        np.testing.assert_array_equal(
            np.asarray(next_state.additional_carry.key)[done_idx], np.asarray(reset_keys)[done_idx]
        )
        np.testing.assert_array_equal(
            np.asarray(next_state.additional_carry.key)[not_done_idx], np.asarray(stepped_key)[not_done_idx]
        )

        expected_cur_step = jnp.where(
            done_mask,
            jnp.ones_like(state.additional_carry.cur_step_in_episode),
            state.additional_carry.cur_step_in_episode + 1,
        )
        np.testing.assert_array_equal(
            np.asarray(next_state.additional_carry.cur_step_in_episode),
            np.asarray(expected_cur_step),
        )

        np.testing.assert_allclose(np.asarray(reward), expected_reward, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.asarray(next_state.reward), expected_reward, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(np.asarray(absorbing), expected_absorbing)
        np.testing.assert_array_equal(np.asarray(next_state.absorbing), expected_absorbing)

        expected_done_count = expected_done_count + np.asarray(done_mask, dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(info["AutoResetWrapper_done_count"]), expected_done_count)
        np.testing.assert_array_equal(np.asarray(next_state.info["AutoResetWrapper_done_count"]), expected_done_count)

        np.testing.assert_array_equal(np.asarray(cleared_done), np.asarray(done_mask))
        np.testing.assert_allclose(
            np.asarray(next_state.additional_carry.autoreset_rng),
            np.asarray(expected_rng_next),
            atol=0.0,
            rtol=0.0,
        )

        state = next_state


def test_autoreset_wrapper_single_key_caches_rng():
    batch = 3
    base_env = _AutoResetSingleKeyEnv(batch)
    env = AutoResetWrapper(base_env)

    rng_key = jax.random.PRNGKey(0)
    _, state = env.reset(rng_key)
    expected_rng = jax.random.split(rng_key, batch)
    np.testing.assert_array_equal(np.asarray(state.additional_carry.autoreset_rng), np.asarray(expected_rng))
    np.testing.assert_array_equal(
        np.asarray(state.info["AutoResetWrapper_done_count"]),
        np.zeros((batch,), dtype=np.int32),
    )


def test_autoreset_wrapper_swaps_rich_carry_fields_and_keeps_unbatched_data():
    base_env = _AutoResetRichEnv()
    env = AutoResetWrapper(base_env)

    batch = 3
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)
    _, _, _, cleared_done, _, next_state, transition_state, _ = env.step_with_transition(
        state, action=jnp.zeros((batch, 1), dtype=jnp.float32)
    )

    done_mask = (np.arange(batch) % 2 == 0)
    split = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split[:, 1]
    stepped_keys = split[:, 0]

    reset_encoded = _encode_keys_to_terrain(reset_keys)
    init_encoded = _encode_keys_to_terrain(rng_keys)

    _assert_done_swaps(
        next_state.additional_carry.last_action,
        _encode_keys_with_offset(reset_keys, 3.0),
        _encode_keys_with_offset(rng_keys, 3.0) + 100.0,
        done_mask,
    )
    _assert_done_swaps(
        next_state.additional_carry.domain_randomizer_state,
        _encode_keys_with_offset(reset_keys, 11.0),
        _encode_keys_with_offset(rng_keys, 11.0) + 100.0,
        done_mask,
    )
    _assert_done_swaps(
        next_state.additional_carry.terrain_state,
        _encode_keys_with_offset(reset_keys, 13.0),
        _encode_keys_with_offset(rng_keys, 13.0) + 100.0,
        done_mask,
    )

    expected_cur_step = np.where(done_mask, 1, 2)
    np.testing.assert_array_equal(
        np.asarray(next_state.additional_carry.cur_step_in_episode), expected_cur_step.astype(np.int32)
    )

    done_idx = np.where(done_mask)[0]
    not_done_idx = np.where(~done_mask)[0]
    np.testing.assert_array_equal(
        np.asarray(next_state.additional_carry.key)[done_idx], np.asarray(reset_keys)[done_idx]
    )
    np.testing.assert_array_equal(
        np.asarray(next_state.additional_carry.key)[not_done_idx], np.asarray(stepped_keys)[not_done_idx]
    )

    _assert_done_swaps(
        next_state.additional_carry.traj_state.phase,
        reset_encoded + 10.0,
        init_encoded + 10.0 + 100.0,
        done_mask,
    )
    _assert_done_swaps(
        next_state.additional_carry.traj_state.scale,
        reset_encoded + 20.0,
        init_encoded + 20.0 + 100.0,
        done_mask,
    )
    np.testing.assert_allclose(
        np.asarray(transition_state.additional_carry.traj_state.phase),
        np.asarray(init_encoded + 10.0 + 100.0),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(transition_state.additional_carry.traj_state.scale),
        np.asarray(init_encoded + 20.0 + 100.0),
        atol=0.0,
        rtol=0.0,
    )

    _assert_done_swaps(
        next_state.additional_carry.observation_states.hist,
        reset_encoded + 50.0,
        init_encoded + 50.0 + 100.0,
        done_mask,
    )
    _assert_done_swaps(
        next_state.additional_carry.observation_states.vel,
        reset_encoded + 60.0,
        init_encoded + 60.0 + 100.0,
        done_mask,
    )

    np.testing.assert_allclose(
        np.asarray(next_state.data["batched"])[done_idx],
        np.asarray(reset_encoded + 70.0)[done_idx],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.data["batched"])[not_done_idx],
        np.asarray(init_encoded + 70.0 + 1000.0)[not_done_idx],
        atol=0.0,
        rtol=0.0,
    )
    _assert_done_swaps(
        next_state.data["matrix"],
        reset_encoded + jnp.arange(batch, dtype=jnp.float32)[None, :],
        init_encoded + jnp.arange(batch, dtype=jnp.float32)[None, :] + 1000.0,
        done_mask,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.data["unbatched"]),
        np.asarray(jnp.array([9.0, 10.0], dtype=jnp.float32)),
        atol=0.0,
        rtol=0.0,
    )

    assert np.asarray(cleared_done).tolist() == done_mask.tolist()

def test_autoreset_wrapper_unwraps_and_rewraps_state():
    base_env = _AutoResetWrappedEnv()
    env = AutoResetWrapper(base_env)

    batch = 2
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)
    assert isinstance(state, _AutoResetWrappedState)
    np.testing.assert_allclose(np.asarray(state.tag), np.zeros((batch,), dtype=np.int32), atol=0.0, rtol=0.0)

    _, _, _, _, _, next_state = env.step(state, action=jnp.zeros((batch, 1), dtype=jnp.float32))
    assert isinstance(next_state, _AutoResetWrappedState)
    np.testing.assert_allclose(np.asarray(next_state.tag), np.ones((batch,), dtype=np.int32), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(next_state.env_state.observation[0]), np.asarray(jnp.zeros((1,))))
    np.testing.assert_allclose(np.asarray(next_state.env_state.observation[1]), np.asarray(jnp.ones((1,))))
