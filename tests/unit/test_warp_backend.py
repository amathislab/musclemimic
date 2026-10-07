"""Warp backend tests for MJX environments.

These tests verify that VecEnv and AutoResetWrapper work correctly with the Warp backend.
This file is separate from test_mjx_reset_parity.py because Warp requires CUDA,
while the main test file sets CUDA_VISIBLE_DEVICES="" for CPU-only testing.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# Ensure default registry entries are available for this minimal test env.
import loco_mujoco.core.reward  # noqa: F401
import loco_mujoco.core.terminal_state_handler  # noqa: F401

from loco_mujoco.core.observations import ObservationType
from musclemimic.core.mujoco_mjx import Mjx
from musclemimic.core.wrappers.mjx import AutoResetWrapper, VecEnv


def _warp_available():
    """Check if Warp is available with CUDA support."""
    try:
        import warp

        return warp.is_cuda_available()
    except ImportError:
        return False


requires_warp = pytest.mark.skipif(not _warp_available(), reason="Warp not available or CUDA not available")


def _carry_keys_from_reset_keys(keys):
    def _carry_key(k):
        k, _ = jax.random.split(k)
        return jax.random.split(k, 8)[0]

    return jax.vmap(_carry_key)(keys)


def _key_sum_from_carry_keys(keys):
    return jnp.sum(keys.astype(jnp.uint32), axis=-1).astype(jnp.float32)


def _assert_done_swaps(actual, reset_value, step_value, done_mask):
    done_mask = np.asarray(done_mask, dtype=bool)
    done_idx = np.where(done_mask)[0]
    not_done_idx = np.where(~done_mask)[0]
    np.testing.assert_allclose(np.asarray(actual)[done_idx], np.asarray(reset_value)[done_idx], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(actual)[not_done_idx], np.asarray(step_value)[not_done_idx], atol=0.0, rtol=0.0
    )


class _WarpTestEnv(Mjx):
    """Minimal MJX env configured to use Warp backend for testing."""

    def __init__(self, xml_path: str, **kwargs):
        # MuJoCo 3.3.7+: nconmax=per-env, naconmax=total across all envs
        self.nconmax = 1
        self.naconmax = 1
        self.njmax = 1
        super().__init__(
            spec=xml_path,
            mjx_backend="warp",
            horizon=10,
            reward_type="NoReward",
            terminal_state_type="NoTerminalStateHandler",
            domain_randomization_type="NoDomainRandomization",
            terrain_type="StaticTerrain",
            init_state_type="DefaultInitialStateHandler",
            control_type="DefaultControl",
            actuation_spec=[],
            observation_spec=[ObservationType.FreeJointPos("root_free", xml_name="root")],
            **kwargs,
        )

    def _mjx_create_observation(self, model, data, carry):
        obs = jnp.asarray(data.qpos[:7], dtype=jnp.float32)
        return obs, carry


class _WarpSwapEnv(Mjx):
    """Warp MJX env with deterministic reset/step signals for swap validation."""

    def __init__(self, xml_path: str, done_mode: str, **kwargs):
        self._done_mode = done_mode
        # MuJoCo 3.3.7+: nconmax=per-env, naconmax=total across all envs
        self.nconmax = 1
        self.naconmax = 1
        self.njmax = 1
        super().__init__(
            spec=xml_path,
            mjx_backend="warp",
            horizon=10,
            reward_type="NoReward",
            terminal_state_type="NoTerminalStateHandler",
            domain_randomization_type="NoDomainRandomization",
            terrain_type="StaticTerrain",
            init_state_type="DefaultInitialStateHandler",
            control_type="DefaultControl",
            actuation_spec=[],
            observation_spec=[ObservationType.FreeJointPos("root_free", xml_name="root")],
            **kwargs,
        )

    def _key_sum(self, carry):
        return jnp.sum(carry.key.astype(jnp.uint32)).astype(jnp.float32)

    def _mjx_reset_carry(self, model, data, carry):
        data, carry = super()._mjx_reset_carry(model, data, carry)
        key_sum = self._key_sum(carry)
        data = data.replace(qpos=data.qpos.at[0].set(key_sum))
        carry = carry.replace(
            domain_randomizer_state=key_sum + 10.0,
            terrain_state=key_sum + 20.0,
        )
        return data, carry

    def _mjx_simulation_post_step(self, model, data, carry):
        key_sum = self._key_sum(carry)
        data = data.replace(qpos=data.qpos.at[0].set(key_sum + 100.0))
        carry = carry.replace(
            domain_randomizer_state=key_sum + 110.0,
            terrain_state=key_sum + 120.0,
        )
        return data, carry

    def _mjx_create_observation(self, model, data, carry):
        obs = jnp.asarray(data.qpos[:1], dtype=jnp.float32)
        return obs, carry

    def _mjx_is_absorbing(self, obs, info, data, carry):
        key_sum = self._key_sum(carry).astype(jnp.uint32)
        absorbing = (key_sum % 2 == 1)
        return absorbing, carry

    def _mjx_reward(self, obs, action, next_obs, absorbing, info, model, data, carry):
        reward = self._key_sum(carry) + 3.0
        return reward, carry, {"reward_total": reward}

    def _mjx_is_done(self, obs, absorbing, info, data, carry):
        if self._done_mode == "all":
            return jnp.array(True)
        if self._done_mode == "none":
            return jnp.array(False)
        if self._done_mode == "parity":
            key_sum = self._key_sum(carry).astype(jnp.uint32)
            return (key_sum % 2 == 0)
        return super()._mjx_is_done(obs, absorbing, info, data, carry)


@pytest.fixture
def warp_test_xml(tmp_path):
    """Create a minimal XML file compatible with Warp backend."""
    xml = """\
<mujoco model="warp_test">
  <option timestep="0.01"/>
  <worldbody>
    <body name="root" pos="0 0 1">
      <joint name="root" type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
    path = tmp_path / "warp_model.xml"
    path.write_text(xml)
    return str(path)


@pytest.fixture
def warp_template_xml(tmp_path):
    """Create an XML with template-only fields used by the reset regression test."""
    xml = """\
<mujoco model="warp_template_test">
  <size nuserdata="3"/>
  <option timestep="0.01"/>
  <worldbody>
    <body name="mocap_target" mocap="true" pos="1 2 3">
      <geom type="sphere" size="0.02" contype="0" conaffinity="0"/>
    </body>
    <body name="root" pos="0 0 1">
      <joint name="root" type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <sensor>
    <framepos objtype="body" objname="root"/>
  </sensor>
</mujoco>
"""
    path = tmp_path / "warp_template_model.xml"
    path.write_text(xml)
    return str(path)


@requires_warp
def test_warp_backend_mjx_env_reset(warp_test_xml):
    """Verify that MJX environment with Warp backend can reset."""
    env = _WarpTestEnv(warp_test_xml)
    assert env._backend_impl == "warp"

    key = jax.random.PRNGKey(0)
    state = env.mjx_reset(key)

    assert state.observation.shape == (7,)
    assert state.done.shape == ()
    assert not np.asarray(state.done)


@requires_warp
def test_warp_backend_reset_uses_put_data_template(monkeypatch, warp_template_xml):
    """Warp reset should preserve the full CPU template instead of hand-copying fields."""
    env = _WarpTestEnv(warp_template_xml)
    env._data.userdata[:] = np.array([7.0, 8.0, 9.0])
    env._data.qacc_warmstart[:] = np.arange(env._model.nv, dtype=np.float64) + 0.25
    mujoco.mj_forward(env._model, env._data)

    expected_userdata = env._data.userdata.copy()
    expected_warmstart = env._data.qacc_warmstart.copy()
    calls = []
    real_put_data = mjx.put_data

    def recording_put_data(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_put_data(*args, **kwargs)

    def fail_make_data(*args, **kwargs):
        raise AssertionError("Warp reset should use mjx.put_data, not mjx.make_data")

    monkeypatch.setattr(mjx, "put_data", recording_put_data)
    monkeypatch.setattr(mjx, "make_data", fail_make_data)

    state = env.mjx_reset(jax.random.PRNGKey(0))

    assert calls == [{"impl": "warp", "naconmax": env._naconmax, "njmax": env._njmax}]
    np.testing.assert_allclose(np.asarray(state.data.userdata), expected_userdata, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(state.data.qacc_warmstart), expected_warmstart, atol=0.0, rtol=0.0)

    reset_state = env._mjx_reset_in_step(state)

    assert calls[-1] == {"impl": "warp", "naconmax": env._naconmax, "njmax": env._njmax}
    np.testing.assert_allclose(np.asarray(reset_state.data.userdata), expected_userdata, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(reset_state.data.qacc_warmstart), expected_warmstart, atol=0.0, rtol=0.0)


@requires_warp
def test_warp_backend_mjx_env_step(warp_test_xml):
    """Verify that MJX environment with Warp backend can step."""
    env = _WarpTestEnv(warp_test_xml)
    key = jax.random.PRNGKey(0)
    state = env.mjx_reset(key)

    action = jnp.zeros((0,), dtype=jnp.float32)
    next_state = env.mjx_step(state, action)

    assert next_state.observation.shape == (7,)
    assert next_state.done.shape == ()


@requires_warp
def test_warp_backend_vecenv_reset(warp_test_xml):
    """Verify that VecEnv works with Warp backend."""
    base_env = _WarpTestEnv(warp_test_xml)
    env = VecEnv(base_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    obs, state = env.reset(rng_keys)

    assert obs.shape == (batch, 7)
    assert state.observation.shape == (batch, 7)
    assert state.done.shape == (batch,)


@requires_warp
def test_warp_backend_vecenv_step(warp_test_xml):
    """Verify that VecEnv can step with Warp backend."""
    base_env = _WarpTestEnv(warp_test_xml)
    env = VecEnv(base_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    obs, reward, absorbing, done, info, next_state = env.step(state, action)

    assert obs.shape == (batch, 7)
    assert reward.shape == (batch,)
    assert done.shape == (batch,)


@requires_warp
def test_warp_backend_autoreset_wrapper_integration(warp_test_xml):
    """Verify AutoResetWrapper works with VecEnv + Warp backend."""
    base_env = _WarpTestEnv(warp_test_xml)
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    obs, state = env.reset(rng_keys)

    assert obs.shape == (batch, 7)
    assert state.additional_carry.autoreset_rng is not None

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    obs, reward, absorbing, done, info, next_state = env.step(state, action)

    assert obs.shape == (batch, 7)
    assert reward.shape == (batch,)
    assert done.shape == (batch,)
    assert "AutoResetWrapper_done_count" in info
    np.testing.assert_array_equal(np.asarray(done), np.zeros((batch,), dtype=bool))


@requires_warp
def test_warp_backend_autoreset_wrapper_rng_chain(warp_test_xml):
    """Verify RNG chain advances correctly with Warp backend."""
    base_env = _WarpTestEnv(warp_test_xml)
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 2
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    initial_rng = np.asarray(state.additional_carry.autoreset_rng)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    _, _, _, _, _, next_state = env.step(state, action)

    next_rng = np.asarray(next_state.additional_carry.autoreset_rng)
    assert not np.array_equal(initial_rng, next_rng), "RNG should advance after step"

    split = jax.vmap(jax.random.split)(rng_keys)
    expected_rng = split[:, 0]
    np.testing.assert_allclose(next_rng, np.asarray(expected_rng), atol=0.0, rtol=0.0)


@requires_warp
def test_warp_backend_autoreset_wrapper_swaps_partial_done(warp_test_xml):
    """Verify AutoResetWrapper swaps reset values only for done envs on Warp."""
    base_env = _WarpSwapEnv(warp_test_xml, done_mode="parity")
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    obs, reward, absorbing, done_out, info, next_state = env.step(state, action)

    initial_carry_keys = _carry_keys_from_reset_keys(rng_keys)
    initial_key_sum = _key_sum_from_carry_keys(initial_carry_keys)
    done_mask = (initial_key_sum.astype(jnp.uint32) % 2 == 0)

    split = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split[:, 1]
    reset_carry_keys = _carry_keys_from_reset_keys(reset_keys)
    reset_key_sum = _key_sum_from_carry_keys(reset_carry_keys)

    expected_step_qpos = initial_key_sum + 100.0
    expected_reset_qpos = reset_key_sum

    _assert_done_swaps(
        np.asarray(next_state.data.qpos[:, 0]),
        expected_reset_qpos,
        expected_step_qpos,
        done_mask,
    )
    _assert_done_swaps(
        np.asarray(obs[:, 0]),
        expected_reset_qpos,
        expected_step_qpos,
        done_mask,
    )
    _assert_done_swaps(
        np.asarray(next_state.additional_carry.domain_randomizer_state),
        reset_key_sum + 10.0,
        initial_key_sum + 110.0,
        done_mask,
    )
    _assert_done_swaps(
        np.asarray(next_state.additional_carry.terrain_state),
        reset_key_sum + 20.0,
        initial_key_sum + 120.0,
        done_mask,
    )

    done_idx = np.where(np.asarray(done_mask))[0]
    not_done_idx = np.where(~np.asarray(done_mask))[0]
    np.testing.assert_array_equal(
        np.asarray(next_state.additional_carry.key)[done_idx],
        np.asarray(reset_carry_keys)[done_idx],
    )
    np.testing.assert_array_equal(
        np.asarray(next_state.additional_carry.key)[not_done_idx],
        np.asarray(initial_carry_keys)[not_done_idx],
    )

    expected_cur_step = np.where(np.asarray(done_mask), 1, 2)
    np.testing.assert_array_equal(
        np.asarray(next_state.additional_carry.cur_step_in_episode), expected_cur_step.astype(np.int32)
    )

    expected_reward = np.asarray(initial_key_sum + 3.0)
    expected_absorbing = (np.asarray(initial_key_sum).astype(np.uint32) % 2 == 1)
    np.testing.assert_allclose(np.asarray(reward), expected_reward, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(next_state.reward), expected_reward, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(np.asarray(absorbing), expected_absorbing)
    np.testing.assert_array_equal(np.asarray(next_state.absorbing), expected_absorbing)

    np.testing.assert_allclose(
        np.asarray(info["AutoResetWrapper_done_count"]),
        np.asarray(done_mask).astype(np.int32),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(done_out), np.asarray(done_mask))
    np.testing.assert_array_equal(np.asarray(next_state.done), np.zeros((batch,), dtype=bool))


@requires_warp
def test_warp_backend_autoreset_wrapper_stress_rollout(warp_test_xml):
    """Stress test swap invariants over multiple steps on Warp backend."""
    base_env = _WarpSwapEnv(warp_test_xml, done_mode="parity")
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    expected_done_count = np.zeros((batch,), dtype=np.int32)
    action = jnp.zeros((batch, 0), dtype=jnp.float32)

    for _ in range(25):
        key_sum = _key_sum_from_carry_keys(state.additional_carry.key)
        done_mask = (key_sum.astype(jnp.uint32) % 2 == 0)

        split = jax.vmap(jax.random.split)(state.additional_carry.autoreset_rng)
        expected_rng_next = split[:, 0]
        reset_keys = split[:, 1]
        reset_carry_keys = _carry_keys_from_reset_keys(reset_keys)
        reset_key_sum = _key_sum_from_carry_keys(reset_carry_keys)

        expected_step_qpos = key_sum + 100.0
        expected_reset_qpos = reset_key_sum

        obs, reward, absorbing, done_out, info, next_state = env.step(state, action)

        _assert_done_swaps(
            np.asarray(next_state.data.qpos[:, 0]),
            expected_reset_qpos,
            expected_step_qpos,
            done_mask,
        )
        _assert_done_swaps(
            np.asarray(obs[:, 0]),
            expected_reset_qpos,
            expected_step_qpos,
            done_mask,
        )
        _assert_done_swaps(
            np.asarray(next_state.additional_carry.domain_randomizer_state),
            reset_key_sum + 10.0,
            key_sum + 110.0,
            done_mask,
        )
        _assert_done_swaps(
            np.asarray(next_state.additional_carry.terrain_state),
            reset_key_sum + 20.0,
            key_sum + 120.0,
            done_mask,
        )

        done_idx = np.where(np.asarray(done_mask))[0]
        not_done_idx = np.where(~np.asarray(done_mask))[0]
        np.testing.assert_array_equal(
            np.asarray(next_state.additional_carry.key)[done_idx],
            np.asarray(reset_carry_keys)[done_idx],
        )
        np.testing.assert_array_equal(
            np.asarray(next_state.additional_carry.key)[not_done_idx],
            np.asarray(state.additional_carry.key)[not_done_idx],
        )

        expected_cur_step = np.where(
            np.asarray(done_mask),
            1,
            np.asarray(state.additional_carry.cur_step_in_episode) + 1,
        )
        np.testing.assert_array_equal(
            np.asarray(next_state.additional_carry.cur_step_in_episode),
            expected_cur_step.astype(np.int32),
        )

        expected_reward = np.asarray(key_sum + 3.0)
        expected_absorbing = np.asarray((key_sum.astype(jnp.uint32) % 2 == 1))
        np.testing.assert_allclose(np.asarray(reward), expected_reward, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.asarray(next_state.reward), expected_reward, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(np.asarray(absorbing), expected_absorbing)
        np.testing.assert_array_equal(np.asarray(next_state.absorbing), expected_absorbing)

        expected_done_count = expected_done_count + np.asarray(done_mask, dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(info["AutoResetWrapper_done_count"]), expected_done_count)
        np.testing.assert_array_equal(np.asarray(next_state.info["AutoResetWrapper_done_count"]), expected_done_count)
        np.testing.assert_array_equal(np.asarray(done_out), np.asarray(done_mask))
        np.testing.assert_array_equal(np.asarray(next_state.done), np.zeros((batch,), dtype=bool))
        np.testing.assert_allclose(
            np.asarray(next_state.additional_carry.autoreset_rng),
            np.asarray(expected_rng_next),
            atol=0.0,
            rtol=0.0,
        )

        state = next_state


@requires_warp
def test_warp_backend_autoreset_wrapper_all_envs_done(warp_test_xml):
    """Verify swap behavior when all envs are done on Warp."""
    base_env = _WarpSwapEnv(warp_test_xml, done_mode="all")
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 3
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    _, _, _, done_out, info, next_state = env.step(state, action)

    split = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split[:, 1]
    reset_carry_keys = _carry_keys_from_reset_keys(reset_keys)
    reset_key_sum = _key_sum_from_carry_keys(reset_carry_keys)

    np.testing.assert_allclose(np.asarray(next_state.data.qpos[:, 0]), np.asarray(reset_key_sum), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.terrain_state),
        np.asarray(reset_key_sum + 20.0),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(next_state.additional_carry.cur_step_in_episode),
        np.ones((batch,), dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(info["AutoResetWrapper_done_count"]),
        np.ones((batch,), dtype=np.int32),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(done_out), np.ones((batch,), dtype=bool))
    np.testing.assert_array_equal(np.asarray(next_state.done), np.zeros((batch,), dtype=bool))


@requires_warp
def test_warp_backend_autoreset_wrapper_no_envs_done(warp_test_xml):
    """Verify swap behavior when no envs are done on Warp."""
    base_env = _WarpSwapEnv(warp_test_xml, done_mode="none")
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 3
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    _, _, _, done_out, info, next_state = env.step(state, action)

    initial_carry_keys = _carry_keys_from_reset_keys(rng_keys)
    initial_key_sum = _key_sum_from_carry_keys(initial_carry_keys)

    np.testing.assert_allclose(
        np.asarray(next_state.data.qpos[:, 0]),
        np.asarray(initial_key_sum + 100.0),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_state.additional_carry.terrain_state),
        np.asarray(initial_key_sum + 120.0),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(next_state.additional_carry.cur_step_in_episode),
        np.full((batch,), 2, dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(info["AutoResetWrapper_done_count"]),
        np.zeros((batch,), dtype=np.int32),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(done_out), np.zeros((batch,), dtype=bool))
    np.testing.assert_array_equal(np.asarray(next_state.done), np.zeros((batch,), dtype=bool))


@requires_warp
def test_warp_backend_autoreset_wrapper_done_count_accumulates(warp_test_xml):
    """Verify done_count accumulates correctly with Warp backend."""
    base_env = _WarpSwapEnv(warp_test_xml, done_mode="all")
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 2
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)

    for step in range(5):
        _, _, _, done_out, info, state = env.step(state, action)
        np.testing.assert_array_equal(
            np.asarray(done_out),
            np.ones((batch,), dtype=bool),
            err_msg=f"done_out should report episode boundaries at step {step}",
        )
        np.testing.assert_array_equal(
            np.asarray(state.done),
            np.zeros((batch,), dtype=bool),
            err_msg=f"inner done should be cleared at step {step}",
        )
        done_count = np.asarray(info["AutoResetWrapper_done_count"])
        expected = np.full((batch,), step + 1, dtype=np.int32)
        np.testing.assert_array_equal(done_count, expected, err_msg=f"done_count mismatch at step {step}")
        np.testing.assert_array_equal(
            np.asarray(state.info["AutoResetWrapper_done_count"]),
            expected,
            err_msg=f"state done_count mismatch at step {step}",
        )


@requires_warp
def test_warp_backend_multiple_rollouts(warp_test_xml):
    """Verify multiple rollouts work correctly with Warp backend."""
    base_env = _WarpTestEnv(warp_test_xml)
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 2
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    prev_rng = None
    for _ in range(10):
        _, _, _, _, _, state = env.step(state, action)
        current_rng = np.asarray(state.additional_carry.autoreset_rng)
        if prev_rng is not None:
            assert not np.array_equal(current_rng, prev_rng), "RNG should change each step"
        prev_rng = current_rng.copy()


@requires_warp
def test_warp_backend_autoreset_wrapper_jit_compatible(warp_test_xml):
    """Verify AutoResetWrapper works under jax.jit with Warp backend."""
    base_env = _WarpSwapEnv(warp_test_xml, done_mode="parity")
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)
    action = jnp.zeros((batch, 0), dtype=jnp.float32)

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
        np.asarray(jit_state.data.qpos[:, 0]),
        np.asarray(ref_state.data.qpos[:, 0]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(jit_state.additional_carry.terrain_state),
        np.asarray(ref_state.additional_carry.terrain_state),
        atol=0.0,
        rtol=0.0,
    )


@pytest.fixture
def warp_contact_xml(tmp_path):
    """Create an XML with contacts/collisions for Warp backend testing."""
    xml = """\
<mujoco model="warp_contact_test">
  <option timestep="0.01"/>
  <worldbody>
    <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0"/>
    <body name="root" pos="0 0 0.5">
      <joint name="root" type="free"/>
      <geom name="ball" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
    path = tmp_path / "warp_contact_model.xml"
    path.write_text(xml)
    return str(path)


class _WarpContactEnv(Mjx):
    """MJX env with contacts for testing Warp collision field handling."""

    def __init__(self, xml_path: str, **kwargs):
        # MuJoCo 3.3.7+: nconmax=per-env, naconmax=total across all envs
        self.nconmax = 64
        self.naconmax = 64  # Single env, so naconmax = nconmax
        self.njmax = 128
        super().__init__(
            spec=xml_path,
            mjx_backend="warp",
            horizon=10,
            reward_type="NoReward",
            terminal_state_type="NoTerminalStateHandler",
            domain_randomization_type="NoDomainRandomization",
            terrain_type="StaticTerrain",
            init_state_type="DefaultInitialStateHandler",
            control_type="DefaultControl",
            actuation_spec=[],
            observation_spec=[ObservationType.FreeJointPos("root_free", xml_name="root")],
            nconmax=self.nconmax,  # Preallocate contact buffer
            njmax=self.njmax,  # Preallocate constraint buffer
            **kwargs,
        )

    def _mjx_create_observation(self, model, data, carry):
        obs = jnp.asarray(data.qpos[:7], dtype=jnp.float32)
        return obs, carry


class _WarpContactSwapEnv(_WarpContactEnv):
    """Warp MJX env with deterministic qpos values for swap validation."""

    def __init__(self, xml_path: str, done_mode: str, **kwargs):
        self._done_mode = done_mode
        # naconmax inherited from _WarpContactEnv
        super().__init__(xml_path, **kwargs)

    def _key_sum(self, carry):
        return jnp.sum(carry.key.astype(jnp.uint32)).astype(jnp.float32)

    def _mjx_reset_carry(self, model, data, carry):
        data, carry = super()._mjx_reset_carry(model, data, carry)
        key_sum = self._key_sum(carry)
        data = data.replace(qpos=data.qpos.at[0].set(key_sum))
        return data, carry

    def _mjx_simulation_post_step(self, model, data, carry):
        key_sum = self._key_sum(carry)
        data = data.replace(qpos=data.qpos.at[0].set(key_sum + 100.0))
        return data, carry

    def _mjx_is_done(self, obs, absorbing, info, data, carry):
        if self._done_mode == "all":
            return jnp.array(True)
        if self._done_mode == "none":
            return jnp.array(False)
        if self._done_mode == "parity":
            key_sum = self._key_sum(carry).astype(jnp.uint32)
            return (key_sum % 2 == 0)
        return super()._mjx_is_done(obs, absorbing, info, data, carry)


@requires_warp
def test_warp_backend_contact_model_reset(warp_contact_xml):
    """Verify Warp backend works with contact-enabled models."""
    env = _WarpContactEnv(warp_contact_xml)
    assert env._backend_impl == "warp"
    assert env.nconmax == 64
    assert env._naconmax == 64
    assert env._njmax == 128

    key = jax.random.PRNGKey(0)
    state = env.mjx_reset(key)

    assert state.observation.shape == (7,)
    assert not np.asarray(state.done)


@requires_warp
def test_warp_backend_contact_vecenv_step(warp_contact_xml):
    """Verify VecEnv stepping with contact model on Warp backend."""
    base_env = _WarpContactEnv(warp_contact_xml)
    env = VecEnv(base_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    for _ in range(5):
        obs, reward, absorbing, done, info, state = env.step(state, action)
        assert obs.shape == (batch, 7)
        assert done.shape == (batch,)


@requires_warp
def test_warp_backend_contact_autoreset_wrapper(warp_contact_xml):
    """Verify AutoResetWrapper handles contact arrays correctly with Warp.

    This tests the where_done swap logic on MJX Data containing collision fields
    (contact, efc_force, etc.) that have shapes based on nconmax/njmax.
    """
    base_env = _WarpContactEnv(warp_contact_xml)
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    obs, state = env.reset(rng_keys)

    assert obs.shape == (batch, 7)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    for step_idx in range(10):
        obs, reward, absorbing, done_out, info, state = env.step(state, action)
        assert obs.shape == (batch, 7), f"Observation shape mismatch at step {step_idx}"
        assert done_out.shape == (batch,), f"Done shape mismatch at step {step_idx}"
        expected_done = np.full((batch,), step_idx == 9, dtype=bool)
        np.testing.assert_array_equal(
            np.asarray(done_out),
            expected_done,
            err_msg=f"done_out mismatch at step {step_idx}",
        )
        np.testing.assert_array_equal(
            np.asarray(state.done),
            np.zeros((batch,), dtype=bool),
            err_msg=f"inner done should be cleared at step {step_idx}",
        )


@requires_warp
def test_warp_backend_contact_autoreset_swaps_done_envs(warp_contact_xml):
    """Verify AutoResetWrapper swap path works with contact-enabled Warp envs."""
    base_env = _WarpContactSwapEnv(warp_contact_xml, done_mode="parity")
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 4
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    _, _, _, _, _, next_state = env.step(state, action)

    initial_carry_keys = _carry_keys_from_reset_keys(rng_keys)
    initial_key_sum = _key_sum_from_carry_keys(initial_carry_keys)
    done_mask = (initial_key_sum.astype(jnp.uint32) % 2 == 0)

    split = jax.vmap(jax.random.split)(rng_keys)
    reset_keys = split[:, 1]
    reset_carry_keys = _carry_keys_from_reset_keys(reset_keys)
    reset_key_sum = _key_sum_from_carry_keys(reset_carry_keys)

    expected_step = initial_key_sum + 100.0
    expected_reset = reset_key_sum

    _assert_done_swaps(
        np.asarray(next_state.data.qpos[:, 0]),
        expected_reset,
        expected_step,
        done_mask,
    )
    contact_pos = np.asarray(next_state.data._impl.contact__pos)
    assert contact_pos.shape[0] == base_env.nconmax


@requires_warp
def test_warp_backend_data_fields_swapped_correctly(warp_contact_xml):
    """Verify MJX Data fields keep expected shapes with contact-enabled Warp models."""
    base_env = _WarpContactEnv(warp_contact_xml)
    vec_env = VecEnv(base_env)
    env = AutoResetWrapper(vec_env)

    batch = 2
    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch)
    _, state = env.reset(rng_keys)

    initial_qpos = np.asarray(state.data.qpos)

    action = jnp.zeros((batch, 0), dtype=jnp.float32)
    _, _, _, _, _, next_state = env.step(state, action)

    next_qpos = np.asarray(next_state.data.qpos)
    assert next_qpos.shape == initial_qpos.shape, "Data qpos shape should be preserved after step"

    if hasattr(next_state.data, "contact"):
        contact = next_state.data.contact
        if hasattr(contact, "pos") and contact.pos is not None:
            contact_pos = np.asarray(contact.pos)
            assert contact_pos.ndim >= 2, "Contact pos should have at least 2 dimensions"
