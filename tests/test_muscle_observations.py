"""
Unit tests for muscle observation flags.
"""

import jax
import mujoco
import numpy as np
import pytest

from musclemimic.environments.humanoids.bimanual import MjxMyoBimanualArm, MyoBimanualArm
from musclemimic.environments.humanoids.myofullbody import MjxMyoFullBody, MyoFullBody

# Configure JAX to use CPU only for tests
jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def base_env_config():
    """Base configuration for test environments."""
    return {
        "timestep": 0.002,
        "n_substeps": 5,
    }


def get_obs_value(env, obs, obs_key):
    """Helper to get observation value from observation array."""
    obs_entry = env.obs_container[obs_key]
    indices = obs_entry.obs_ind
    if obs_entry.dim == 1:
        # Single value observation
        return obs[indices[0]]
    else:
        # Multi-dimensional observation
        return obs[indices]


def get_actuator_length_from_mujoco(model, data, actuator_name):
    """Get actuator length directly from MuJoCo data."""
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    return data.actuator_length[act_id]


def get_actuator_velocity_from_mujoco(model, data, actuator_name):
    """Get actuator velocity directly from MuJoCo data."""
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    return data.actuator_velocity[act_id]


def get_actuator_force_from_mujoco(model, data, actuator_name):
    """Get actuator force directly from MuJoCo data."""
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    return data.actuator_force[act_id]


def get_actuator_activation_from_mujoco(model, data, actuator_name):
    """Get actuator activation directly from MuJoCo data."""
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    return data.act[act_id] if act_id < len(data.act) else 0.0


def get_touch_sensor_from_mujoco(model, data, sensor_name):
    """Get touch sensor value directly from MuJoCo data."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    # Touch sensors return a single scalar value (contact force magnitude)
    sensor_adr = model.sensor_adr[sensor_id]
    return data.sensordata[sensor_adr]


class TestMyoBimanualArmObservations:
    """Test suite for MyoBimanualArm observation flags."""

    @pytest.mark.parametrize(
        ("enable_joint_pos_observations", "enable_joint_vel_observations", "expected_keys"),
        [
            (False, False, set()),
            (True, False, {"q_all_pos"}),
            (False, True, {"dq_all_vel"}),
        ],
    )
    def test_joint_observation_flags_control_joint_observation_presence(
        self,
        base_env_config,
        enable_joint_pos_observations,
        enable_joint_vel_observations,
        expected_keys,
    ):
        """Test that joint position/velocity flags are respected by MyoBimanualArm."""
        env = MyoBimanualArm(
            disable_fingers=True,
            enable_joint_pos_observations=enable_joint_pos_observations,
            enable_joint_vel_observations=enable_joint_vel_observations,
            enable_muscle_length_observations=False,
            enable_muscle_velocity_observations=False,
            enable_muscle_force_observations=False,
            enable_muscle_activation_observations=False,
            **base_env_config,
        )

        env.reset()
        obs_keys = set(env.obs_container.keys())

        assert ("q_all_pos" in obs_keys) == ("q_all_pos" in expected_keys)
        assert ("dq_all_vel" in obs_keys) == ("dq_all_vel" in expected_keys)

    def test_no_muscle_observations_default(self, base_env_config):
        """Test that by default, no muscle observations are included."""
        env = MyoBimanualArm(
            disable_fingers=True,
            enable_muscle_length_observations=False,
            enable_muscle_velocity_observations=False,
            enable_muscle_force_observations=False,
            enable_muscle_activation_observations=False,
            **base_env_config,
        )

        env.reset()
        obs_keys = list(env.obs_container.keys())

        # Should only have joint positions and velocities
        assert "q_all_pos" in obs_keys
        assert "dq_all_vel" in obs_keys

        # Should NOT have any muscle observations
        muscle_obs = [key for key in obs_keys if key.startswith("muscle_")]
        assert len(muscle_obs) == 0, f"Found unexpected muscle observations: {muscle_obs}"

    def test_muscle_length_observations_match_mujoco(self, base_env_config):
        """Test that muscle length observations match MuJoCo actuator_length values."""
        env = MyoBimanualArm(
            disable_fingers=True,
            enable_muscle_length_observations=True,
            enable_muscle_velocity_observations=False,
            enable_muscle_force_observations=False,
            enable_muscle_activation_observations=False,
            **base_env_config,
        )

        obs = env.reset()
        model = env._model
        data = env._data
        obs_keys = list(env.obs_container.keys())

        # Check a few actuators (not all to keep test fast)
        num_to_check = min(5, model.nu)
        for i in range(num_to_check):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            obs_key = f"muscle_length_{actuator_name.lower()}"

            # Check that observation exists
            assert obs_key in obs_keys, f"Missing observation: {obs_key}"

            # Get expected value from MuJoCo
            expected_value = get_actuator_length_from_mujoco(model, data, actuator_name)

            # Get observed value
            observed_value = get_obs_value(env, obs, obs_key)

            # Compare values (allow small numerical differences)
            np.testing.assert_allclose(
                observed_value,
                expected_value,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Mismatch for {obs_key}: observed={observed_value}, expected={expected_value}",
            )

        # Verify no other muscle observations are present
        muscle_velocity_obs = [key for key in obs_keys if key.startswith("muscle_velocity_")]
        muscle_force_obs = [key for key in obs_keys if key.startswith("muscle_force_")]
        muscle_activation_obs = [key for key in obs_keys if key.startswith("muscle_activation_")]

        assert len(muscle_velocity_obs) == 0, "Found unexpected muscle_velocity observations"
        assert len(muscle_force_obs) == 0, "Found unexpected muscle_force observations"
        assert len(muscle_activation_obs) == 0, "Found unexpected muscle_activation observations"

    def test_muscle_velocity_observations_match_mujoco(self, base_env_config):
        """Test that muscle velocity observations match MuJoCo actuator_velocity values."""
        env = MyoBimanualArm(
            disable_fingers=True,
            enable_muscle_length_observations=False,
            enable_muscle_velocity_observations=True,
            enable_muscle_force_observations=False,
            enable_muscle_activation_observations=False,
            **base_env_config,
        )

        obs = env.reset()
        model = env._model
        data = env._data
        obs_keys = list(env.obs_container.keys())

        # Check a few actuators
        num_to_check = min(5, model.nu)
        for i in range(num_to_check):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            obs_key = f"muscle_velocity_{actuator_name.lower()}"

            assert obs_key in obs_keys, f"Missing observation: {obs_key}"

            expected_value = get_actuator_velocity_from_mujoco(model, data, actuator_name)
            observed_value = get_obs_value(env, obs, obs_key)

            np.testing.assert_allclose(
                observed_value, expected_value, rtol=1e-5, atol=1e-8, err_msg=f"Mismatch for {obs_key}"
            )

    def test_muscle_force_observations_match_mujoco(self, base_env_config):
        """Test that muscle force observations match MuJoCo actuator_force values."""
        env = MyoBimanualArm(
            disable_fingers=True,
            enable_muscle_length_observations=False,
            enable_muscle_velocity_observations=False,
            enable_muscle_force_observations=True,
            enable_muscle_activation_observations=False,
            **base_env_config,
        )

        obs = env.reset()
        model = env._model
        data = env._data
        obs_keys = list(env.obs_container.keys())

        # Check a few actuators
        num_to_check = min(5, model.nu)
        for i in range(num_to_check):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            obs_key = f"muscle_force_{actuator_name.lower()}"

            assert obs_key in obs_keys, f"Missing observation: {obs_key}"

            expected_value = get_actuator_force_from_mujoco(model, data, actuator_name)
            observed_value = get_obs_value(env, obs, obs_key)

            np.testing.assert_allclose(
                observed_value, expected_value, rtol=1e-5, atol=1e-8, err_msg=f"Mismatch for {obs_key}"
            )

    def test_muscle_activation_observations_match_mujoco(self, base_env_config):
        """Test that muscle activation observations match MuJoCo act values."""
        env = MyoBimanualArm(
            disable_fingers=True,
            enable_muscle_length_observations=False,
            enable_muscle_velocity_observations=False,
            enable_muscle_force_observations=False,
            enable_muscle_activation_observations=True,
            **base_env_config,
        )

        obs = env.reset()
        model = env._model
        data = env._data
        obs_keys = list(env.obs_container.keys())

        # Check a few actuators
        num_to_check = min(5, model.nu)
        for i in range(num_to_check):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            obs_key = f"muscle_activation_{actuator_name.lower()}"

            assert obs_key in obs_keys, f"Missing observation: {obs_key}"

            expected_value = get_actuator_activation_from_mujoco(model, data, actuator_name)
            observed_value = get_obs_value(env, obs, obs_key)

            np.testing.assert_allclose(
                observed_value, expected_value, rtol=1e-5, atol=1e-8, err_msg=f"Mismatch for {obs_key}"
            )

    def test_all_muscle_observations_enabled(self, base_env_config):
        """Test that all muscle observations are present when all flags are True."""
        env = MyoBimanualArm(
            disable_fingers=True,
            enable_muscle_length_observations=True,
            enable_muscle_velocity_observations=True,
            enable_muscle_force_observations=True,
            enable_muscle_activation_observations=True,
            **base_env_config,
        )

        obs = env.reset()
        obs_keys = list(env.obs_container.keys())

        # Check that all muscle observation types exist
        muscle_length_obs = [key for key in obs_keys if key.startswith("muscle_length_")]
        muscle_velocity_obs = [key for key in obs_keys if key.startswith("muscle_velocity_")]
        muscle_force_obs = [key for key in obs_keys if key.startswith("muscle_force_")]
        muscle_activation_obs = [key for key in obs_keys if key.startswith("muscle_activation_")]

        assert len(muscle_length_obs) > 0, "No muscle_length observations found"
        assert len(muscle_velocity_obs) > 0, "No muscle_velocity observations found"
        assert len(muscle_force_obs) > 0, "No muscle_force observations found"
        assert len(muscle_activation_obs) > 0, "No muscle_activation observations found"

        # All counts should be equal (one per actuator)
        num_actuators = len(muscle_length_obs)
        assert len(muscle_velocity_obs) == num_actuators
        assert len(muscle_force_obs) == num_actuators
        assert len(muscle_activation_obs) == num_actuators

        # Verify a few values match MuJoCo
        model = env._model
        data = env._data
        num_to_check = min(3, model.nu)

        for i in range(num_to_check):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

            # Check muscle length
            obs_key = f"muscle_length_{actuator_name.lower()}"
            expected = get_actuator_length_from_mujoco(model, data, actuator_name)
            observed = get_obs_value(env, obs, obs_key)
            np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-8)

            # Check muscle force
            obs_key = f"muscle_force_{actuator_name.lower()}"
            expected = get_actuator_force_from_mujoco(model, data, actuator_name)
            observed = get_obs_value(env, obs, obs_key)
            np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-8)


class TestMjxMyoBimanualArmObservations:
    """Test suite for MjxMyoBimanualArm observation flags (JAX backend)."""

    def test_mjx_jax_all_muscle_observations_present(self, base_env_config):
        """Test MjxMyoBimanualArm with JAX backend has all muscle observations when flags are True."""
        env = MjxMyoBimanualArm(
            disable_fingers=True,
            enable_muscle_length_observations=True,
            enable_muscle_velocity_observations=True,
            enable_muscle_force_observations=True,
            enable_muscle_activation_observations=True,
            mjx_backend="jax",
            **base_env_config,
        )

        env.reset()
        obs_keys = list(env.obs_container.keys())

        # Verify all muscle observation types exist
        muscle_length_obs = [key for key in obs_keys if key.startswith("muscle_length_")]
        muscle_velocity_obs = [key for key in obs_keys if key.startswith("muscle_velocity_")]
        muscle_force_obs = [key for key in obs_keys if key.startswith("muscle_force_")]
        muscle_activation_obs = [key for key in obs_keys if key.startswith("muscle_activation_")]

        assert len(muscle_length_obs) > 0, "No muscle_length observations found"
        assert len(muscle_velocity_obs) > 0, "No muscle_velocity observations found"
        assert len(muscle_force_obs) > 0, "No muscle_force observations found"
        assert len(muscle_activation_obs) > 0, "No muscle_activation observations found"

    def test_mjx_jax_no_muscle_observations(self, base_env_config):
        """Test MjxMyoBimanualArm with JAX backend has no muscle observations when flags are False."""
        env = MjxMyoBimanualArm(
            disable_fingers=True,
            enable_muscle_length_observations=False,
            enable_muscle_velocity_observations=False,
            enable_muscle_force_observations=False,
            enable_muscle_activation_observations=False,
            mjx_backend="jax",
            **base_env_config,
        )

        env.reset()
        obs_keys = list(env.obs_container.keys())

        # Should NOT have any muscle observations
        muscle_obs = [key for key in obs_keys if key.startswith("muscle_")]
        assert len(muscle_obs) == 0, f"Found unexpected muscle observations: {muscle_obs}"


class TestMyoFullBodyObservations:
    """Test suite for MyoFullBody observation flags."""

    def test_egocentric_root_observations_replace_world_frame_root_state(self, base_env_config):
        env = MyoFullBody(
            disable_fingers=True,
            enable_muscle_length_observations=False,
            enable_muscle_velocity_observations=False,
            enable_muscle_force_observations=False,
            enable_muscle_activation_observations=False,
            enable_touch_sensor_observations=False,
            use_egocentric_root_observations=True,
            **base_env_config,
        )

        observation = env.reset()
        observation_names = set(env.obs_container.keys())

        assert {"q_root_height", "projected_gravity", "dq_free_joint_heading"} <= observation_names
        assert "q_free_joint" not in observation_names
        assert "dq_free_joint" not in observation_names
        assert env.obs_container["q_root_height"].dim == 1
        assert env.obs_container["projected_gravity"].dim == 3
        assert env.obs_container["dq_free_joint_heading"].dim == 6
        assert np.all(np.isfinite(observation))

    def test_myofullbody_all_observations_match_mujoco(self, base_env_config):
        """Test MyoFullBody with all observation flags enabled match MuJoCo values."""
        env = MyoFullBody(
            disable_fingers=True,
            enable_muscle_length_observations=True,
            enable_muscle_velocity_observations=True,
            enable_muscle_force_observations=True,
            enable_muscle_activation_observations=True,
            enable_touch_sensor_observations=True,
            **base_env_config,
        )

        obs = env.reset()
        model = env._model
        data = env._data
        obs_keys = list(env.obs_container.keys())

        # Check a subset of muscle actuators
        num_to_check = min(3, model.nu)
        for i in range(num_to_check):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

            # Check muscle length
            obs_key = f"muscle_length_{actuator_name.lower()}"
            if obs_key in obs_keys:
                expected = get_actuator_length_from_mujoco(model, data, actuator_name)
                observed = get_obs_value(env, obs, obs_key)
                np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-8)

    def test_myofullbody_touch_sensors_match_mujoco(self, base_env_config):
        """Test MyoFullBody touch sensor observations match MuJoCo sensor data."""
        env = MyoFullBody(
            disable_fingers=True,
            enable_muscle_length_observations=False,
            enable_muscle_velocity_observations=False,
            enable_muscle_force_observations=False,
            enable_muscle_activation_observations=False,
            enable_touch_sensor_observations=True,
            **base_env_config,
        )

        obs = env.reset()
        model = env._model
        data = env._data
        obs_keys = list(env.obs_container.keys())

        # Check expected touch sensors
        touch_sensors = ["r_foot", "r_toes", "l_foot", "l_toes"]
        for sensor_name in touch_sensors:
            obs_key = f"touch_{sensor_name}"
            if obs_key in obs_keys:
                expected = get_touch_sensor_from_mujoco(model, data, sensor_name)
                observed = get_obs_value(env, obs, obs_key)
                # Touch sensor returns a single scalar value (contact force magnitude)
                np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-8)

    def test_myofullbody_no_muscle_observations(self, base_env_config):
        """Test MyoFullBody with all muscle observation flags disabled."""
        env = MyoFullBody(
            disable_fingers=True,
            enable_muscle_length_observations=False,
            enable_muscle_velocity_observations=False,
            enable_muscle_force_observations=False,
            enable_muscle_activation_observations=False,
            enable_touch_sensor_observations=False,
            **base_env_config,
        )

        env.reset()
        obs_keys = list(env.obs_container.keys())

        # Should NOT have any muscle or touch observations
        muscle_obs = [key for key in obs_keys if key.startswith("muscle_")]
        touch_obs = [key for key in obs_keys if key.startswith("touch_")]

        assert len(muscle_obs) == 0, f"Found unexpected muscle observations: {muscle_obs}"
        assert len(touch_obs) == 0, f"Found unexpected touch observations: {touch_obs}"


class TestMjxMyoFullBodyObservations:
    """Test suite for MjxMyoFullBody observation flags (JAX backend)."""

    def test_mjx_myofullbody_all_observations_present(self, base_env_config):
        """Test MjxMyoFullBody with all observation flags enabled."""
        env = MjxMyoFullBody(
            disable_fingers=True,
            enable_muscle_length_observations=True,
            enable_muscle_velocity_observations=True,
            enable_muscle_force_observations=True,
            enable_muscle_activation_observations=True,
            enable_touch_sensor_observations=True,
            mjx_backend="jax",
            **base_env_config,
        )

        env.reset()
        obs_keys = list(env.obs_container.keys())

        # Verify all observation types exist
        muscle_length_obs = [key for key in obs_keys if key.startswith("muscle_length_")]
        muscle_velocity_obs = [key for key in obs_keys if key.startswith("muscle_velocity_")]
        muscle_force_obs = [key for key in obs_keys if key.startswith("muscle_force_")]
        muscle_activation_obs = [key for key in obs_keys if key.startswith("muscle_activation_")]
        touch_sensor_obs = [key for key in obs_keys if key.startswith("touch_")]

        assert len(muscle_length_obs) > 0, "No muscle_length observations found"
        assert len(muscle_velocity_obs) > 0, "No muscle_velocity observations found"
        assert len(muscle_force_obs) > 0, "No muscle_force observations found"
        assert len(muscle_activation_obs) > 0, "No muscle_activation observations found"
        assert len(touch_sensor_obs) > 0, "No touch sensor observations found"


if __name__ == "__main__":
    # Run tests with: pytest tests/test_muscle_observations.py -v
    pytest.main([__file__, "-v"])
