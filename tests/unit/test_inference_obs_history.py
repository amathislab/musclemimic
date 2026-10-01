import numpy as np
import jax.numpy as jnp
from types import SimpleNamespace

from omegaconf import OmegaConf

import musclemimic.algorithms.ppo.inference as inference


class FakePolicy:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def sample(self, seed):
        return jnp.zeros((self.action_dim,), dtype=jnp.float32)


class FakeNetwork:
    def __init__(self, action_dim: int = 2):
        self.action_dim = action_dim
        self.seen_obs = []

    def apply(self, vars_in, obs_b, mutable=None):
        self.seen_obs.append(np.array(obs_b))
        return FakePolicy(self.action_dim), {"run_stats": vars_in["run_stats"]}


class FakeTrainState:
    def __init__(self, params=None, run_stats=None):
        if params is None:
            params = {"log_std": np.zeros(2, dtype=np.float32)}
        if run_stats is None:
            run_stats = {}
        self.params = params
        self.run_stats = run_stats

    def replace(self, **kwargs):
        params = kwargs.get("params", self.params)
        run_stats = kwargs.get("run_stats", self.run_stats)
        return FakeTrainState(params=params, run_stats=run_stats)


class FakeEnv:
    def __init__(self):
        self._obs = np.array([1.0, 2.0], dtype=np.float32)

    def reset(self):
        self._obs = np.array([1.0, 2.0], dtype=np.float32)
        return self._obs.copy()

    def step(self, action):
        self._obs = self._obs + 1.0
        return self._obs.copy(), 0.0, False, False, {}

    def render(self, record=False):
        pass

    def stop(self):
        pass


def test_observation_history_buffer_rolls():
    buffer = inference.ObservationHistoryBuffer(3)
    obs0 = np.array([1.0, 2.0], dtype=np.float32)
    obs1 = np.array([3.0, 4.0], dtype=np.float32)
    obs2 = np.array([5.0, 6.0], dtype=np.float32)

    out0 = buffer.reset(obs0)
    out1 = buffer.step(obs1)
    out2 = buffer.step(obs2)

    np.testing.assert_allclose(out0, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(out1, np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(out2, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32))


def test_play_policy_mujoco_stacks_history(monkeypatch):
    monkeypatch.setattr(inference.jax, "jit", lambda f: f)

    env = FakeEnv()
    network = FakeNetwork(action_dim=2)
    agent_conf = SimpleNamespace(
        config=OmegaConf.create({"experiment": {"len_obs_history": 3, "n_seeds": 1}}),
        network=network,
    )
    agent_state = SimpleNamespace(train_state=FakeTrainState())

    inference.play_policy(
        env,
        agent_conf,
        agent_state,
        n_envs=1,
        n_steps=2,
        render=False,
        record=False,
        use_mujoco=True,
        do_wrap_env=False,
    )

    assert len(network.seen_obs) == 2
    np.testing.assert_allclose(
        network.seen_obs[0].reshape(-1),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        network.seen_obs[1].reshape(-1),
        np.array([0.0, 0.0, 1.0, 2.0, 2.0, 3.0], dtype=np.float32),
    )


def test_play_policy_can_stop_after_one_mujoco_episode(monkeypatch):
    monkeypatch.setattr(inference.jax, "jit", lambda f: f)

    class OneEpisodeEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.reset_count = 0
            self.step_count = 0

        def reset(self):
            self.reset_count += 1
            return super().reset()

        def step(self, action):
            self.step_count += 1
            self._obs = self._obs + 1.0
            done = self.step_count == 2
            return self._obs.copy(), 0.0, False, done, {"truncated": done}

    env = OneEpisodeEnv()
    network = FakeNetwork(action_dim=2)
    agent_conf = SimpleNamespace(
        config=OmegaConf.create({"experiment": {"len_obs_history": 1, "n_seeds": 1}}),
        network=network,
    )
    agent_state = SimpleNamespace(train_state=FakeTrainState())

    inference.play_policy(
        env,
        agent_conf,
        agent_state,
        n_envs=1,
        n_steps=10,
        render=False,
        record=False,
        use_mujoco=True,
        do_wrap_env=False,
        stop_after_first_episode=True,
    )

    assert env.step_count == 2
    assert env.reset_count == 1
    assert len(network.seen_obs) == 2


def test_observation_history_buffer_split_goal_reset():
    """Test that split_goal only stacks state obs, keeps goal at current timestep."""
    # obs = [state0, state1, goal0, goal1] with state_indices=[0,1], goal_indices=[2,3]
    state_indices = np.array([0, 1])
    goal_indices = np.array([2, 3])
    buffer = inference.ObservationHistoryBuffer(
        n_steps=3, split_goal=True, state_indices=state_indices, goal_indices=goal_indices
    )
    obs0 = np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float32)

    out0 = buffer.reset(obs0)

    # Expected: [state_hist(zeros, zeros, state0), goal0]
    # state_hist = [0, 0, 0, 0, 1, 2] (3 steps * 2 state_dim)
    # goal = [10, 20]
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 10.0, 20.0], dtype=np.float32)
    np.testing.assert_allclose(out0, expected)


def test_observation_history_buffer_split_goal_step():
    """Test that split_goal updates state history but goal always from current obs."""
    state_indices = np.array([0, 1])
    goal_indices = np.array([2, 3])
    buffer = inference.ObservationHistoryBuffer(
        n_steps=3, split_goal=True, state_indices=state_indices, goal_indices=goal_indices
    )
    obs0 = np.array([1.0, 2.0, 10.0, 20.0], dtype=np.float32)
    obs1 = np.array([3.0, 4.0, 30.0, 40.0], dtype=np.float32)
    obs2 = np.array([5.0, 6.0, 50.0, 60.0], dtype=np.float32)

    buffer.reset(obs0)
    out1 = buffer.step(obs1)
    out2 = buffer.step(obs2)

    # After step 1: state_hist = [0, 0, 1, 2, 3, 4], goal = [30, 40]
    expected1 = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 30.0, 40.0], dtype=np.float32)
    np.testing.assert_allclose(out1, expected1)

    # After step 2: state_hist = [1, 2, 3, 4, 5, 6], goal = [50, 60]
    expected2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0, 60.0], dtype=np.float32)
    np.testing.assert_allclose(out2, expected2)


def test_observation_history_buffer_split_goal_maintains_shape():
    """Test that split_goal output shape is (n_steps * state_dim + goal_dim)."""
    obs_dim = 10
    goal_dim = 3
    state_dim = obs_dim - goal_dim
    n_steps = 4

    state_indices = np.arange(state_dim)
    goal_indices = np.arange(state_dim, obs_dim)
    buffer = inference.ObservationHistoryBuffer(
        n_steps=n_steps, split_goal=True, state_indices=state_indices, goal_indices=goal_indices
    )
    obs = np.random.randn(obs_dim).astype(np.float32)

    out = buffer.reset(obs)
    expected_shape = n_steps * state_dim + goal_dim
    assert out.shape == (expected_shape,), f"Expected shape ({expected_shape},), got {out.shape}"

    out2 = buffer.step(obs)
    assert out2.shape == (expected_shape,), f"Expected shape ({expected_shape},), got {out2.shape}"


class FakeObsContainer:
    """Fake obs_container for testing."""

    def __init__(self, goal_indices):
        self._goal_indices = np.array(goal_indices)

    def get_obs_ind_by_group(self, group_name):
        if group_name == "goal":
            return self._goal_indices
        return np.array([])


class FakeEnvWithGoal:
    """Fake env with goal dimension for split_goal testing."""

    def __init__(self, state_dim=2, goal_dim=2):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self._goal = SimpleNamespace(dim=goal_dim)
        self._step = 0
        # Goal indices are at the end
        self.obs_container = FakeObsContainer(
            goal_indices=np.arange(state_dim, state_dim + goal_dim)
        )
        self.info = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(state_dim + goal_dim,))
        )

    def reset(self):
        self._step = 0
        state = np.arange(self.state_dim, dtype=np.float32)
        goal = np.arange(100, 100 + self.goal_dim, dtype=np.float32)
        return np.concatenate([state, goal])

    def step(self, action):
        self._step += 1
        state = np.arange(self.state_dim, dtype=np.float32) + self._step
        goal = np.arange(100, 100 + self.goal_dim, dtype=np.float32) + self._step * 10
        obs = np.concatenate([state, goal])
        return obs, 0.0, False, False, {}

    def render(self, record=False):
        pass

    def stop(self):
        pass


def test_play_policy_mujoco_split_goal(monkeypatch):
    """Test that play_policy correctly handles split_goal in MuJoCo mode."""
    monkeypatch.setattr(inference.jax, "jit", lambda f: f)

    env = FakeEnvWithGoal(state_dim=2, goal_dim=2)
    network = FakeNetwork(action_dim=2)
    agent_conf = SimpleNamespace(
        config=OmegaConf.create({
            "experiment": {
                "len_obs_history": 3,
                "split_goal": True,
                "n_seeds": 1,
            }
        }),
        network=network,
    )
    agent_state = SimpleNamespace(train_state=FakeTrainState())

    inference.play_policy(
        env,
        agent_conf,
        agent_state,
        n_envs=1,
        n_steps=2,
        render=False,
        record=False,
        use_mujoco=True,
        do_wrap_env=False,
    )

    assert len(network.seen_obs) == 2

    # After reset: state_hist = [0, 0, 0, 0, 0, 1], goal = [100, 101]
    expected0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 101.0], dtype=np.float32)
    np.testing.assert_allclose(network.seen_obs[0].reshape(-1), expected0)

    # After step 1: state_hist = [0, 0, 0, 1, 1, 2], goal = [110, 111]
    expected1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 110.0, 111.0], dtype=np.float32)
    np.testing.assert_allclose(network.seen_obs[1].reshape(-1), expected1)
