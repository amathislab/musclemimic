"""Complete-dataset evaluation through the public fullbody and bimanual CLIs."""

import importlib
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.runner import eval_utils


class TrajectoryLibrary:
    n_trajectories = 3
    is_numpy = False
    lengths = (2, 4, 3)

    def len_trajectory(self, index):
        return self.lengths[index]


class CpuEnvironment:
    dt = 0.01

    def __init__(self):
        self.th = TrajectoryLibrary()
        self.info = SimpleNamespace(horizon=10)
        self.starts = []

    def reset(self):
        self.trajectory, self.step_number = self.th.fixed_start_conf
        self.starts.append((self.trajectory, self.step_number))
        return np.zeros(1)

    def step(self, action):
        self.step_number += 1
        early = self.trajectory == 1 and self.step_number == 2
        done = early or self.step_number == self.th.len_trajectory(self.trajectory)
        return (
            np.zeros(1),
            1.0,
            early,
            done,
            {
                "traj_no": self.trajectory,
                "traj_len": self.th.len_trajectory(self.trajectory),
                "subtraj_step_no": self.step_number - 1,
            },
        )


class MjxEnvironment:
    dt = 0.01

    def __init__(self):
        self.th = TrajectoryLibrary()
        self.info = SimpleNamespace(horizon=10)

    def reset(self, key):
        return self.reset_to(key, jnp.asarray(0))

    def reset_to(self, key, trajectory):
        return self.reset_to_step(key, trajectory, jnp.asarray(0))

    def reset_to_step(self, key, trajectory, step):
        return jnp.zeros(1), {"trajectory": trajectory, "step": step}

    def step(self, state, action):
        step = state["step"] + 1
        trajectory = state["trajectory"]
        early = (trajectory == 1) & (step >= 2)
        done = early | (step >= jnp.asarray(self.th.lengths)[trajectory])
        return jnp.zeros(1), jnp.asarray(1.0), early, done, {}, {"trajectory": trajectory, "step": step}


@pytest.fixture(params=["fullbody.eval", "bimanual.eval"])
def entrypoint(request, monkeypatch):
    module = importlib.import_module(request.param)
    config = OmegaConf.create(
        {
            "experiment": {
                "n_seeds": 1,
                "env_params": {"env_name": "TestEnvironment"},
                "task_factory": {"name": "ImitationFactory", "params": {"amass_dataset_conf": {}}},
            }
        }
    )
    action_modes = []

    def policy_factory(_conf, deterministic):
        action_modes.append(deterministic)
        return lambda state, obs, key: (jnp.zeros_like(obs), state)

    def unexpected_playback(*args, **kwargs):
        pytest.fail("Dataset evaluation must finish after the complete sweep, without ordinary playback")

    algorithm = SimpleNamespace(
        init_agent_conf=lambda *args: SimpleNamespace(config=config),
        restore_agent_state=lambda *args: SimpleNamespace(train_state=None),
        play_policy=unexpected_playback,
        play_policy_mujoco=unexpected_playback,
    )
    monkeypatch.setattr(module, "load_checkpoint", lambda path: (config, None, None))
    monkeypatch.setattr(module, "pick_algorithm", lambda config: algorithm)
    monkeypatch.setattr(module, "setup_headless", lambda args: (True, None))
    monkeypatch.setattr(eval_utils, "make_policy_action_fn", policy_factory)

    def run(flags, use_mujoco):
        env = CpuEnvironment() if use_mujoco else MjxEnvironment()
        monkeypatch.setattr(
            module.TaskFactory, "get_factory_cls", lambda name: SimpleNamespace(make=lambda **kwargs: env)
        )
        arguments = ["eval", "--path", "unused", "--evaluate_all", "--no_render", "--n_steps", "1", *flags]
        if use_mujoco:
            arguments.append("--use_mujoco")
        monkeypatch.setattr(sys, "argv", arguments)
        assert module.main() == 0
        return env, action_modes

    return module.__name__, run


@pytest.mark.parametrize("use_mujoco", [False, True])
@pytest.mark.parametrize("flags", [[], ["--metrics"], ["--metrics_only"]])
def test_evaluate_all_visits_complete_library_once(entrypoint, capsys, flags, use_mujoco):
    _, run = entrypoint
    env, _ = run([*flags, "--metrics_steps", "1", "--metrics_envs", "2"], use_mujoco)
    output = capsys.readouterr().out
    assert "val_total_frame: 9.000000" in output
    assert "val_mean_episode_length: 2.333333" in output
    assert "val_early_termination_count: 1.000000" in output
    if use_mujoco:
        assert env.starts == [(0, 0), (1, 0), (2, 0)]


@pytest.mark.parametrize(
    ("flags", "deterministic"),
    [([], True), (["--stochastic"], False), (["--metrics_stochastic"], False), (["--metrics_deterministic"], True)],
)
def test_implicit_evaluate_all_preserves_requested_action_mode(entrypoint, flags, deterministic):
    _, run = entrypoint
    _, modes = run(flags, True)
    assert modes == [deterministic]


def test_explicit_metrics_preserves_entrypoint_default(entrypoint):
    name, run = entrypoint
    _, modes = run(["--metrics"], True)
    assert modes == [name == "bimanual.eval"]
