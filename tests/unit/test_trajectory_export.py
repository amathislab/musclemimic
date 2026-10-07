"""State timing and reproducibility contracts for the shared rollout exporter."""

import argparse
import json
from types import SimpleNamespace

import jax
import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.runner import eval_utils


@pytest.mark.parametrize(
    "arguments",
    [
        ["--export_trajectory"],
        ["--use_mujoco", "--export_kinetics"],
        ["--use_mujoco", "--export_trajectory", "--metrics_only"],
    ],
)
def test_unsupported_export_modes_fail_before_loading_checkpoint(arguments):
    parser = argparse.ArgumentParser()
    eval_utils.add_common_eval_args(parser, 1)
    args = parser.parse_args(["--path", "unused", *arguments])
    with pytest.raises(ValueError):
        eval_utils.normalize_eval_args(args)


def test_native_kinetics_export_cli_is_supported():
    parser = argparse.ArgumentParser()
    eval_utils.add_common_eval_args(parser, 1)
    args = parser.parse_args(["--path", "unused", "--use_mujoco", "--export_trajectory", "--export_kinetics"])
    eval_utils.normalize_eval_args(args)
    assert args.export_kinetics and args.eval_seed == 0


def test_evaluate_all_preserves_trajectory_export_mode():
    parser = argparse.ArgumentParser()
    eval_utils.add_common_eval_args(parser, 1)
    args = parser.parse_args(["--path", "unused", "--use_mujoco", "--export_trajectory"])
    args.evaluate_all = True
    eval_utils.normalize_eval_args(args)
    assert args.export_trajectory
    assert not args.metrics and not args.metrics_only


class ExportEnvironment:
    """Small deterministic trajectory environment with visibly changing activation."""

    dt = 0.01

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint name="joint"/>'
            '<geom type="sphere" size=".1"/></body></worldbody><actuator>'
            '<general name="muscle" joint="joint" dyntype="filter" dynprm=".1"/>'
            "</actuator></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        self.obs_container = {
            "touch_" + name: SimpleNamespace(obs_ind=np.array([i]))
            for i, name in enumerate(("r_foot", "r_toes", "l_foot", "l_toes"))
        }
        self.th = SimpleNamespace(
            n_trajectories=2,
            new_traj_no=0,
            new_subtraj_step_no_init=0,
            fixed_start_conf=[0, 0],
            len_trajectory=lambda _: 5,
            traj=SimpleNamespace(data=SimpleNamespace(get=lambda *_: SimpleNamespace(qpos=np.zeros(1)))),
        )

    def reset(self, key=None):
        mujoco.mj_resetData(self.model, self.data)
        self.th.new_traj_no = self.th.fixed_start_conf[0]
        self.data.act[0] = 0.2
        self.steps = 0
        return np.full(4, self.data.act[0])

    def step(self, action):
        self.data.qpos[0] += 1
        self.data.act[0] += 0.1
        self.data.ctrl[:] = np.asarray(action).ravel()
        self.data.time += self.dt
        self.steps += 1
        return np.full(4, self.data.act[0]), 1.0, False, self.steps == 4, {"subtraj_step_no": self.steps}


def export(monkeypatch, tmp_path, **kwargs):
    env = ExportEnvironment()
    config = OmegaConf.create({"experiment": {"n_seeds": 1, "len_obs_history": 1}})

    def policy_factory(_conf, deterministic):
        @jax.jit
        def policy(state, obs, key):
            action = obs[:1] if deterministic else jax.random.uniform(key, (1,))
            return action, state

        return policy

    monkeypatch.setattr(eval_utils, "make_policy_action_fn", policy_factory)
    _, archive = eval_utils.run_with_trajectory_export(
        env,
        SimpleNamespace(config=config),
        SimpleNamespace(train_state=None),
        n_steps=4,
        use_mujoco=True,
        trajectory_dir=str(tmp_path),
        **kwargs,
    )
    return archive


def test_export_activation_and_state_share_pre_step_time(monkeypatch, tmp_path):
    archive = export(monkeypatch, tmp_path, deterministic=True)
    np.testing.assert_allclose(archive["episode_0_muscle_activations"][:, 0], [0.2, 0.3, 0.4, 0.5])
    np.testing.assert_array_equal(archive["episode_0_joint_positions"][:, 0], [0, 1, 2, 3])
    np.testing.assert_allclose(archive["episode_0_policy_actions"][:, 0], [0.2, 0.3, 0.4, 0.5])
    assert archive["activation_sample_phase"] == "pre_step"
    assert archive["state_sample_phase"] == "pre_step"


@pytest.mark.parametrize("trajectory_length", [5, 6])
def test_export_serializes_completion_with_numpy_trajectory_length(monkeypatch, tmp_path, trajectory_length):
    reset = ExportEnvironment.reset

    def reset_with_numpy_length(self, key=None):
        self.th.len_trajectory = lambda _: np.int64(trajectory_length)
        return reset(self, key)

    monkeypatch.setattr(ExportEnvironment, "reset", reset_with_numpy_length)
    export(monkeypatch, tmp_path, deterministic=True)
    (export_path,) = tmp_path.glob("*.npz")
    with np.load(export_path, allow_pickle=False) as archive:
        metadata = json.loads(archive["episode_metadata_json"].item())
    assert metadata[0]["complete"] is (trajectory_length == 5)
    assert metadata[0]["episode_steps"] == 4


def test_export_filename_uses_motion_group_or_unique_timestamp(monkeypatch, tmp_path):
    grouped = tmp_path / "grouped"
    export(monkeypatch, grouped, deterministic=True, motion_group="KIT_KINESIS_TESTING_MOTIONS")
    assert [path.name for path in grouped.glob("*.npz")] == ["episodes_mujoco_KIT_KINESIS_TESTING_MOTIONS.npz"]

    ungrouped = tmp_path / "ungrouped"
    export(monkeypatch, ungrouped, deterministic=True)
    export(monkeypatch, ungrouped, deterministic=True)
    names = [path.name for path in ungrouped.glob("*.npz")]
    assert len(names) == 2
    assert not any("None" in name for name in names)


def test_export_seed_controls_reproducible_sampling(monkeypatch, tmp_path):
    first = export(monkeypatch, tmp_path / "first", eval_seed=7)
    repeated = export(monkeypatch, tmp_path / "repeat", eval_seed=7)
    other = export(monkeypatch, tmp_path / "other", eval_seed=8)
    key = "episode_0_policy_actions"
    np.testing.assert_array_equal(first[key], repeated[key])
    assert not np.array_equal(first[key], other[key])
    assert first["eval_seed"] == 7


def test_export_seeds_native_reset_without_changing_global_numpy_rng(monkeypatch, tmp_path):
    reset = ExportEnvironment.reset

    def random_reset(self, key=None):
        reset(self, key)
        self.data.act[0] = np.random.uniform()
        return np.full(4, self.data.act[0])

    monkeypatch.setattr(ExportEnvironment, "reset", random_reset)
    before = np.random.get_state()
    first = export(monkeypatch, tmp_path / "first", eval_seed=7)
    repeat = export(monkeypatch, tmp_path / "repeat", eval_seed=7)
    other = export(monkeypatch, tmp_path / "other", eval_seed=8)
    for expected, actual in zip(before, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(expected, actual)
    key = "episode_0_muscle_activations"
    np.testing.assert_array_equal(first[key], repeat[key])
    assert not np.array_equal(first[key], other[key])


@pytest.mark.parametrize("compiled", [False, True])
def test_export_keeps_reference_identity_across_passes(monkeypatch, tmp_path, compiled):
    with jax.disable_jit(not compiled):
        archive = export(monkeypatch, tmp_path, evaluate_all=True, evaluate_all_passes=2)
    assert archive["n_episodes"] == 4
    assert [int(archive[f"episode_{i}_traj_id"]) for i in range(4)] == [0, 1, 0, 1]
    for episode in range(4):
        np.testing.assert_array_equal(archive[f"episode_{episode}_reference_steps"], np.arange(4))
    assert not np.array_equal(archive["episode_0_policy_actions"], archive["episode_2_policy_actions"])
