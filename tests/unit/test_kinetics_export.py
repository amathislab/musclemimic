"""Native contact-force and observer regression tests without external assets."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from loco_mujoco.core.mujoco_base import Mujoco
from loco_mujoco.core.observations.base import TouchSensor
from musclemimic.runner import eval_utils
from musclemimic.runner.kinetics_export import KineticsRecorder


def native_environment(*, substeps=5, ground_quat="1 0 0 0"):
    feet = "".join(
        f'<body name="{body}" pos="{x} {y} 0"><geom type="sphere" size=".05" mass="1"/>'
        f'<site name="{sensor}" type="sphere" size=".055"/></body>'
        for body, sensor, x, y in (
            ("calcn_r", "r_foot", "-.08", "-.08"),
            ("toes_r", "r_toes", ".08", "-.08"),
            ("calcn_l", "l_foot", "-.08", ".08"),
            ("toes_l", "l_toes", ".08", ".08"),
        )
    )
    sensors = "".join(f'<touch name="{name}" site="{name}"/>' for name in ("r_foot", "r_toes", "l_foot", "l_toes"))
    spec = mujoco.MjSpec.from_string(
        f'<mujoco><option timestep=".002" integrator="implicitfast"/>'
        f'<worldbody><geom name="ground" type="plane" size="2 2 .1" quat="{ground_quat}"/>'
        f'<body name="root" pos="0 0 .048"><freejoint/>{feet}'
        '<body pos="0 0 .2"><joint name="hinge"/>'
        '<geom type="sphere" size=".02" mass=".1"/></body></body></worldbody>'
        '<actuator><general name="activation" joint="hinge" dyntype="filter" dynprm=".1"'
        f' ctrllimited="true" ctrlrange="0 1"/></actuator><sensor>{sensors}</sensor></mujoco>'
    )
    observations = [TouchSensor("touch_" + name, name) for name in ("r_foot", "r_toes", "l_foot", "l_toes")]
    return Mujoco(spec, ["activation"], observations, n_substeps=substeps)


@pytest.mark.parametrize("ground_quat", ["1 0 0 0", ".9961947 .0871557 0 0"])
def test_world_grf_matches_independent_root_constraint_force(ground_quat):
    env = native_environment(ground_quat=ground_quat)
    env.reset()
    recorder = KineticsRecorder(env.model)
    maximum_error = 0.0

    def observe(model, data, start):
        nonlocal maximum_error
        recorder.record_physics_step(model, data, start)
        error = np.abs(recorder.physics_forces[-1].sum(0) - data.qfrc_constraint[:3]).max()
        maximum_error = max(maximum_error, float(error))

    with env.observe_physics_steps(observe):
        for step in range(20):
            env.step(np.array([0.0]))
            recorder.finish_control_step(step + 1)
    arrays = recorder.episode_arrays()
    assert maximum_error < 1e-10  # Double-precision contact-frame rotation and summation roundoff.
    assert arrays["physics_ground_reaction_forces"].shape == (100, 2, 3)
    assert arrays["ground_reaction_forces"].shape == (20, 2, 3)
    assert arrays["physics_ground_reaction_forces"][:, :, 2].max() > 0
    np.testing.assert_allclose(arrays["physics_step_end_times"] - arrays["physics_step_start_times"], 0.002)
    np.testing.assert_array_equal(arrays["physics_control_steps"], np.repeat(np.arange(20), 5))
    np.testing.assert_array_equal(arrays["ground_reaction_forces"], arrays["physics_ground_reaction_forces"][4::5])


@pytest.mark.parametrize("substeps", [1, 5, 10])
def test_observation_does_not_change_dynamics_or_control_frequency(substeps):
    plain = native_environment(substeps=substeps)
    observed = native_environment(substeps=substeps)
    plain.reset()
    observed.reset()
    recorder = KineticsRecorder(observed.model)
    for i in range(10):
        action = np.array([np.sin(i)])
        plain.step(action)
        with observed.observe_physics_steps(recorder.record_physics_step):
            observed.step(action)
        recorder.finish_control_step(i + 1)
        for field in ("qpos", "qvel", "act", "ctrl", "sensordata"):
            np.testing.assert_array_equal(getattr(plain.data, field), getattr(observed.data, field))
    assert len(recorder.physics_forces) == substeps * 10


def test_observer_restores_registration_after_exception():
    env = native_environment()
    env.reset()

    def fail(*_):
        raise RuntimeError("observer failure")

    with pytest.raises(RuntimeError, match="observer failure"), env.observe_physics_steps(fail):
        env.step(np.array([0.0]))
    with env.observe_physics_steps(lambda *_: None):
        with pytest.raises(RuntimeError, match="already registered"), env.observe_physics_steps(fail):
            pass
        env.step(np.array([0.0]))


def test_missing_foot_geometry_and_missing_physics_samples_fail_explicitly():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match=r"mass|foot bodies"):
        KineticsRecorder(model)
    recorder = KineticsRecorder(native_environment().model)
    with pytest.raises(RuntimeError, match="no physics samples"):
        recorder.finish_control_step(1)


@pytest.mark.parametrize("gravity", [np.zeros(3), np.full(3, np.nan)])
def test_invalid_gravity_is_rejected_before_recording(gravity):
    env = native_environment()
    env.model.opt.gravity[:] = gravity
    with pytest.raises(ValueError, match="gravity"):
        KineticsRecorder(env.model)


def test_shared_export_writes_control_and_native_kinetics(monkeypatch, tmp_path):
    env = native_environment()
    env.th = SimpleNamespace(
        new_traj_no=0,
        new_subtraj_step_no_init=0,
        len_trajectory=lambda _: 5,
        traj=SimpleNamespace(data=SimpleNamespace(get=lambda *_: SimpleNamespace(qpos=env.model.qpos0.copy()))),
    )
    conf = SimpleNamespace(config=OmegaConf.create({"experiment": {"n_seeds": 1}}))
    monkeypatch.setattr(
        eval_utils, "make_policy_action_fn", lambda *_args, **_kwargs: lambda state, *_: (np.zeros(1), state)
    )
    path, arrays = eval_utils.run_with_trajectory_export(
        env,
        conf,
        SimpleNamespace(train_state=None),
        n_steps=4,
        use_mujoco=True,
        export_kinetics=True,
        trajectory_dir=str(tmp_path),
        motion_group="fixture",
        eval_seed=3,
    )
    with np.load(path, allow_pickle=False) as archive:
        assert archive["activation_sample_phase"].item() == "pre_step"
        assert archive["episode_0_physics_ground_reaction_forces"].shape == (20, 2, 3)
        assert archive["episode_0_ground_reaction_forces"].shape == (4, 2, 3)
        np.testing.assert_array_equal(archive["episode_0_kinetics_reference_steps"], [1, 2, 3, 4])
        np.testing.assert_array_equal(archive["episode_0_reference_steps"], [0, 1, 2, 3])
        np.testing.assert_array_equal(archive["episode_0_muscle_activations"][0], [0.0])
        assert archive["episode_0_muscle_commands"][0, 0] > 0
        np.testing.assert_allclose(archive["episode_0_kinetics_interval_start_times"], [0.008, 0.018, 0.028, 0.038])
    assert arrays["motion_group"] == "fixture"
    assert list(tmp_path.glob("*.mjb"))
