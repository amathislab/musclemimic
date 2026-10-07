# Physiological rollout export

`fullbody.eval --export_trajectory` exports joint states, actions and muscle
activation. Add `--export_kinetics` for actuator forces and ground reaction
forces (GRF). MM-EvalKit compares these signals with human EMG and force-plate
recordings.

Export requires `--use_mujoco`. Kinetics collection supports MyoFullBody foot
contacts with fixed plane or heightfield ground.

Run the following command from the MuscleMimic repository root, after configuring
the dataset caches required by the checkpoint. The selected motion group must
exist in the checkpoint's retargeting namespace.

```bash
uv run python -m fullbody.eval \
  --path /path/to/flashsac/checkpoint_123 \
  --use_mujoco --no_render \
  --export_trajectory --export_kinetics \
  --evaluate_all --motion_group KIT_KINESIS_TESTING_MOTIONS \
  --eval_seed 0 \
  --trajectory_dir outputs/physiology/flashsac
```

Run the same command with the PPO checkpoint and a separate output directory.
Use the same ordered motion group, retargeted references, initialization,
termination overrides, action mode, control frequency and evaluation seed.

Export defaults to deterministic actions. Add `--stochastic` for sampled
actions. `--eval_seed` controls export initialization and action sampling;
`--train_state_seed` selects weights from a checkpoint with multiple training
seeds. Fullbody `--metrics` defaults to stochastic actions; use
`--metrics_deterministic` for deterministic metrics.

`--evaluate_all` visits every reference from frame zero and ignores `--n_steps`.
With export, `--evaluate_all_passes 3` repeats the dataset three times. For a
single motion, use `--motion_path KIT/348/walking_run09_poses --start_from_beginning`
and `--n_steps` to limit the control steps. Episode metadata records termination
and completion.

Without export, `--evaluate_all` computes metrics for the whole dataset and
exits. It defaults to deterministic actions; add `--stochastic` for sampled
actions. This behavior applies to both `fullbody.eval` and `bimanual.eval`.
An early termination ends that trajectory's episode and evaluation continues
with the next trajectory.

The example writes
`myofullbody_episodes_mujoco_KIT_KINESIS_TESTING_MOTIONS.npz` and a matching `.mjb` compiled
model. Episode arrays use the prefixes `episode_0_*`, `episode_1_*`, and so on.

To record a playback video, run:

```bash
uv run python -m fullbody.eval \
  --path /path/to/flashsac/checkpoint_123 \
  --use_mujoco --no_render --record \
  --motion_path KIT/348/walking_run09_poses --start_from_beginning \
  --n_steps 200 --record_dir outputs/physiology/videos
```

Set `MUJOCO_GL=egl` before starting Python on a machine with EGL support.
`--no_render --record` records offscreen without opening a window. Run recording
and data export as separate commands.

State row `i` describes the observation used to compute action `i`:

```text
time t_i: qpos_i, qvel_i, activation_i, touch_observation_i
          -> policy(observation_i) -> action_i
          -> physics substeps over [t_i, t_(i+1)]
time t_(i+1): next observation and reward
```

`joint_positions`, `joint_velocities` and `muscle_activations` are sampled before
`env.step`. Their times are `episode_i_timesteps`, and their reference indices
are `episode_i_reference_steps`. `muscle_activations` contains internal actuator
activation states, not surface EMG voltages or policy actions. `muscle_names`
labels activation columns; `actuator_names` labels physical actuator
control columns. `muscle_commands` records the controls applied in that step.

Joint accelerations use `acceleration_sample_phase=post_step`, with end labels
in `episode_i_post_step_timesteps`. Accelerations and forces describe the solver
interval.

GRF sums ground forces on each foot's heel and toes in world coordinates.
Array side order is `right, left`; vector axes are world `x, y, z`.
For the usual 0.002 s physics step and five substeps, control samples are 100 Hz
and raw physics samples are 500 Hz.

For each episode, the kinetics arrays are:

```text
ground_reaction_forces          [T, 2, 3]  N, last physics sample per control step
generalized_actuator_forces     [T, J]     Nm for hinge joints, N for slide joints
force_sensor_values            [T, 4]     N, scalar touch-force channels
kinetics_reference_steps        [T]        end-labelled reference frame indices
kinetics_interval_start_times   [T]        s, last solver interval start
kinetics_interval_end_times     [T]        s, last solver interval end
physics_ground_reaction_forces  [P, 2, 3]  N, every native physics step
physics_step_start_times        [P]        s
physics_step_end_times          [P]        s
physics_control_steps           [P]        owning control row, starting at zero
```

`T` is the episode's control-step count and `P` is its physics-step count. With
five substeps, the first control row's force interval is [0.008, 0.010] s,
while its pre-step state time is 0 s. `kinetics_sample_phase=post_step` describes
the readout convention; `grf_sample_phase=solver_interval` and the interval
arrays specify its physical timing. `grf_control_reduction=last_physics_step`
declares the reduction. The archive does not filter or normalize raw forces.
Use the physics arrays for impact peaks, impulse and contact timing. Control
samples can miss short force peaks. Generalized actuator force describes the
model's actuators and does not include all contributions to an OpenSim net
inverse-dynamics joint moment.

[MM-EvalKit](https://github.com/amathislab/mm-evalkit) reads these archives.
Its README describes the supported comparison workflows and their reference
data requirements.
