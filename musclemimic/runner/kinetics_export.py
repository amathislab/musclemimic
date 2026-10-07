"""Read native MuJoCo kinetics during policy export, retaining solver timing."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

FOOT_SENSOR_NAMES = ("r_foot", "r_toes", "l_foot", "l_toes")
FOOT_BODY_NAMES = (("calcn_r", "toes_r"), ("calcn_l", "toes_l"))
GRF_SIDES = ("right", "left")
GROUND_TYPES = (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)


class KineticsRecorder:
    """Collect foot GRF [N] for a fixed native model, with right/left XYZ columns.

    Register ``record_physics_step`` with ``env.observe_physics_steps`` only
    during policy steps. The recorder performs read-only queries of solved
    contact forces. It supports MyoFullBody foot bodies and scalar touch sensors
    on static world-attached plane or heightfield ground geometry.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        if not isinstance(model, mujoco.MjModel):
            raise ValueError("Kinetics export requires a native MuJoCo CPU model.")
        self.model = model
        self.mass = float(model.body_mass.sum())
        self.gravity = model.opt.gravity.copy()
        if (
            self.mass <= 0
            or not np.isfinite(self.mass)
            or not np.isfinite(self.gravity).all()
            or np.linalg.norm(self.gravity) == 0
        ):
            raise ValueError("Kinetics export requires positive model mass and nonzero gravity.")
        self.foot_bodies = []
        for names in FOOT_BODY_NAMES:
            ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names}
            if -1 in ids:
                raise ValueError(f"Kinetics export requires foot bodies {FOOT_BODY_NAMES}.")
            self.foot_bodies.append(ids)
        self.ground_geoms = set(
            np.flatnonzero(np.isin(model.geom_type, GROUND_TYPES) & (model.geom_bodyid == 0)).tolist()
        )
        if not self.ground_geoms:
            raise ValueError("Kinetics export requires world-attached plane or heightfield ground.")
        sensor_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) for name in FOOT_SENSOR_NAMES]
        if any(
            i < 0 or model.sensor_type[i] != mujoco.mjtSensor.mjSENS_TOUCH or model.sensor_dim[i] != 1
            for i in sensor_ids
        ):
            raise ValueError(f"Kinetics export requires scalar touch sensors {FOOT_SENSOR_NAMES}.")
        self.sensor_addresses = model.sensor_adr[sensor_ids]
        joint_ids = np.flatnonzero(np.isin(model.jnt_type, (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)))
        self.force_addresses = model.jnt_dofadr[joint_ids]
        self.joint_names = [model.joint(i).name for i in joint_ids]
        self.joint_units = ["Nm" if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE else "N" for i in joint_ids]
        self.reset_episode()

    def reset_episode(self) -> None:
        """Clear samples between episodes while retaining validated model metadata."""
        self.physics_forces: list[np.ndarray] = []
        self.physics_starts: list[float] = []
        self.physics_ends: list[float] = []
        self.physics_control_steps: list[int] = []
        self.control_forces: list[np.ndarray] = []
        self.control_starts: list[float] = []
        self.control_ends: list[float] = []
        self.generalized: list[np.ndarray] = []
        self.sensor_forces: list[np.ndarray] = []
        self.reference_steps: list[int] = []

    def record_physics_step(self, model: mujoco.MjModel, data: mujoco.MjData, start_time: float) -> None:
        """Read one solved step on [start_time, data.time] [s], without advancing it."""
        if model is not self.model:
            raise ValueError("Kinetics export cannot change the native model during a rollout.")
        force = np.zeros((len(GRF_SIDES), 3), dtype=float)
        wrench = np.zeros(6, dtype=float)
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if (first in self.ground_geoms) == (second in self.ground_geoms):
                continue
            ground_first = first in self.ground_geoms
            body = int(model.geom_bodyid[second if ground_first else first])
            side = next((i for i, ids in enumerate(self.foot_bodies) if body in ids), None)
            if side is None:
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            # Contact-frame rows are world axes. MuJoCo reports force on geom2.
            force[side] += (1 if ground_first else -1) * (contact.frame.reshape(3, 3).T @ wrench[:3])
        if not np.isfinite(force).all():
            raise FloatingPointError("Non-finite ground reaction force during export.")
        self.physics_forces.append(force)
        self.physics_starts.append(start_time)
        self.physics_ends.append(float(data.time))
        self.physics_control_steps.append(len(self.control_forces))
        # Snapshot before environment post-step hooks can recompute derived values.
        self.last_generalized = data.qfrc_actuator[self.force_addresses].copy()
        self.last_sensors = data.sensordata[self.sensor_addresses].copy()

    def finish_control_step(self, reference_step: int) -> None:
        """Retain the last solver sample and its end-labelled reference frame."""
        if not self.physics_control_steps or self.physics_control_steps[-1] != len(self.control_forces):
            raise RuntimeError("Kinetics export received no physics samples for the control step.")
        if self.mass != float(self.model.body_mass.sum()) or not np.array_equal(self.gravity, self.model.opt.gravity):
            raise ValueError("Kinetics export requires fixed model mass and gravity across episodes.")
        self.control_forces.append(self.physics_forces[-1])
        self.control_starts.append(self.physics_starts[-1])
        self.control_ends.append(self.physics_ends[-1])
        self.generalized.append(self.last_generalized)
        self.sensor_forces.append(self.last_sensors)
        self.reference_steps.append(reference_step)

    def metadata(self) -> dict[str, Any]:
        """Return units, coordinate frame, solver settings and sample-phase metadata."""
        return {
            "kinetics_sample_phase": "post_step",
            "kinetics_time_convention": "last_solver_interval_end",
            "generalized_actuator_force_joint_names": np.asarray(self.joint_names, dtype=str),
            "generalized_actuator_force_units": np.asarray(self.joint_units, dtype=str),
            "force_sensor_names": np.asarray(FOOT_SENSOR_NAMES),
            "force_sensor_units": np.asarray(["N"] * len(FOOT_SENSOR_NAMES)),
            "model_mass_kg": self.mass,
            "gravity_m_s2": self.gravity,
            "physics_dt": float(self.model.opt.timestep),
            "grf_coordinate_frame": "world",
            "grf_units": "N",
            "grf_sides": np.asarray(GRF_SIDES),
            "grf_sample_phase": "solver_interval",
            "grf_control_reduction": "last_physics_step",
            "grf_body_names": np.asarray(FOOT_BODY_NAMES),
            "grf_ground_geom_ids": np.asarray(sorted(self.ground_geoms)),
            "mujoco_integrator": int(self.model.opt.integrator),
            "mujoco_solver": int(self.model.opt.solver),
            "mujoco_solver_iterations": int(self.model.opt.iterations),
            "mujoco_solver_tolerance": float(self.model.opt.tolerance),
        }

    def episode_arrays(self) -> dict[str, np.ndarray]:
        """Return control [T,...] and physics [P,...] signals, with raw GRF [N]."""
        return {
            "generalized_actuator_forces": np.asarray(self.generalized),
            "force_sensor_values": np.asarray(self.sensor_forces),
            "kinetics_reference_steps": np.asarray(self.reference_steps, dtype=np.int64),
            "ground_reaction_forces": np.asarray(self.control_forces),
            "kinetics_interval_start_times": np.asarray(self.control_starts),
            "kinetics_interval_end_times": np.asarray(self.control_ends),
            "physics_ground_reaction_forces": np.asarray(self.physics_forces),
            "physics_step_start_times": np.asarray(self.physics_starts),
            "physics_step_end_times": np.asarray(self.physics_ends),
            "physics_control_steps": np.asarray(self.physics_control_steps, dtype=np.int64),
        }
