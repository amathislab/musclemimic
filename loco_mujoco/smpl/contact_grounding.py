"""Contact-aware root-height correction for retargeted motion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import ndimage, sparse
from scipy.optimize import lsq_linear

from loco_mujoco.core.utils.ground import FloorSurfaceDistanceEvaluator

FLOOR_NAME = "floor"
ROOT_Z_QPOS_OFFSET = 2


@dataclass(frozen=True)
class SupportDefinition:
    """Source frames and robot bodies for one potential support region."""

    source_frame_names: tuple[str, ...]
    robot_body_names: frozenset[str]
    enter_height_m: float
    exit_height_m: float


MYOFULLBODY_SUPPORT_DEFINITIONS: Mapping[str, SupportDefinition] = {
    "left_foot": SupportDefinition(
        source_frame_names=("calcn_l", "toes_l"),
        robot_body_names=frozenset(("calcn_l", "toes_l")),
        enter_height_m=0.06,
        exit_height_m=0.08,
    ),
    "right_foot": SupportDefinition(
        source_frame_names=("calcn_r", "toes_r"),
        robot_body_names=frozenset(("calcn_r", "toes_r")),
        enter_height_m=0.06,
        exit_height_m=0.08,
    ),
    "left_knee": SupportDefinition(
        source_frame_names=("tibia_l",),
        robot_body_names=frozenset(("tibia_l",)),
        enter_height_m=0.14,
        exit_height_m=0.18,
    ),
    "right_knee": SupportDefinition(
        source_frame_names=("tibia_r",),
        robot_body_names=frozenset(("tibia_r",)),
        enter_height_m=0.14,
        exit_height_m=0.18,
    ),
    "left_hand": SupportDefinition(
        source_frame_names=("lunate_l",),
        robot_body_names=frozenset(("radius_l",)),
        enter_height_m=0.10,
        exit_height_m=0.14,
    ),
    "right_hand": SupportDefinition(
        source_frame_names=("lunate_r",),
        robot_body_names=frozenset(("radius_r",)),
        enter_height_m=0.10,
        exit_height_m=0.14,
    ),
}


@dataclass(frozen=True)
class ContactDetectionConfig:
    """Thresholds for source support-contact inference."""

    baseline_quantile: float = 0.02
    enter_speed_m_s: float = 0.40
    exit_speed_m_s: float = 1.00
    minimum_contact_frames: int = 2


@dataclass(frozen=True)
class RootZCorrectionConfig:
    """Weights and constraints for smooth root-z correction."""

    contact_weight: float = 400.0
    zero_prior_weight: float = 0.02
    velocity_weight: float = 8.0
    acceleration_weight: float = 80.0
    penetration_tolerance_m: float = 1e-6
    solver_max_iterations: int = 2000
    solver_iteration_limit_optimality_tolerance: float = 1e-7


@dataclass(frozen=True)
class ContactGroundingResult:
    """Corrected trajectory and diagnostics for one grounding solve."""

    qpos: np.ndarray
    root_z_correction_m: np.ndarray
    global_surface_clearance_m: np.ndarray
    support_surface_clearance_m: Mapping[str, np.ndarray]
    source_contacts: Mapping[str, np.ndarray]
    source_contact_frame_clearance_m: np.ndarray
    source_contact_frame_fraction: float
    solver_cost: float
    solver_optimality: float
    solver_iterations: int


def _minimum_run_filter(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    """Remove true runs shorter than ``minimum_frames``."""
    if minimum_frames < 1:
        raise ValueError("minimum_contact_frames must be at least one.")
    output = np.asarray(mask, dtype=bool).copy()
    labels, num_labels = ndimage.label(output)
    for label_index in range(1, num_labels + 1):
        indices = labels == label_index
        if int(np.sum(indices)) < minimum_frames:
            output[indices] = False
    return output


def infer_support_contacts(
    target_positions_m: np.ndarray,
    frame_names: Sequence[str],
    frequency_hz: float,
    definitions: Mapping[str, SupportDefinition],
    config: ContactDetectionConfig,
) -> dict[str, np.ndarray]:
    """Infer source support contacts from target height and speed.

    Args:
        target_positions_m: Source target positions shaped
            ``[num_frames, num_source_frames, 3]`` [m].
        frame_names: Names for the source-frame axis.
        frequency_hz: Sampling frequency [Hz].
        definitions: Support-specific source frames and thresholds.
        config: Contact hysteresis and duration thresholds.

    Returns:
        Boolean contact masks shaped ``[num_frames]`` for each support.
    """
    positions = np.asarray(target_positions_m, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3 or len(positions) < 2:
        raise ValueError("target_positions_m must have shape [num_frames >= 2, num_source_frames, 3].")
    if positions.shape[1] != len(frame_names):
        raise ValueError("frame_names must match the target_positions_m source-frame axis.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("target_positions_m must contain only finite values.")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    if not 0.0 <= config.baseline_quantile <= 1.0:
        raise ValueError("baseline_quantile must be in [0, 1].")

    name_to_index = {name: index for index, name in enumerate(frame_names)}
    foot_names = tuple(
        frame_name
        for group_name in ("left_foot", "right_foot")
        for frame_name in definitions[group_name].source_frame_names
    )
    missing = sorted(
        {name for definition in definitions.values() for name in definition.source_frame_names} - name_to_index.keys()
    )
    if missing:
        raise ValueError(f"Missing source support frames: {', '.join(missing)}")

    foot_points_m = positions[:, [name_to_index[name] for name in foot_names]]
    ground_height_m = float(np.quantile(foot_points_m[..., 2], config.baseline_quantile))
    contacts = {}
    for group_name, definition in definitions.items():
        support_points_m = positions[:, [name_to_index[name] for name in definition.source_frame_names]]
        relative_height_m = np.min(support_points_m[..., 2], axis=1) - ground_height_m
        velocities_m_s = np.gradient(support_points_m, 1.0 / frequency_hz, axis=0)
        support_speed_m_s = np.min(np.linalg.norm(velocities_m_s, axis=2), axis=1)
        enter = (relative_height_m <= definition.enter_height_m) & (support_speed_m_s <= config.enter_speed_m_s)
        remain = (relative_height_m <= definition.exit_height_m) & (support_speed_m_s <= config.exit_speed_m_s)
        contact = ndimage.binary_propagation(enter, mask=remain)
        contact = ndimage.binary_closing(contact, structure=np.ones(3, dtype=bool))
        contacts[group_name] = _minimum_run_filter(contact, config.minimum_contact_frames)
    return contacts


def _floor_collision_geom_ids(
    model: mujoco.MjModel,
    body_names: frozenset[str],
    floor_name: str,
) -> tuple[int, ...]:
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_name)
    if floor_geom_id < 0:
        raise ValueError(f"No geom named '{floor_name}' found.")
    explicit_pairs = {
        tuple(sorted((int(geom1), int(geom2)))) for geom1, geom2 in zip(model.pair_geom1, model.pair_geom2, strict=True)
    }
    geom_ids = []
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        )
        if body_name not in body_names:
            continue
        pair = tuple(sorted((floor_geom_id, geom_id)))
        masks_enable_collision = bool(
            (model.geom_contype[floor_geom_id] & model.geom_conaffinity[geom_id])
            or (model.geom_contype[geom_id] & model.geom_conaffinity[floor_geom_id])
        )
        if masks_enable_collision or pair in explicit_pairs:
            geom_ids.append(geom_id)
    if not geom_ids:
        raise ValueError(f"No floor-collidable geoms found for bodies {sorted(body_names)}.")
    return tuple(geom_ids)


def measure_support_clearances(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    definitions: Mapping[str, SupportDefinition],
    *,
    floor_name: str = FLOOR_NAME,
) -> dict[str, np.ndarray]:
    """Measure support-specific collision-surface clearances [m]."""
    qpos_array = np.asarray(qpos, dtype=np.float64)
    if qpos_array.ndim != 2 or qpos_array.shape[1] != model.nq or len(qpos_array) == 0:
        raise ValueError(f"qpos must have shape [num_frames, {model.nq}].")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_name)
    support_geom_ids = {
        group_name: _floor_collision_geom_ids(
            model,
            definition.robot_body_names,
            floor_name,
        )
        for group_name, definition in definitions.items()
    }
    clearances_m = {group_name: np.empty(len(qpos_array), dtype=np.float64) for group_name in definitions}
    data = mujoco.MjData(model)
    for frame_index, qpos_frame in enumerate(qpos_array):
        data.qpos[:] = qpos_frame
        mujoco.mj_forward(model, data)
        for group_name, geom_ids in support_geom_ids.items():
            clearances_m[group_name][frame_index] = min(
                float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        floor_geom_id,
                        geom_id,
                        np.inf,
                        None,
                    )
                )
                for geom_id in geom_ids
            )
    return clearances_m


def solve_contact_aware_root_z_shift(
    global_clearance_m: np.ndarray,
    support_clearance_m: Mapping[str, np.ndarray],
    contacts: Mapping[str, np.ndarray],
    config: RootZCorrectionConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Solve a smooth per-frame root-z shift with no penetration."""
    global_clearance = np.asarray(global_clearance_m, dtype=np.float64)
    if global_clearance.ndim != 1 or len(global_clearance) < 2:
        raise ValueError("global_clearance_m must contain at least two frames.")
    if not np.all(np.isfinite(global_clearance)):
        raise ValueError("global_clearance_m must contain only finite values.")
    if set(support_clearance_m) != set(contacts):
        raise ValueError("support clearances and contact masks must have identical keys.")

    num_frames = len(global_clearance)
    contact_active = np.logical_or.reduce(tuple(np.asarray(mask, dtype=bool) for mask in contacts.values()))
    contact_anchor_m = np.zeros(num_frames, dtype=np.float64)
    for group_name in contacts:
        clearance = np.asarray(support_clearance_m[group_name], dtype=np.float64)
        contact = np.asarray(contacts[group_name], dtype=bool)
        if clearance.shape != (num_frames,) or contact.shape != (num_frames,):
            raise ValueError("Every support clearance and contact mask must match the trajectory length.")
        if not np.all(np.isfinite(clearance)):
            raise ValueError("Support clearances must contain only finite values.")
    for frame_index in np.flatnonzero(contact_active):
        active_clearances = [
            float(support_clearance_m[group_name][frame_index])
            for group_name, contact in contacts.items()
            if contact[frame_index]
        ]
        contact_anchor_m[frame_index] = -min(active_clearances)

    matrices = []
    targets = []
    contact_indices = np.flatnonzero(contact_active)
    if len(contact_indices):
        contact_matrix = sparse.coo_matrix(
            (
                np.full(len(contact_indices), np.sqrt(config.contact_weight)),
                (np.arange(len(contact_indices)), contact_indices),
            ),
            shape=(len(contact_indices), num_frames),
        ).tocsr()
        matrices.append(contact_matrix)
        targets.append(np.sqrt(config.contact_weight) * contact_anchor_m[contact_indices])

    matrices.append(np.sqrt(config.zero_prior_weight) * sparse.eye(num_frames, format="csr"))
    targets.append(np.zeros(num_frames, dtype=np.float64))
    first_difference = sparse.diags(
        (-np.ones(num_frames - 1), np.ones(num_frames - 1)),
        (0, 1),
        shape=(num_frames - 1, num_frames),
        format="csr",
    )
    matrices.append(np.sqrt(config.velocity_weight) * first_difference)
    targets.append(np.zeros(num_frames - 1, dtype=np.float64))
    if num_frames >= 3:
        second_difference = sparse.diags(
            (
                np.ones(num_frames - 2),
                -2.0 * np.ones(num_frames - 2),
                np.ones(num_frames - 2),
            ),
            (0, 1, 2),
            shape=(num_frames - 2, num_frames),
            format="csr",
        )
        matrices.append(np.sqrt(config.acceleration_weight) * second_difference)
        targets.append(np.zeros(num_frames - 2, dtype=np.float64))

    design_matrix = sparse.vstack(matrices, format="csr")
    target = np.concatenate(targets)
    bounds = (
        -global_clearance - config.penetration_tolerance_m,
        np.full(num_frames, np.inf),
    )
    result = lsq_linear(
        design_matrix,
        target,
        bounds=bounds,
        max_iter=config.solver_max_iterations,
    )
    acceptable_iteration_limit = (
        result.status == 0
        and result.optimality <= config.solver_iteration_limit_optimality_tolerance
        and np.all(np.isfinite(result.x))
        and np.all(result.x >= bounds[0])
    )
    if not result.success and not acceptable_iteration_limit:
        raise RuntimeError(
            "Contact-aware root-z solve failed: "
            f"status={result.status}, iterations={result.nit}, "
            f"optimality={result.optimality:.6g}, cost={result.cost:.6g}. {result.message}"
        )
    return np.asarray(result.x), {
        "root_z_solver_cost": float(result.cost),
        "root_z_solver_optimality": float(result.optimality),
        "root_z_solver_iterations": int(result.nit),
        "root_z_solver_reached_iteration_limit": int(not result.success),
    }


def _root_z_qpos_id(model: mujoco.MjModel, root_joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
    if joint_id < 0:
        raise ValueError(f"No joint named '{root_joint_name}' found.")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Joint '{root_joint_name}' is not a free joint.")
    return int(model.jnt_qposadr[joint_id]) + ROOT_Z_QPOS_OFFSET


def contact_frame_clearance(
    support_clearance_m: Mapping[str, np.ndarray],
    contacts: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Return the highest expected-support clearance in each contact frame."""
    if set(support_clearance_m) != set(contacts) or not contacts:
        raise ValueError("support clearances and nonempty contact masks must have identical keys.")
    num_frames = len(next(iter(contacts.values())))
    result = np.full(num_frames, np.nan, dtype=np.float64)
    for frame_index in range(num_frames):
        active = [
            float(support_clearance_m[group_name][frame_index])
            for group_name, contact in contacts.items()
            if contact[frame_index]
        ]
        if active:
            result[frame_index] = max(active)
    return result


def ground_contact_aware_trajectory(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    root_joint_name: str,
    target_positions_m: np.ndarray,
    frame_names: Sequence[str],
    frequency_hz: float,
    *,
    definitions: Mapping[str, SupportDefinition] = MYOFULLBODY_SUPPORT_DEFINITIONS,
    excluded_body_subtree_names: Sequence[str] = (),
    contact_config: ContactDetectionConfig = ContactDetectionConfig(),
    correction_config: RootZCorrectionConfig = RootZCorrectionConfig(),
) -> ContactGroundingResult:
    """Apply source-contact-aware root-z correction to a trajectory."""
    contacts = infer_support_contacts(
        target_positions_m,
        frame_names,
        frequency_hz,
        definitions,
        contact_config,
    )
    evaluator = FloorSurfaceDistanceEvaluator(
        model,
        excluded_body_subtree_names=excluded_body_subtree_names,
    )
    global_clearance_m, _closest_geom_ids = evaluator.measure_trajectory(qpos)
    support_clearance_m = measure_support_clearances(model, qpos, definitions)
    correction_m, solver = solve_contact_aware_root_z_shift(
        global_clearance_m,
        support_clearance_m,
        contacts,
        correction_config,
    )
    corrected_qpos = np.asarray(qpos, dtype=np.float64).copy()
    corrected_qpos[:, _root_z_qpos_id(model, root_joint_name)] += correction_m
    corrected_global_clearance_m, _closest_geom_ids = evaluator.measure_trajectory(corrected_qpos)
    corrected_support_clearance_m = measure_support_clearances(
        model,
        corrected_qpos,
        definitions,
    )
    per_contact_frame_clearance_m = contact_frame_clearance(
        corrected_support_clearance_m,
        contacts,
    )
    contact_active = np.logical_or.reduce(tuple(contacts.values()))
    return ContactGroundingResult(
        qpos=corrected_qpos,
        root_z_correction_m=correction_m,
        global_surface_clearance_m=corrected_global_clearance_m,
        support_surface_clearance_m=corrected_support_clearance_m,
        source_contacts=contacts,
        source_contact_frame_clearance_m=per_contact_frame_clearance_m,
        source_contact_frame_fraction=float(np.mean(contact_active)),
        solver_cost=float(solver["root_z_solver_cost"]),
        solver_optimality=float(solver["root_z_solver_optimality"]),
        solver_iterations=int(solver["root_z_solver_iterations"]),
    )
