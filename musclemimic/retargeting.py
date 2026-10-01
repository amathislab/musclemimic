"""Public integration API for external motion-retargeting packages.

This module deliberately exposes only stable, method-neutral primitives.  External
packages such as TERRA can depend on MuscleMimic without importing private implementation
modules throughout their codebase.
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from loco_mujoco.core.mujoco_base import Mujoco
from loco_mujoco.core.terrain import BoxSpec, BoxTerrain, TerrainSpec
from loco_mujoco.core.terrain.boxes import SEAT_GEOM_PREFIX
from loco_mujoco.smpl import SMPLH_BONE_ORDER_NAMES
from loco_mujoco.smpl.retargeting import (
    OPTIMIZED_SHAPE_FILE_NAME,
    extend_motion,
    fit_gmr_motion,
    fit_smpl_motion,
    fit_smpl_shape,
    get_converted_amass_dataset_path,
    get_smpl_model_path,
    load_robot_conf_file,
    retargeting_cache_dir,
)
from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryCacheType,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
    materialize_trajectory,
)
from musclemimic.web_viewer.c3d_pipeline import retarget_c3d_to_trajectory


def joint_couplers(model: mujoco.MjModel) -> list[tuple[int, int, np.ndarray]]:
    """Return active nonconstant polynomial joint equalities from a MuJoCo model."""
    couplers = []
    for index in range(model.neq):
        if model.eq_type[index] != mujoco.mjtEq.mjEQ_JOINT or not model.eq_active0[index]:
            continue
        dependent = int(model.eq_obj1id[index])
        independent = int(model.eq_obj2id[index])
        if independent < 0:
            continue
        couplers.append(
            (
                int(model.jnt_qposadr[dependent]),
                int(model.jnt_qposadr[independent]),
                np.array(model.eq_data[index][:5], dtype=float),
            )
        )
    return couplers


def scene_geom_ids(model: mujoco.MjModel, floor_name: str = "floor") -> list[int]:
    """Return the floor and static box-terrain geometry IDs."""
    ids = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == floor_name or name.startswith(BoxTerrain.GEOM_PREFIX):
            ids.append(geom_id)
    return ids


def max_penetration_with_geoms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: Sequence[int],
) -> float:
    """Return the deepest signed contact distance against selected environment geoms."""
    selected = {int(geom_id) for geom_id in geom_ids}
    penetration = 0.0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if (int(contact.geom1) in selected) != (int(contact.geom2) in selected):
            penetration = min(penetration, float(contact.dist))
    return penetration


__all__ = [
    "OPTIMIZED_SHAPE_FILE_NAME",
    "SEAT_GEOM_PREFIX",
    "SMPLH_BONE_ORDER_NAMES",
    "BoxSpec",
    "Mujoco",
    "TerrainSpec",
    "Trajectory",
    "TrajectoryCacheType",
    "TrajectoryData",
    "TrajectoryInfo",
    "TrajectoryModel",
    "extend_motion",
    "fit_gmr_motion",
    "fit_smpl_motion",
    "fit_smpl_shape",
    "get_converted_amass_dataset_path",
    "get_smpl_model_path",
    "joint_couplers",
    "load_robot_conf_file",
    "materialize_trajectory",
    "max_penetration_with_geoms",
    "retarget_c3d_to_trajectory",
    "retargeting_cache_dir",
    "scene_geom_ids",
]
