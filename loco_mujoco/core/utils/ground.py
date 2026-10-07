"""MuJoCo collision-surface distance utilities for flat ground."""

from collections.abc import Iterable

import mujoco
import numpy as np

FREE_JOINT_Z_TRANSLATION_OFFSET = 2


class FloorSurfaceDistanceEvaluator:
    """Measure signed distances from moving collision geoms to a floor geom."""

    def __init__(
        self,
        model: mujoco.MjModel,
        floor_name: str = "floor",
        *,
        excluded_body_subtree_names: Iterable[str] = (),
    ) -> None:
        """Create an evaluator for one MuJoCo model.

        Args:
            model: MuJoCo model whose moving collision geoms are measured.
            floor_name: Name of the flat floor geom.
            excluded_body_subtree_names: Body names whose geoms and descendant
                body geoms do not participate in floor-distance measurements.
        """
        floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_name)
        if floor_geom_id < 0:
            raise ValueError(f"No geom named '{floor_name}' found.")

        excluded_root_ids = set()
        for body_name in excluded_body_subtree_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"No body named '{body_name}' found for floor-distance exclusion.")
            excluded_root_ids.add(body_id)

        excluded_body_ids = set(excluded_root_ids)
        for body_id in range(1, model.nbody):
            ancestor_id = body_id
            while ancestor_id != 0:
                if ancestor_id in excluded_root_ids:
                    excluded_body_ids.add(body_id)
                    break
                ancestor_id = int(model.body_parentid[ancestor_id])

        explicit_pairs = {
            tuple(sorted((int(geom1), int(geom2))))
            for geom1, geom2 in zip(model.pair_geom1, model.pair_geom2, strict=True)
        }
        collision_geom_ids = []
        for geom_id in range(model.ngeom):
            if geom_id == floor_geom_id:
                continue
            body_id = int(model.geom_bodyid[geom_id])
            if model.body_weldid[body_id] == 0:
                continue
            if body_id in excluded_body_ids:
                continue

            pair = tuple(sorted((floor_geom_id, geom_id)))
            masks_enable_collision = bool(
                (model.geom_contype[floor_geom_id] & model.geom_conaffinity[geom_id])
                or (model.geom_contype[geom_id] & model.geom_conaffinity[floor_geom_id])
            )
            if masks_enable_collision or pair in explicit_pairs:
                collision_geom_ids.append(geom_id)

        if not collision_geom_ids:
            raise ValueError(f"No moving collision geoms can collide with floor geom '{floor_name}'.")

        self._model = model
        self._floor_geom_id = floor_geom_id
        self._collision_geom_ids = tuple(collision_geom_ids)

    def measure(self, data: mujoco.MjData) -> tuple[float, int]:
        """Return the minimum signed surface distance in meters and geom ID.

        Positive distance indicates separation. Negative distance indicates
        penetration. Call ``mujoco.mj_forward`` after changing state first.
        """
        minimum_distance_m = np.inf
        closest_geom_id = -1
        for geom_id in self._collision_geom_ids:
            distance_m = float(
                mujoco.mj_geomDistance(
                    self._model,
                    data,
                    self._floor_geom_id,
                    geom_id,
                    np.inf,
                    None,
                )
            )
            if distance_m < minimum_distance_m:
                minimum_distance_m = distance_m
                closest_geom_id = geom_id

        if not np.isfinite(minimum_distance_m):
            raise ValueError("MuJoCo could not compute a finite distance to the floor geom.")
        return minimum_distance_m, closest_geom_id

    def measure_trajectory(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Measure qpos shaped ``[num_frames, model.nq]``.

        Returns signed distances in meters and closest geom IDs, each shaped
        ``[num_frames]``.
        """
        qpos_array = self._validated_qpos(qpos)
        data = mujoco.MjData(self._model)
        distances_m = np.empty(len(qpos_array), dtype=np.float64)
        closest_geom_ids = np.empty(len(qpos_array), dtype=np.int32)
        for frame, qpos_frame in enumerate(qpos_array):
            data.qpos[:] = qpos_frame
            mujoco.mj_forward(self._model, data)
            distances_m[frame], closest_geom_ids[frame] = self.measure(data)
        return distances_m, closest_geom_ids

    def align_trajectory(
        self,
        qpos: np.ndarray,
        root_free_joint_name: str,
        *,
        surface_distances_m: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Align to flat ground with one world-z translation in meters.

        The single translation makes the closest collision surface touch the
        floor and preserves all frame-to-frame root-height changes.
        """
        qpos_array = self._validated_qpos(qpos)
        distances_m = self._resolve_surface_distances(qpos_array, surface_distances_m)
        root_z_qpos_id = self._root_z_qpos_id(root_free_joint_name)
        world_z_translation_m = -float(np.min(distances_m))
        aligned_qpos = qpos_array.copy()
        aligned_qpos[:, root_z_qpos_id] += world_z_translation_m
        return aligned_qpos, world_z_translation_m

    def ensure_minimum_trajectory_clearance(
        self,
        qpos: np.ndarray,
        root_free_joint_name: str,
        minimum_clearance_m: float,
        *,
        surface_distances_m: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Lift a trajectory by one world-z translation when it penetrates.

        The method never lowers a trajectory. It preserves every frame-to-frame
        root displacement and is suitable as a guard after trajectory
        interpolation.

        Args:
            qpos: Joint positions shaped ``[num_frames, model.nq]``.
            root_free_joint_name: Name of the root free joint.
            minimum_clearance_m: Required minimum collision-surface clearance
                [m].
            surface_distances_m: Optional precomputed signed clearances [m].

        Returns:
            Corrected qpos and the nonnegative world-z translation [m].
        """
        if minimum_clearance_m < 0.0:
            raise ValueError("minimum_clearance_m must be nonnegative.")
        qpos_array = self._validated_qpos(qpos)
        distances_m = self._resolve_surface_distances(qpos_array, surface_distances_m)
        root_z_qpos_id = self._root_z_qpos_id(root_free_joint_name)
        world_z_translation_m = max(
            0.0,
            minimum_clearance_m - float(np.min(distances_m)),
        )
        aligned_qpos = qpos_array.copy()
        aligned_qpos[:, root_z_qpos_id] += world_z_translation_m
        return aligned_qpos, world_z_translation_m

    def _resolve_surface_distances(
        self,
        qpos: np.ndarray,
        surface_distances_m: np.ndarray | None,
    ) -> np.ndarray:
        if surface_distances_m is None:
            distances_m, _closest_geom_ids = self.measure_trajectory(qpos)
            return distances_m
        distances_m = np.asarray(surface_distances_m, dtype=np.float64)
        if distances_m.shape != (len(qpos),):
            raise ValueError(f"surface_distances_m must have shape ({len(qpos)},), got {distances_m.shape}.")
        if not np.all(np.isfinite(distances_m)):
            raise ValueError("surface_distances_m must contain only finite values.")
        return distances_m

    def _root_z_qpos_id(self, root_free_joint_name: str) -> int:
        root_joint_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_JOINT,
            root_free_joint_name,
        )
        if root_joint_id < 0:
            raise ValueError(f"No joint named '{root_free_joint_name}' found.")
        if self._model.jnt_type[root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint '{root_free_joint_name}' is not a free joint.")
        return int(self._model.jnt_qposadr[root_joint_id]) + FREE_JOINT_Z_TRANSLATION_OFFSET

    def _validated_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos_array = np.asarray(qpos)
        if qpos_array.ndim != 2 or qpos_array.shape[1] != self._model.nq:
            raise ValueError(
                f"qpos must have shape [num_frames, model.nq], got {qpos_array.shape} for model.nq={self._model.nq}."
            )
        if len(qpos_array) == 0:
            raise ValueError("qpos must contain at least one frame.")
        return qpos_array
