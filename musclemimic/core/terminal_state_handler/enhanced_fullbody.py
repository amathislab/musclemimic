from types import ModuleType
from typing import Any

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from loco_mujoco.core.terminal_state_handler.height import HeightBasedTerminalStateHandler
from loco_mujoco.core.utils.backend import assert_backend_is_supported
from loco_mujoco.core.utils.math import calculate_relative_site_quantities
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__, identifier="[FullBodyTerminal]")


class EnhancedFullBodyTerminalStateHandler(HeightBasedTerminalStateHandler):
    """
    Terminal state handler for full-body imitation tasks.

    Episodes terminate when either:
    - the root height leaves the healthy range inherited from
      `HeightBasedTerminalStateHandler`, or
    - tracked ankle/root mimic sites drift too far from the reference trajectory.
    """

    def __init__(
        self,
        env: Any,
        root_height_healthy_range: list | None = None,
        ankle_deviation: float = 0.5,
        root_deviation: float = 0.3,
        ankle_sites: list | None = None,
        root_site: str = "pelvis_mimic",
        enable_site_check: bool = True,
        site_deviation_mode: str = "mean",
        **handler_config: dict[str, Any],
    ):
        """
        Initialize enhanced full-body terminal state handler.

        Args:
            env: The environment instance
            root_height_healthy_range: [min, max] healthy height range for root
            ankle_deviation: Max/mean ankle position deviation from reference (meters)
            root_deviation: Max/mean root/pelvis position deviation from reference (meters)
            ankle_sites: List of ankle site names to track
            root_site: Name of root/pelvis site to track
            enable_site_check: Whether to check site deviations
            site_deviation_mode: "max" (any site) or "mean" (average across sites)
            **handler_config: Additional configuration passed to parent
        """
        # Pass height range to parent if provided
        if root_height_healthy_range is not None:
            handler_config["root_height_healthy_range"] = root_height_healthy_range

        super().__init__(env, **handler_config)

        # Site-based termination parameters
        self.ankle_deviation = ankle_deviation
        self.root_deviation = root_deviation
        self.enable_site_check = enable_site_check

        self.ankle_sites = ankle_sites if ankle_sites is not None else []
        self.root_site = root_site

        # Validate and store deviation mode
        valid_modes = ["max", "mean"]
        if site_deviation_mode not in valid_modes:
            raise ValueError(f"site_deviation_mode must be one of {valid_modes}")
        self.site_deviation_mode = site_deviation_mode

        # Track model site IDs and thresholds for ankle/root checks
        self.tracked_model_site_ids: list[int] = []
        self.tracked_thresholds: np.ndarray | list[float] = []

        if self.enable_site_check and hasattr(env, "_goal") and hasattr(env._goal, "_site_mapper"):
            # Get model site IDs for tracked sites
            model_site_ids: list[int] = []
            thresholds = []

            # Add ankle sites
            for ankle_site in self.ankle_sites:
                site_id = env.model.site(ankle_site).id
                model_site_ids.append(site_id)
                thresholds.append(self.ankle_deviation)

            # Add root site
            if self.root_site:
                site_id = env.model.site(self.root_site).id
                model_site_ids.append(site_id)
                thresholds.append(self.root_deviation)

            if model_site_ids:
                self.tracked_model_site_ids = list(model_site_ids)
                self.tracked_thresholds = np.array(thresholds)
        logger.info("EnhancedFullBodyTerminalStateHandler initialized")
        logger.info("  Height range: %s", self.root_height_range)
        logger.info("  Ankle sites: %s", self.ankle_sites)
        logger.info("  Root site: %s", self.root_site)
        logger.info("  Max ankle deviation: %sm", self.ankle_deviation)
        logger.info("  Max root deviation: %sm", self.root_deviation)
        logger.info("  Site deviation mode: %s", self.site_deviation_mode)
        logger.debug("  Tracked model IDs: %s", self.tracked_model_site_ids)
        logger.debug("  Tracked thresholds: %s", self.tracked_thresholds)

    def _check_site_deviations(self, env: Any, state: Any, carry: Any, backend: ModuleType) -> bool | jnp.ndarray:
        """
        Compare tracked ankle/root mimic sites against the current reference frame.

        Args:
            env: The environment instance
            state: Current simulation state
            carry: Additional carry information
            backend: Backend module (numpy or jax.numpy)

        Returns:
            Boolean indicating if termination should occur
        """
        # Exit early when trajectory or site data is unavailable.
        if not self.enable_site_check:
            return backend.array(False)

        if not hasattr(env, "th") or env.th is None:
            return backend.array(False)

        if not (hasattr(env, "_goal") and hasattr(env._goal, "_rel_site_ids")):
            return backend.array(False)

        # No current site data means there is nothing to compare.
        if not hasattr(state, "site_xpos"):
            return backend.array(False)

        # Reference data for the current trajectory step.
        ref_data = env.th.get_current_traj_data(carry, backend)

        # No reference site data means there is nothing to compare against.
        if not hasattr(ref_data, "site_xpos"):
            return backend.array(False)

        # Goal stores the model site ids used for mimic tracking.
        site_mapping = env._goal._rel_site_ids

        # Extract the tracked sites from the full simulation state.
        current_mapped_sites = state.site_xpos[site_mapping]

        # Trajectory caches may store only a reduced site set, so resolve model
        # site ids into trajectory indices when needed.
        if env._goal._site_mapper.requires_mapping:
            traj_indices = env._goal._site_mapper.model_ids_to_traj_indices(site_mapping)
            ref_mapped_sites = ref_data.site_xpos[traj_indices]
        else:
            ref_mapped_sites = ref_data.site_xpos

        # Compute per-site Euclidean errors.
        site_deviations = backend.linalg.norm(current_mapped_sites - ref_mapped_sites, axis=-1)

        # Nothing to evaluate if no tracked sites were configured.
        if len(self.tracked_model_site_ids) == 0:
            return backend.array(False)

        # Resolve tracked model ids against the current goal mapping so the
        # selection remains valid even if the mapped site ordering changes.
        site_mapping = np.asarray(env._goal._rel_site_ids)
        id_to_env_index = {int(mid): idx for idx, mid in enumerate(site_mapping)}
        missing = [mid for mid in self.tracked_model_site_ids if mid not in id_to_env_index]
        if missing:
            raise ValueError(f"Tracked site IDs {missing} are not present in env._goal._rel_site_ids")

        tracked_env_indices = backend.asarray([id_to_env_index[mid] for mid in self.tracked_model_site_ids], dtype=int)

        # Select only the tracked ankle/root sites.
        tracked_deviations = backend.take(site_deviations, tracked_env_indices, axis=0)
        tracked_thresholds = backend.array(self.tracked_thresholds)

        # Apply either a per-site max threshold or a mean threshold.
        violations = backend.greater(tracked_deviations, tracked_thresholds)

        is_max_mode = self.site_deviation_mode == "max"

        max_violation = backend.any(violations)

        mean_deviation = backend.mean(tracked_deviations)
        mean_threshold = backend.mean(tracked_thresholds)
        mean_violation = backend.greater(mean_deviation, mean_threshold)

        # Select the configured reduction without Python branching.
        return backend.where(is_max_mode, max_violation, mean_violation)

    def is_absorbing(self, env: Any, obs: np.ndarray, info: dict[str, Any], data: MjData, carry: Any) -> bool | Any:
        """
        Check if the current state is terminal (CPU Mujoco version).

        Args:
            env: The environment instance
            obs: Observations
            info: Info dictionary
            data: Mujoco data structure
            carry: Additional carry information

        Returns:
            Tuple of (is_absorbing, carry)
        """
        return self._is_absorbing_compat(env, obs, info, data, carry, np)

    def mjx_is_absorbing(self, env: Any, obs: jnp.ndarray, info: dict[str, Any], data: Data, carry: Any) -> bool | Any:
        """
        Check if the current state is terminal (Mjx version).

        Args:
            env: The environment instance
            obs: Observations
            info: Info dictionary
            data: Mjx data structure
            carry: Additional carry information

        Returns:
            Tuple of (is_absorbing, carry)
        """
        return self._is_absorbing_compat(env, obs, info, data, carry, jnp)

    def _is_absorbing_compat(
        self,
        env: Any,
        obs: np.ndarray | jnp.ndarray,
        info: dict[str, Any],
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> bool | Any:
        """
        Shared termination path for numpy and JAX backends.

        Args:
            env: The environment instance
            obs: Observations
            info: Info dictionary
            data: Simulation data
            carry: Additional carry information
            backend: Backend module (numpy or jax.numpy)

        Returns:
            Tuple of (should_terminate, carry)
        """
        # Combine inherited height checks with reference-tracking checks.
        height_violation, carry = super()._is_absorbing_compat(env, obs, info, data, carry, backend)
        site_violation = self._check_site_deviations(env, data, carry, backend)
        should_terminate = height_violation | site_violation

        return should_terminate, carry


# Register handler on import for availability via string
EnhancedFullBodyTerminalStateHandler.register()


class MeanSiteDeviationTerminalStateHandler(TerminalStateHandler):
    """
    Terminal state handler using mean absolute mimic-site deviation.

    This operates in world coordinates, so it is useful when the character is
    expected to stay close to the reference trajectory in absolute space.
    """

    def __init__(
        self,
        env: Any,
        mean_site_deviation_threshold: float = 0.3,
        enable_site_check: bool = True,
        exclude_sites: list[str] | None = None,
        **handler_config: dict[str, Any],
    ):
        """
        Args:
            env: The environment instance
            mean_site_deviation_threshold: Max mean deviation across all mimic sites (meters)
            enable_site_check: Whether to check site deviations
            exclude_sites: Optional list of mimic site names to exclude from mean calculation
            **handler_config: Additional configuration passed to parent
        """
        super().__init__(env, **handler_config)

        self.mean_site_deviation_threshold = mean_site_deviation_threshold
        self.enable_site_check = enable_site_check
        self._include_indices = np.array([], dtype=int)
        self._has_exclusions = False
        self._n_included: int | None = None
        self._root_qpos_ids_xy: np.ndarray | None = None

        exclude_sites_list = exclude_sites or []
        sites_for_mimic = []
        if hasattr(env, "_goal") and hasattr(env._goal, "_info_props"):
            sites_for_mimic = env._goal._info_props.get("sites_for_mimic", [])
        if not sites_for_mimic:
            sites_for_mimic = getattr(env, "sites_for_mimic", [])
        if sites_for_mimic:
            n_sites = len(sites_for_mimic)
            include_mask = np.ones(n_sites, dtype=bool)
            for exc_name in exclude_sites_list:
                if exc_name in sites_for_mimic:
                    include_mask[sites_for_mimic.index(exc_name)] = False
                else:
                    logger.warning("exclude site '%s' not in sites_for_mimic", exc_name)
            self._include_indices = np.nonzero(include_mask)[0]
            self._n_included = int(self._include_indices.size)
            self._has_exclusions = self._n_included < n_sites
            if exclude_sites_list:
                logger.debug("Excluding sites: %s", exclude_sites_list)
            if self._n_included == 0:
                logger.warning("All sites excluded - handler will never terminate")
        elif exclude_sites_list:
            logger.warning("exclude_sites provided but sites_for_mimic unavailable; ignoring")

        root_joint_name = self._info_props.get("root_free_joint_xml_name")
        model = getattr(env, "_model", None)
        if root_joint_name and model is not None:
            try:
                root_qpos_ids = np.array(mj_jntname2qposid(root_joint_name, model), dtype=int)
                if root_qpos_ids.size >= 2:
                    self._root_qpos_ids_xy = root_qpos_ids[:2]
            except Exception as exc:
                logger.warning("failed to resolve root qpos indices for '%s': %s", root_joint_name, exc)

        logger.debug("MeanSiteDeviationTerminalStateHandler initialized")
        logger.debug("  Mean site deviation threshold (default): %sm", self.mean_site_deviation_threshold)
        logger.debug("  Uses: Absolute world positions (no height termination)")

    def reset(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[MjData | Data, Any]:
        assert_backend_is_supported(backend)
        return data, carry

    def _check_mean_site_deviation(self, env: Any, state: Any, carry: Any, backend: ModuleType) -> bool | jnp.ndarray:
        """Check if mean absolute position deviation exceeds threshold."""
        if not self.enable_site_check:
            return backend.array(False)

        if not hasattr(env, "th") or env.th is None:
            return backend.array(False)

        if not (hasattr(env, "_goal") and hasattr(env._goal, "_rel_site_ids")):
            return backend.array(False)

        if not hasattr(state, "site_xpos"):
            return backend.array(False)

        ref_data = env.th.get_current_traj_data(carry, backend)

        if not hasattr(ref_data, "site_xpos"):
            return backend.array(False)

        site_mapping = env._goal._rel_site_ids
        current_mapped_sites = state.site_xpos[site_mapping]

        if env._goal._site_mapper.requires_mapping:
            traj_indices = env._goal._site_mapper.model_ids_to_traj_indices(site_mapping)
            ref_mapped_sites = ref_data.site_xpos[traj_indices]
        else:
            ref_mapped_sites = ref_data.site_xpos

        # Align reference sites to the per-episode root XY origin used when
        # the simulator is initialized from trajectory data.
        if self._root_qpos_ids_xy is not None:
            init_ref = env.th.get_init_traj_data(carry, backend)
            if hasattr(init_ref, "qpos"):
                root_xy = init_ref.qpos[self._root_qpos_ids_xy]
                offset = backend.concatenate([root_xy, backend.zeros(1, dtype=root_xy.dtype)])
                ref_mapped_sites = ref_mapped_sites - offset

        site_deviations = backend.linalg.norm(current_mapped_sites - ref_mapped_sites, axis=-1)
        if self._has_exclusions:
            if self._n_included == 0:
                return backend.array(False)
            site_deviations = backend.take(site_deviations, self._include_indices, axis=0)

        mean_deviation = backend.mean(site_deviations)

        threshold = carry.termination_threshold
        return backend.greater(mean_deviation, threshold)

    def is_absorbing(self, env: Any, obs: np.ndarray, info: dict[str, Any], data: MjData, carry: Any) -> bool | Any:
        """Check if the current state is terminal (CPU Mujoco version)."""
        return self._is_absorbing_compat(env, obs, info, data, carry, np)

    def mjx_is_absorbing(self, env: Any, obs: jnp.ndarray, info: dict[str, Any], data: Data, carry: Any) -> bool | Any:
        """Check if the current state is terminal (Mjx version)."""
        return self._is_absorbing_compat(env, obs, info, data, carry, jnp)

    def _is_absorbing_compat(
        self,
        env: Any,
        obs: np.ndarray | jnp.ndarray,
        info: dict[str, Any],
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> bool | Any:
        """Check termination based on mean absolute site deviation."""
        site_violation = self._check_mean_site_deviation(env, data, carry, backend)

        return site_violation, carry


MeanSiteDeviationTerminalStateHandler.register()


class MeanRelativeSiteDeviationTerminalStateHandler(TerminalStateHandler):
    """
    Terminal state handler using mean relative mimic-site deviation.

    This compares site positions in the root-relative frame, making it
    insensitive to global translation.
    """

    def __init__(
        self,
        env: Any,
        mean_site_deviation_threshold: float = 0.3,
        enable_site_check: bool = True,
        exclude_sites: list[str] | None = None,
        **handler_config: dict[str, Any],
    ):
        """
        Args:
            env: The environment instance
            mean_site_deviation_threshold: Max mean deviation across all mimic sites (meters)
            enable_site_check: Whether to check site deviations
            exclude_sites: Optional list of mimic site names to exclude from mean calculation
            **handler_config: Additional configuration passed to parent
        """
        super().__init__(env, **handler_config)

        self.mean_site_deviation_threshold = mean_site_deviation_threshold
        self.enable_site_check = enable_site_check
        self._include_indices = np.array([], dtype=int)
        self._has_exclusions = False
        self._n_included: int | None = None

        exclude_sites_list = exclude_sites or []
        sites_for_mimic = []
        if hasattr(env, "_goal") and hasattr(env._goal, "_info_props"):
            sites_for_mimic = env._goal._info_props.get("sites_for_mimic", [])
        if not sites_for_mimic:
            sites_for_mimic = getattr(env, "sites_for_mimic", [])
        if sites_for_mimic:
            n_sites_relative = len(sites_for_mimic) - 1
            include_mask = np.ones(n_sites_relative, dtype=bool)
            for exc_name in exclude_sites_list:
                if exc_name == sites_for_mimic[0]:
                    logger.warning("'%s' is already excluded in relative handler (main site)", exc_name)
                elif exc_name in sites_for_mimic:
                    include_mask[sites_for_mimic.index(exc_name) - 1] = False
                else:
                    logger.warning("exclude site '%s' not in sites_for_mimic", exc_name)
            self._include_indices = np.nonzero(include_mask)[0]
            self._n_included = int(self._include_indices.size)
            self._has_exclusions = self._n_included < n_sites_relative
            if exclude_sites_list:
                logger.debug("Excluding sites: %s", exclude_sites_list)
            if self._n_included == 0:
                logger.warning("All sites excluded - handler will never terminate")
        elif exclude_sites_list:
            logger.warning("exclude_sites provided but sites_for_mimic unavailable; ignoring")

        logger.debug("MeanRelativeSiteDeviationTerminalStateHandler initialized")
        logger.debug("  Mean site deviation threshold (default): %sm", self.mean_site_deviation_threshold)
        logger.debug("  Uses: Relative positions w.r.t root (no height termination)")

    def reset(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[MjData | Data, Any]:
        assert_backend_is_supported(backend)
        return data, carry

    def _check_mean_site_deviation(self, env: Any, state: Any, carry: Any, backend: ModuleType) -> bool | jnp.ndarray:
        """Check if mean relative position deviation exceeds threshold."""
        if not self.enable_site_check:
            return backend.array(False)

        if not hasattr(env, "th") or env.th is None:
            return backend.array(False)

        if not (hasattr(env, "_goal") and hasattr(env._goal, "_rel_site_ids")):
            return backend.array(False)

        if not hasattr(state, "site_xpos"):
            return backend.array(False)

        ref_data = env.th.get_current_traj_data(carry, backend)

        if not hasattr(ref_data, "site_xpos"):
            return backend.array(False)

        rel_body_ids = env._goal._site_bodyid[env._goal._rel_site_ids]

        # Relative site positions for the current simulation state.
        site_rpos, _, _ = calculate_relative_site_quantities(
            state, env._goal._rel_site_ids, rel_body_ids, env.model.body_rootid, backend
        )

        # Relative site positions for the reference trajectory frame.
        if env._goal._site_mapper.requires_mapping:
            traj_indices = env._goal._site_mapper.model_ids_to_traj_indices(env._goal._rel_site_ids)
            site_rpos_traj, _, _ = calculate_relative_site_quantities(
                ref_data,
                env._goal._rel_site_ids,
                rel_body_ids,
                env.model.body_rootid,
                backend,
                trajectory_site_indices=traj_indices,
            )
        else:
            site_rpos_traj, _, _ = calculate_relative_site_quantities(
                ref_data, env._goal._rel_site_ids, rel_body_ids, env.model.body_rootid, backend
            )

        # Compute per-site Euclidean errors in the root-relative frame.
        site_deviations = backend.linalg.norm(site_rpos - site_rpos_traj, axis=-1)
        if self._has_exclusions:
            if self._n_included == 0:
                return backend.array(False)
            site_deviations = backend.take(site_deviations, self._include_indices, axis=0)

        mean_deviation = backend.mean(site_deviations)

        threshold = carry.termination_threshold
        return backend.greater(mean_deviation, threshold)

    def is_absorbing(self, env: Any, obs: np.ndarray, info: dict[str, Any], data: MjData, carry: Any) -> bool | Any:
        """Check if the current state is terminal (CPU Mujoco version)."""
        return self._is_absorbing_compat(env, obs, info, data, carry, np)

    def mjx_is_absorbing(self, env: Any, obs: jnp.ndarray, info: dict[str, Any], data: Data, carry: Any) -> bool | Any:
        """Check if the current state is terminal (Mjx version)."""
        return self._is_absorbing_compat(env, obs, info, data, carry, jnp)

    def _is_absorbing_compat(
        self,
        env: Any,
        obs: np.ndarray | jnp.ndarray,
        info: dict[str, Any],
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> bool | Any:
        """Check termination based on mean relative site deviation."""
        site_violation = self._check_mean_site_deviation(env, data, carry, backend)

        return site_violation, carry


MeanRelativeSiteDeviationTerminalStateHandler.register()


class MeanRelativeSiteDeviationWithRootTerminalStateHandler(TerminalStateHandler):
    """
    Terminal state handler combining relative pose tracking with root drift checks.

    It enforces three conditions:
    - mean root-relative site deviation,
    - root position deviation,
    - optional root orientation deviation.
    """

    def __init__(
        self,
        env: Any,
        mean_site_deviation_threshold: float = 0.3,
        root_deviation_threshold: float = 0.3,
        root_orientation_threshold: float | None = None,
        root_site: str = "pelvis_mimic",
        enable_site_check: bool = True,
        **handler_config: dict[str, Any],
    ):
        """
        Args:
            env: The environment instance
            mean_site_deviation_threshold: Max mean deviation across all mimic sites (meters)
            root_deviation_threshold: Max root position deviation from reference (meters)
            root_orientation_threshold: Max root orientation deviation from reference (radians).
                None disables the check. Typical values: pi/3 (~60 deg) to pi/2 (~90 deg).
            root_site: Name of root/pelvis site to track
            enable_site_check: Whether to check site deviations
            **handler_config: Additional configuration passed to parent
        """
        super().__init__(env, **handler_config)

        self.mean_site_deviation_threshold = mean_site_deviation_threshold
        self.root_deviation_threshold = root_deviation_threshold
        self.root_orientation_threshold = root_orientation_threshold
        self.root_site = root_site
        self.enable_site_check = enable_site_check

        # Pre-compute root-site lookup data for the trajectory caches.
        self.root_site_id = None
        self.root_traj_index = None
        self._root_qpos_ids_xy = None
        self._root_qpos_ids_quat = None

        if self.enable_site_check and hasattr(env, "_goal") and hasattr(env._goal, "_site_mapper"):
            self.root_site_id = env.model.site(self.root_site).id
            if env._goal._site_mapper.requires_mapping:
                self.root_traj_index = env._goal._site_mapper.model_ids_to_traj_indices([self.root_site_id])[0]

        info_props = env._get_all_info_properties() if hasattr(env, "_get_all_info_properties") else {}
        root_joint_name = info_props.get("root_free_joint_xml_name") or getattr(env, "root_free_joint_xml_name", None)
        model = getattr(env, "_model", None)
        if root_joint_name and model is not None:
            try:
                root_qpos_ids = np.array(mj_jntname2qposid(root_joint_name, model), dtype=int)
                if root_qpos_ids.size >= 2:
                    self._root_qpos_ids_xy = root_qpos_ids[:2]
                if root_qpos_ids.size >= 7:
                    self._root_qpos_ids_quat = root_qpos_ids[3:7]
            except Exception as exc:
                logger.warning("failed to resolve root qpos indices for '%s': %s", root_joint_name, exc)

        logger.info("MeanRelativeSiteDeviationWithRootTerminalStateHandler initialized")
        logger.info("  Mean site deviation threshold (default): %sm", self.mean_site_deviation_threshold)
        logger.info("  Root deviation threshold: %sm", self.root_deviation_threshold)
        logger.info(
            "  Root orientation threshold: %s%s",
            self.root_orientation_threshold,
            "" if self.root_orientation_threshold is None else " rad",
        )
        logger.info("  Root site: %s (id=%s)", self.root_site, self.root_site_id)
        logger.info("  Uses: Relative positions w.r.t root + root absolute position check")

    def reset(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[MjData | Data, Any]:
        assert_backend_is_supported(backend)
        return data, carry

    def _check_mean_site_deviation(self, env: Any, state: Any, carry: Any, backend: ModuleType) -> bool | jnp.ndarray:
        """Check if mean relative position deviation exceeds threshold."""
        if not self.enable_site_check:
            return backend.array(False)

        if not hasattr(env, "th") or env.th is None:
            return backend.array(False)

        if not (hasattr(env, "_goal") and hasattr(env._goal, "_rel_site_ids")):
            return backend.array(False)

        if not hasattr(state, "site_xpos"):
            return backend.array(False)

        ref_data = env.th.get_current_traj_data(carry, backend)

        if not hasattr(ref_data, "site_xpos"):
            return backend.array(False)

        rel_body_ids = env._goal._site_bodyid[env._goal._rel_site_ids]

        # Relative site positions for the current simulation state.
        site_rpos, _, _ = calculate_relative_site_quantities(
            state, env._goal._rel_site_ids, rel_body_ids, env.model.body_rootid, backend
        )

        # Relative site positions for the reference trajectory frame.
        if env._goal._site_mapper.requires_mapping:
            traj_indices = env._goal._site_mapper.model_ids_to_traj_indices(env._goal._rel_site_ids)
            site_rpos_traj, _, _ = calculate_relative_site_quantities(
                ref_data,
                env._goal._rel_site_ids,
                rel_body_ids,
                env.model.body_rootid,
                backend,
                trajectory_site_indices=traj_indices,
            )
        else:
            site_rpos_traj, _, _ = calculate_relative_site_quantities(
                ref_data, env._goal._rel_site_ids, rel_body_ids, env.model.body_rootid, backend
            )

        # Compute per-site Euclidean errors in the root-relative frame.
        site_deviations = backend.linalg.norm(site_rpos - site_rpos_traj, axis=-1)

        mean_deviation = backend.mean(site_deviations)

        threshold = carry.termination_threshold
        return backend.greater(mean_deviation, threshold)

    def _check_root_deviation(self, env: Any, state: Any, carry: Any, backend: ModuleType) -> bool | jnp.ndarray:
        """Check if root site absolute position deviation exceeds threshold."""
        if not self.enable_site_check:
            return backend.array(False)

        if self.root_site_id is None:
            return backend.array(False)

        if not hasattr(env, "th") or env.th is None:
            return backend.array(False)

        if not hasattr(state, "site_xpos"):
            return backend.array(False)

        ref_data = env.th.get_current_traj_data(carry, backend)

        if not hasattr(ref_data, "site_xpos"):
            return backend.array(False)

        # Current root position in world coordinates.
        current_root_pos = state.site_xpos[self.root_site_id]

        # Reference root position, mapped into the trajectory cache layout if needed.
        if self.root_traj_index is not None:
            ref_root_pos = ref_data.site_xpos[self.root_traj_index]
        else:
            ref_root_pos = ref_data.site_xpos[self.root_site_id]

        # Align the reference root to the per-episode XY origin used when the
        # simulator is initialized from trajectory data.
        if (
            not getattr(env, "preserve_trajectory_root_xy", False)
            and self._root_qpos_ids_xy is not None
            and hasattr(env.th, "get_init_traj_data")
        ):
            init_ref = env.th.get_init_traj_data(carry, backend)
            if hasattr(init_ref, "qpos"):
                root_xy = init_ref.qpos[self._root_qpos_ids_xy]
                offset = backend.concatenate([root_xy, backend.zeros(1, dtype=root_xy.dtype)])
                ref_root_pos = ref_root_pos - offset

        # Compute Euclidean root-position error.
        root_deviation = backend.linalg.norm(current_root_pos - ref_root_pos)

        return backend.greater(root_deviation, self.root_deviation_threshold)

    def _check_root_orientation(self, env: Any, state: Any, carry: Any, backend: ModuleType) -> bool | jnp.ndarray:
        """Check if root orientation deviation from reference exceeds threshold.

        Uses geodesic distance on SO(3): `2 * arccos(|q1 · q2|)`.
        """
        if self.root_orientation_threshold is None:
            return backend.array(False)

        if self._root_qpos_ids_quat is None:
            return backend.array(False)

        if not hasattr(env, "th") or env.th is None:
            return backend.array(False)

        ref_data = env.th.get_current_traj_data(carry, backend)
        if not hasattr(ref_data, "qpos"):
            return backend.array(False)

        cur_quat = state.qpos[self._root_qpos_ids_quat]
        ref_quat = ref_data.qpos[self._root_qpos_ids_quat]

        # Normalize before computing the geodesic distance.
        cur_quat = cur_quat / backend.linalg.norm(cur_quat)
        ref_quat = ref_quat / backend.linalg.norm(ref_quat)

        # Geodesic distance on unit quaternions.
        dot = backend.abs(backend.dot(cur_quat, ref_quat))
        angular_distance = 2 * backend.arccos(backend.clip(dot, 0.0, 1.0))

        return backend.greater(angular_distance, self.root_orientation_threshold)

    def is_absorbing(self, env: Any, obs: np.ndarray, info: dict[str, Any], data: MjData, carry: Any) -> bool | Any:
        """Check if the current state is terminal (CPU Mujoco version)."""
        return self._is_absorbing_compat(env, obs, info, data, carry, np)

    def mjx_is_absorbing(self, env: Any, obs: jnp.ndarray, info: dict[str, Any], data: Data, carry: Any) -> bool | Any:
        """Check if the current state is terminal (Mjx version)."""
        return self._is_absorbing_compat(env, obs, info, data, carry, jnp)

    def _is_absorbing_compat(
        self,
        env: Any,
        obs: np.ndarray | jnp.ndarray,
        info: dict[str, Any],
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> bool | Any:
        """Check termination based on mean relative site deviation, root position, and root orientation."""
        site_violation = self._check_mean_site_deviation(env, data, carry, backend)
        root_violation = self._check_root_deviation(env, data, carry, backend)
        rot_violation = self._check_root_orientation(env, data, carry, backend)

        return site_violation | root_violation | rot_violation, carry


MeanRelativeSiteDeviationWithRootTerminalStateHandler.register()
