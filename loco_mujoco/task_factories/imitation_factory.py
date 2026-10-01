import dataclasses
import random
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, ListConfig

from loco_mujoco.datasets.humanoids.LAFAN1 import (
    LAFAN1_ALL_DATASETS,
    LAFAN1_DANCE_DATASETS,
    LAFAN1_LOCOMOTION_DATASETS,
    load_lafan1_trajectory,
)
from loco_mujoco.smpl.retargeting import (
    get_converted_amass_dataset_path,
    load_retargeted_amass_trajectory,
    retarget_smpl_to_bimanual_via_intermediate,
    retargeting_cache_dir,
)
from loco_mujoco.trajectory import Trajectory, TrajectoryCacheType
from loco_mujoco.trajectory.materialized_cache import MaterializedTrajectoryCache

from .base import TaskFactory
from .dataset_confs import (
    AMASSDatasetConf,
    C3DDatasetConf,
    CustomDatasetConf,
    LAFAN1DatasetConf,
    expand_amass_dataset_group_spec,
    get_amass_dataset_groups,
)


class ImitationFactory(TaskFactory):
    """
    A factory class for creating imitation learning environments with arbitrary trajectories.

    Methods:
        make(env_name: str, task: str, dataset_type: str, debug: bool = False, **kwargs) -> LocoEnv:
            Creates an environment, loads a trajectory based on the task and dataset type, and returns the environment.

        get_traj_path(env_cls, dataset_type: str, task: str, debug: bool) -> str:
            Determines the path to the trajectory file based on the dataset type, task, and debug mode.
    """

    @classmethod
    def make(
        cls,
        env_name: str,
        # default_dataset_conf: DefaultDatasetConf | dict | DictConfig = None,  # Not used
        amass_dataset_conf: AMASSDatasetConf | dict | DictConfig = None,
        c3d_dataset_conf: C3DDatasetConf | dict | DictConfig = None,
        lafan1_dataset_conf: LAFAN1DatasetConf | dict | DictConfig = None,
        custom_dataset_conf: CustomDatasetConf | dict | DictConfig = None,
        trajectory_cache_root: str | None = None,
        trajectory_cache_key: str | None = None,
        trajectory_handler=None,
        terminal_state_type: str = "RootPoseTrajTerminalStateHandler",
        init_state_type: str = "TrajInitialStateHandler",
        **kwargs,
    ):
        """
        Creates and returns an imitation learning environment given different configurations.

        Args:
            env_name (str): The name of the registered environment to create.
            default_dataset_conf (DefaultDatasetConf, optional): The configuration for the default trajectory.
            amass_dataset_conf (AMASSDatasetConf, optional): The configuration for the AMASS trajectory.
            c3d_dataset_conf (C3DDatasetConf, optional): The configuration for converted C3D trajectories.
            lafan1_dataset_conf (LAFAN1DatasetConf, optional): The configuration for the LAFAN1 trajectory.
            custom_dataset_conf (CustomDatasetConf, optional): The configuration for a custom trajectory.
            terminal_state_type (str, optional): The terminal state handler to use.
                Defaults to "RootPoseTrajTerminalStateHandler".
            init_state_type (str, optional): The initial state handler to use. Defaults to "TrajInitialStateHandler".
            **kwargs: Additional keyword arguments to pass to the environment constructor.

        Returns:
            LocoEnv: An instance of the requested imitation learning environment with the trajectory preloaded.

        Raises:
            ValueError: If the `dataset_type` is unknown.
        """

        from musclemimic.environments.base import LocoEnv

        if env_name not in LocoEnv.registered_envs:
            raise KeyError(f"Environment '{env_name}' is not a registered MuscleMimic environment.")

        if isinstance(amass_dataset_conf, (dict, DictConfig)):
            valid_keys = {f.name for f in dataclasses.fields(AMASSDatasetConf)}
            filtered_conf = {k: v for k, v in amass_dataset_conf.items() if k in valid_keys}
            amass_dataset_conf = AMASSDatasetConf(**filtered_conf)
        if isinstance(c3d_dataset_conf, (dict, DictConfig)):
            valid_keys = {f.name for f in dataclasses.fields(C3DDatasetConf)}
            filtered_conf = {k: v for k, v in c3d_dataset_conf.items() if k in valid_keys}
            c3d_dataset_conf = C3DDatasetConf(**filtered_conf)
        if isinstance(lafan1_dataset_conf, (dict, DictConfig)):
            lafan1_dataset_conf = LAFAN1DatasetConf(**lafan1_dataset_conf)
        if isinstance(custom_dataset_conf, (dict, DictConfig)):
            custom_dataset_conf = CustomDatasetConf(**custom_dataset_conf)

        # Get environment class
        env_cls = LocoEnv.registered_envs[env_name]

        # Auto-select appropriate terminal state handler for bimanual environments
        if "Bimanual" in env_name and terminal_state_type == "RootPoseTrajTerminalStateHandler":
            terminal_state_type = "BimanualTerminalStateHandler"

        # Extract goal-related parameters from kwargs to avoid passing them to environment
        visualize_goal = kwargs.pop("visualize_goal", False)
        goal_params = kwargs.pop("goal_params", {})
        goal_type = kwargs.pop("goal_type", "GoalTrajMimic")  # Default goal for imitation

        if visualize_goal:
            goal_params["visualize_goal"] = visualize_goal

        # Extract and unpack env_params if provided
        env_params = kwargs.pop("env_params", {})
        # Merge env_params into kwargs, with kwargs taking precedence
        merged_kwargs = {**env_params, **kwargs}

        paired_artifacts = None
        if amass_dataset_conf is not None and amass_dataset_conf.load_paired_terrain:
            if "terrain_type" in merged_kwargs or "terrain_params" in merged_kwargs:
                raise ValueError("Paired AMASS terrain cannot be combined with explicit terrain_type/terrain_params.")
            paired_artifacts = cls.get_paired_amass_terrain(env_name, amass_dataset_conf)
            merged_kwargs["terrain_type"] = "BoxTerrain"
            merged_kwargs["terrain_params"] = paired_artifacts["sidecar"].target_terrain.to_env_params()

        # Create and return the environment
        env = env_cls(
            init_state_type=init_state_type,
            terminal_state_type=terminal_state_type,
            goal_type=goal_type,
            goal_params=goal_params,
            **merged_kwargs,
        )
        if paired_artifacts is not None:
            env.paired_motion_path = paired_artifacts["motion_path"]
            env.paired_terrain_path = paired_artifacts["terrain_path"]
            env.paired_terrain_sidecar = paired_artifacts["sidecar"]

        cache_type = cls._get_trajectory_cache_type(
            amass_dataset_conf,
            c3d_dataset_conf,
            lafan1_dataset_conf,
            custom_dataset_conf,
        )
        site_names = (
            getattr(env, "sites_for_mimic", None)
            if cache_type == TrajectoryCacheType.SPARSE
            else None
        )

        if trajectory_handler is not None:
            env.install_shared_trajectory_handler(trajectory_handler)
            return env

        def load_source_trajectory():
            all_trajs = []
            if amass_dataset_conf is not None:
                all_trajs.append(
                    cls.get_amass_traj(env, amass_dataset_conf, visualize_goal=visualize_goal)
                )
            if c3d_dataset_conf is not None:
                all_trajs.append(cls.get_c3d_traj(env, c3d_dataset_conf))
            if lafan1_dataset_conf is not None:
                all_trajs.append(cls.get_lafan1_traj(env, lafan1_dataset_conf))
            if custom_dataset_conf is not None:
                all_trajs.append(cls.get_custom_dataset(env, custom_dataset_conf))
            return Trajectory.concatenate(all_trajs, backend=np) if all_trajs else None

        cache_root = str(trajectory_cache_root or "").strip()
        cache_key = str(trajectory_cache_key or "").strip()
        if bool(cache_root) != bool(cache_key):
            raise ValueError("trajectory_cache_root and trajectory_cache_key must be configured together")

        persistent_cache = None
        if cache_root:
            th_params = env._th_params or {}
            persistent_cache = MaterializedTrajectoryCache(
                cache_root,
                cache_key,
                model=env._model,
                control_dt=env.dt,
                cache_type=cache_type,
                site_names=site_names,
                clip_joint_ranges=bool(th_params.get("clip_trajectory_to_joint_ranges", False)),
            )

        if persistent_cache is None:
            trajectory = load_source_trajectory()
            if trajectory is not None:
                env.load_trajectory(
                    traj=trajectory,
                    warn=False,
                    cache_type=cache_type,
                    site_names=site_names,
                )
            return env

        with persistent_cache.writer_lock():
            trajectory = persistent_cache.load()
            if trajectory is not None:
                print(f"[TrajectoryCache] INFO: Loaded {persistent_cache.path}")
                env.load_trajectory(
                    materialized_traj=trajectory,
                    warn=False,
                    cache_type=cache_type,
                    site_names=site_names,
                )
                return env

            trajectory = load_source_trajectory()
            if trajectory is not None:
                env.load_trajectory(
                    traj=trajectory,
                    warn=False,
                    cache_type=cache_type,
                    site_names=site_names,
                )
                persistent_cache.publish(env.th.traj)
                print(f"[TrajectoryCache] INFO: Published {persistent_cache.path}")

        return env

    @staticmethod
    def get_amass_dataset_paths(amass_dataset_conf: AMASSDatasetConf) -> list[str]:
        """Resolve an AMASS config to a stable, de-duplicated motion list."""
        dataset_paths: list[str] = []
        if amass_dataset_conf.dataset_group is not None:
            groups = get_amass_dataset_groups()
            for group_name in expand_amass_dataset_group_spec(amass_dataset_conf.dataset_group):
                if group_name not in groups:
                    raise ValueError(f"Unknown dataset group: {group_name}")
                dataset_paths.extend(groups[group_name])
        if amass_dataset_conf.rel_dataset_path is not None:
            dataset_paths.extend(
                amass_dataset_conf.rel_dataset_path
                if isinstance(amass_dataset_conf.rel_dataset_path, (ListConfig, list))
                else [amass_dataset_conf.rel_dataset_path]
            )
        return list(dict.fromkeys(str(path) for path in dataset_paths))

    @classmethod
    def get_paired_amass_terrain(cls, env_name: str, amass_dataset_conf: AMASSDatasetConf) -> dict:
        """Resolve and validate the trajectory and terrain sidecar for one AMASS motion."""
        try:
            from terra import TerrainSidecar
        except ImportError as exc:
            raise ImportError(
                "load_paired_terrain requires the standalone terra-retargeting package"
            ) from exc

        dataset_paths = cls.get_amass_dataset_paths(amass_dataset_conf)
        if len(dataset_paths) != 1:
            raise ValueError(
                "load_paired_terrain requires exactly one AMASS motion, "
                f"but the configuration resolved to {len(dataset_paths)}."
            )

        cache_root = (
            get_converted_amass_dataset_path()
            if amass_dataset_conf.cache_root is None
            else amass_dataset_conf.cache_root
        )
        cache_env_name = env_name.replace("Mjx", "")
        cache_dir = retargeting_cache_dir(
            cache_root,
            cache_env_name,
            amass_dataset_conf.retargeting_method,
            amass_dataset_conf.output_cache_subdir,
        )
        motion_path = cache_dir / f"{dataset_paths[0]}.npz"
        terrain_path = motion_path.with_name(f"{motion_path.stem}_terrain.json")
        if not motion_path.is_file():
            raise FileNotFoundError(f"Paired retargeted motion not found: {motion_path}")
        if not terrain_path.is_file():
            raise FileNotFoundError(f"Paired terrain sidecar not found: {terrain_path}")

        sidecar = TerrainSidecar.load(terrain_path)
        if amass_dataset_conf.require_nonflat_terrain and not sidecar.target_terrain.boxes:
            raise ValueError(f"Expected a non-flat paired terrain, but {terrain_path} contains no boxes.")
        return {
            "motion_path": Path(motion_path),
            "terrain_path": Path(terrain_path),
            "sidecar": sidecar,
        }

    @staticmethod
    def _get_trajectory_cache_type(*dataset_confs) -> TrajectoryCacheType:
        cache_types = [TrajectoryCacheType(conf.trajectory_cache_type) for conf in dataset_confs if conf is not None]
        if not cache_types:
            return TrajectoryCacheType.FULL
        if len(set(cache_types)) != 1:
            raise ValueError(f"Mixed trajectory_cache_type values are not supported: {cache_types}")
        return cache_types[0]

    @classmethod
    def get_amass_traj(cls, env, amass_dataset_conf: AMASSDatasetConf, visualize_goal: bool = False) -> Trajectory:
        """
        Determines the path to the trajectory file based on the dataset type, task, and debug mode.

        Args:
            env: The environment, which provides dataset paths.
            amass_dataset_conf (AMASSDatasetConf): The configuration for the AMASS trajectory
            visualize_goal (bool): If True we are constructing a visualization / evaluation environment.

        Returns:
            Trajectory: The AMASS trajectories.

        Raises:
            ValueError: If the `dataset_group` is unknown.
        """
        # Accept both dataclass instances and raw dict/DictConfig inputs
        if isinstance(amass_dataset_conf, (dict, DictConfig)):
            amass_dataset_conf = AMASSDatasetConf(**amass_dataset_conf)

        dataset_paths = cls.get_amass_dataset_paths(amass_dataset_conf)

        # Optionally cap the number of motions to load when datasets are very large.
        if amass_dataset_conf.max_motions is not None and isinstance(dataset_paths, list):
            if len(dataset_paths) > amass_dataset_conf.max_motions:
                before = len(dataset_paths)
                dataset_paths = random.Random(amass_dataset_conf.max_motions_seed).sample(
                    dataset_paths,
                    amass_dataset_conf.max_motions,
                )
                print(
                    f"[AMASS] INFO: Sampled {amass_dataset_conf.max_motions} trajectories out of {before} "
                    f"(max_motions={amass_dataset_conf.max_motions}, "
                    f"seed={amass_dataset_conf.max_motions_seed})."
                )

        env_name = env.__class__.__name__
        if visualize_goal:
            print(f"[Visualization] Building trajectories for env={env_name} with {len(dataset_paths)} paths.")

        # Load trajectories from AMASS datasets
        # Extract retargeting configs
        retargeting_method = amass_dataset_conf.retargeting_method
        output_cache_subdir = amass_dataset_conf.output_cache_subdir
        gmr_config = amass_dataset_conf.gmr_config
        terra_config = amass_dataset_conf.terra_config
        clear_cache = amass_dataset_conf.clear_cache
        allow_cache_download = amass_dataset_conf.allow_cache_download
        cache_root = amass_dataset_conf.cache_root

        if "MyoBimanualArm" in env_name:
            method_name = retargeting_method.upper() if retargeting_method else "SMPL"
            print(
                f"[MuscleMimic] Detected MyoBimanualArm environment. "
                f"Using three-stage retargeting pipeline with {method_name} for Stage 1."
            )
            traj = retarget_smpl_to_bimanual_via_intermediate(
                dataset_paths,
                retargeting_method=retargeting_method,
                gmr_config=gmr_config,
                clear_cache=clear_cache,
            )
        else:
            traj = load_retargeted_amass_trajectory(
                env_name,
                dataset_paths,
                retargeting_method=retargeting_method,
                gmr_config=gmr_config,
                terra_config=terra_config,
                clear_cache=clear_cache,
                output_cache_subdir=output_cache_subdir,
                allow_cache_download=allow_cache_download,
                cache_root=cache_root,
            )

        return traj

    @staticmethod
    def get_c3d_traj(env, c3d_dataset_conf: C3DDatasetConf) -> Trajectory:
        """
        Load converted C3D-derived trajectories from the converted C3D cache.

        Args:
            env: The environment, used to select the saved model namespace.
            c3d_dataset_conf (C3DDatasetConf): Converted C3D trajectory configuration.

        Returns:
            Trajectory: The converted C3D trajectories.

        """
        from musclemimic.web_viewer.c3d_pipeline import (
            get_converted_c3d_dataset_path,
            normalize_c3d_dataset_name,
        )

        dataset_paths = (
            c3d_dataset_conf.rel_dataset_path
            if isinstance(c3d_dataset_conf.rel_dataset_path, (ListConfig, list))
            else [c3d_dataset_conf.rel_dataset_path]
        )
        dataset_paths = list(dict.fromkeys(dataset_paths))

        if c3d_dataset_conf.max_motions is not None and len(dataset_paths) > c3d_dataset_conf.max_motions:
            before = len(dataset_paths)
            dataset_paths = random.sample(dataset_paths, c3d_dataset_conf.max_motions)
            print(
                f"[C3D] INFO: Sampled {c3d_dataset_conf.max_motions} trajectories out of {before} "
                f"(max_motions={c3d_dataset_conf.max_motions})."
            )

        cache_env_name = env.__class__.__name__.replace("Mjx", "")
        converted_root = get_converted_c3d_dataset_path()
        method = c3d_dataset_conf.retargeting_method

        trajectories = []
        for rel_dataset_path in dataset_paths:
            normalized = normalize_c3d_dataset_name(rel_dataset_path)
            trajectory_path = converted_root / cache_env_name / method / normalized.with_suffix(".npz")
            if not trajectory_path.exists():
                raise FileNotFoundError(
                    f"Converted C3D trajectory not found: {trajectory_path}. "
                    "Create it with `python -m musclemimic.web_viewer.run --c3d-file ... "
                    "--c3d-dataset-name <name>`."
                )
            trajectories.append(Trajectory.load(trajectory_path, backend=np))

        if len(trajectories) == 1:
            traj = trajectories[0]
        else:
            traj = Trajectory.concatenate(trajectories, backend=np)

        return traj

    @staticmethod
    def get_lafan1_traj(env, lafan1_dataset_conf: LAFAN1DatasetConf) -> Trajectory:
        """
        Determines the path to the trajectory file based on the dataset type, task, and debug mode.

        Args:
            env: The environment, which provides dataset paths.
            lafan1_dataset_conf (LAFAN1DatasetConf): The configuration for the LAFAN1 trajectory.

        Returns:
            Trajectory: The LAFAN1 trajectories.

        Raises:
            ValueError: If the `dataset_group` is unknown.
        """
        # Determine dataset paths
        if lafan1_dataset_conf.dataset_group:
            if lafan1_dataset_conf.dataset_group == "LAFAN1_LOCOMOTION_DATASETS":
                dataset_paths = LAFAN1_LOCOMOTION_DATASETS
            elif lafan1_dataset_conf.dataset_group == "LAFAN1_DANCE_DATASETS":
                dataset_paths = LAFAN1_DANCE_DATASETS
            elif lafan1_dataset_conf.dataset_group == "LAFAN1_ALL_DATASETS":
                dataset_paths = LAFAN1_ALL_DATASETS
            else:
                raise ValueError(f"Unknown dataset group: {lafan1_dataset_conf.dataset_group}")
        else:
            dataset_paths = (
                lafan1_dataset_conf.dataset_name
                if isinstance(lafan1_dataset_conf.dataset_name, (ListConfig, list))
                else [lafan1_dataset_conf.dataset_name]
            )

        # Load LAFAN1 Trajectory
        traj = load_lafan1_trajectory(env.__class__.__name__, dataset_paths)

        return traj

    @staticmethod
    def get_custom_dataset(env, custom_dataset_conf: CustomDatasetConf) -> Trajectory:
        """
        Loads the custom trajectory based on the dataset type, task, and debug mode.

        Args:
            env: The environment, which provides dataset paths.
            custom_dataset_conf (CustomDatasetConf): The configuration for the custom trajectory.

        Returns:
            Trajectory: The custom trajectories.

        """
        from loco_mujoco.smpl.retargeting import extend_motion

        traj = custom_dataset_conf.traj
        # Retargeted custom motions often start with qpos/qvel plus partial site data.
        # Extend them to full body/site kinematics before handing them to the trajectory handler.
        if not traj.data.is_complete:
            env_name = env.__class__.__name__
            env_params = {}
            traj = extend_motion(env_name, env_params, traj)

        return traj
