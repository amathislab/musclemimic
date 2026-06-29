import dataclasses
import random

import numpy as np
from omegaconf import DictConfig, ListConfig

from loco_mujoco.datasets.humanoids.LAFAN1 import (
    LAFAN1_ALL_DATASETS,
    LAFAN1_DANCE_DATASETS,
    LAFAN1_LOCOMOTION_DATASETS,
    load_lafan1_trajectory,
)
from loco_mujoco.smpl.retargeting import (
    load_retargeted_amass_trajectory,
    retarget_smpl_to_bimanual_via_intermediate,
)
from loco_mujoco.trajectory import Trajectory, TrajectoryCacheType

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

        # Create and return the environment
        env = env_cls(
            init_state_type=init_state_type,
            terminal_state_type=terminal_state_type,
            goal_type=goal_type,
            goal_params=goal_params,
            **merged_kwargs,
        )

        all_trajs = []

        # Load the default trajectory if available
        # if default_dataset_conf is not None:
        #     if isinstance(default_dataset_conf, (dict, DictConfig)):
        #         default_dataset_conf = DefaultDatasetConf(**default_dataset_conf)
        #     all_trajs.append(cls.get_default_traj(env, default_dataset_conf))

        # Load the AMASS trajectory if available
        if amass_dataset_conf is not None:
            if isinstance(amass_dataset_conf, (dict, DictConfig)):
                # Filter out unsupported keys
                valid_keys = {f.name for f in dataclasses.fields(AMASSDatasetConf)}
                filtered_conf = {k: v for k, v in amass_dataset_conf.items() if k in valid_keys}
                amass_dataset_conf = AMASSDatasetConf(**filtered_conf)
            # Pass along visualization flag for optional logging
            all_trajs.append(cls.get_amass_traj(env, amass_dataset_conf, visualize_goal=visualize_goal))

        # Load converted C3D trajectories if available
        if c3d_dataset_conf is not None:
            if isinstance(c3d_dataset_conf, (dict, DictConfig)):
                valid_keys = {f.name for f in dataclasses.fields(C3DDatasetConf)}
                filtered_conf = {k: v for k, v in c3d_dataset_conf.items() if k in valid_keys}
                c3d_dataset_conf = C3DDatasetConf(**filtered_conf)
            all_trajs.append(cls.get_c3d_traj(env, c3d_dataset_conf))

        # Load the LAFAN1 trajectory if available
        if lafan1_dataset_conf is not None:
            if isinstance(lafan1_dataset_conf, (dict, DictConfig)):
                lafan1_dataset_conf = LAFAN1DatasetConf(**lafan1_dataset_conf)
            all_trajs.append(cls.get_lafan1_traj(env, lafan1_dataset_conf))

        # Load the custom trajectory if available
        if custom_dataset_conf is not None:
            if isinstance(custom_dataset_conf, (dict, DictConfig)):
                custom_dataset_conf = CustomDatasetConf(**custom_dataset_conf)
            all_trajs.append(cls.get_custom_dataset(env, custom_dataset_conf))

        # Only process trajectories if we have any to load
        if all_trajs:
            cache_type = cls._get_trajectory_cache_type(
                amass_dataset_conf,
                c3d_dataset_conf,
                lafan1_dataset_conf,
                custom_dataset_conf,
            )
            all_trajs = Trajectory.concatenate(all_trajs, backend=np)

            # add to the environment
            env.load_trajectory(
                traj=all_trajs,
                warn=False,
                cache_type=cache_type,
                site_names=getattr(env, "sites_for_mimic", None) if cache_type == TrajectoryCacheType.SPARSE else None,
            )

        return env

    @staticmethod
    def _get_trajectory_cache_type(*dataset_confs) -> TrajectoryCacheType:
        cache_types = [
            TrajectoryCacheType(conf.trajectory_cache_type)
            for conf in dataset_confs
            if conf is not None
        ]
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

        # Determine dataset paths
        dataset_paths = []
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
        dataset_paths = list(dict.fromkeys(dataset_paths))

        # Optionally cap the number of motions to load when datasets are very large.
        if amass_dataset_conf.max_motions is not None and isinstance(dataset_paths, list):
            if len(dataset_paths) > amass_dataset_conf.max_motions:
                before = len(dataset_paths)
                dataset_paths = random.sample(dataset_paths, amass_dataset_conf.max_motions)
                print(
                    f"[AMASS] INFO: Sampled {amass_dataset_conf.max_motions} trajectories out of {before} "
                    f"(max_motions={amass_dataset_conf.max_motions})."
                )

        env_name = env.__class__.__name__
        if visualize_goal:
            print(f"[Visualization] Building trajectories for env={env_name} with {len(dataset_paths)} paths.")

        # Load trajectories from AMASS datasets
        # Extract retargeting configs
        retargeting_method = amass_dataset_conf.retargeting_method
        gmr_config = amass_dataset_conf.gmr_config
        clear_cache = amass_dataset_conf.clear_cache

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
                clear_cache=clear_cache,
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
