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
from loco_mujoco.smpl.motion_spec import AMASSMotionSpec, as_amass_motion_spec
from loco_mujoco.smpl.retargeting import (
    get_converted_amass_dataset_path,
    get_gmr_cache_identity,
    get_gmr_motion_cache_path,
    gmr_quality_rejection_path,
    load_retargeted_amass_trajectory_set,
    retarget_smpl_to_bimanual_via_intermediate_set,
)
from loco_mujoco.trajectory import LoadedTrajectorySet, Trajectory, TrajectoryCacheType, materialize_trajectory
from loco_mujoco.trajectory.materialized_cache import (
    MATERIALIZED_MOTION_NAMES_METADATA_KEY,
    load_or_build_materialized_trajectory,
    materialized_trajectory_fingerprint,
)

from .base import TaskFactory
from .dataset_confs import (
    AMASSDatasetConf,
    C3DDatasetConf,
    CustomDatasetConf,
    LAFAN1DatasetConf,
    expand_amass_dataset_group_spec,
    get_amass_dataset_groups,
    get_amass_motion_spec_groups,
    resolve_training_gmr_cache_namespace,
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

        loaded_sets = []

        # Load the default trajectory if available
        # if default_dataset_conf is not None:
        #     if isinstance(default_dataset_conf, (dict, DictConfig)):
        #         default_dataset_conf = DefaultDatasetConf(**default_dataset_conf)
        #     all_trajs.append(cls.get_default_traj(env, default_dataset_conf))

        # Load the AMASS trajectory if available
        if amass_dataset_conf is not None:
            if isinstance(amass_dataset_conf, dict | DictConfig):
                # Filter out unsupported keys
                valid_keys = {f.name for f in dataclasses.fields(AMASSDatasetConf)}
                filtered_conf = {k: v for k, v in amass_dataset_conf.items() if k in valid_keys}
                amass_dataset_conf = AMASSDatasetConf(**filtered_conf)
            has_other_dataset = any(
                conf is not None for conf in (c3d_dataset_conf, lafan1_dataset_conf, custom_dataset_conf)
            )
            if amass_dataset_conf.write_materialized_cache:
                if has_other_dataset:
                    raise ValueError("AMASS materialized cache cannot be combined with other trajectory datasets.")
                cls._load_materialized_amass_trajectory(env, amass_dataset_conf, visualize_goal)
                return env
            if not has_other_dataset:
                cls._load_runtime_amass_trajectory(env, amass_dataset_conf, visualize_goal)
                return env
            loaded_sets.append(cls.get_amass_trajectory_set(env, amass_dataset_conf, visualize_goal))

        # Load converted C3D trajectories if available
        if c3d_dataset_conf is not None:
            if isinstance(c3d_dataset_conf, dict | DictConfig):
                valid_keys = {f.name for f in dataclasses.fields(C3DDatasetConf)}
                filtered_conf = {k: v for k, v in c3d_dataset_conf.items() if k in valid_keys}
                c3d_dataset_conf = C3DDatasetConf(**filtered_conf)
            loaded_sets.append(cls.get_c3d_trajectory_set(env, c3d_dataset_conf))

        # Load the LAFAN1 trajectory if available
        if lafan1_dataset_conf is not None:
            if isinstance(lafan1_dataset_conf, dict | DictConfig):
                lafan1_dataset_conf = LAFAN1DatasetConf(**lafan1_dataset_conf)
            loaded_sets.append(cls.get_lafan1_trajectory_set(env, lafan1_dataset_conf))

        # Load the custom trajectory if available
        if custom_dataset_conf is not None:
            if isinstance(custom_dataset_conf, dict | DictConfig):
                custom_dataset_conf = CustomDatasetConf(**custom_dataset_conf)
            loaded_sets.append(cls.get_custom_trajectory_set(env, custom_dataset_conf))

        # Only process trajectories if we have any to load
        if loaded_sets:
            cache_type = cls._get_trajectory_cache_type(
                amass_dataset_conf,
                c3d_dataset_conf,
                lafan1_dataset_conf,
                custom_dataset_conf,
            )
            loaded = LoadedTrajectorySet.concatenate(loaded_sets, backend=np)

            # add to the environment
            env.load_trajectory(
                traj=loaded.trajectory,
                motion_names=loaded.motion_names,
                warn=False,
                cache_type=cache_type,
                site_names=getattr(env, "sites_for_mimic", None) if cache_type == TrajectoryCacheType.SPARSE else None,
            )

        return env

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
        """Load AMASS trajectory data."""
        return cls.get_amass_trajectory_set(env, amass_dataset_conf, visualize_goal).trajectory

    @classmethod
    def get_amass_trajectory_set(
        cls,
        env,
        amass_dataset_conf: AMASSDatasetConf,
        visualize_goal: bool = False,
    ) -> LoadedTrajectorySet:
        """Load AMASS trajectories with their source names."""
        # Accept both dataclass instances and raw dict/DictConfig inputs
        if isinstance(amass_dataset_conf, dict | DictConfig):
            amass_dataset_conf = AMASSDatasetConf(**amass_dataset_conf)

        motion_specs = cls.select_amass_motion_specs(amass_dataset_conf)
        return cls._load_amass_dataset_paths(env, amass_dataset_conf, motion_specs, visualize_goal)

    @staticmethod
    def resolve_amass_dataset_paths(
        amass_dataset_conf: AMASSDatasetConf | dict | DictConfig,
    ) -> list[str]:
        """Expand an AMASS configuration to its canonical motion set."""
        if isinstance(amass_dataset_conf, dict | DictConfig):
            amass_dataset_conf = AMASSDatasetConf(**amass_dataset_conf)

        dataset_paths = []
        if amass_dataset_conf.dataset_group is not None:
            groups = get_amass_dataset_groups()
            for group_name in expand_amass_dataset_group_spec(amass_dataset_conf.dataset_group):
                if group_name not in groups:
                    raise ValueError(f"Unknown dataset group: {group_name}")
                dataset_paths.extend(groups[group_name])
        if amass_dataset_conf.rel_dataset_path is not None:
            configured_paths = (
                amass_dataset_conf.rel_dataset_path
                if isinstance(amass_dataset_conf.rel_dataset_path, ListConfig | list | tuple)
                else [amass_dataset_conf.rel_dataset_path]
            )
            dataset_paths.extend(as_amass_motion_spec(path).path for path in configured_paths)
        return sorted(set(dataset_paths))

    @staticmethod
    def resolve_amass_motion_specs(
        amass_dataset_conf: AMASSDatasetConf | dict | DictConfig,
    ) -> list[AMASSMotionSpec]:
        """Expand an AMASS configuration while retaining source-time bounds."""
        if isinstance(amass_dataset_conf, dict | DictConfig):
            amass_dataset_conf = AMASSDatasetConf(**amass_dataset_conf)

        motion_specs = []
        if amass_dataset_conf.dataset_group is not None:
            groups = get_amass_motion_spec_groups()
            for group_name in expand_amass_dataset_group_spec(amass_dataset_conf.dataset_group):
                if group_name not in groups:
                    raise ValueError(f"Unknown dataset group: {group_name}")
                motion_specs.extend(groups[group_name])
        if amass_dataset_conf.rel_dataset_path is not None:
            configured_paths = (
                amass_dataset_conf.rel_dataset_path
                if isinstance(amass_dataset_conf.rel_dataset_path, ListConfig | list | tuple)
                else [amass_dataset_conf.rel_dataset_path]
            )
            motion_specs.extend(as_amass_motion_spec(path) for path in configured_paths)

        specs_by_path = {}
        for spec in motion_specs:
            previous = specs_by_path.setdefault(spec.path, spec)
            if previous != spec:
                raise ValueError(
                    f"AMASS motion {spec.path!r} has conflicting time bounds: "
                    f"{previous.cache_key!r} and {spec.cache_key!r}."
                )
        return [specs_by_path[path] for path in sorted(specs_by_path)]

    @classmethod
    def select_amass_dataset_paths(
        cls,
        amass_dataset_conf: AMASSDatasetConf | dict | DictConfig,
    ) -> list[str]:
        """Return the reproducible motion subset selected by an AMASS configuration."""
        if isinstance(amass_dataset_conf, dict | DictConfig):
            amass_dataset_conf = AMASSDatasetConf(**amass_dataset_conf)

        dataset_paths = cls.resolve_amass_dataset_paths(amass_dataset_conf)
        if amass_dataset_conf.max_motions is not None and len(dataset_paths) > amass_dataset_conf.max_motions:
            before = len(dataset_paths)
            rng = random.Random(amass_dataset_conf.max_motions_seed)
            dataset_paths = sorted(rng.sample(dataset_paths, amass_dataset_conf.max_motions))
            print(
                f"[AMASS] INFO: Sampled {amass_dataset_conf.max_motions} trajectories out of {before} "
                f"(max_motions={amass_dataset_conf.max_motions}, "
                f"seed={amass_dataset_conf.max_motions_seed})."
            )
        return dataset_paths

    @classmethod
    def select_amass_motion_specs(
        cls,
        amass_dataset_conf: AMASSDatasetConf | dict | DictConfig,
    ) -> list[AMASSMotionSpec]:
        """Return the selected motion specifications, including time bounds."""
        if isinstance(amass_dataset_conf, dict | DictConfig):
            amass_dataset_conf = AMASSDatasetConf(**amass_dataset_conf)

        specs_by_path = {spec.path: spec for spec in cls.resolve_amass_motion_specs(amass_dataset_conf)}
        selected_paths = cls.select_amass_dataset_paths(amass_dataset_conf)
        return [specs_by_path[path] for path in selected_paths]

    @staticmethod
    def _load_amass_dataset_paths(
        env,
        amass_dataset_conf: AMASSDatasetConf,
        dataset_paths: list[AMASSMotionSpec],
        visualize_goal: bool,
    ) -> LoadedTrajectorySet:
        """Load already-resolved AMASS motion names."""
        env_name = env.__class__.__name__
        if visualize_goal:
            print(f"[Visualization] Building trajectories for env={env_name} with {len(dataset_paths)} paths.")

        # Load trajectories from AMASS datasets
        # Extract retargeting configs
        retargeting_method = amass_dataset_conf.retargeting_method
        gmr_config = amass_dataset_conf.gmr_config
        gmr_cache_namespace = resolve_training_gmr_cache_namespace(
            retargeting_method,
            amass_dataset_conf.gmr_cache_namespace,
        )
        clear_cache = amass_dataset_conf.clear_cache

        if "MyoBimanualArm" in env_name:
            method_name = retargeting_method.upper() if retargeting_method else "SMPL"
            print(
                f"[MuscleMimic] Detected MyoBimanualArm environment. "
                f"Using three-stage retargeting pipeline with {method_name} for Stage 1."
            )
            loaded = retarget_smpl_to_bimanual_via_intermediate_set(
                dataset_paths,
                retargeting_method=retargeting_method,
                gmr_config=gmr_config,
                clear_cache=clear_cache,
            )
        else:
            loaded = load_retargeted_amass_trajectory_set(
                env_name,
                dataset_paths,
                retargeting_method=retargeting_method,
                gmr_config=gmr_config,
                gmr_cache_namespace=gmr_cache_namespace,
                clear_cache=clear_cache,
            )

        return loaded

    @classmethod
    def _load_materialized_amass_trajectory(
        cls,
        env,
        amass_dataset_conf: AMASSDatasetConf,
        visualize_goal: bool,
    ) -> None:
        """Load or create the persistent runtime cache for one AMASS motion set."""
        motion_specs = cls.select_amass_motion_specs(amass_dataset_conf)
        motion_names = [spec.path for spec in motion_specs]
        cache_type = TrajectoryCacheType(amass_dataset_conf.trajectory_cache_type)
        site_names = list(env.sites_for_mimic) if cache_type == TrajectoryCacheType.SPARSE else None
        retargeting_method = amass_dataset_conf.retargeting_method or "default"
        gmr_cache_namespace = resolve_training_gmr_cache_namespace(
            retargeting_method,
            amass_dataset_conf.gmr_cache_namespace,
        )
        gmr_config = amass_dataset_conf.gmr_config
        if retargeting_method == "gmr":
            gmr_cache_identity, _cache_variant = get_gmr_cache_identity(gmr_config)
            retargeting_identity = f"gmr:{gmr_cache_namespace}:{gmr_cache_identity}"
        else:
            retargeting_identity = retargeting_method
        control_dt = env.dt
        fingerprint = materialized_trajectory_fingerprint(
            motion_names=[spec.cache_key for spec in motion_specs],
            model=env._model,
            control_dt=control_dt,
            cache_type=cache_type,
            site_names=site_names,
            retargeting_method=retargeting_identity,
            model_spec=getattr(env, "mjspec", None),
        )

        cache_env_name = env.__class__.__name__.replace("Mjx", "")
        converted_root = Path(get_converted_amass_dataset_path())
        cache_path = converted_root / cache_env_name / retargeting_method / "_materialized" / f"{fingerprint}.npz"

        def build() -> Trajectory:
            print(f"[AMASS] INFO: Building materialized trajectory cache at {cache_path}")
            source = cls._load_amass_dataset_paths(env, amass_dataset_conf, motion_specs, visualize_goal)
            if source.motion_names != tuple(motion_names):
                loaded_names = set(source.motion_names)
                missing_specs = [spec for spec in motion_specs if spec.path not in loaded_names]
                unexpected_missing = [
                    spec.path
                    for spec in missing_specs
                    if retargeting_method != "gmr"
                    or not gmr_quality_rejection_path(
                        get_gmr_motion_cache_path(
                            cache_env_name,
                            spec,
                            converted_root,
                            cache_namespace=gmr_cache_namespace,
                            gmr_config=gmr_config,
                        )
                    ).exists()
                ]
                if unexpected_missing:
                    raise RuntimeError(
                        "Cannot materialize an incomplete AMASS motion set. "
                        f"Failed motions: {sorted(unexpected_missing)}"
                    )
                print(f"[AMASS] INFO: Excluding {len(missing_specs)} persistent GMR quality rejection(s).")
            trajectory = materialize_trajectory(
                source.trajectory,
                model=env._model,
                control_dt=control_dt,
                cache_type=cache_type,
                site_names=site_names,
            )
            metadata = dict(trajectory.info.metadata or {})
            metadata[MATERIALIZED_MOTION_NAMES_METADATA_KEY] = list(source.motion_names)
            return dataclasses.replace(
                trajectory,
                info=dataclasses.replace(trajectory.info, metadata=metadata),
            )

        rebuild = amass_dataset_conf.rebuild_materialized_cache or amass_dataset_conf.clear_cache
        cache_hit = cache_path.exists() and not rebuild
        if cache_hit:
            print(f"[AMASS] INFO: Loading materialized trajectory cache from {cache_path}")

        trajectory = load_or_build_materialized_trajectory(
            cache_path=cache_path,
            fingerprint=fingerprint,
            build=build,
            rebuild=rebuild,
        )
        if not cache_hit:
            print(f"[AMASS] INFO: Saved materialized trajectory cache to {cache_path}")
        cached_motion_names = tuple(
            (trajectory.info.metadata or {}).get(
                MATERIALIZED_MOTION_NAMES_METADATA_KEY,
                motion_names,
            )
        )
        env.load_trajectory(
            traj=trajectory,
            motion_names=cached_motion_names,
            warn=False,
            cache_type=cache_type,
            site_names=site_names,
            materialized=True,
        )

    @classmethod
    def _load_runtime_amass_trajectory(
        cls,
        env,
        amass_dataset_conf: AMASSDatasetConf,
        visualize_goal: bool,
    ) -> None:
        """Load AMASS motions without writing a persistent materialized cache."""
        cache_type = TrajectoryCacheType(amass_dataset_conf.trajectory_cache_type)
        sites_for_mimic = getattr(env, "sites_for_mimic", None)
        site_names = (
            list(sites_for_mimic) if cache_type == TrajectoryCacheType.SPARSE and sites_for_mimic is not None else None
        )
        loaded = cls.get_amass_trajectory_set(env, amass_dataset_conf, visualize_goal)
        env.load_trajectory(
            traj=loaded.trajectory,
            motion_names=loaded.motion_names,
            warn=False,
            cache_type=cache_type,
            site_names=site_names,
        )

    @staticmethod
    def get_c3d_traj(env, c3d_dataset_conf: C3DDatasetConf) -> Trajectory:
        """Load converted C3D trajectory data."""
        return ImitationFactory.get_c3d_trajectory_set(env, c3d_dataset_conf).trajectory

    @staticmethod
    def get_c3d_trajectory_set(env, c3d_dataset_conf: C3DDatasetConf) -> LoadedTrajectorySet:
        """Load converted C3D trajectories with their source names."""
        from musclemimic.web_viewer.c3d_pipeline import (
            get_converted_c3d_dataset_path,
            normalize_c3d_dataset_name,
        )

        dataset_paths = (
            c3d_dataset_conf.rel_dataset_path
            if isinstance(c3d_dataset_conf.rel_dataset_path, ListConfig | list)
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

        loaded_sets = []
        for rel_dataset_path in dataset_paths:
            normalized = normalize_c3d_dataset_name(rel_dataset_path)
            trajectory_path = converted_root / cache_env_name / method / normalized.with_suffix(".npz")
            if not trajectory_path.exists():
                raise FileNotFoundError(
                    f"Converted C3D trajectory not found: {trajectory_path}. "
                    "Create it with `python -m musclemimic.web_viewer.run --c3d-file ... "
                    "--c3d-dataset-name <name>`."
                )
            trajectory = Trajectory.load(trajectory_path, backend=np)
            motion_names = (rel_dataset_path,) * int(trajectory.data.n_trajectories)
            loaded_sets.append(LoadedTrajectorySet(trajectory, motion_names))

        return LoadedTrajectorySet.concatenate(loaded_sets, backend=np)

    @staticmethod
    def get_lafan1_traj(env, lafan1_dataset_conf: LAFAN1DatasetConf) -> Trajectory:
        """Load LAFAN1 trajectory data."""
        return ImitationFactory.get_lafan1_trajectory_set(env, lafan1_dataset_conf).trajectory

    @staticmethod
    def get_lafan1_trajectory_set(env, lafan1_dataset_conf: LAFAN1DatasetConf) -> LoadedTrajectorySet:
        """Load LAFAN1 trajectories with their source names."""
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
                if isinstance(lafan1_dataset_conf.dataset_name, ListConfig | list)
                else [lafan1_dataset_conf.dataset_name]
            )

        loaded_sets = []
        for dataset_path in dataset_paths:
            trajectory = load_lafan1_trajectory(env.__class__.__name__, dataset_path)
            motion_names = (dataset_path,) * int(trajectory.data.n_trajectories)
            loaded_sets.append(LoadedTrajectorySet(trajectory, motion_names))

        return LoadedTrajectorySet.concatenate(loaded_sets, backend=np)

    @staticmethod
    def get_custom_dataset(env, custom_dataset_conf: CustomDatasetConf) -> Trajectory:
        """Load custom trajectory data."""
        return ImitationFactory.get_custom_trajectory_set(env, custom_dataset_conf).trajectory

    @staticmethod
    def get_custom_trajectory_set(env, custom_dataset_conf: CustomDatasetConf) -> LoadedTrajectorySet:
        """Load custom trajectories without inferred source names."""
        from loco_mujoco.smpl.retargeting import extend_motion

        traj = custom_dataset_conf.traj
        # Custom trajectories must match the environment kinematic layout.
        if not traj.data.is_complete:
            env_name = env.__class__.__name__
            env_params = {}
            traj = extend_motion(env_name, env_params, traj)

        motion_names = (None,) * int(traj.data.n_trajectories)
        return LoadedTrajectorySet(traj, motion_names)
