import inspect
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx
from mujoco.mjx import Data, Model

from loco_mujoco.core.mujoco_base import AdditionalCarry, Mujoco
from loco_mujoco.core.visuals import MujocoViewer
from loco_mujoco.trajectory import TrajectoryData


_NUMERICS_INFO_KEYS = (
    "numerics_nonfinite_observation",
    "numerics_nonfinite_qpos",
    "numerics_nonfinite_qvel",
    "numerics_nonfinite_reward",
    "numerics_nonfinite_reward_info",
    "numerics_nonfinite_transition",
)


def _is_inexact(value: Any) -> bool:
    """Return whether a JAX-compatible value can contain NaN or infinity."""
    try:
        return jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact)
    except (TypeError, ValueError):
        return False


def _tree_has_nonfinite(tree: Any) -> jax.Array:
    """Return a scalar flag for nonfinite floating-point leaves in ``tree``."""
    flags = [
        jnp.any(~jnp.isfinite(jnp.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(tree)
        if _is_inexact(leaf)
    ]
    if not flags:
        return jnp.asarray(False)
    return jnp.any(jnp.stack(flags))


def _sanitize_inexact_tree(tree: Any) -> Any:
    """Replace NaN and infinities in floating-point leaves with finite zeros."""

    def sanitize(value):
        if not _is_inexact(value):
            return value
        return jnp.nan_to_num(jnp.asarray(value), nan=0.0, posinf=0.0, neginf=0.0)

    return jax.tree_util.tree_map(sanitize, tree)


def _preserve_scan_carry_axis_types(next_tree: Any, previous_tree: Any) -> Any:
    """Preserve manual-axis variation erased by opaque simulation calls.

    MJX-Warp steps through an FFI boundary that can return arrays without the
    manual-axis variation carried by their inputs.  A ``lax.scan`` requires the
    input and output carry types to match exactly, including those annotations.
    Make every non-empty output depend on its corresponding input without
    changing its value.  Empty buffers keep the previous value, which is
    equivalent and retains its annotation.
    """

    def preserve(next_value: Any, previous_value: Any) -> Any:
        if next_value.size == 0:
            return previous_value
        input_dependent = previous_value == previous_value
        return jnp.where(input_dependent, next_value, next_value)

    return jax.tree_util.tree_map(preserve, next_tree, previous_tree)


def _parse_warp_graph_mode(value: str):
    """Translate the public configuration value to MuJoCo's Warp enum."""
    from mujoco.mjx.warp.types import GraphMode

    graph_modes = {
        "jax": GraphMode.JAX,
        "warp": GraphMode.WARP,
        "warp_staged_ex": GraphMode.WARP_STAGED_EX,
    }
    requested_mode = str(value).lower()
    if requested_mode not in graph_modes:
        valid_modes = ", ".join(repr(mode) for mode in graph_modes)
        raise ValueError(f"mjx_warp_graph_mode must be one of {{{valid_modes}}}; got {value!r}")
    return graph_modes[requested_mode]


@struct.dataclass
class MjxAdditionalCarry(AdditionalCarry):
    """
    Additional carry for the Mjx environment.

    Includes final observation/info for episode boundaries and RNG state for AutoResetWrapper.
    """

    final_observation: jax.Array
    final_info: dict[str, Any]
    # AutoResetWrapper RNG state (None when not using AutoResetWrapper)
    autoreset_rng: jax.Array | None = None


@struct.dataclass
class MjxState:
    """
    State of the Mjx environment.

    Args:
        data (Data): Mjx data structure.
        observation (jax.Array): Observation of the environment.
        reward (float): Reward of the environment.
        absorbing (bool): Whether the state is absorbing.
        done (bool): Whether the episode is done.
        additional_carry (Any): Additional carry information.
        info (Dict[str, Any]): Information dictionary.

    """

    data: Data
    observation: jax.Array
    reward: float
    absorbing: bool
    done: bool
    additional_carry: MjxAdditionalCarry
    info: dict[str, Any] = struct.field(default_factory=dict)


class Mjx(Mujoco):
    """
    Base class for Mujoco environments using JAX.

    Args:
        n_envs (int): Number of environments to run in parallel.
        **kwargs: Additional arguments to pass to the Mujoco base class.

    """

    def __init__(
        self,
        mjx_backend="jax",
        mjx_warp_graph_mode: str | None = None,
        **kwargs,
    ):
        # Extract mjx_backend parameter before passing to parent
        self.mjx_backend = mjx_backend
        self.mjx_warp_graph_mode = mjx_warp_graph_mode

        # call base mujoco env
        super().__init__(**kwargs)

        # add information to mdp_info
        self._mdp_info.mjx_env = True

        # setup mjx model and data
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        # Select backend implementation and store it for later
        backend_impl = self._get_mjx_backend()
        self._backend_impl = backend_impl

        put_model_kwargs = {}
        if backend_impl == "warp" and mjx_warp_graph_mode is not None:
            put_model_kwargs["graph_mode"] = _parse_warp_graph_mode(mjx_warp_graph_mode)
        sys = mjx.put_model(self._model, impl=backend_impl, **put_model_kwargs)
        # Warp's FFI buffers must first be materialized by each transformed
        # replica, not eagerly bound to device zero during Python setup.
        self.sys = jax.device_get(sys) if backend_impl == "warp" else sys

        # Handle data creation based on backend
        if backend_impl == "warp":
            # The model/spec nconmax remains the per-world contact budget, while
            # MJX-Warp 3.11 allocates contacts through naconmax across all worlds.
            # Use the calculated values from the environment if available, otherwise model defaults.
            nconmax_val = getattr(self, "nconmax", self._model.nconmax)
            njmax_val = getattr(self, "njmax", self._model.njmax)
            # naconmax is the total active-contact capacity across all environments.
            naconmax_val = getattr(self, "naconmax", nconmax_val)
            # Store budgets for use in reset
            self._nconmax = int(nconmax_val)
            self._njmax = int(njmax_val)
            self._naconmax = int(naconmax_val)
        # NOTE: No longer storing _first_data - it will be created fresh in each reset
        # This avoids sharing Data objects across vmapped instances (causes BatchTracer leaks)
        self._backend_impl = backend_impl

    def _make_device_data(self) -> Data:
        """Create fresh backend data from the reset and forwarded CPU template."""
        if self._backend_impl == "warp":
            return mjx.put_data(
                self._model,
                self._data,
                impl="warp",
                njmax=self._njmax,
                naconmax=self._naconmax,
            )
        return mjx.put_data(self._model, self._data, impl=self._backend_impl)

    def _get_mjx_backend(self):
        """
        Determines the MJX backend implementation to use.

        Returns:
            str: Backend implementation ('jax' or 'warp')

        Raises:
            ImportError: If Warp backend requested but warp-lang not installed
            RuntimeError: If Warp backend requested but CUDA not available
        """
        if self.mjx_backend == "warp":
            # Check if Warp is available - raise error if not
            try:
                import warp
            except ImportError as error:
                raise ImportError(
                    "Warp backend requested but warp-lang not installed. Install with: pip install warp-lang"
                ) from error

            # Check if CUDA is available for Warp - raise error if not
            if not warp.is_cuda_available():
                raise RuntimeError("Warp backend requested but CUDA not available. Warp requires CUDA.")

            print(f"Using Warp backend (version {warp.__version__})")
            return "warp"
        else:
            # Default to JAX backend
            return "jax"

    def mjx_reset(
        self,
        key: jax.random.PRNGKey,
        selected_traj_idx: jax.Array | None = None,
        selected_subtraj_step: jax.Array | None = None,
    ) -> MjxState:
        """
        Resets the environment.

        Args:
            key (jax.random.PRNGKey): Random key for the reset.

        Returns:
            MjxState: The reset state of the environment.

        """

        key, subkey = jax.random.split(key)

        data = self._make_device_data()

        # Forward kinematics for initial carry initialization
        data = mjx.forward(self.sys, data)

        carry = self._init_additional_carry(key, self._model, data, jnp)
        if selected_traj_idx is not None and hasattr(carry, "selected_traj_idx"):
            carry = carry.replace(
                selected_traj_idx=jnp.asarray(selected_traj_idx, dtype=jnp.int32),
            )
        if selected_subtraj_step is not None and hasattr(carry, "selected_subtraj_step"):
            carry = carry.replace(
                selected_subtraj_step=jnp.asarray(selected_subtraj_step, dtype=jnp.int32),
            )

        # Explicit trajectory resets have a separate static path so data-parallel
        # compilation does not trace the ordinary random-start branch.
        if selected_traj_idx is None:
            data, carry = self._mjx_reset_carry(self.sys, data, carry)
        else:
            data, carry = self._mjx_reset_carry_to_step(
                self.sys,
                data,
                carry,
                selected_traj_idx,
                selected_subtraj_step,
            )

        # Apply terrain/domain randomization and recompute derived quantities so the initial
        # observation uses the correct model parameters and kinematics.
        sys, data, carry = self._mjx_simulation_pre_step(self.sys, data, carry)
        data = mjx.forward(sys, data)

        # reset all stateful entities
        data, carry = self.obs_container.reset_state(self, sys, data, carry, jnp)

        obs, carry = self._mjx_create_observation(sys, data, carry)
        reward = 0.0
        absorbing = jnp.array(False, dtype=bool)
        done = jnp.array(False, dtype=bool)
        info = self._mjx_reset_info_dictionary(obs, data, subkey)

        # Initialize reward_info keys with zeros for JAX scan compatibility
        dummy_action = jnp.zeros(self._mdp_info.action_space.shape[0])
        _, carry, reward_info = self._mjx_reward(obs, dummy_action, obs, absorbing, info, self._model, data, carry)
        for key, val in reward_info.items():
            info[key] = jnp.zeros_like(val)
        for key in _NUMERICS_INFO_KEYS:
            info[key] = jnp.asarray(False)

        return MjxState(
            data=data, observation=obs, reward=reward, absorbing=absorbing, done=done, info=info, additional_carry=carry
        )

    def _mjx_reset_in_step(self, state: MjxState) -> MjxState:
        """Reset environment in-step (for auto-reset when done).

        Unlike mjx_reset(), this preserves the pytree structure by modifying
        the existing state object rather than creating a new one. This helper
        is only available on MJX backends that support ``mjx.put_data``.

        Args:
            state: Current state to reset

        Returns:
            Reset state with same pytree structure as input
        """
        carry = state.additional_carry

        # Rebuild a fresh MJX Data from the CPU template and then reapply environment-specific reset logic.
        # NOTE: We use self._data as template - it's read-only, not shared mutable state
        data = self._make_device_data()
        data, carry = self._mjx_reset_carry(self.sys, data, carry)

        # Reset carry (no special filtering needed - state.info has same structure)
        carry = carry.replace(
            cur_step_in_episode=1,
            final_observation=state.observation,
            last_action=jnp.zeros_like(carry.last_action),
            final_info=state.info,
        )

        # Apply terrain/domain randomization and recompute derived quantities after reset_carry updates.
        sys, data, carry = self._mjx_simulation_pre_step(self.sys, data, carry)
        data = mjx.forward(sys, data)

        # Update all stateful entities
        data, carry = self.obs_container.reset_state(self, sys, data, carry, jnp)

        # Create new observation
        obs, carry = self._mjx_create_observation(sys, data, carry)

        # Return modified state (preserves pytree structure)
        return state.replace(data=data, observation=obs, additional_carry=carry)

    def mjx_step(self, state: MjxState, action: jax.Array) -> MjxState:
        """

        Args:
            state (MjxState): Current state of the environment.
            action (jax.Array): Action to take in the environment.

        Returns:
            MjxState: The next state of the environment.

        """

        data = state.data
        cur_info = state.info
        carry = state.additional_carry
        carry = carry.replace(last_action=action)

        # reset dones
        state = state.replace(done=jnp.zeros_like(state.done, dtype=bool))

        # preprocess action
        processed_action, carry = self._mjx_preprocess_action(action, self._model, data, carry)

        # modify data and model *before* step if needed
        sys, data, carry = self._mjx_simulation_pre_step(self.sys, data, carry)

        def _inner_loop(_runner_state, _):
            """Single intermediate step using scan for vmap compatibility."""
            _data, _carry = _runner_state

            ctrl_action, _carry = self._mjx_compute_action(processed_action, self._model, _data, _carry)

            # step in the environment using the action
            action_idx = jnp.asarray(self._action_indices, dtype=jnp.int32)
            ctrl = _data.ctrl.at[action_idx].set(ctrl_action)
            _data = _data.replace(ctrl=ctrl)

            # Use scan instead of fori_loop for vmap compatibility
            def single_step(data, _):
                next_data = mjx.step(sys, data)
                next_data = _preserve_scan_carry_axis_types(next_data, data)
                return next_data, None

            _data = jax.lax.scan(single_step, _data, (), self._n_substeps)[0]

            return (_data, _carry), None

        # run inner loop with scan for vmap compatibility
        (data, carry), _ = jax.lax.scan(_inner_loop, (data, carry), (), self._n_intermediate_steps)

        # modify data *after* step if needed (does nothing by default)
        data, carry = self._mjx_simulation_post_step(self._model, data, carry)

        # create the observation
        cur_obs, carry = self._mjx_create_observation(sys, data, carry)

        # modify the observation and the data if needed (does nothing by default)
        cur_obs, data, cur_info, carry = self._mjx_step_finalize(cur_obs, self._model, data, cur_info, carry)

        # create info
        cur_info = self._mjx_update_info_dictionary(cur_info, cur_obs, data, carry)

        # check if the next obs is an absorbing state
        absorbing, carry = self._mjx_is_absorbing(cur_obs, cur_info, data, carry)

        # calculate the reward
        reward, carry, reward_info = self._mjx_reward(
            state.observation, action, cur_obs, absorbing, cur_info, self._model, data, carry
        )

        # Contain a bad transition before it reaches episode/reward wrappers.
        # Reward implementations may sanitize their scalar output, so honor an
        # explicit pre-sanitization marker in addition to inspecting values.
        reward_marker = jnp.any(
            jnp.asarray(reward_info.get("numerics_nonfinite_reward", False), dtype=bool)
        )
        nonfinite_observation = _tree_has_nonfinite(cur_obs)
        nonfinite_qpos = _tree_has_nonfinite(data.qpos)
        nonfinite_qvel = _tree_has_nonfinite(data.qvel)
        nonfinite_reward = reward_marker | _tree_has_nonfinite(reward)
        nonfinite_reward_info = _tree_has_nonfinite(reward_info)
        nonfinite_transition = (
            nonfinite_observation
            | nonfinite_qpos
            | nonfinite_qvel
            | nonfinite_reward
            | nonfinite_reward_info
        )

        reward = jnp.where(nonfinite_transition, jnp.zeros_like(reward), reward)
        cur_obs = jnp.nan_to_num(cur_obs, nan=0.0, posinf=0.0, neginf=0.0)
        reward_info = dict(_sanitize_inexact_tree(reward_info))
        reward_info.update(
            {
                "numerics_nonfinite_observation": nonfinite_observation,
                "numerics_nonfinite_qpos": nonfinite_qpos,
                "numerics_nonfinite_qvel": nonfinite_qvel,
                "numerics_nonfinite_reward": nonfinite_reward,
                "numerics_nonfinite_reward_info": nonfinite_reward_info,
                "numerics_nonfinite_transition": nonfinite_transition,
            }
        )
        # Merge finite reward components and numerical flags into info for logging.
        cur_info.update(reward_info)
        absorbing = jnp.logical_or(absorbing, nonfinite_transition)

        # check if done
        done = self._mjx_is_done(cur_obs, absorbing, cur_info, data, carry)
        done = jnp.logical_or(done, nonfinite_transition)

        # create state
        carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode + 1)

        state = state.replace(
            data=data,
            observation=cur_obs,
            reward=reward,
            absorbing=absorbing,
            done=done,
            info=cur_info,
            additional_carry=carry,
        )

        # NOTE: mjx_step intentionally returns the real transition state only.
        # Training code is responsible for any autoreset policy (for example via
        # AutoResetWrapper), which keeps rollout control flow separate from the
        # base physics step across both JAX and Warp backends.

        return state

    def _mjx_create_observation(self, model: Model, data: Data, carry: MjxAdditionalCarry) -> jax.Array:
        """
        Creates the observation for the environment.

        Args:
            model (Model): Mjx model.
            data (Data): Mjx data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            jax.Array: The observation of the environment.

        """
        return self._create_observation_compat(model, data, carry, jnp)

    def _mjx_reset_info_dictionary(self, obs: jnp.ndarray, data: Data, key: jax.random.PRNGKey) -> dict:
        """
        Resets the info dictionary.

        Args:
            obs (jnp.ndarray): Observation of the environment.
            data (Data): Mjx data structure.
            key (jax.random.PRNGKey): Random key.

        Returns:
            Dict: The updated info dictionary.

        """
        return {}

    def _mjx_update_info_dictionary(self, info: dict, obs: jnp.ndarray, data: Data, carry: MjxAdditionalCarry) -> dict:
        """
        Updates the info dictionary.

        Args:
            obs (jnp.ndarray): Observation of the environment.
            data (Data): Mjx data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Dict: The updated info dictionary.

        """
        return info

    def _mjx_reward(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        absorbing: bool,
        info: dict,
        model: Model,
        data: Data,
        carry: MjxAdditionalCarry,
    ) -> tuple[float, MjxAdditionalCarry, dict]:
        """
        Calls the reward function of the environment.

        Args:
            obs (jnp.ndarray): Observation of the environment.
            action (jnp.ndarray): Action taken in the environment.
            next_obs (jnp.ndarray): Next observation of the environment.
            absorbing (bool): Whether the next state is absorbing.
            info (Dict): Information dictionary.
            model (Model): Mjx model.
            data (Data): Mjx data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[float, MjxAdditionalCarry, dict]: The reward, updated carry, and reward component info.

        """
        result = self._reward_function(obs, action, next_obs, absorbing, info, self, model, data, carry, jnp)
        # Backwards compatibility: handle reward functions that return (reward, carry) without reward_info
        if len(result) == 2:
            reward, carry = result
            reward_info = {}
        else:
            reward, carry, reward_info = result
        return reward, carry, reward_info

    def _mjx_is_absorbing(self, obs: jnp.ndarray, info: dict, data: Data, carry: MjxAdditionalCarry) -> bool:
        """
        Determines if the current state is absorbing.

        Args:
            obs (jnp.ndarray): Current observation.
            info (Dict): Information dictionary.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            bool: True if the state is absorbing, False otherwise.
        """
        # Check if JAX-compatible curriculum learning is enabled
        if hasattr(self, "_curriculum_state_info") and hasattr(
            self._terminal_state_handler, "mjx_is_absorbing_with_state"
        ):
            return self._terminal_state_handler.mjx_is_absorbing_with_state(
                self, obs, info, data, carry, self._curriculum_state_info
            )
        else:
            return self._terminal_state_handler.mjx_is_absorbing(self, obs, info, data, carry)

    def _mjx_is_done(
        self, obs: jnp.ndarray, absorbing: bool, info: dict, data: Data, carry: MjxAdditionalCarry
    ) -> bool:
        """
        Determines if the episode is done.

        Args:
            obs (jnp.ndarray): Current observation.
            absorbing (bool): Whether the next state is absorbing.
            info (Dict): Information dictionary.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            bool: True if the episode is done, False otherwise.
        """
        done = jnp.greater_equal(carry.cur_step_in_episode, self.info.horizon)
        done = jnp.logical_or(done, absorbing)
        return done

    def _mjx_simulation_pre_step(
        self, model: Model, data: Data, carry: MjxAdditionalCarry
    ) -> tuple[Model, Data, MjxAdditionalCarry]:
        """
        Applies pre-step modifications to the model, data, and carry.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Model, Data, MjxAdditionalCarry]: Updated model, data, and carry.
        """
        # Multi-GPU Warp keeps ``self.sys`` host-resident until reset/step is
        # transformed so it is not eagerly bound to device zero.  Terrain and
        # domain-randomizer hooks use JAX functional updates (``array.at``), so
        # materialize a replica-local working model at their shared boundary.
        model = jax.tree.map(jnp.asarray, model)
        model, data, carry = self._terrain.update(self, model, data, carry, jnp)
        model, data, carry = self._domain_randomizer.update(self, model, data, carry, jnp)
        return model, data, carry

    def _mjx_simulation_post_step(
        self, model: Model, data: Data, carry: MjxAdditionalCarry
    ) -> tuple[Data, MjxAdditionalCarry]:
        """
        Applies post-step modifications to the data and carry.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Data, MjxAdditionalCarry]: Updated data and carry.
        """
        return data, carry

    def _mjx_preprocess_action(
        self, action: jnp.ndarray, model: Model, data: Data, carry: MjxAdditionalCarry
    ) -> tuple[jnp.ndarray, MjxAdditionalCarry]:
        """
        Transforms the action before applying it to the environment.

        Args:
            action (jnp.ndarray): Action input.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[jnp.ndarray, MjxAdditionalCarry]: Processed action and updated carry.
        """
        action, carry = self._domain_randomizer.update_action(self, action, model, data, carry, jnp)
        return action, carry

    def _mjx_compute_action(
        self, action: jnp.ndarray, model: Model, data: Data, carry: MjxAdditionalCarry
    ) -> tuple[jnp.ndarray, MjxAdditionalCarry]:
        """
        Applies transformations to the action at intermediate steps.

        Args:
            action (jnp.ndarray): Action at the current step.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[jnp.ndarray, MjxAdditionalCarry]: Computed action and updated carry.
        """
        action, carry = self._control_func.generate_action(self, action, model, data, carry, jnp)
        return action, carry

    def _mjx_reset_carry(self, model: Model, data: Data, carry: MjxAdditionalCarry) -> tuple[Data, MjxAdditionalCarry]:
        """
        Resets the additional carry and allows modification to the Mujoco data.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Data, MjxAdditionalCarry]: Updated data and carry.
        """
        data, carry = self._terminal_state_handler.reset(self, model, data, carry, jnp)
        data, carry = self._terrain.reset(self, model, data, carry, jnp)
        data, carry = self._init_state_handler.reset(self, model, data, carry, jnp)
        data, carry = self._domain_randomizer.reset(self, model, data, carry, jnp)
        data, carry = self._reward_function.reset(self, model, data, carry, jnp)
        return data, carry

    def _mjx_reset_carry_to_step(
        self,
        model: Model,
        data: Data,
        carry: MjxAdditionalCarry,
        selected_traj_idx: jax.Array,
        selected_subtraj_step: jax.Array | None,
    ) -> tuple[Data, MjxAdditionalCarry]:
        """Reset hook for an explicitly selected trajectory frame.

        Non-trajectory environments retain the normal reset behavior.  LocoEnv
        overrides this hook to install the selected trajectory state directly.
        """
        del selected_traj_idx, selected_subtraj_step
        return self._mjx_reset_carry(model, data, carry)

    def _mjx_step_finalize(
        self, obs: jnp.ndarray, model: Model, data: Data, info: dict, carry: MjxAdditionalCarry
    ) -> tuple[jnp.ndarray, Data, dict, MjxAdditionalCarry]:
        """
        Allows information to be accessed at the end of a step.

        Args:
            obs (jnp.ndarray): Observation.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            info (Dict): Information dictionary.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[jnp.ndarray, Data, Dict, MjxAdditionalCarry]: Updated observation, data, info, and carry.
        """
        obs, carry = self._domain_randomizer.update_observation(self, obs, model, data, carry, jnp)
        return obs, data, info, carry

    @staticmethod
    def mjx_set_sim_state_from_traj_data(data: Data, traj_data: TrajectoryData, carry: MjxAdditionalCarry) -> Data:
        """
        Sets the simulation state from the trajectory data.

        Args:
            data (Data): Current Mujoco data.
            traj_data (TrajectoryData): Data from the trajectory.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Data: Updated Mujoco data.
        """
        return data.replace(
            xpos=traj_data.xpos if traj_data.xpos.size > 0 else data.xpos,
            xquat=traj_data.xquat if traj_data.xquat.size > 0 else data.xquat,
            cvel=traj_data.cvel if traj_data.cvel.size > 0 else data.cvel,
            qpos=traj_data.qpos if traj_data.qpos.size > 0 else data.qpos,
            qvel=traj_data.qvel if traj_data.qvel.size > 0 else data.qvel,
        )

    def _mjx_set_sim_state_from_obs(self, data: Data, obs: jnp.ndarray) -> Data:
        """
        Updates the simulation state from an observation.

        .. note:: This may not fully set the state of the simulation if the observation does not contain all the
                  necessary information.

        Args:
            data (Data): Current Mujoco data.
            obs (jnp.ndarray): Observation containing state information.

        Returns:
            Data: Updated Mujoco data.
        """
        data = data.replace(
            qpos=data.qpos.at[self._data_indices.free_joint_qpos].set(obs[self._obs_indices.free_joint_qpos]),
            qvel=data.qvel.at[self._data_indices.free_joint_qvel].set(obs[self._obs_indices.free_joint_qvel]),
        )

        return data.replace(
            xpos=data.xpos.at[self._data_indices.body_xpos].set(obs[self._obs_indices.body_xpos].reshape(-1, 3)),
            xquat=data.xquat.at[self._data_indices.body_xquat].set(obs[self._obs_indices.body_xquat].reshape(-1, 4)),
            cvel=data.cvel.at[self._data_indices.body_cvel].set(obs[self._obs_indices.body_cvel].reshape(-1, 6)),
            qpos=data.qpos.at[self._data_indices.joint_qpos].set(obs[self._obs_indices.joint_qpos]),
            qvel=data.qvel.at[self._data_indices.joint_qvel].set(obs[self._obs_indices.joint_qvel]),
            site_xpos=data.site_xpos.at[self._data_indices.site_xpos].set(
                obs[self._obs_indices.site_xpos].reshape(-1, 3)
            ),
            site_xmat=data.site_xmat.at[self._data_indices.site_xmat].set(
                obs[self._obs_indices.site_xmat].reshape(-1, 9)
            ),
        )

    def mjx_render(self, state, record: bool = False, debug_info: dict | None = None) -> np.ndarray:
        """
        Renders all environments in parallel.

        Args:
            state: Current environment state.
            record (bool): Whether to record the rendering.
            debug_info (dict): Optional debug info to overlay on recorded frames.

        Returns:
            np.ndarray: Rendered image.
        """
        # Unwrap wrapper states (e.g., LogWrapper/NormalizeVecReward) to bare MjxState
        unwrapped_state = state
        while hasattr(unwrapped_state, "env_state"):
            unwrapped_state = unwrapped_state.env_state

        if self._viewer is None:
            if "default_camera_mode" not in self._viewer_params.keys():
                self._viewer_params["default_camera_mode"] = "static"
            # Only pass parameters that MujocoViewer.__init__ accepts
            viewer_sig = inspect.signature(MujocoViewer.__init__)
            viewer_accepted_params = set(viewer_sig.parameters.keys()) - {"self", "model", "dt", "record"}
            viewer_params_filtered = {k: v for k, v in self._viewer_params.items() if k in viewer_accepted_params}
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **viewer_params_filtered)

        if self._terrain.is_dynamic:
            terrain_state = unwrapped_state.additional_carry.terrain_state
            assert hasattr(terrain_state, "height_field_raw"), "Terrain state does not have height_field_raw."
            assert self._terrain.hfield_id is not None, "Terrain hfield id is not set."
            hfield_data = np.array(terrain_state.height_field_raw)
            self._model.hfield_data = hfield_data[0]
            self._viewer.upload_hfield(self._model, hfield_id=self._terrain.hfield_id)

        return self._viewer.parallel_render(unwrapped_state, record, debug_info=debug_info)

    def mjx_render_trajectory(self, trajectory, record: bool = False) -> None:
        """
        Renders a trajectory sequence.

        Args:
            trajectory: A sequence of environment states.
            record (bool): Whether to record the rendering.

        """
        assert len(trajectory) > 0, "Mjx render got provided with an empty trajectory."

        # Unwrap first element if trajectory contains wrapper states
        first_state = trajectory[0]
        while hasattr(first_state, "env_state"):
            first_state = first_state.env_state

        if self._viewer is None:
            # Only pass parameters that MujocoViewer.__init__ accepts
            viewer_sig = inspect.signature(MujocoViewer.__init__)
            viewer_accepted_params = set(viewer_sig.parameters.keys()) - {"self", "model", "dt", "record"}
            viewer_params_filtered = {k: v for k, v in self._viewer_params.items() if k in viewer_accepted_params}
            self._viewer = MujocoViewer(self._model, self.dt, record=record, **viewer_params_filtered)

        n_envs = first_state.data.qpos.shape[0]

        for i in range(n_envs):
            for state in trajectory:
                cur_state = state
                while hasattr(cur_state, "env_state"):
                    cur_state = cur_state.env_state
                self._data.qpos, self._data.qvel = cur_state.data.qpos[i, :], cur_state.data.qvel[i, :]
                mujoco.mj_forward(self._model, self._data)
                self._viewer.render(self._data, record)

    def _init_additional_carry(self, key, model: Model, data: Data, backend: ModuleType) -> MjxAdditionalCarry:
        """
        Initializes additional carry parameters.

        Args:
            key: Random key for initialization.
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            backend (ModuleType): Computational backend (either numpy or jax.numpy).

        Returns:
            MjxAdditionalCarry: Initialized carry object.
        """
        carry = super()._init_additional_carry(key, model, data, backend)
        return MjxAdditionalCarry(
            final_observation=backend.zeros(self.info.observation_space.shape), final_info={}, **vars(carry)
        )

    @property
    def n_envs(self) -> int:
        """Returns the number of environments."""
        return self._n_envs

    @property
    def mjx_env(self) -> bool:
        """Indicates whether this environment supports MJX backend."""
        return bool(getattr(self, "mjx_enabled", False))
