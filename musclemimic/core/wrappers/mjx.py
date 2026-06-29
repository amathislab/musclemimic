from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from loco_mujoco.core.utils.env import Box
from musclemimic.core.mujoco_mjx import Mjx, MjxState


class LocoMjxWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        """
        Allow proxy access to regular attributes of Mjx env.
        """
        return getattr(self.env, name)

    def reset(self, rng_key):
        if isinstance(rng_key, (int, jnp.integer)):
            rng_key = jax.random.PRNGKey(int(rng_key))

        state = self.env.mjx_reset(rng_key)
        return state.observation, state

    def reset_to(self, rng_key, traj_idx):
        if isinstance(rng_key, (int, jnp.integer)):
            rng_key = jax.random.PRNGKey(int(rng_key))

        state = self.env.mjx_reset(
            rng_key,
            selected_traj_idx=traj_idx,
        )
        return state.observation, state

    def step(self, state, action):
        next_state = self.env.mjx_step(state, action)
        next_obs = jnp.where(next_state.done, next_state.additional_carry.final_observation, next_state.observation)
        return (next_obs, next_state.reward, next_state.absorbing, next_state.done, next_state.info, next_state)


def is_vectorized(env) -> bool:
    """Check if environment is already wrapped with VecEnv.

    Traverses the wrapper chain to detect if VecEnv is present,
    preventing duplicate vectorization that causes nested vmap issues.

    Args:
        env: Environment to check (can be wrapped or unwrapped)

    Returns:
        True if VecEnv is in the wrapper chain, False otherwise
    """
    # Import here to avoid circular dependency
    # VecEnv is defined later in this file
    current = env
    max_depth = 20  # Prevent infinite loops in case of circular references

    for _ in range(max_depth):
        # Check class name to avoid forward reference issues
        if current.__class__.__name__ == "VecEnv":
            return True
        # Traverse wrapper chain
        if not hasattr(current, "env"):
            break
        current = current.env

    return False


@struct.dataclass
class BaseWrapperState:
    def __getattr__(self, name):
        """
        Allow proxy access to all attributes of all States.
        """
        try:
            if name in self.__dict__.keys():
                return self.__dict__[name]
            else:
                return getattr(self.env_state, name)
        except AttributeError as e:
            raise AttributeError(f"Attribute '{name}' not found in any env state nor the MjxState.") from e

    def find(self, cls):
        if isinstance(self, cls):
            return self
        elif isinstance(self.env_state, MjxState) and cls != MjxState:
            raise AttributeError(f"Class '{cls}' not found")
        else:
            return self.env_state.find(cls)


class BaseWrapper:
    def __init__(self, env):
        # if it's the bare Mjx class, wrap it in the LocoMjxWrapper first
        if issubclass(env.__class__, Mjx):
            self.env = LocoMjxWrapper(env)
        else:
            self.env = env

    def reset(self, rng_key):
        return self.env.reset(rng_key)

    def reset_to(self, rng_key, traj_idx):
        return self.env.reset_to(rng_key, traj_idx)

    def step(self, state, action):
        return self.env.step(state, action)

    def step_with_transition(self, state, action):
        """Step the environment and expose both rollout-carry and transition state.

        Contract:
            - ``carry_state`` is the state that should be fed into the next rollout
              step.
            - ``transition_state`` is the state that semantically belongs to the
              transition that just happened and should be used for metrics,
              diagnostics, or logging that refer to "this step".
            - ``transition_obs`` is the actual next observation for this
              transition. With autoreset it differs from the next rollout obs.

        Most wrappers do not change episode-boundary semantics, so they return the
        same state for both roles. Wrappers that perform autoreset can diverge:
        the carry state may already contain the reset state for the next episode,
        while the transition state still refers to the terminal pre-reset state.

        Returns:
            ``(obs, reward, absorbing, done, info, carry_state,
            transition_state, transition_obs)``.
        """
        res = self.step(state, action)
        return (*res, res[-1], res[0])

    def __getattr__(self, name):
        return getattr(self.env, name)

    def find_attr(self, state, attr_name):
        # Recursively search for the attribute
        if hasattr(state, attr_name):
            return getattr(state, attr_name)

        # If the attribute is not found, check env_state recursively
        if hasattr(state, "env_state") and state.env_state is not None:
            return self.find_attr(state.env_state, attr_name)

        # If the attribute or env_state isn't found
        raise AttributeError(f"Attribute '{attr_name}' not found")

    def unwrapped(self):
        # find first env which is not a subclass of BaseWrapper
        if isinstance(self.env, BaseWrapper):
            return self.env.unwrapped()
        else:
            return self.env.env


@struct.dataclass
class SummaryMetrics:
    mean_episode_return: float = 0.0
    mean_episode_length: float = 0.0
    max_timestep: int = 0.0
    early_termination_count: float = 0.0
    early_termination_rate: float = 0.0
    horizon_timeout_count: float = 0.0
    horizon_timeout_rate: float = 0.0
    # Reward sub-terms (mean over trajectory batch)
    reward_total: float = 0.0
    reward_qpos: float = 0.0
    reward_qvel: float = 0.0
    reward_root_pos: float = 0.0
    reward_rpos: float = 0.0
    reward_rquat: float = 0.0
    reward_rvel_rot: float = 0.0
    reward_rvel_lin: float = 0.0
    reward_root_vel: float = 0.0
    penalty_total: float = 0.0
    penalty_activation_energy: float = 0.0
    # Diagnostic error metrics
    err_root_xyz: float = 0.0
    err_root_yaw: float = 0.0
    err_joint_pos: float = 0.0
    err_joint_vel: float = 0.0
    err_site_abs: float = 0.0
    err_rpos: float = 0.0


@struct.dataclass
class Metrics:
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int
    done: bool
    absorbing: bool


@struct.dataclass
class LogEnvState(BaseWrapperState):
    env_state: MjxState
    metrics: Metrics


class LogWrapper(BaseWrapper):
    """Log the episode returns and lengths."""

    def reset(self, rng_key):
        obs, env_state = self.env.reset(rng_key)
        return self._wrap_reset_output(obs, env_state)

    def reset_to(self, rng_key, traj_idx):
        obs, env_state = self.env.reset_to(rng_key, traj_idx)
        return self._wrap_reset_output(obs, env_state)

    def _wrap_reset_output(self, obs, env_state):
        # Detect batch size from observation shape and initialize batched metrics
        # This handles both VecEnv → LogWrapper (batched) and LogWrapper → VecEnv (unbatched)
        if obs.ndim > 1:
            # Batched: obs.shape = (num_envs, obs_dim)
            batch_size = obs.shape[0]
            state = LogEnvState(
                env_state,
                metrics=Metrics(
                    jnp.zeros(batch_size, dtype=jnp.float32),
                    jnp.zeros(batch_size, dtype=jnp.int32),
                    jnp.zeros(batch_size, dtype=jnp.float32),
                    jnp.zeros(batch_size, dtype=jnp.int32),
                    jnp.zeros(batch_size, dtype=jnp.int32),
                    jnp.zeros(batch_size, dtype=bool),
                    jnp.zeros(batch_size, dtype=bool),
                ),
            )
        else:
            # Unbatched: obs.shape = (obs_dim,)
            state = LogEnvState(env_state, metrics=Metrics(0, 0, 0, 0, 0, False, False))
        return obs, state

    def step(self, state: LogEnvState, action: int | float):
        # make a step
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action)

        new_episode_return = state.metrics.episode_returns + reward
        new_episode_length = state.metrics.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            metrics=Metrics(
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=state.metrics.returned_episode_returns * (1 - done)
                + new_episode_return * done,
                returned_episode_lengths=state.metrics.returned_episode_lengths * (1 - done)
                + new_episode_length * done,
                timestep=state.metrics.timestep + 1,
                done=done,
                absorbing=absorbing,
            ),
        )
        return next_observation, reward, absorbing, done, info, state


@struct.dataclass
class NStepWrapperState(BaseWrapperState):
    env_state: MjxState
    observation_buffer: jnp.ndarray


class NStepWrapper(BaseWrapper):
    """Wrapper that stacks observations over multiple timesteps.

    When split_goal=True, only the state portion of the observation is stacked,
    while the goal observation is kept at the current timestep (not stacked).
    This prevents redundant stacking of future trajectory information.

    Output format when split_goal=True: [state_hist, goal]
        - state_hist: flattened buffer of shape (n_steps * state_dim,)
        - goal: current goal observation of shape (goal_dim,)
    """

    def __init__(self, env, n_steps, split_goal=False):
        super().__init__(env)
        self.n_steps = n_steps
        self.split_goal = split_goal

        if split_goal:
            if not hasattr(env, "obs_container"):
                raise ValueError("split_goal=True requires env.obs_container with goal group indices")
            goal_indices = env.obs_container.get_obs_ind_by_group("goal")
            if goal_indices.size == 0:
                raise ValueError("split_goal=True requires goal observations grouped as 'goal'")

            self.raw_obs_dim = env.info.observation_space.shape[0]
            self.goal_indices = np.asarray(goal_indices, dtype=int)
            state_mask = np.ones(self.raw_obs_dim, dtype=bool)
            state_mask[self.goal_indices] = False
            self.state_indices = np.arange(self.raw_obs_dim, dtype=int)[state_mask]
            self.goal_dim = self.goal_indices.size
            self.state_dim = self.state_indices.size
        else:
            self.goal_dim = 0
            self.state_dim = 0
            self.raw_obs_dim = 0
            self.goal_indices = None
            self.state_indices = None

        self.info = self.update_info(env.info)

    def update_info(self, info):
        new_info = deepcopy(info)

        if self.split_goal:
            state_high = info.observation_space.high[self.state_indices]
            state_low = info.observation_space.low[self.state_indices]
            goal_high = info.observation_space.high[self.goal_indices]
            goal_low = info.observation_space.low[self.goal_indices]
            high = np.concatenate([
                np.tile(state_high, self.n_steps),
                goal_high,
            ])
            low = np.concatenate([
                np.tile(state_low, self.n_steps),
                goal_low,
            ])
        else:
            # Original behavior: stack entire obs
            high = np.tile(info.observation_space.high, self.n_steps)
            low = np.tile(info.observation_space.low, self.n_steps)

        observation_space = Box(low, high)
        new_info.observation_space = observation_space
        return new_info

    def reset(self, rng_key):
        obs, env_state = self.env.reset(rng_key)
        return self._wrap_reset_output(obs, env_state)

    def reset_to(self, rng_key, traj_idx):
        obs, env_state = self.env.reset_to(rng_key, traj_idx)
        return self._wrap_reset_output(obs, env_state)

    def _wrap_reset_output(self, obs, env_state):
        if self.split_goal:
            # Split obs into state and goal
            state_obs = obs[self.state_indices]
            goal_obs = obs[self.goal_indices]

            # Buffer only stores state observations: (n_steps, state_dim)
            observation_buffer = jnp.zeros((self.n_steps, self.state_dim), dtype=obs.dtype)
            observation_buffer = observation_buffer.at[-1].set(state_obs)

            state = NStepWrapperState(env_state, observation_buffer)

            # Output: [flattened state history, current goal]
            state_hist = jnp.reshape(observation_buffer, (-1,))
            out_obs = jnp.concatenate([state_hist, goal_obs], axis=0)
        else:
            # Original behavior
            observation_buffer = jnp.tile(jnp.zeros_like(obs), (self.n_steps, 1))
            observation_buffer = observation_buffer.at[-1].set(obs)
            state = NStepWrapperState(env_state, observation_buffer)
            out_obs = jnp.reshape(observation_buffer, (-1,))

        return out_obs, state

    def step(self, state: NStepWrapperState, action: int | float):
        # Make a step
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action)

        observation_buffer = state.observation_buffer

        if self.split_goal:
            # Split next_observation into state and goal
            next_state = next_observation[self.state_indices]
            next_goal = next_observation[self.goal_indices]

            # Roll buffer and append new state
            observation_buffer = jnp.roll(observation_buffer, shift=-1, axis=0)
            observation_buffer = observation_buffer.at[-1].set(next_state)

            state = NStepWrapperState(env_state, observation_buffer)

            # Output: [flattened state history, current goal]
            state_hist = jnp.reshape(observation_buffer, (-1,))
            out_obs = jnp.concatenate([state_hist, next_goal], axis=0)
        else:
            # Original behavior
            observation_buffer = jnp.roll(observation_buffer, shift=-1, axis=0)
            observation_buffer = observation_buffer.at[-1].set(next_observation)
            state = NStepWrapperState(env_state, observation_buffer)
            out_obs = jnp.reshape(observation_buffer, (-1,))

        return out_obs, reward, absorbing, done, info, state


class VecEnv(BaseWrapper):
    """Vectorized environment wrapper using standard JAX vmap.

    Compatible with both JAX and Warp backends through MJX's unified interface.
    """

    def __init__(self, env):
        super().__init__(env)
        self._base_reset = jax.vmap(self.env.reset, in_axes=(0,))
        self._base_reset_to = jax.vmap(self.env.reset_to, in_axes=(0, 0))
        self._base_step = jax.vmap(self.env.step, in_axes=(0, 0))

    def reset(self, rng_key):
        """Reset all parallel environments.

        Args:
            rng_key: Array of shape (num_envs,) containing RNG keys for each environment

        Returns:
            Tuple of (observations, states) for all environments
        """
        return self._base_reset(rng_key)

    def reset_to(self, rng_key, traj_idx):
        """Reset all parallel environments to specific trajectories."""
        return self._base_reset_to(rng_key, traj_idx)

    def step(self, state, action):
        """Step all parallel environments.

        Args:
            state: Batched state from previous step/reset
            action: Batched actions of shape (num_envs, action_dim)

        Returns:
            Tuple of (obs, reward, absorbing, done, info, next_state) for all environments
        """
        return self._base_step(state, action)


class AutoResetWrapper(BaseWrapper):
    """Auto-reset done envs outside of JIT/vmap under the tuple-based MJX interface.

    - Apply after VecEnv for both backends.
    - Keeps Data creation out of transforms.
    - Expects the wrapped env to follow the standard MJX wrapper contract:
      ``reset -> (obs, state)`` and
      ``step -> (obs, reward, absorbing, done, info, state)``.

    Architecture (post-refactor):
        step() is decomposed into 4 pure functions for maintainability:
        1. _prepare_reset_keys() - RNG management and key splitting
        2. _compute_reset_candidates() - Build reset state for done envs
        3. _step_inner() - Execute env step under the standard tuple interface
        4. _apply_autoreset() - Conditional swap + carry reset + info update

    Done semantics:
        - done_out: returned to training (True when episode ended THIS step)
        - done_in: written to inner state (always False, ensures clean next step)
    """

    # =========================================================================
    # Carry reset policy: table-driven specification
    # =========================================================================
    # Each entry defines how a carry field should be reset when done=True.
    # Format: (field_name, reset_mode)
    #   - "tree_map": use jax.tree_util.tree_map(where_done, reset_val, cur_val)
    #   - "where_done": use where_done(reset_val, cur_val) directly (for arrays)
    #   - "constant_one": reset to ones_like(cur_val) where done
    #   - "field_by_field": iterate dataclass fields individually (for incompatible types)
    _CARRY_RESET_SPEC = (
        ("traj_state", "tree_map"),
        ("key", "where_done"),
        ("cur_step_in_episode", "constant_one"),
        ("last_action", "where_done"),
        ("domain_randomizer_state", "tree_map"),
        ("terrain_state", "tree_map"),
        ("observation_states", "field_by_field"),  # make_dataclass creates incompatible types
    )

    def __init__(self, env):
        super().__init__(env)
        self._info_key = "AutoResetWrapper"

    # =========================================================================
    # State utility methods
    # =========================================================================

    def _vmapped_split(self, keys):
        return jax.vmap(jax.random.split)(keys)

    def _unwrap_state(self, st):
        """Return inner MjxState and rewrap function (peel all wrapper layers)."""
        chain = []
        inner = st
        while hasattr(inner, "env_state") and not isinstance(inner, MjxState):
            chain.append(inner)
            inner = inner.env_state

        def rewrap(s):
            cur = s
            for wrapper in reversed(chain):
                cur = wrapper.replace(env_state=cur)
            return cur

        return inner, rewrap

    def _has_field(self, st, field_name: str) -> bool:
        return hasattr(st, "__dataclass_fields__") and field_name in st.__dataclass_fields__

    def _find_observation_buffer(self, st):
        if self._has_field(st, "observation_buffer"):
            return st.observation_buffer
        if self._has_field(st, "env_state"):
            return self._find_observation_buffer(st.env_state)
        return None

    def _replace_observation_buffer(self, st, new_buffer):
        if self._has_field(st, "observation_buffer"):
            return st.replace(observation_buffer=new_buffer)
        if self._has_field(st, "env_state"):
            return st.replace(env_state=self._replace_observation_buffer(st.env_state, new_buffer))
        return st

    # =========================================================================
    # reset()
    # =========================================================================

    def reset(self, rng_keys):
        """Initialize environment with autoreset RNG cache."""
        obs, state = self.env.reset(rng_keys)

        inner, rewrap = self._unwrap_state(state)

        # Derive per-env RNG cache; handle both batched and single-key inputs
        batch = inner.observation.shape[0]
        rk = rng_keys
        try:
            is_batched_keys = hasattr(rk, "shape") and rk.shape and (rk.shape[0] == batch)
        except Exception:
            is_batched_keys = False
        cached_rng = rk if is_batched_keys else jax.random.split(rk, batch)

        # Store reset RNG in additional_carry (NOT info) to avoid PPO batch reshaping issues
        new_carry = inner.additional_carry.replace(autoreset_rng=cached_rng)

        # Only store scalar metrics in info (safe to batch)
        info = dict(inner.info)
        info[f"{self._info_key}_done_count"] = jnp.zeros((batch,), dtype=jnp.int32)
        info["final_traj_no"] = jnp.zeros((batch,), dtype=jnp.int32)
        info["imitation_error_total"] = jnp.zeros((batch,), dtype=jnp.float32)

        new_inner = inner.replace(info=info, additional_carry=new_carry)
        new_state = rewrap(new_inner)
        return obs, new_state

    # =========================================================================
    # step() - decomposed into pure functions
    # =========================================================================

    def _prepare_reset_keys(self, inner):
        """Split RNG keys for reset candidates.

        Returns:
            rng_next: Updated RNG cache for next step
            reset_keys: Keys for building reset candidates
        """
        split = self._vmapped_split(inner.additional_carry.autoreset_rng)
        # jax.random.split returns (batch, 2) for KeyArray or (batch, 2, 2) for uint32
        rng_next, reset_keys = split[:, 0], split[:, 1]
        return rng_next, reset_keys

    def _compute_reset_candidates(self, reset_keys):
        """Build reset state candidates for done environments.

        Returns:
            reset_obs: Observations from fresh reset
            reset_inner: Inner MjxState from fresh reset
            reset_state: Full wrapped state from fresh reset
        """
        reset_obs, reset_state = self.env.reset(reset_keys)
        reset_inner, _ = self._unwrap_state(reset_state)
        return reset_obs, reset_inner, reset_state

    def _step_inner(self, state, action):
        """Execute environment step with normalized interface.

        Returns:
            obs, reward, absorbing, done, info, next_state, next_inner, rewrap_next
        """
        obs, reward, absorbing, done, info, next_state = self.env.step(state, action)
        next_inner, rewrap_next = self._unwrap_state(next_state)
        return obs, reward, absorbing, done, info, next_state, next_inner, rewrap_next

    def _make_where_done(self, done):
        """Create a where_done closure for conditional swapping.

        Args:
            done: Boolean mask of shape [num_envs]

        Returns:
            Closure that selects reset values where done=True
        """

        def where_done(reset_val, cur_val):
            """Select reset_val where done=True, cur_val where done=False."""
            if hasattr(cur_val, "shape") and cur_val.shape:
                if done.ndim > 0 and cur_val.shape[0] == done.shape[0]:
                    d = done
                    while d.ndim < cur_val.ndim:
                        d = d[..., None]
                    return jnp.where(d, reset_val, cur_val)
            return cur_val

        return where_done

    def _apply_carry_reset(self, cur_carry, reset_carry, where_done):
        """Apply table-driven carry reset policy.

        Uses _CARRY_RESET_SPEC to determine how each field should be reset.
        """
        new_carry = cur_carry

        for field_name, reset_mode in self._CARRY_RESET_SPEC:
            reset_val = getattr(reset_carry, field_name, None)
            cur_val = getattr(new_carry, field_name, None)

            if reset_val is None or cur_val is None:
                continue

            if reset_mode == "tree_map":
                new_val = jax.tree_util.tree_map(where_done, reset_val, cur_val)
            elif reset_mode == "where_done":
                new_val = where_done(reset_val, cur_val)
            elif reset_mode == "constant_one":
                one_like = jnp.ones_like(cur_val)
                new_val = where_done(one_like, cur_val)
            elif reset_mode == "field_by_field":
                # For dynamically-created dataclasses with incompatible types
                swapped_fields = {}
                for fname in cur_val.__dataclass_fields__.keys():
                    reset_field = getattr(reset_val, fname)
                    cur_field = getattr(cur_val, fname)
                    swapped_fields[fname] = jax.tree_util.tree_map(where_done, reset_field, cur_field)
                new_val = cur_val.replace(**swapped_fields)
            else:
                raise ValueError(f"Unknown reset_mode: {reset_mode}")

            new_carry = new_carry.replace(**{field_name: new_val})

        return new_carry

    def _extract_info_metrics(self, cur_carry, done):
        """Extract metrics for info dict before reset swap."""
        metrics = {}

        # Extract pre-reset trajectory ID
        traj_state = getattr(cur_carry, "traj_state", None)
        if traj_state is not None:
            metrics["final_traj_no"] = traj_state.traj_no
        else:
            metrics["final_traj_no"] = jnp.zeros(done.shape, dtype=jnp.int32)

        # Extract imitation error from reward_state
        reward_state = getattr(cur_carry, "reward_state", None)
        if reward_state is not None and hasattr(reward_state, "imitation_error_total"):
            metrics["imitation_error_total"] = reward_state.imitation_error_total
        else:
            metrics["imitation_error_total"] = jnp.zeros(done.shape, dtype=jnp.float32)

        return metrics

    def _apply_autoreset(
        self, done, inner, next_inner, reset_obs, reset_inner, reset_state, obs, next_state, rng_next, where_done
    ):
        """Apply autoreset: conditional swap + carry reset + info update.

        Args:
            done: Boolean mask for which envs terminated (done_out semantics)
            inner: Pre-step inner state (for info accumulation)
            next_inner: Post-step inner state
            reset_obs, reset_inner, reset_state: Reset candidates
            obs: Post-step observation
            next_state: Post-step wrapped state
            rng_next: Updated RNG cache
            where_done: Closure for conditional swapping

        Returns:
            new_obs: Swapped observation (reset obs where done)
            updated_state: State with autoreset applied
            next_info: Updated info dict
        """
        # Swap data and observations where done
        new_data = jax.tree_util.tree_map(where_done, reset_inner.data, next_inner.data)
        new_obs = where_done(reset_obs, obs)
        new_inner_obs = where_done(reset_inner.observation, next_inner.observation)

        # Build updated info dict
        next_info = dict(next_inner.info)
        next_info[f"{self._info_key}_done_count"] = (
            inner.info[f"{self._info_key}_done_count"] + done.astype(jnp.int32)
        )

        # Extract metrics before reset swap
        info_metrics = self._extract_info_metrics(next_inner.additional_carry, done)
        next_info.update(info_metrics)

        # Apply table-driven carry reset
        cur_carry = next_inner.additional_carry
        new_carry = self._apply_carry_reset(cur_carry, reset_inner.additional_carry, where_done)
        new_carry = new_carry.replace(autoreset_rng=rng_next)

        # done_in: Clear done flag for inner state (ensures clean next step)
        # done_out: The original 'done' is returned to training (handled in step())
        done_in = jnp.zeros_like(done, dtype=bool)

        _, rewrap_next = self._unwrap_state(next_state)
        updated_inner = next_inner.replace(
            data=new_data, observation=new_inner_obs, info=next_info, done=done_in, additional_carry=new_carry
        )
        updated_state = rewrap_next(updated_inner)

        # Handle observation buffer (from NStepWrapper)
        reset_obs_buffer = self._find_observation_buffer(reset_state)
        current_obs_buffer = self._find_observation_buffer(next_state)
        if reset_obs_buffer is not None and current_obs_buffer is not None:
            new_obs_buffer = where_done(reset_obs_buffer, current_obs_buffer)
            updated_state = self._replace_observation_buffer(updated_state, new_obs_buffer)

        return new_obs, updated_state, next_info

    def _step_with_autoreset(self, state, action):
        """Execute a step and split post-reset carry from pre-reset transition.

        ``next_state`` is the raw stepped state from the wrapped env. This is the
        state whose data/info/carry semantically belong to the transition that
        just happened. ``updated_state`` is the state after autoreset has swapped
        reset values into done environments. Rollout must continue from
        ``updated_state``, but per-step metrics must read from ``next_state``.
        """
        # 1. Prepare reset keys
        inner, _ = self._unwrap_state(state)
        rng_next, reset_keys = self._prepare_reset_keys(inner)

        # 2. Build reset candidates (computed every step for JIT compatibility)
        reset_obs, reset_inner, reset_state = self._compute_reset_candidates(reset_keys)

        # 3. Step inner env
        obs, reward, absorbing, done, _info, next_state, next_inner, _ = self._step_inner(state, action)

        # done_out: This is what we return to training (episode ended THIS step)
        done_out = done

        # 4. Apply autoreset with conditional swapping
        where_done = self._make_where_done(done_out)
        new_obs, updated_state, next_info = self._apply_autoreset(
            done_out, inner, next_inner, reset_obs, reset_inner, reset_state, obs, next_state, rng_next, where_done
        )

        return new_obs, reward, absorbing, done_out, next_info, updated_state, next_state, obs

    def step(self, state, action):
        """Execute step with automatic reset for done environments.

        Done semantics (explicit):
            - done_out: Returned to training, True when episode ended at this step
            - done_in: Written to inner state, always False for clean next step
        """
        new_obs, reward, absorbing, done_out, next_info, updated_state, _, _ = self._step_with_autoreset(state, action)
        return new_obs, reward, absorbing, done_out, next_info, updated_state

    def step_with_transition(self, state, action):
        """Execute step and expose both rollout-carry and pre-reset transition.

        Returned values intentionally have mixed semantics:
            - ``new_obs`` and ``updated_state`` are post-reset values for the next
              rollout step.
            - ``reward`` / ``absorbing`` / ``done`` / ``info`` describe the
              transition that just happened.
            - ``transition_state`` is the pre-reset state for the transition that
              just happened.
            - ``transition_obs`` is the pre-reset next observation for value
              bootstrap on truncation.

        The split only matters on done steps. On non-terminal steps,
        ``updated_state`` and ``transition_state`` are equivalent.
        """
        (
            new_obs,
            reward,
            absorbing,
            done_out,
            next_info,
            updated_state,
            transition_state,
            transition_obs,
        ) = self._step_with_autoreset(state, action)
        return new_obs, reward, absorbing, done_out, next_info, updated_state, transition_state, transition_obs


@struct.dataclass
class NormalizeVecRewEnvState(BaseWrapperState):
    env_state: MjxState
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float


class NormalizeVecReward(BaseWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key):
        obs, state = self.env.reset(key)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, state, action):
        next_observation, reward, absorbing, done, info, env_state = self.env.step(state.env_state, action)

        state, normalized_reward = self._normalize_step_output(state, next_observation, reward, done, env_state)

        return next_observation, normalized_reward, absorbing, done, info, state

    def step_with_transition(self, state, action):
        """Step inner env and preserve transition/carry split through normalization.

        Reward normalization updates exactly once from the transition reward and
        then wraps both returned states with the same running statistics. This
        keeps the outer state consistent while still exposing the inner
        pre-reset/post-reset distinction from autoreset wrappers below.
        """
        (
            next_observation,
            reward,
            absorbing,
            done,
            info,
            env_state,
            transition_env_state,
            transition_observation,
        ) = self.env.step_with_transition(state.env_state, action)

        next_state, normalized_reward = self._normalize_step_output(state, next_observation, reward, done, env_state)
        transition_state = next_state.replace(env_state=transition_env_state)
        return next_observation, normalized_reward, absorbing, done, info, next_state, transition_state, transition_observation

    def _normalize_step_output(self, state, next_observation, reward, done, env_state):
        """Update running reward stats once and reuse them for carry/transition states."""

        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = next_observation.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count

        next_state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )

        normalized_reward = reward / jnp.sqrt(next_state.var + 1e-8)
        return next_state, normalized_reward
