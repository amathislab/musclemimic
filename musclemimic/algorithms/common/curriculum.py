from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class CurriculumParams:
    init_ema_val: jnp.ndarray
    low_band: jnp.ndarray
    high_band: jnp.ndarray
    adjust_factor: jnp.ndarray
    consecutive_k: jnp.ndarray
    min_threshold: jnp.ndarray
    ema_alpha: jnp.ndarray


@struct.dataclass
class CurriculumState:
    current_threshold: jnp.ndarray
    ema_rate: jnp.ndarray
    consecutive_count: jnp.ndarray
    last_direction: jnp.ndarray
    # threshold can never exceed initial value
    initial_threshold: jnp.ndarray


def _get_cfg_value(cfg, name, default=None):
    try:
        return cfg.get(name, default)
    except AttributeError:
        return getattr(cfg, name, default)


def validate_curriculum_config(cfg) -> None:
    adjust_factor = float(_get_cfg_value(cfg, "adjust_factor"))
    init_threshold = _get_cfg_value(cfg, "init_threshold")
    min_threshold = float(_get_cfg_value(cfg, "min_threshold"))
    low_band = float(_get_cfg_value(cfg, "low_band"))
    high_band = float(_get_cfg_value(cfg, "high_band"))

    assert 0.0 < adjust_factor < 1.0, "adjust_factor must be in (0, 1)"
    assert init_threshold is not None, "init_threshold is required when adaptive termination is enabled"
    assert min_threshold <= float(init_threshold), "init_threshold must be >= min_threshold"
    assert low_band < high_band, "low_band must be less than high_band"

    ema_alpha = float(_get_cfg_value(cfg, "ema_alpha"))
    assert 0.0 < ema_alpha <= 1.0, "ema_alpha must be in (0, 1]"

    consecutive_k = int(_get_cfg_value(cfg, "consecutive_k"))
    assert consecutive_k >= 1, "consecutive_k must be >= 1"


def create_curriculum_params(cfg) -> CurriculumParams:
    return CurriculumParams(
        init_ema_val=jnp.asarray(_get_cfg_value(cfg, "init_ema_val", 1.0), dtype=jnp.float32),
        low_band=jnp.asarray(_get_cfg_value(cfg, "low_band"), dtype=jnp.float32),
        high_band=jnp.asarray(_get_cfg_value(cfg, "high_band"), dtype=jnp.float32),
        adjust_factor=jnp.asarray(_get_cfg_value(cfg, "adjust_factor"), dtype=jnp.float32),
        consecutive_k=jnp.asarray(int(_get_cfg_value(cfg, "consecutive_k")), dtype=jnp.int32),
        min_threshold=jnp.asarray(_get_cfg_value(cfg, "min_threshold"), dtype=jnp.float32),
        ema_alpha=jnp.asarray(_get_cfg_value(cfg, "ema_alpha"), dtype=jnp.float32),
    )


def create_curriculum_state(init_threshold, init_ema_val) -> CurriculumState:
    return CurriculumState(
        current_threshold=jnp.asarray(init_threshold, dtype=jnp.float32),
        ema_rate=jnp.asarray(init_ema_val, dtype=jnp.float32),
        consecutive_count=jnp.asarray(0, dtype=jnp.int32),
        last_direction=jnp.asarray(0, dtype=jnp.int32),
        initial_threshold=jnp.asarray(init_threshold, dtype=jnp.float32),
    )


def compute_early_termination_stats(
    done: jnp.ndarray,
    absorbing: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    done_f = done.astype(jnp.float32)
    absorbing_f = absorbing.astype(jnp.float32)
    num_done = jnp.sum(done_f)
    early_count = jnp.sum(done_f * absorbing_f)
    rate = jnp.where(
        num_done > 0,
        early_count / num_done,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    return early_count, rate


def update_curriculum_state(
    state: CurriculumState,
    early_rate: jnp.ndarray,
    params: CurriculumParams,
) -> tuple[CurriculumState, jnp.ndarray, jnp.ndarray]:
    early_rate = early_rate.astype(jnp.float32)
    ema_rate = (1.0 - params.ema_alpha) * state.ema_rate + params.ema_alpha * early_rate

    direction = jnp.where(
        ema_rate < params.low_band,
        -jnp.ones((), dtype=jnp.int32),
        jnp.where(
            ema_rate > params.high_band,
            jnp.ones((), dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        ),
    )

    same_dir = direction == state.last_direction
    non_zero = direction != 0
    consecutive = jnp.where(
        non_zero & same_dir,
        state.consecutive_count + jnp.ones((), dtype=jnp.int32),
        jnp.where(non_zero, jnp.ones((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.int32)),
    )

    should_adjust = non_zero & (consecutive >= params.consecutive_k)
    decreased = state.current_threshold * params.adjust_factor
    increased = state.current_threshold / params.adjust_factor
    new_threshold = jnp.where(
        should_adjust & (direction < 0),
        decreased,
        jnp.where(should_adjust & (direction > 0), increased, state.current_threshold),
    )
    # Ratchet: never exceed initial threshold (one-way curriculum)
    new_threshold = jnp.clip(new_threshold, params.min_threshold, state.initial_threshold)

    consecutive = jnp.where(should_adjust | (direction == 0), jnp.zeros((), dtype=jnp.int32), consecutive)
    last_direction = jnp.where(direction == 0, jnp.zeros((), dtype=jnp.int32), direction)

    new_state = state.replace(
        current_threshold=new_threshold,
        ema_rate=ema_rate,
        consecutive_count=consecutive,
        last_direction=last_direction,
    )
    return new_state, direction, should_adjust.astype(jnp.float32)


# =============================================================================
# Reward Curriculum
# =============================================================================


@struct.dataclass
class RewardCurriculumState:
    """State for adaptive reward curriculum based on termination rate threshold.

    This curriculum increases qvel_w_sum and root_vel_w_sum when termination rate
    stays below a threshold for K consecutive updates, indicating the policy is
    stable enough to handle stricter velocity tracking.
    """

    qvel_w_sum: jnp.ndarray  # Current qvel weight (scalar)
    root_vel_w_sum: jnp.ndarray  # Current root_vel weight (scalar)
    ema_term_rate: jnp.ndarray  # EMA of termination rate (scalar)
    consecutive_below: jnp.ndarray  # Consecutive updates below threshold (int32)
    initialized: jnp.ndarray  # bool, shape=()


def create_reward_curriculum_state(cfg) -> RewardCurriculumState:
    """Initialize reward curriculum state.

    Args:
        cfg: Config dict with keys:
            - qvel_w_sum_init: Initial qvel weight (default 0.2)
            - root_vel_w_sum_init: Initial root_vel weight (default 0.2)

    Returns:
        Initialized RewardCurriculumState
    """
    return RewardCurriculumState(
        qvel_w_sum=jnp.array(_get_cfg_value(cfg, "qvel_w_sum_init", 0.2), dtype=jnp.float32),
        root_vel_w_sum=jnp.array(_get_cfg_value(cfg, "root_vel_w_sum_init", 0.2), dtype=jnp.float32),
        ema_term_rate=jnp.array(0.0, dtype=jnp.float32),
        consecutive_below=jnp.array(0, dtype=jnp.int32),
        initialized=jnp.array(False, dtype=jnp.bool_),
    )


def update_reward_curriculum_state(
    state: RewardCurriculumState,
    term_rate: jnp.ndarray,
    alpha: float,
    eta: float,
    qvel_w_max: float,
    root_vel_w_max: float,
    term_rate_threshold: float,
    consecutive_k: int,
) -> RewardCurriculumState:
    """Update reward curriculum based on termination rate threshold.

    When termination rate stays below threshold for K consecutive updates,
    increase both qvel_w_sum and root_vel_w_sum. This indicates the policy
    is stable enough to handle stricter velocity tracking.

    Args:
        state: Current reward curriculum state
        term_rate: Current termination rate
        alpha: EMA smoothing factor (Python float, static)
        eta: Multiplicative growth rate for weights (Python float, static)
        qvel_w_max: Maximum value for qvel_w_sum (Python float, static)
        root_vel_w_max: Maximum value for root_vel_w_sum (Python float, static)
        term_rate_threshold: Threshold below which we count consecutive updates
        consecutive_k: Number of consecutive updates below threshold to trigger increase

    Returns:
        Updated RewardCurriculumState
    """
    tr = jnp.asarray(term_rate, dtype=jnp.float32)

    def _init_branch(s):
        # First observation: initialize EMA with current term_rate
        return s.replace(
            ema_term_rate=tr,
            initialized=jnp.array(True, dtype=jnp.bool_),
        )

    def _update_branch(s):
        # Update EMA
        ema_new = (1.0 - alpha) * s.ema_term_rate + alpha * tr

        # Check if below threshold
        is_below = ema_new < term_rate_threshold

        # Update consecutive counter
        new_consecutive = jnp.where(
            is_below,
            s.consecutive_below + 1,
            jnp.array(0, dtype=jnp.int32),
        )

        # Check if we should increase weights (K consecutive below threshold)
        should_increase = new_consecutive >= consecutive_k

        # Increase both weights if condition met
        new_qvel_w = jnp.where(
            should_increase,
            jnp.minimum(s.qvel_w_sum * (1.0 + eta), qvel_w_max),
            s.qvel_w_sum,
        )
        new_root_vel_w = jnp.where(
            should_increase,
            jnp.minimum(s.root_vel_w_sum * (1.0 + eta), root_vel_w_max),
            s.root_vel_w_sum,
        )

        # Reset counter after weight increase
        final_consecutive = jnp.where(
            should_increase,
            jnp.array(0, dtype=jnp.int32),
            new_consecutive,
        )

        return s.replace(
            qvel_w_sum=new_qvel_w,
            root_vel_w_sum=new_root_vel_w,
            ema_term_rate=ema_new,
            consecutive_below=final_consecutive,
        )

    return jax.lax.cond(
        state.initialized,
        _update_branch,
        _init_branch,
        state,
    )
