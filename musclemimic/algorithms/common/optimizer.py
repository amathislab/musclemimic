"""
Optimizer construction and learning rate schedules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import optax
import optax.contrib

LRSchedule = Callable[[int], jnp.ndarray]


def linear_lr_schedule(
    count: int,
    num_minibatches: int,
    update_epochs: int,
    lr: float,
    num_updates: int,
) -> jnp.ndarray:
    """
    Linear decay from lr to 0 over training.

    Args:
        count: optimizer step counter
        num_minibatches: minibatches per update
        update_epochs: epochs per update
        lr: base learning rate
        num_updates: total updates

    Returns:
        learning rate at current step
    """
    steps_per_update = num_minibatches * update_epochs
    frac = 1.0 - (count // steps_per_update) / num_updates
    return lr * frac


def warmup_cosine_lr_schedule(
    count: int,
    num_minibatches: int,
    update_epochs: int,
    lr: float,
    num_updates: int,
    warmup_steps: int | None = None,
    min_lr_ratio: float = 0.0,
) -> jnp.ndarray:
    """
    Linear warmup followed by cosine annealing to min_lr.

    Args:
        count: optimizer step counter
        num_minibatches: minibatches per update
        update_epochs: epochs per update
        lr: base learning rate
        num_updates: total updates
        warmup_steps: warmup duration in updates (default: 10% of total)
        min_lr_ratio: final lr as fraction of base lr

    Returns:
        learning rate at current step
    """
    steps_per_update = num_minibatches * update_epochs
    current_update = count // steps_per_update
    schedule = _build_warmup_cosine_schedule(lr, num_updates, warmup_steps, min_lr_ratio)
    return schedule(current_update)


def warmup_hold_cosine_lr_schedule(
    count: int,
    num_minibatches: int,
    update_epochs: int,
    lr: float,
    num_updates: int,
    warmup_steps: int | None = None,
    hold_fraction: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> jnp.ndarray:
    """Warm up, hold the peak rate, then cosine-decay by update index."""
    steps_per_update = num_minibatches * update_epochs
    current_update = count // steps_per_update
    schedule = _build_warmup_hold_cosine_schedule(
        lr,
        num_updates,
        warmup_steps,
        hold_fraction,
        min_lr_ratio,
    )
    return schedule(current_update)


def _build_warmup_cosine_schedule(
    lr: float,
    num_updates: int,
    warmup_steps: int | None = None,
    min_lr_ratio: float = 0.0,
) -> LRSchedule:
    """Build an update-indexed warmup+cosine schedule backed by optax."""
    if warmup_steps is None:
        warmup_steps = max(1, int(0.1 * num_updates))
    else:
        warmup_steps = max(0, int(warmup_steps))

    # optax requires decay_steps > warmup_steps. When warmup is as long as or
    # longer than the planned training horizon, keep the previous behavior:
    # finish the warmup, hit the peak once, then decay to the minimum in 1 step.
    decay_steps = max(int(num_updates), warmup_steps + 1)

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=lr * min_lr_ratio,
    )


def _build_warmup_hold_cosine_schedule(
    lr: float,
    num_updates: int,
    warmup_steps: int | None = None,
    hold_fraction: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> LRSchedule:
    """Build an update-indexed warmup, peak hold, and cosine schedule."""
    if num_updates < 1:
        raise ValueError("num_updates must be positive")
    if not 0.0 <= hold_fraction <= 1.0:
        raise ValueError("hold_fraction must be between 0 and 1")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    if warmup_steps is None:
        warmup_steps = max(1, int(0.1 * num_updates))
    else:
        warmup_steps = max(0, int(warmup_steps))
    warmup_steps = min(warmup_steps, int(num_updates))
    hold_end = max(warmup_steps, int(hold_fraction * num_updates))
    hold_end = min(hold_end, int(num_updates))
    decay_steps = max(int(num_updates) - hold_end, 1)

    def schedule(update_index: int) -> jnp.ndarray:
        update = jnp.asarray(update_index, dtype=jnp.float32)
        peak = jnp.asarray(lr, dtype=jnp.float32)
        minimum = peak * min_lr_ratio
        warmup_rate = peak * update / max(warmup_steps, 1)
        cosine_progress = jnp.clip((update - hold_end) / decay_steps, 0.0, 1.0)
        cosine_rate = minimum + 0.5 * (peak - minimum) * (
            1.0 + jnp.cos(jnp.pi * cosine_progress)
        )
        return jnp.where(
            update < warmup_steps,
            warmup_rate,
            jnp.where(update < hold_end, peak, cosine_rate),
        )

    return schedule


def build_optimizer(
    optimizer_type: str,
    learning_rate: float | LRSchedule,
    weight_decay: float,
    max_grad_norm: float,
    muon_config: dict | None = None,
) -> optax.GradientTransformation:
    """
    Build optimizer by type with gradient clipping.

    Args:
        optimizer_type: "adamw" or "muon"
        learning_rate: constant or scheduled learning rate
        weight_decay: weight decay for AdamW (also Muon's adam_weight_decay)
        max_grad_norm: gradient clipping threshold
        muon_config: optional Muon params (beta, nesterov, ns_steps, etc.)

    Returns:
        optax gradient transformation (clipping + optimizer)
    """
    if optimizer_type == "adamw":
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eps=1e-5,
        )
    elif optimizer_type == "muon":
        cfg = muon_config or {}
        optimizer = optax.contrib.muon(
            learning_rate=learning_rate,
            beta=cfg.get("beta", 0.95),
            nesterov=cfg.get("nesterov", True),
            ns_steps=cfg.get("ns_steps", 5),
            weight_decay=cfg.get("weight_decay", 0.0),
            adam_weight_decay=cfg.get("adam_weight_decay", 0.0),
            adaptive=cfg.get("adaptive", False),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Use 'adamw' or 'muon'")

    return optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer)


def get_optimizer(exp: Any) -> optax.GradientTransformation:
    """
    Build optax optimizer from experiment config.

    Schedule modes (exp.schedule):
        - "fixed": constant lr
        - "adaptive": constant lr, runtime adjustment via gradient scaling
        - anneal_lr=True: linear, warmup_cosine, or warmup_hold_cosine decay

    Args:
        exp: experiment config with training hyperparameters

    Returns:
        optax gradient transformation
    """
    schedule = getattr(exp, "schedule", None) or exp.get("schedule", "fixed")

    # Helper to get config value (supports both DictConfig and namespace)
    def _get(key, default=None):
        if hasattr(exp, key):
            return getattr(exp, key)
        return exp.get(key, default) if hasattr(exp, "get") else default

    lr = _get("lr")
    anneal_lr = _get("anneal_lr", False)
    num_minibatches = _get("num_minibatches")
    update_epochs = _get("update_epochs")
    num_updates = _get("num_updates")

    # Determine learning rate (constant or scheduled)
    if schedule == "adaptive":
        print("[optimizer] using adaptive lr (kl-based adjustment)")
        learning_rate: float | LRSchedule = lr
    elif anneal_lr:
        lr_type = _get("lr_schedule_type", "linear")
        warmup = _get("warmup_steps", None)
        min_ratio = _get("min_lr_ratio", 0.0)
        steps_per_update = num_minibatches * update_epochs

        if lr_type == "warmup_cosine":
            schedule = _build_warmup_cosine_schedule(lr, num_updates, warmup, min_ratio)

            def learning_rate(c):
                return schedule(c // steps_per_update)
        elif lr_type == "warmup_hold_cosine":
            hold_fraction = _get("hold_fraction", 0.5)
            schedule = _build_warmup_hold_cosine_schedule(
                lr,
                num_updates,
                warmup,
                hold_fraction,
                min_ratio,
            )

            def learning_rate(c):
                return schedule(c // steps_per_update)
        else:
            def learning_rate(c):
                return linear_lr_schedule(
                    c, num_minibatches, update_epochs, lr, num_updates,
                )
    else:
        learning_rate = lr

    optimizer_type = _get("optimizer_type", "adamw")
    muon_cfg = _get("muon_config") if optimizer_type == "muon" else None

    tx = build_optimizer(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=_get("weight_decay", 0.0),
        max_grad_norm=_get("max_grad_norm", 1.0),
        muon_config=muon_cfg,
    )
    return optax.apply_if_finite(tx, max_consecutive_errors=10_000_000)
