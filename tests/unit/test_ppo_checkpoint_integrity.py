"""Tests for fail-closed PPO checkpoint integrity checks."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from musclemimic.algorithms.ppo import checkpoint
from musclemimic.algorithms.ppo.checkpoint_integrity import (
    PPOCheckpointIntegrityError,
    validate_ppo_checkpoint_train_state,
)


def _state(
    *,
    step: int = 12,
    last_finite: bool = True,
    notfinite_count: int = 0,
    total_notfinite: int = 0,
    counts: tuple[int, ...] = (12,),
):
    return {
        "step": jnp.asarray(step, dtype=jnp.int32),
        "params": {},
        "run_stats": {},
        "opt_state": {
            "last_finite": jnp.asarray(last_finite),
            "notfinite_count": jnp.asarray(notfinite_count, dtype=jnp.int32),
            "total_notfinite": jnp.asarray(total_notfinite, dtype=jnp.int32),
            "inner_state": tuple({"count": jnp.asarray(count, dtype=jnp.int32)} for count in counts),
        },
    }


def test_accepts_clean_optimizer_state():
    validate_ppo_checkpoint_train_state(_state(), context="test")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"last_finite": False}, "last_finite.*False"),
        ({"notfinite_count": 1}, "notfinite_count.*=1"),
        ({"total_notfinite": 1}, "total_notfinite.*=1"),
    ],
)
def test_rejects_apply_if_finite_failure_state(overrides, message):
    with pytest.raises(PPOCheckpointIntegrityError, match=message):
        validate_ppo_checkpoint_train_state(_state(**overrides), context="test")


def test_rejects_nonfinite_optimizer_counter():
    state = _state()
    state["opt_state"]["inner_state"] = ({"count": jnp.asarray(jnp.inf)},)

    with pytest.raises(PPOCheckpointIntegrityError, match="contains NaN or infinity"):
        validate_ppo_checkpoint_train_state(state, context="test")


@pytest.mark.parametrize("invalid", [jnp.nan, jnp.inf, -jnp.inf])
def test_rejects_nonfinite_parameter_or_optimizer_leaves(invalid):
    for field, value in (
        ("params", {"actor": {"kernel": jnp.asarray([1.0, invalid])}}),
        ("opt_state", {"moments": jnp.asarray([0.0, invalid])}),
    ):
        state = _state()
        if field == "opt_state":
            state[field]["inner_state"] = (
                {
                    "count": jnp.asarray(12, dtype=jnp.int32),
                    **value,
                },
            )
        else:
            state[field] = value

        with pytest.raises(PPOCheckpointIntegrityError, match="contains NaN or infinity"):
            validate_ppo_checkpoint_train_state(state, context="test")


def test_rejects_stalled_inner_optimizer_progress():
    with pytest.raises(PPOCheckpointIntegrityError, match=r"outer step=12.*minimum inner optimizer count=11"):
        validate_ppo_checkpoint_train_state(_state(counts=(11, 11)), context="test")


def test_accepts_lr_schedule_reset_progress():
    validate_ppo_checkpoint_train_state(_state(step=12, counts=(12, 112)), context="test")


def test_accepts_legacy_state_without_named_optimizer_counters():
    validate_ppo_checkpoint_train_state(
        {"step": jnp.asarray(12), "opt_state": {"moments": {}}},
        context="test",
    )


def test_resume_rejects_poisoned_state_before_typed_restore(monkeypatch, tmp_path):
    manager = SimpleNamespace(
        load_checkpoint=lambda *_args, **_kwargs: (({}, _state(last_finite=False)), SimpleNamespace()),
        close=lambda: None,
    )
    monkeypatch.setattr(checkpoint, "UnifiedCheckpointManager", lambda **_kwargs: manager)
    agent_conf = SimpleNamespace(tx=object())

    with pytest.raises(PPOCheckpointIntegrityError, match=r"last_finite.*False"):
        checkpoint.load_checkpoint_for_resume(str(tmp_path / "checkpoint_12"), agent_conf)
