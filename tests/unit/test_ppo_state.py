"""Tests for the explicit PPO runtime-state boundary."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from musclemimic.algorithms.common.curriculum import (
    create_curriculum_state,
    create_reward_curriculum_state,
)
from musclemimic.algorithms.common.distributed import make_data_mesh
from musclemimic.algorithms.ppo.checkpoint import build_runtime_state
from musclemimic.algorithms.ppo.state import PPORuntimeState


@struct.dataclass
class _TestTrainState:
    step: jax.Array


def _runtime_state() -> PPORuntimeState:
    return PPORuntimeState(
        train_state=_TestTrainState(step=jnp.asarray(3, dtype=jnp.int32)),
        env_state={"position": jnp.asarray([1.0, 2.0], dtype=jnp.float32)},
        last_obs=jnp.asarray([[4.0, 5.0]], dtype=jnp.float32),
        rng=jnp.asarray([6, 7], dtype=jnp.uint32),
        lr=jnp.asarray(3e-4, dtype=jnp.float32),
        val_rng=jnp.asarray([8, 9], dtype=jnp.uint32),
        curriculum_state=create_curriculum_state(0.3, 1.0),
        reward_curriculum_state=create_reward_curriculum_state({}),
    )


def _advance_runtime_state(state: PPORuntimeState) -> PPORuntimeState:
    return state.replace(
        train_state=state.train_state.replace(step=state.train_state.step + 1),
        env_state={"position": state.env_state["position"] + 1.0},
        last_obs=state.last_obs + 1.0,
        rng=state.rng + jnp.asarray([1, 2], dtype=jnp.uint32),
        lr=state.lr * 0.5,
        val_rng=state.val_rng + jnp.asarray([2, 1], dtype=jnp.uint32),
        curriculum_state=state.curriculum_state.replace(
            current_threshold=state.curriculum_state.current_threshold * 0.9,
        ),
        reward_curriculum_state=state.reward_curriculum_state.replace(
            qvel_w_sum=state.reward_curriculum_state.qvel_w_sum + 0.1,
        ),
    )


def _assert_trees_equal(actual, expected) -> None:
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    assert actual_tree == expected_tree
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_leaf), np.asarray(expected_leaf))


def test_runtime_state_has_equivalent_eager_and_jit_updates():
    state = _runtime_state()

    eager = _advance_runtime_state(state)
    compiled = jax.jit(_advance_runtime_state)(state)

    _assert_trees_equal(compiled, eager)


def test_runtime_state_is_a_valid_shard_map_carry():
    state = _runtime_state()
    mesh = make_data_mesh(1)
    replicated = jax.sharding.PartitionSpec()
    specs = jax.tree.map(
        lambda _value: replicated,
        state,
    )
    updated = jax.jit(
        jax.shard_map(
            _advance_runtime_state,
            mesh=mesh,
            in_specs=(specs,),
            out_specs=specs,
        )
    )(state)
    expected = _advance_runtime_state(state)

    _assert_trees_equal(updated, expected)


def test_legacy_v1_checkpoint_runtime_state_round_trips_without_schema_changes():
    state = _runtime_state()
    legacy_v1 = {
        "env_state": state.env_state,
        "last_obs": state.last_obs,
        "rng": state.rng,
        "lr": state.lr,
        "val_rng": state.val_rng,
        "curriculum_state": state.curriculum_state,
        "reward_curriculum_state": state.reward_curriculum_state,
    }

    restored = PPORuntimeState.from_checkpoint_state(state.train_state, legacy_v1)
    checkpoint_builder_output = build_runtime_state(
        restored.env_state,
        restored.last_obs,
        restored.rng,
        restored.lr,
        restored.val_rng,
        restored.curriculum_state,
        restored.reward_curriculum_state,
    )

    assert tuple(restored.to_checkpoint_state()) == tuple(legacy_v1)
    _assert_trees_equal(restored.to_checkpoint_state(), legacy_v1)
    _assert_trees_equal(restored.to_checkpoint_state(), checkpoint_builder_output)


def test_checkpoint_runtime_state_does_not_guess_missing_fields():
    state = _runtime_state()
    incomplete = state.to_checkpoint_state()
    del incomplete["rng"]

    with pytest.raises(KeyError, match="rng"):
        PPORuntimeState.from_checkpoint_state(state.train_state, incomplete)
