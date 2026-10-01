"""PPO rollout numerical-fault aggregation tests."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.algorithms.ppo.numerics import summarize_numerical_transitions


class _TrajectoryBatch(NamedTuple):
    reward: jax.Array
    info: dict[str, jax.Array]


def test_numerical_diagnostics_count_reasons_and_offending_trajectories():
    trajectory_ids = jnp.asarray(
        [[0, 1, 2, 3], [0, 1, 2, 3]],
        dtype=jnp.int32,
    )
    observation = jnp.asarray([[True, False, False, False], [True, False, False, False]])
    qpos = jnp.asarray([[False, True, False, False], [False, False, False, False]])
    qvel = jnp.asarray([[False, False, True, False], [False, False, False, False]])
    reward = jnp.asarray([[False, True, False, False], [False, False, False, False]])
    reward_info = jnp.asarray([[False, False, False, False], [False, False, True, False]])
    transition = observation | qpos | qvel | reward | reward_info
    transition = transition.at[1, 3].set(True)
    batch = _TrajectoryBatch(
        reward=jnp.zeros((2, 4), dtype=jnp.float32),
        info={
            "final_traj_no": trajectory_ids,
            "numerics_nonfinite_observation": observation,
            "numerics_nonfinite_qpos": qpos,
            "numerics_nonfinite_qvel": qvel,
            "numerics_nonfinite_reward": reward,
            "numerics_nonfinite_reward_info": reward_info,
            "numerics_nonfinite_transition": transition,
        },
    )

    eager = summarize_numerical_transitions(batch, {"n_trajectories": 4})
    compiled = jax.jit(
        lambda rollout: summarize_numerical_transitions(
            rollout,
            {"n_trajectories": 4},
        )
    )(batch)

    for compiled_leaf, eager_leaf in zip(
        jax.tree.leaves(compiled),
        jax.tree.leaves(eager),
        strict=True,
    ):
        np.testing.assert_allclose(compiled_leaf, eager_leaf)
    assert float(eager.invalid_transition_count) == 6.0
    assert float(eager.invalid_transition_rate) == 0.75
    np.testing.assert_array_equal(eager.reason_counts, [2.0, 1.0, 1.0, 1.0, 1.0])
    assert int(eager.offending_trajectory_count) == 4
    np.testing.assert_array_equal(eager.topk_trajectory_counts, [2.0, 2.0, 1.0])
    assert set(np.asarray(eager.topk_trajectory_ids[:2]).tolist()) == {0, 2}
    assert int(eager.topk_trajectory_ids[2]) in {1, 3}


def test_numerical_diagnostics_are_zero_without_wrapper_flags():
    batch = _TrajectoryBatch(
        reward=jnp.zeros((2, 3), dtype=jnp.float32),
        info={},
    )

    diagnostics = summarize_numerical_transitions(batch, {})

    assert float(diagnostics.invalid_transition_count) == 0.0
    assert float(diagnostics.invalid_transition_rate) == 0.0
    np.testing.assert_array_equal(diagnostics.reason_counts, np.zeros(5))
    assert int(diagnostics.offending_trajectory_count) == 0
    np.testing.assert_array_equal(diagnostics.topk_trajectory_ids, [-1, -1, -1])
    np.testing.assert_array_equal(diagnostics.topk_trajectory_counts, [0.0, 0.0, 0.0])


def test_missing_trajectory_identity_is_not_misattributed():
    batch = _TrajectoryBatch(
        reward=jnp.zeros((1, 2), dtype=jnp.float32),
        info={"numerics_nonfinite_transition": jnp.asarray([[True, False]])},
    )

    diagnostics = summarize_numerical_transitions(
        batch,
        {"n_trajectories": 2},
    )

    assert float(diagnostics.invalid_transition_count) == 1.0
    assert int(diagnostics.offending_trajectory_count) == 0
    np.testing.assert_array_equal(diagnostics.topk_trajectory_ids, [-1, -1, -1])
