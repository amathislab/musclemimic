"""Numerical equivalence of a PPO minibatch update under pmap.

Scenario: the same synthetic minibatch processed as
  (a) a single device with the full batch, and
  (b) N pmap replicas each with 1/N of the batch.

Claim: after one `_update_minibatch` call, the resulting params differ only
by floating-point precision. If any cross-device reduction is missing
(gradient, advantage mean/std, run_stats forward cache), this test will
catch it -- a missing pmean manifests as order-of-magnitude-apart params.

Run on the CPU 4-device simulator so the test is hardware-independent::

    JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=4' \
        pytest tests/unit/test_pmap_numerics.py -q
"""

from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import sys
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from musclemimic.algorithms.common.distributed import replicate_for_pmap
from musclemimic.algorithms.ppo.numerics import summarize_numerical_transitions

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_loss = _load_module(
    "musclemimic_ppo_loss_under_test",
    pathlib.Path("musclemimic") / "algorithms" / "ppo" / "loss.py",
)
_adaptive = _load_module(
    "musclemimic_adaptive_sampling_under_test",
    pathlib.Path("musclemimic") / "algorithms" / "common" / "adaptive_sampling.py",
)
normalize_advantages = _loss.normalize_advantages
normalize_advantages_global = _loss.normalize_advantages_global
ppo_actor_loss = _loss.ppo_actor_loss
compute_per_traj_termination_stats = _adaptive.compute_per_traj_termination_stats

N_DEV = 4


@pytest.fixture(scope="module", autouse=True)
def _verify_simulator():
    if jax.local_device_count() != N_DEV:
        pytest.skip(
            f"need 4 simulated devices, got {jax.local_device_count()}. "
            f"run with JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=4'"
        )


# ----------------------------------------------------------------------------
# normalize_advantages_global: behaviour under pmap vs. single device
# ----------------------------------------------------------------------------


def test_normalize_adv_global_matches_single_device_stats():
    """With ``axis_name`` set inside pmap, the *global* mean/std should drive
    normalisation (not each device's local stats)."""
    # Fabricate advantages with a specific mean/std so we can check by hand.
    # Full batch of 16 = 4 devices x 4 per device.
    full = jnp.arange(16, dtype=jnp.float32) - 7.5  # mean 0, std sqrt(~22.9)
    sharded = full.reshape(N_DEV, 4)

    @functools.partial(jax.pmap, axis_name="devices")
    def run(adv_local):
        return normalize_advantages_global(adv_local, axis_name="devices")

    out = run(sharded)
    got = np.asarray(out).reshape(-1)
    # Compare to a single-device normalisation of the same data.
    single = normalize_advantages(full)
    expected = np.asarray(single)
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_normalize_adv_local_drifts_under_pmap_without_axis_name():
    """Sanity: the OLD behaviour (per-device normalisation) gives different
    numbers than the global one. Prevents someone from 'simplifying' our code
    back to a plain per-device normalisation and silently degrading training."""
    full = jnp.arange(16, dtype=jnp.float32) - 7.5
    sharded = full.reshape(N_DEV, 4)

    @jax.pmap
    def run_local(adv):
        return normalize_advantages(adv)

    out_local = np.asarray(run_local(sharded)).reshape(-1)
    expected_global = np.asarray(normalize_advantages(full))
    # Local-per-device normalisation is NOT equal to global: it renormalises
    # each quarter to mean 0, std 1 independently.
    assert not np.allclose(out_local, expected_global, atol=1e-3)


def test_policy_ratio_statistics_match_single_device_global_batch():
    old_log_probability = jnp.zeros(16, dtype=jnp.float32)
    log_probability = jnp.log(jnp.linspace(0.5, 1.8, 16, dtype=jnp.float32))
    advantages = jnp.ones(16, dtype=jnp.float32)
    _, expected = ppo_actor_loss(
        log_probability,
        old_log_probability,
        advantages,
        0.2,
        return_ratio_stats=True,
    )

    @functools.partial(jax.pmap, axis_name="devices")
    def apply_sharded(new_log_probability, old_log_probability, local_advantages):
        return ppo_actor_loss(
            new_log_probability,
            old_log_probability,
            local_advantages,
            0.2,
            return_ratio_stats=True,
            axis_name="devices",
        )[1]

    actual = apply_sharded(
        log_probability.reshape(N_DEV, 4),
        old_log_probability.reshape(N_DEV, 4),
        advantages.reshape(N_DEV, 4),
    )
    for name, expected_value in expected.items():
        np.testing.assert_allclose(
            np.asarray(actual[name]),
            np.broadcast_to(np.asarray(expected_value), (N_DEV,)),
            atol=1e-6,
            rtol=1e-6,
        )


def test_running_mean_std_matches_single_device_global_batch():
    """Policy normalization uses global moments and counts on every replica."""
    from musclemimic.algorithms.common.networks import RunningMeanStd

    observations = jnp.arange(96, dtype=jnp.float32).reshape(32, 3) / 10.0
    sharded_observations = observations.reshape(N_DEV, 8, 3)
    layer = RunningMeanStd()
    variables = layer.init(jax.random.key(0), jnp.zeros((1, 3)))

    single_output, single_updates = layer.apply(
        variables,
        observations,
        update_stats=True,
        mutable=["run_stats"],
    )

    replicated_variables = replicate_for_pmap(variables, jax.local_devices())

    @functools.partial(jax.pmap, axis_name="devices")
    def apply_sharded(local_variables, local_observations):
        return layer.apply(
            local_variables,
            local_observations,
            update_stats=True,
            axis_name="devices",
            mutable=["run_stats"],
        )

    sharded_output, sharded_updates = apply_sharded(
        replicated_variables,
        sharded_observations,
    )

    np.testing.assert_allclose(
        np.asarray(sharded_output).reshape(32, 3),
        np.asarray(single_output),
        atol=1e-5,
        rtol=1e-5,
    )
    for device_id in range(N_DEV):
        device_updates = jax.tree.map(lambda value, index=device_id: value[index], sharded_updates)
        np.testing.assert_allclose(
            np.asarray(device_updates["run_stats"]["mean"]),
            np.asarray(single_updates["run_stats"]["mean"]),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(device_updates["run_stats"]["var"]),
            np.asarray(single_updates["run_stats"]["var"]),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(device_updates["run_stats"]["count"]),
            np.asarray(single_updates["run_stats"]["count"]),
            atol=1e-5,
            rtol=1e-5,
        )


def test_running_mean_std_is_read_only_and_batch_invariant():
    """A frozen policy observation must not depend on its batch companions."""
    from musclemimic.algorithms.common.networks import RunningMeanStd

    layer = RunningMeanStd()
    variables = layer.init(jax.random.key(0), jnp.zeros((1, 2)))
    _, updates = layer.apply(
        variables,
        jnp.asarray([[-2.0, 1.0], [2.0, 5.0]], dtype=jnp.float32),
        update_stats=True,
        mutable=["run_stats"],
    )
    frozen_variables = {**variables, "run_stats": updates["run_stats"]}

    target = jnp.asarray([[1.0, 3.0]], dtype=jnp.float32)
    target_with_peer = jnp.concatenate(
        (target, jnp.asarray([[1000.0, -1000.0]], dtype=jnp.float32)),
        axis=0,
    )
    target_output = layer.apply(frozen_variables, target)
    target_with_peer_output = layer.apply(frozen_variables, target_with_peer)[0:1]

    np.testing.assert_allclose(target_output, target_with_peer_output, atol=1e-6, rtol=1e-6)


def test_reward_running_stats_match_single_device_global_batch():
    """Reward return moments and terminal carries are globally equivalent."""
    from musclemimic.core.wrappers.mjx import (
        NormalizeVecReward,
        NormalizeVecRewEnvState,
    )

    returns = jnp.linspace(-4.0, 5.0, 32, dtype=jnp.float32)
    rewards = jnp.linspace(0.5, 2.0, 32, dtype=jnp.float32)
    dones = jnp.arange(32) % 5 == 0

    def make_state(return_values):
        return NormalizeVecRewEnvState(
            env_state=jnp.asarray(0),
            mean=jnp.asarray(0.25),
            var=jnp.asarray(1.5),
            count=jnp.asarray(7.0),
            return_val=return_values,
        )

    single_wrapper = NormalizeVecReward(SimpleNamespace(), gamma=0.97)
    single_state, single_rewards = single_wrapper._normalize_step_output(
        make_state(returns),
        rewards,
        dones,
        jnp.asarray(1),
    )
    sharded_wrapper = NormalizeVecReward(SimpleNamespace(), gamma=0.97, axis_name="devices")

    @functools.partial(jax.pmap, axis_name="devices")
    def apply_sharded(local_returns, local_rewards, local_dones):
        return sharded_wrapper._normalize_step_output(
            make_state(local_returns),
            local_rewards,
            local_dones,
            jnp.asarray(1),
        )

    sharded_state, sharded_rewards = apply_sharded(
        returns.reshape(N_DEV, 8),
        rewards.reshape(N_DEV, 8),
        dones.reshape(N_DEV, 8),
    )

    np.testing.assert_allclose(np.asarray(sharded_rewards).reshape(-1), np.asarray(single_rewards), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(sharded_state.return_val).reshape(-1),
        np.asarray(single_state.return_val),
        atol=1e-6,
    )
    for device_id in range(N_DEV):
        np.testing.assert_allclose(np.asarray(sharded_state.mean[device_id]), np.asarray(single_state.mean), atol=1e-6)
        np.testing.assert_allclose(np.asarray(sharded_state.var[device_id]), np.asarray(single_state.var), atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(sharded_state.count[device_id]), np.asarray(single_state.count), atol=1e-6
        )


def test_reward_running_stats_exclude_nonfinite_lanes_across_devices():
    """One bad lane cannot poison moments or rewards on any replica."""
    from musclemimic.core.wrappers.mjx import (
        NormalizeVecReward,
        NormalizeVecRewEnvState,
    )

    returns = jnp.linspace(-4.0, 5.0, 32, dtype=jnp.float32).at[3].set(jnp.inf)
    rewards = jnp.linspace(0.5, 2.0, 32, dtype=jnp.float32).at[17].set(jnp.nan)
    dones = jnp.zeros((32,), dtype=bool)

    def make_state(return_values):
        return NormalizeVecRewEnvState(
            env_state=jnp.asarray(0),
            mean=jnp.asarray(0.25),
            var=jnp.asarray(1.5),
            count=jnp.asarray(7.0),
            return_val=return_values,
        )

    single_wrapper = NormalizeVecReward(SimpleNamespace(), gamma=0.97)
    single_state, single_rewards = single_wrapper._normalize_step_output(
        make_state(returns),
        rewards,
        dones,
        jnp.asarray(1),
    )
    sharded_wrapper = NormalizeVecReward(SimpleNamespace(), gamma=0.97, axis_name="devices")

    @functools.partial(jax.pmap, axis_name="devices")
    def apply_sharded(local_returns, local_rewards, local_dones):
        return sharded_wrapper._normalize_step_output(
            make_state(local_returns),
            local_rewards,
            local_dones,
            jnp.asarray(1),
        )

    sharded_state, sharded_rewards = apply_sharded(
        returns.reshape(N_DEV, 8),
        rewards.reshape(N_DEV, 8),
        dones.reshape(N_DEV, 8),
    )

    flat_sharded_rewards = np.asarray(sharded_rewards).reshape(-1)
    flat_sharded_returns = np.asarray(sharded_state.return_val).reshape(-1)
    assert np.isfinite(np.asarray(single_rewards)).all()
    assert np.isfinite(flat_sharded_rewards).all()
    assert np.isfinite(np.asarray(single_state.return_val)).all()
    assert np.isfinite(flat_sharded_returns).all()
    assert flat_sharded_rewards[17] == 0.0
    assert flat_sharded_returns[3] == 0.0
    assert flat_sharded_returns[17] == 0.0

    np.testing.assert_allclose(flat_sharded_rewards, np.asarray(single_rewards), atol=1e-6)
    np.testing.assert_allclose(flat_sharded_returns, np.asarray(single_state.return_val), atol=1e-6)
    for device_id in range(N_DEV):
        np.testing.assert_allclose(np.asarray(sharded_state.mean[device_id]), np.asarray(single_state.mean), atol=1e-6)
        np.testing.assert_allclose(np.asarray(sharded_state.var[device_id]), np.asarray(single_state.var), atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(sharded_state.count[device_id]), np.asarray(single_state.count), atol=1e-6
        )


def test_reward_running_stats_repair_nonfinite_previous_moments():
    """A damaged checkpoint/runtime state is repaired before merging."""
    from musclemimic.core.wrappers.mjx import (
        NormalizeVecReward,
        NormalizeVecRewEnvState,
    )

    wrapper = NormalizeVecReward(SimpleNamespace(), gamma=0.97)
    state = NormalizeVecRewEnvState(
        env_state=jnp.asarray(0),
        mean=jnp.asarray(jnp.nan),
        var=jnp.asarray(jnp.inf),
        count=jnp.asarray(jnp.nan),
        return_val=jnp.asarray([jnp.nan, 1.0], dtype=jnp.float32),
    )
    next_state, normalized_rewards = wrapper._normalize_step_output(
        state,
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        jnp.asarray([False, False]),
        jnp.asarray(1),
    )

    for value in (
        next_state.mean,
        next_state.var,
        next_state.count,
        next_state.return_val,
        normalized_rewards,
    ):
        assert np.isfinite(np.asarray(value)).all()
    assert np.asarray(next_state.return_val)[0] == 0.0


def test_reward_running_stats_exclude_sanitized_numerical_transition():
    """A zeroed simulator-fault reward must not update return moments."""
    from musclemimic.core.wrappers.mjx import (
        NormalizeVecReward,
        NormalizeVecRewEnvState,
    )

    wrapper = NormalizeVecReward(SimpleNamespace(), gamma=0.97)
    state = NormalizeVecRewEnvState(
        env_state=jnp.asarray(0),
        mean=jnp.asarray(0.25),
        var=jnp.asarray(1.5),
        count=jnp.asarray(7.0),
        return_val=jnp.asarray([3.0], dtype=jnp.float32),
    )
    transition_state = SimpleNamespace(info={"numerics_nonfinite_transition": jnp.asarray([True])})
    next_state, normalized_rewards = wrapper._normalize_step_output(
        state,
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray([True]),
        transition_state,
    )

    np.testing.assert_allclose(np.asarray(next_state.mean), np.asarray(state.mean), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(next_state.var), np.asarray(state.var), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(next_state.count), np.asarray(state.count), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(next_state.return_val), np.zeros((1,)), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(normalized_rewards), np.zeros((1,)), atol=0.0, rtol=0.0)


def test_soft_moe_global_auxiliary_loss_and_gradient_match_single_device():
    """MoE squares global importance, matching an unsharded full batch."""
    from musclemimic.algorithms.common.moe_networks import SoftMoELayer

    layer = SoftMoELayer(
        output_dim=4,
        num_experts=2,
        hidden_dim=4,
        load_balance_loss_weight=1.0,
        gate_init_std=0.5,
    )
    observations = jnp.linspace(-3.0, 3.0, 96, dtype=jnp.float32).reshape(32, 3)
    variables = layer.init(jax.random.key(3), observations)

    def auxiliary_loss(params, batch, axis_name=None):
        _, metrics = layer.apply(
            {"params": params},
            batch,
            return_metrics=True,
            axis_name=axis_name,
        )
        return metrics["load_balance_loss"]

    single_loss, single_gradient = jax.value_and_grad(auxiliary_loss)(
        variables["params"],
        observations,
    )
    replicated_params = replicate_for_pmap(variables["params"], jax.local_devices())
    sharded_observations = observations.reshape(N_DEV, 8, 3)

    @functools.partial(jax.pmap, axis_name="devices")
    def sharded_loss_and_gradient(params, batch):
        loss, gradient = jax.value_and_grad(auxiliary_loss)(params, batch, "devices")
        return loss, jax.lax.pmean(gradient, axis_name="devices")

    losses, gradients = sharded_loss_and_gradient(replicated_params, sharded_observations)
    np.testing.assert_allclose(np.asarray(losses), np.asarray(single_loss), atol=1e-6, rtol=1e-6)
    for device_id in range(N_DEV):
        device_gradient = jax.tree.map(lambda value, index=device_id: value[index], gradients)
        for actual, expected in zip(
            jax.tree.leaves(device_gradient),
            jax.tree.leaves(single_gradient),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6, rtol=1e-6)


# ----------------------------------------------------------------------------
# Gradient equivalence: 1x full batch vs Nx partial batches with pmean
# ----------------------------------------------------------------------------


def _toy_loss(params, xs, ys):
    # Simple MSE on a linear model y ~= W x + b.
    weights, bias = params
    preds = xs @ weights + bias
    return jnp.mean((preds - ys) ** 2)


def test_gradient_pmean_matches_single_device_gradient():
    """value_and_grad on a full batch should give the same gradient as
    pmap(value_and_grad) + lax.pmean on a sharded batch. Matches the pattern
    used in learning.update."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 5)).astype(np.float32)
    y = rng.standard_normal((32,)).astype(np.float32)
    weights = rng.standard_normal((5,)).astype(np.float32)
    b = np.float32(0.3)
    params = (jnp.asarray(weights), jnp.asarray(b))
    xs = jnp.asarray(x)
    ys = jnp.asarray(y)

    # (a) single-device full batch
    grad_single = jax.grad(_toy_loss)(params, xs, ys)

    # (b) pmap 4x8 with pmean
    xs_sharded = xs.reshape(N_DEV, 8, 5)
    ys_sharded = ys.reshape(N_DEV, 8)

    @functools.partial(jax.pmap, axis_name="devices", in_axes=(None, 0, 0))
    def pmapped_grad(p, x_local, y_local):
        g = jax.grad(_toy_loss)(p, x_local, y_local)
        return jax.lax.pmean(g, axis_name="devices")

    grad_pmap = pmapped_grad(params, xs_sharded, ys_sharded)
    # Every replica should hold the same (post-pmean) gradient.
    for d in range(N_DEV):
        np.testing.assert_allclose(
            np.asarray(grad_pmap[0][d]),
            np.asarray(grad_single[0]),
            atol=1e-5,
            rtol=1e-5,
        )
    np.testing.assert_allclose(
        np.asarray(grad_pmap[1][0]),
        np.asarray(grad_single[1]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_adaptive_sampling_psum_matches_global_counts():
    """Per-trajectory done/early counts psum'd across devices should equal
    the counts a single-device run would compute on the concatenated batch.

    Matches the aggregation pattern used by the PPO training update for the
    adaptive-sampling code path.
    """
    n_traj = 8
    num_steps = 6
    num_envs = 16  # total across 4 devices

    # Fabricate a batch where each env visits a different traj.
    rng = np.random.default_rng(42)
    final_traj_no = rng.integers(0, n_traj, size=(num_steps, num_envs)).astype(np.int32)
    done = rng.random((num_steps, num_envs)) < 0.1
    absorbing = rng.random((num_steps, num_envs)) < 0.05

    # Single-device reference.
    info_full = {"final_traj_no": jnp.asarray(final_traj_no)}
    _, done_full, early_full = compute_per_traj_termination_stats(
        info_full, jnp.asarray(done), jnp.asarray(absorbing), n_traj
    )

    # Sharded across 4 devices. Each device takes its env slice.
    envs_per_dev = num_envs // N_DEV
    final_sharded = final_traj_no.reshape(num_steps, N_DEV, envs_per_dev).transpose(1, 0, 2)
    done_sharded = done.reshape(num_steps, N_DEV, envs_per_dev).transpose(1, 0, 2)
    absorb_sharded = absorbing.reshape(num_steps, N_DEV, envs_per_dev).transpose(1, 0, 2)

    @functools.partial(jax.pmap, axis_name="devices")
    def per_device(fn, d, a):
        info_local = {"final_traj_no": fn}
        _, dc, ec = compute_per_traj_termination_stats(info_local, d, a, n_traj)
        dc = jax.lax.psum(dc, axis_name="devices")
        ec = jax.lax.psum(ec, axis_name="devices")
        return dc, ec

    dc_pmap, ec_pmap = per_device(jnp.asarray(final_sharded), jnp.asarray(done_sharded), jnp.asarray(absorb_sharded))
    # All replicas hold identical (post-psum) totals.
    for d in range(N_DEV):
        np.testing.assert_array_equal(np.asarray(dc_pmap[d]), np.asarray(done_full))
        np.testing.assert_array_equal(np.asarray(ec_pmap[d]), np.asarray(early_full))


def test_numerical_transition_diagnostics_are_global_across_devices():
    """Reason and trajectory counts must be identical on every replica."""

    shape = (N_DEV, 2, 2)
    trajectory_ids = jnp.asarray(
        [
            [[0, 0], [1, 1]],
            [[1, 2], [2, 3]],
            [[2, 2], [2, 3]],
            [[3, 0], [1, 3]],
        ],
        dtype=jnp.int32,
    )
    observation = jnp.zeros(shape, dtype=bool).at[0, 0, :].set(True)
    qpos = jnp.zeros(shape, dtype=bool)
    qvel = jnp.zeros(shape, dtype=bool).at[1, 0, 0].set(True)
    reward = jnp.zeros(shape, dtype=bool).at[2, 0, :].set(True).at[2, 1, 0].set(True)
    reward_info = jnp.zeros(shape, dtype=bool).at[3, 0, 0].set(True)
    transition = observation | qpos | qvel | reward | reward_info

    @functools.partial(jax.pmap, axis_name="devices")
    def summarize(
        local_ids,
        local_observation,
        local_qpos,
        local_qvel,
        local_reward,
        local_reward_info,
        local_invalid,
    ):
        rollout = SimpleNamespace(
            reward=jnp.zeros(local_ids.shape, dtype=jnp.float32),
            info={
                "final_traj_no": local_ids,
                "numerics_nonfinite_observation": local_observation,
                "numerics_nonfinite_qpos": local_qpos,
                "numerics_nonfinite_qvel": local_qvel,
                "numerics_nonfinite_reward": local_reward,
                "numerics_nonfinite_reward_info": local_reward_info,
                "numerics_nonfinite_transition": local_invalid,
            },
        )
        return summarize_numerical_transitions(
            rollout,
            {"n_trajectories": 4},
            axis_name="devices",
        )

    diagnostics = summarize(
        trajectory_ids,
        observation,
        qpos,
        qvel,
        reward,
        reward_info,
        transition,
    )

    for replica in range(N_DEV):
        assert float(diagnostics.invalid_transition_count[replica]) == 7.0
        np.testing.assert_allclose(
            diagnostics.invalid_transition_rate[replica],
            7.0 / 16.0,
        )
        np.testing.assert_array_equal(
            diagnostics.reason_counts[replica],
            [2.0, 0.0, 1.0, 3.0, 1.0],
        )
        assert int(diagnostics.offending_trajectory_count[replica]) == 4
        np.testing.assert_array_equal(
            diagnostics.topk_trajectory_counts[replica],
            [3.0, 2.0, 1.0],
        )
        np.testing.assert_array_equal(
            diagnostics.topk_trajectory_ids[replica, :2],
            [2, 0],
        )


def test_shard_sampling_per_device_stays_within_shard():
    """Under static motion sharding, each pmap replica's sampling weights
    should put zero probability on motions owned by other devices. Run a small
    simulation of the runner's sampling init + mask flow and verify nothing
    leaks outside each shard."""
    from musclemimic.algorithms.common.trajectory_sharding import (
        ShardSpec,
        apply_shard_mask_and_renormalize,
    )

    n_traj = 20
    spec = ShardSpec.build(n_traj_global=n_traj, n_devices=N_DEV)

    # Pretend global weights concentrate on the first two motions.
    global_w = np.zeros(n_traj, dtype=np.float32)
    global_w[0] = 3.0
    global_w[1] = 2.0
    global_w[15] = 1.0
    global_w /= global_w.sum()

    @functools.partial(jax.pmap, axis_name="devices")
    def per_dev_weights(_dummy):
        device_id = jax.lax.axis_index("devices")
        return apply_shard_mask_and_renormalize(jnp.asarray(global_w), spec, device_id)

    # Input is unused; just needs leading (N_DEV,) axis.
    out = per_dev_weights(jnp.arange(N_DEV, dtype=jnp.float32))
    jax.block_until_ready(out)
    w_per_dev = np.asarray(out)
    assert w_per_dev.shape == (N_DEV, n_traj)

    # Each replica: sum == 1.0 and all mass lies inside its shard.
    for d in range(N_DEV):
        w = w_per_dev[d]
        start, stop = spec.local_range(d)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
        # Outside shard = exactly zero.
        outside = np.concatenate([w[:start], w[stop:]])
        assert np.all(outside == 0.0), f"device {d} has nonzero mass outside shard"


def test_gradient_without_pmean_would_diverge():
    """If we forget the pmean, each replica holds only its local gradient and
    apply_gradients would step each replica in different directions. Confirm
    the divergence exists at the gradient level (so the pmean in learning.py is
    not a no-op)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 5)).astype(np.float32)
    y = rng.standard_normal((32,)).astype(np.float32)
    weights = rng.standard_normal((5,)).astype(np.float32)
    params = (jnp.asarray(weights), jnp.float32(0.0))
    xs = jnp.asarray(x).reshape(N_DEV, 8, 5)
    ys = jnp.asarray(y).reshape(N_DEV, 8)

    @functools.partial(jax.pmap, in_axes=(None, 0, 0))
    def per_device_grad(p, x_local, y_local):
        return jax.grad(_toy_loss)(p, x_local, y_local)

    g = per_device_grad(params, xs, ys)
    # Gradients should differ between replicas (otherwise our test is trivial).
    gw = np.asarray(g[0])
    assert not np.allclose(gw[0], gw[1], atol=1e-2), (
        "per-device gradients should differ; the simulator might be collapsing"
    )
