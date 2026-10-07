"""Numerical equivalence of a PPO minibatch update under shard_map.

Scenario: the same synthetic minibatch processed as
  (a) a single device with the full batch, and
  (b) N data shards each with 1/N of the batch.

Claim: after one `_update_minibatch` call, the resulting params differ only
by floating-point precision. If any cross-device reduction is missing
(gradient, advantage mean/std, run_stats forward cache), this test will
catch it -- a missing pmean manifests as order-of-magnitude-apart params.

Run on the CPU 4-device simulator so the test is hardware-independent::

    JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=4' \
        pytest tests/unit/test_data_parallel_numerics.py -q
"""

from __future__ import annotations

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
import optax
import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.common.distributed import (
    DATA_AXIS_NAME,
    make_data_mesh,
    mark_data_varying,
)

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
P = jax.sharding.PartitionSpec
MESH = make_data_mesh(N_DEV)
DATA = P(DATA_AXIS_NAME)
REPLICATED = P()


def _put(tree, specs):
    return jax.tree.map(
        lambda value, spec: jax.device_put(
            value,
            jax.sharding.NamedSharding(MESH, spec),
        ),
        tree,
        specs,
    )


def _specs(tree, spec):
    return jax.tree.map(lambda _value: spec, tree)


def _run_sharded(function, in_specs, out_specs, *args):
    compiled = jax.jit(
        jax.shard_map(
            function,
            mesh=MESH,
            in_specs=in_specs,
            out_specs=out_specs,
        )
    )
    placed = tuple(_put(value, specs) for value, specs in zip(args, in_specs, strict=True))
    return compiled(*placed)


@pytest.fixture(scope="module", autouse=True)
def _verify_simulator():
    if jax.local_device_count() != N_DEV:
        pytest.skip(
            f"need 4 simulated devices, got {jax.local_device_count()}. "
            f"run with JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=4'"
        )


# ----------------------------------------------------------------------------
# normalize_advantages_global: shard_map versus single-device behavior
# ----------------------------------------------------------------------------


def test_normalize_adv_global_matches_single_device_stats():
    """With ``axis_name`` set, the *global* mean/std should drive
    normalisation (not each device's local stats)."""
    # Fabricate advantages with a specific mean/std so we can check by hand.
    # Full batch of 16 = 4 devices x 4 per device.
    full = jnp.arange(16, dtype=jnp.float32) - 7.5  # mean 0, std sqrt(~22.9)
    out = _run_sharded(
        lambda adv: normalize_advantages_global(adv, axis_name=DATA_AXIS_NAME),
        (DATA,),
        DATA,
        full,
    )
    got = np.asarray(out)
    # Compare to a single-device normalisation of the same data.
    single = normalize_advantages(full)
    expected = np.asarray(single)
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_normalize_adv_local_drifts_under_shard_map_without_axis_name():
    """Sanity: the OLD behaviour (per-device normalisation) gives different
    numbers than the global one. Prevents someone from 'simplifying' our code
    back to a plain per-device normalisation and silently degrading training."""
    full = jnp.arange(16, dtype=jnp.float32) - 7.5
    out_local = np.asarray(_run_sharded(normalize_advantages, (DATA,), DATA, full))
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

    def apply_sharded(new_log_probability, old_log_probability, local_advantages):
        return ppo_actor_loss(
            new_log_probability,
            old_log_probability,
            local_advantages,
            0.2,
            return_ratio_stats=True,
            axis_name=DATA_AXIS_NAME,
        )[1]

    actual = _run_sharded(
        apply_sharded,
        (DATA, DATA, DATA),
        _specs(expected, REPLICATED),
        log_probability,
        old_log_probability,
        advantages,
    )
    for name, expected_value in expected.items():
        np.testing.assert_allclose(
            np.asarray(actual[name]),
            np.asarray(expected_value),
            atol=1e-6,
            rtol=1e-6,
        )


def test_running_mean_std_matches_single_device_global_batch():
    """Policy normalization uses global moments and counts on every replica."""
    from musclemimic.algorithms.common.networks import RunningMeanStd

    observations = jnp.arange(96, dtype=jnp.float32).reshape(32, 3) / 10.0
    layer = RunningMeanStd()
    variables = layer.init(jax.random.key(0), jnp.zeros((1, 3)))

    single_output, single_updates = layer.apply(
        variables,
        observations,
        update_stats=True,
        mutable=["run_stats"],
    )

    def apply_sharded(local_variables, local_observations):
        return layer.apply(
            local_variables,
            local_observations,
            update_stats=True,
            axis_name=DATA_AXIS_NAME,
            mutable=["run_stats"],
        )

    sharded_output, sharded_updates = _run_sharded(
        apply_sharded,
        (_specs(variables, REPLICATED), DATA),
        (DATA, _specs(single_updates, REPLICATED)),
        variables,
        observations,
    )

    np.testing.assert_allclose(
        np.asarray(sharded_output).reshape(32, 3),
        np.asarray(single_output),
        atol=1e-5,
        rtol=1e-5,
    )
    for name in ("mean", "var", "count"):
        np.testing.assert_allclose(
            np.asarray(sharded_updates["run_stats"][name]),
            np.asarray(single_updates["run_stats"][name]),
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
    sharded_wrapper = NormalizeVecReward(
        SimpleNamespace(),
        gamma=0.97,
        axis_name=DATA_AXIS_NAME,
    )

    def apply_sharded(local_returns, local_rewards, local_dones):
        return sharded_wrapper._normalize_step_output(
            make_state(local_returns),
            local_rewards,
            local_dones,
            jnp.asarray(1),
        )

    state_specs = _specs(make_state(returns[:8]), REPLICATED).replace(
        return_val=DATA,
    )
    sharded_state, sharded_rewards = _run_sharded(
        apply_sharded,
        (DATA, DATA, DATA),
        (state_specs, DATA),
        returns,
        rewards,
        dones,
    )

    np.testing.assert_allclose(np.asarray(sharded_rewards).reshape(-1), np.asarray(single_rewards), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(sharded_state.return_val).reshape(-1),
        np.asarray(single_state.return_val),
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(sharded_state.mean), np.asarray(single_state.mean), atol=1e-6)
    np.testing.assert_allclose(np.asarray(sharded_state.var), np.asarray(single_state.var), atol=1e-6)
    np.testing.assert_allclose(np.asarray(sharded_state.count), np.asarray(single_state.count), atol=1e-6)


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

    def sharded_loss_and_gradient(params, batch):
        loss = auxiliary_loss(
            params,
            batch,
            DATA_AXIS_NAME,
        )
        gradient = jax.grad(
            lambda current: auxiliary_loss(
                current,
                batch,
                DATA_AXIS_NAME,
            )
        )(params)
        return loss, gradient

    losses, gradients = _run_sharded(
        sharded_loss_and_gradient,
        (_specs(variables["params"], REPLICATED), DATA),
        (REPLICATED, _specs(single_gradient, REPLICATED)),
        variables["params"],
        observations,
    )
    np.testing.assert_allclose(np.asarray(losses), np.asarray(single_loss), atol=1e-6, rtol=1e-6)
    for actual, expected in zip(
        jax.tree.leaves(gradients),
        jax.tree.leaves(single_gradient),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6, rtol=1e-6)


# ----------------------------------------------------------------------------
# Gradient equivalence: one full batch versus data shards with pmean
# ----------------------------------------------------------------------------


def _toy_loss(params, xs, ys):
    # Simple MSE on a linear model y ~= W x + b.
    weights, bias = params
    preds = xs @ weights + bias
    return jnp.mean((preds - ys) ** 2)


def test_gradient_pmean_matches_single_device_gradient():
    """value_and_grad on a full batch should give the same gradient as
    shard_map(value_and_grad) + lax.pmean on a sharded batch. Matches the pattern
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

    # (b) four axis-0 shards with shard_map's replicated-input transpose
    def sharded_grad(p, x_local, y_local):
        axis_size = jax.lax.psum(jnp.asarray(1.0), DATA_AXIS_NAME)
        return jax.grad(lambda current: _toy_loss(current, x_local, y_local) / axis_size)(p)

    sharded_gradient = _run_sharded(
        sharded_grad,
        (_specs(params, REPLICATED), DATA, DATA),
        _specs(grad_single, REPLICATED),
        params,
        xs,
        ys,
    )
    np.testing.assert_allclose(
        np.asarray(sharded_gradient[0]),
        np.asarray(grad_single[0]),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(sharded_gradient[1]),
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

    def per_device(fn, d, a):
        info_local = {"final_traj_no": fn}
        _, dc, ec = compute_per_traj_termination_stats(info_local, d, a, n_traj)
        dc = jax.lax.psum(dc, axis_name=DATA_AXIS_NAME)
        ec = jax.lax.psum(ec, axis_name=DATA_AXIS_NAME)
        return dc, ec

    env_data = P(None, DATA_AXIS_NAME)
    done_sharded, early_sharded = _run_sharded(
        per_device,
        (env_data, env_data, env_data),
        (REPLICATED, REPLICATED),
        jnp.asarray(final_traj_no),
        jnp.asarray(done),
        jnp.asarray(absorbing),
    )
    np.testing.assert_array_equal(np.asarray(done_sharded), np.asarray(done_full))
    np.testing.assert_array_equal(np.asarray(early_sharded), np.asarray(early_full))


def test_shard_sampling_per_device_stays_within_shard():
    """Under static motion sharding, each device's sampling weights
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

    def per_dev_weights(_dummy):
        device_id = jax.lax.axis_index(DATA_AXIS_NAME)
        weights = apply_shard_mask_and_renormalize(
            jnp.asarray(global_w),
            spec,
            device_id,
        )
        return weights

    out = _run_sharded(
        per_dev_weights,
        (DATA,),
        DATA,
        jnp.arange(N_DEV, dtype=jnp.float32),
    )
    jax.block_until_ready(out)
    w_per_dev = np.asarray(out).reshape(N_DEV, n_traj)
    assert w_per_dev.shape == (N_DEV, n_traj)

    # Each replica: sum == 1.0 and all mass lies inside its shard.
    for d in range(N_DEV):
        w = w_per_dev[d]
        start, stop = spec.local_range(d)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
        # Outside shard = exactly zero.
        outside = np.concatenate([w[:start], w[stop:]])
        assert np.all(outside == 0.0), f"device {d} has nonzero mass outside shard"


def test_unscaled_local_loss_multiplies_replicated_gradient():
    """The transpose sums gradients of replicated shard_map inputs."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 5)).astype(np.float32)
    y = rng.standard_normal((32,)).astype(np.float32)
    weights = rng.standard_normal((5,)).astype(np.float32)
    params = (jnp.asarray(weights), jnp.float32(0.0))
    xs = jnp.asarray(x)
    ys = jnp.asarray(y)

    expected = jax.grad(_toy_loss)(params, xs, ys)

    def unscaled_gradient(p, x_local, y_local):
        return jax.grad(_toy_loss)(p, x_local, y_local)

    g = _run_sharded(
        unscaled_gradient,
        (_specs(params, REPLICATED), DATA, DATA),
        _specs(params, REPLICATED),
        params,
        xs,
        ys,
    )
    for actual, reference in zip(
        jax.tree.leaves(g),
        jax.tree.leaves(expected),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual),
            N_DEV * np.asarray(reference),
            atol=1e-5,
            rtol=1e-5,
        )


def test_flashsac_batch_norm_matches_global_batch_moments():
    from musclemimic.algorithms.flashsac.networks import UnitBatchNorm

    inputs = jnp.linspace(-3.0, 5.0, 96, dtype=jnp.float32).reshape(32, 3)
    norm = UnitBatchNorm()
    variables = norm.init(jax.random.key(0), inputs[:8], training=False)
    single_output, single_updates = norm.apply(
        variables,
        inputs,
        training=True,
        mutable=["batch_stats"],
    )

    def apply_sharded(local_variables, local_inputs):
        return norm.apply(
            local_variables,
            local_inputs,
            training=True,
            axis_name=DATA_AXIS_NAME,
            mutable=["batch_stats"],
        )

    sharded_output, sharded_updates = _run_sharded(
        apply_sharded,
        (_specs(variables, REPLICATED), DATA),
        (DATA, _specs(single_updates, REPLICATED)),
        variables,
        inputs,
    )
    np.testing.assert_allclose(sharded_output, single_output, atol=1e-5, rtol=1e-5)
    for name in ("mean", "var"):
        np.testing.assert_allclose(
            sharded_updates["batch_stats"][name],
            single_updates["batch_stats"][name],
            atol=1e-5,
            rtol=1e-5,
        )


def test_flashsac_replicated_update_matches_one_identical_shard():
    from musclemimic.algorithms.flashsac.config import FlashSACAgentConf
    from musclemimic.algorithms.flashsac.networks import (
        FlashSACActor,
        FlashSACDoubleCritic,
    )
    from musclemimic.algorithms.flashsac.replay_buffer import (
        ReplayBatch,
        create_replay_buffer,
    )
    from musclemimic.algorithms.flashsac.train import initialize_agent_state
    from musclemimic.algorithms.flashsac.update import update_from_batch
    from musclemimic.runner.flashsac_sharding import train_state_specs

    experiment = OmegaConf.create(
        {
            "initial_temperature": 0.01,
            "buffer_capacity": 16,
            "replay_storage_dtype": "bfloat16",
            "actor_update_period": 2,
            "target_sigma": 0.15,
            "target_update_tau": 0.01,
            "gamma": 0.99,
            "critic_min_value": -5.0,
            "critic_max_value": 5.0,
            "normalize_reward": True,
            "normalized_return_max": 5.0,
        }
    )
    schedule = optax.constant_schedule(3e-4)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    agent_conf = FlashSACAgentConf(
        config=OmegaConf.create({"experiment": experiment}),
        actor=FlashSACActor(action_dim=2, hidden_dim=8, num_blocks=1),
        critic=FlashSACDoubleCritic(hidden_dim=8, num_blocks=1, num_bins=5),
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
        temperature_optimizer=optax.adam(3e-4),
        learning_rate_schedule=schedule,
        actor_observation_indices=jnp.arange(4),
        critic_observation_indices=jnp.arange(4),
    )
    local_state = initialize_agent_state(jax.random.key(0), 4, 2, agent_conf)
    global_state = local_state.replace(
        replay_buffer=create_replay_buffer(
            16 * N_DEV,
            observation_dim=4,
            action_dim=2,
            storage_dtype="bfloat16",
        )
    )
    local_batch = ReplayBatch(
        observations=jnp.linspace(-1.0, 1.0, 32).reshape(8, 4),
        actions=jnp.linspace(-0.5, 0.5, 16).reshape(8, 2),
        rewards=jnp.linspace(0.1, 0.8, 8),
        next_observations=jnp.linspace(-0.8, 1.2, 32).reshape(8, 4),
        terminated=jnp.arange(8) % 3 == 0,
    )
    global_batch = jax.tree.map(
        lambda value: jnp.concatenate([value] * N_DEV),
        local_batch,
    )
    key = jax.random.PRNGKey(9)
    single_state, single_metrics = update_from_batch(
        local_state,
        local_batch,
        key,
        agent_conf,
    )
    state_specs = train_state_specs(global_state)

    def update_shard(state, batch, keys):
        return update_from_batch(
            state,
            batch,
            jnp.squeeze(keys, axis=0),
            agent_conf,
            axis_name=DATA_AXIS_NAME,
        )

    sharded_state, sharded_metrics = _run_sharded(
        update_shard,
        (state_specs, _specs(global_batch, DATA), DATA),
        (state_specs, _specs(single_metrics, REPLICATED)),
        global_state,
        global_batch,
        jnp.broadcast_to(key, (N_DEV, *key.shape)),
    )
    for actual_tree, expected_tree in (
        (sharded_state.actor_params, single_state.actor_params),
        (sharded_state.critic_params, single_state.critic_params),
        (sharded_metrics, single_metrics),
    ):
        for actual, expected in zip(
            jax.tree.leaves(actual_tree),
            jax.tree.leaves(expected_tree),
            strict=True,
        ):
            # Collective reduction order differs from the unsharded CPU
            # reduction; the resulting parameter drift stays below 1e-4.
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_mark_data_varying_preserves_scan_carry_layout():
    def scan_shard(values):
        initial = mark_data_varying(
            {
                "changing": values,
                "constant": jnp.zeros(values.shape, dtype=values.dtype),
            },
            DATA_AXIS_NAME,
        )

        def step(state, _unused):
            next_state = {
                "changing": state["changing"] + 1,
                "constant": jnp.zeros(state["constant"].shape, dtype=values.dtype),
            }
            return mark_data_varying(next_state, DATA_AXIS_NAME), None

        return jax.lax.scan(step, initial, None, length=2)[0]

    values = jnp.arange(8, dtype=jnp.float32)
    result = _run_sharded(
        scan_shard,
        (DATA,),
        {"changing": DATA, "constant": DATA},
        values,
    )

    np.testing.assert_array_equal(result["changing"], values + 2)
    np.testing.assert_array_equal(result["constant"], jnp.zeros_like(values))
