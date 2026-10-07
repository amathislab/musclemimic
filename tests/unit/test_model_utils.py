"""Unit tests for model utilities."""

import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

from musclemimic.utils.model import (  # noqa: E402
    count_actor_critic_params,
    count_params,
    count_params_by_path,
    count_trainable_params,
)


@pytest.fixture
def flat_params():
    return {
        "kernel": jnp.zeros((10, 5)),
        "bias": jnp.zeros(5),
    }


@pytest.fixture
def nested_params():
    return {
        "layer1": {
            "kernel": jnp.zeros((10, 20)),
            "bias": jnp.zeros(20),
        },
        "layer2": {
            "kernel": jnp.zeros((20, 5)),
            "bias": jnp.zeros(5),
        },
    }


@pytest.fixture
def actor_critic_params():
    return {
        "actor": {
            "Dense_0": {"kernel": jnp.zeros((100, 256)), "bias": jnp.zeros(256)},
            "Dense_1": {"kernel": jnp.zeros((256, 10)), "bias": jnp.zeros(10)},
        },
        "critic": {
            "Dense_0": {"kernel": jnp.zeros((100, 256)), "bias": jnp.zeros(256)},
            "Dense_1": {"kernel": jnp.zeros((256, 1)), "bias": jnp.zeros(1)},
        },
        "log_std": jnp.zeros(10),
        "RunningMeanStd_0": {
            "mean": jnp.zeros(100),
            "var": jnp.ones(100),
            "count": jnp.array(1.0),
        },
    }


def test_count_params_flat_dict(flat_params):
    assert count_params(flat_params) == 55


def test_count_params_nested_dict(nested_params):
    # 10*20 + 20 + 20*5 + 5 = 200 + 20 + 100 + 5 = 325
    assert count_params(nested_params) == 325


def test_count_params_single_array():
    params = jnp.zeros((100, 50))
    assert count_params(params) == 5000


def test_count_params_empty_dict():
    assert count_params({}) == 0


def test_count_params_scalar():
    params = {"log_std": jnp.array(0.5)}
    assert count_params(params) == 1


def test_count_params_1d_array():
    params = {"log_std": jnp.zeros(10)}
    assert count_params(params) == 10


def test_count_params_by_path_single_prefix():
    params = {
        "actor": {"kernel": jnp.zeros((10, 5)), "bias": jnp.zeros(5)},
        "critic": {"kernel": jnp.zeros((10, 1)), "bias": jnp.zeros(1)},
    }
    assert count_params_by_path(params, "actor") == 55
    assert count_params_by_path(params, "critic") == 11


def test_count_params_by_path_prefix_matching():
    params = {
        "actor_layer1": {"kernel": jnp.zeros((10, 5))},
        "actor_layer2": {"kernel": jnp.zeros((5, 3))},
        "critic": {"kernel": jnp.zeros((10, 1))},
    }
    # actor_layer1 + actor_layer2 = 50 + 15 = 65
    assert count_params_by_path(params, "actor") == 65


def test_count_params_by_path_no_match():
    params = {
        "encoder": {"kernel": jnp.zeros((10, 5))},
        "decoder": {"kernel": jnp.zeros((5, 10))},
    }
    assert count_params_by_path(params, "actor") == 0


def test_count_params_by_path_empty_params():
    assert count_params_by_path({}, "actor") == 0


def test_count_actor_critic_params_standard_structure(actor_critic_params):
    counts = count_actor_critic_params(actor_critic_params)

    # Actor: Dense_0 (100*256 + 256) + Dense_1 (256*10 + 10) + log_std (10)
    actor_expected = 100 * 256 + 256 + 256 * 10 + 10 + 10
    assert counts["actor"] == actor_expected

    # Critic: Dense_0 (100*256 + 256) + Dense_1 (256*1 + 1)
    critic_expected = 100 * 256 + 256 + 256 * 1 + 1
    assert counts["critic"] == critic_expected

    # Shared: RunningMeanStd (100 + 100 + 1) = 201
    shared_expected = 100 + 100 + 1
    assert counts["shared"] == shared_expected

    # Total
    assert counts["total"] == actor_expected + critic_expected + shared_expected


def test_count_actor_critic_params_log_std_counted_as_actor():
    params = {
        "actor": {"kernel": jnp.zeros((10, 5))},
        "critic": {"kernel": jnp.zeros((10, 1))},
        "log_std": jnp.zeros(5),
    }
    counts = count_actor_critic_params(params)
    assert counts["actor"] == 50 + 5  # actor kernel + log_std
    assert counts["critic"] == 10


def test_count_actor_critic_params_prefix_named_submodules():
    params = {
        "actor_dense_0": {"kernel": jnp.zeros((8, 4)), "bias": jnp.zeros(4)},
        "actor_moe_layer_1": {"gate": {"kernel": jnp.zeros((4, 2))}},
        "critic_dense_0": {"kernel": jnp.zeros((8, 1))},
        "log_std": jnp.zeros(4),
        "shared_norm": {"scale": jnp.zeros(3)},
    }
    counts = count_actor_critic_params(params)
    actor_expected = 8 * 4 + 4 + 4 * 2 + 4
    critic_expected = 8 * 1
    shared_expected = 3
    assert counts["actor"] == actor_expected
    assert counts["critic"] == critic_expected
    assert counts["shared"] == shared_expected
    assert counts["total"] == actor_expected + critic_expected + shared_expected


def test_count_actor_critic_params_no_shared():
    params = {
        "actor": {"kernel": jnp.zeros((10, 5))},
        "critic": {"kernel": jnp.zeros((10, 1))},
    }
    counts = count_actor_critic_params(params)
    assert counts["shared"] == 0


def test_count_actor_critic_params_empty():
    counts = count_actor_critic_params({})
    assert counts["actor"] == 0
    assert counts["critic"] == 0
    assert counts["shared"] == 0
    assert counts["total"] == 0


def test_count_actor_critic_params_total_equals_sum():
    params = {
        "actor": {"kernel": jnp.zeros((50, 30))},
        "critic": {"kernel": jnp.zeros((50, 20))},
        "log_std": jnp.zeros(30),
        "other": {"bias": jnp.zeros(10)},
    }
    counts = count_actor_critic_params(params)
    assert counts["total"] == counts["actor"] + counts["critic"] + counts["shared"]


def test_count_trainable_params_alias(flat_params):
    assert count_trainable_params(flat_params) == count_params(flat_params)


def test_count_params_with_real_actor_critic_network():
    from musclemimic.algorithms.common.networks import ActorCritic

    network = ActorCritic(
        action_dim=10,
        hidden_layer_dims=(64, 32),
        critic_hidden_layer_dims=(64, 32),
    )

    key = jax.random.PRNGKey(0)
    obs_dim = 50
    dummy_obs = jnp.zeros((1, obs_dim))
    variables = network.init(key, dummy_obs)
    params = variables["params"]

    counts = count_actor_critic_params(params)

    assert "actor" in counts
    assert "critic" in counts
    assert "shared" in counts
    assert "total" in counts

    assert counts["actor"] > 0
    assert counts["critic"] > 0
    assert counts["total"] == counts["actor"] + counts["critic"] + counts["shared"]
    assert counts["total"] == count_params(params)


def test_actor_critic_actor_only_does_not_require_critic_params():
    from flax.core import unfreeze

    from musclemimic.algorithms.common.networks import ActorCritic

    network = ActorCritic(
        action_dim=4,
        hidden_layer_dims=(16, 8),
        critic_hidden_layer_dims=(16, 8),
    )

    key = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((2, 12))
    variables = network.init(key, dummy_obs)

    full_pi, _ = network.apply(variables, dummy_obs)
    actor_only_variables = unfreeze(variables)
    actor_only_variables["params"].pop("critic")
    actor_pi = network.apply(actor_only_variables, dummy_obs, actor_only=True)

    np.testing.assert_allclose(np.asarray(actor_pi.mode()), np.asarray(full_pi.mode()))


def test_soft_moe_actor_only_does_not_require_critic_params():
    from flax.core import unfreeze

    from musclemimic.algorithms.common.moe_networks import SoftMoEActorCritic

    network = SoftMoEActorCritic(
        action_dim=4,
        hidden_layer_dims=(16, 8),
        moe_at_layers=(1,),
        num_experts=2,
    )
    dummy_obs = jnp.zeros((2, 12))
    variables = network.init(jax.random.key(0), dummy_obs)
    full_pi, _ = network.apply(variables, dummy_obs)

    actor_only_variables = unfreeze(variables)
    for name in tuple(actor_only_variables["params"]):
        if name.startswith("critic_"):
            actor_only_variables["params"].pop(name)
    actor_pi = network.apply(actor_only_variables, dummy_obs, actor_only=True)

    np.testing.assert_allclose(np.asarray(actor_pi.mode()), np.asarray(full_pi.mode()))


def test_count_params_jit_compatible(flat_params):
    @jax.jit
    def jitted_count(p):
        return count_params(p)

    result = jitted_count(flat_params)
    assert result == 55
