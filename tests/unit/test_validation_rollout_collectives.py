"""Regression coverage for collective-free PPO and FlashSAC validation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.common.trajectory_evaluation import evaluate_exact_trajectory_set
from musclemimic.algorithms.flashsac import evaluation as flashsac_evaluation
from musclemimic.algorithms.flashsac.state import FlashSACPolicyState
from musclemimic.algorithms.ppo import evaluation as ppo_evaluation
from musclemimic.utils.metrics import ValidationSummary

DATA_AXIS = "data"
ROLLOUT_STEPS = 5


class _Environment:
    def reset_to(self, keys, ids):
        del keys
        return jnp.zeros((ids.size, 1)), {"step": jnp.zeros_like(ids), "id": ids}

    def step(self, state, action):
        del action
        lengths = 2 + 2 * state["id"]
        step = state["step"] + 1
        done = step >= lengths
        observation = step[:, None].astype(jnp.float32)
        reward = jnp.ones_like(step, dtype=jnp.float32)
        # Completed slots keep being simulated until the static scan ends.
        error = jnp.where(state["step"] >= lengths, jnp.nan, 2.0)
        return observation, reward, done, done, {"err_root_quat": error}, {"step": step, "id": state["id"]}


class _Policy:
    def __init__(self, observation):
        self.observation = observation

    def mode(self):
        return jnp.zeros_like(self.observation)


def _ppo_apply(variables, observation):
    del variables
    return _Policy(observation), jnp.zeros(observation.shape[0])


class _Actor:
    @staticmethod
    def apply(variables, observation, *, training):
        del variables, training
        return jnp.zeros_like(observation), jnp.ones_like(observation)


def _validate(algorithm, rng, mask, *, should_validate=True, axis_name=DATA_AXIS):
    config = OmegaConf.create({"validation": {"evaluate_all": True, "deterministic": True, "num_steps": ROLLOUT_STEPS}})
    kwargs = {
        "env": _Environment(),
        "config": config,
        "metrics_handler": SimpleNamespace(get_zero_container=ValidationSummary),
        "trajectory_ids": jnp.asarray([0, 1]),
        "trajectory_lengths": jnp.asarray([2, 4]),
        "trajectory_mask": mask,
        "axis_name": axis_name,
    }
    if algorithm == "ppo":
        return ppo_evaluation.run_validation_step(
            SimpleNamespace(apply_fn=_ppo_apply, params={}, run_stats={}),
            rng,
            should_validate,
            **kwargs,
        )
    return flashsac_evaluation.run_validation_step(
        SimpleNamespace(actor=_Actor(), actor_observation_indices=jnp.asarray([0])),
        FlashSACPolicyState(actor_params={}, actor_batch_stats={}),
        rng,
        **kwargs,
    )


@pytest.mark.parametrize("algorithm", ["ppo", "flashsac"])
def test_exact_validation_rollout_has_no_device_collectives(algorithm):
    if jax.local_device_count() < 2:
        pytest.skip("requires at least two devices, including simulated CPU devices")
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()), (DATA_AXIS,))
    data = jax.sharding.PartitionSpec(DATA_AXIS)

    def shard(keys):
        replica = jax.lax.axis_index(DATA_AXIS)
        mask = jnp.arange(2) < (replica % 2 + 1)
        result = _validate(algorithm, jnp.squeeze(keys, axis=0), mask)
        return jax.tree.map(lambda value: jnp.expand_dims(value, axis=0), result)

    compiled = jax.jit(jax.shard_map(shard, mesh=mesh, in_specs=data, out_specs=data))
    keys = jax.device_put(jax.random.split(jax.random.key(7), mesh.size), jax.sharding.NamedSharding(mesh, data))
    lowered = compiled.lower(keys)
    hlo = lowered.as_text().lower()
    for collective in ("all_reduce", "all_gather", "reduce_scatter", "collective_permute"):
        assert collective not in hlo
    result = jax.block_until_ready(lowered.compile()(keys))
    np.testing.assert_array_equal(result.metrics.episode_count, np.arange(mesh.size) % 2 + 1)


def test_exact_validation_ignores_nonfinite_post_terminal_diagnostics():
    result = jax.jit(
        lambda rng: evaluate_exact_trajectory_set(
            lambda observation, key: jnp.zeros_like(observation),
            rng,
            _Environment(),
            trajectory_ids=jnp.asarray([0, 1]),
            trajectory_lengths=jnp.asarray([2, 4]),
            trajectory_mask=jnp.asarray([True, True]),
            num_steps=ROLLOUT_STEPS,
            axis_name=None,
        )
    )(jax.random.key(7))
    assert float(result.metrics.err_root_quat) == pytest.approx(2.0)


def test_exact_ppo_validation_preserves_traced_skip_rng_and_tree():
    compiled = jax.jit(
        lambda rng, enabled: _validate("ppo", rng, jnp.asarray([True, True]), should_validate=enabled, axis_name=None)
    )
    rng = jax.random.key(13)
    evaluated = compiled(rng, jnp.asarray(True))
    skipped = compiled(rng, jnp.asarray(False))
    assert float(evaluated.metrics.episode_count) == 2.0
    assert float(skipped.metrics.episode_count) == 0.0
    assert jax.tree.structure(evaluated.metrics) == jax.tree.structure(skipped.metrics)
    np.testing.assert_array_equal(jax.random.key_data(skipped.rng), jax.random.key_data(rng))
