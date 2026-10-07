"""Compare donated and retained buffers on the current multi-GPU Warp path."""

from time import monotonic

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from musclemimic.algorithms.ppo.evaluation import ValidationStepResult
from musclemimic.runner.data_parallel import DATA, REPLICATED, compile_sharded, put_tree
from musclemimic.runner.validation_reduction import add_validation_shard_axis, finish_validation
from musclemimic.runner.warp_graph_cache import clear_warp_graph_cache, disable_warp_graph_eviction
from musclemimic.utils.metrics import ValidationSummary

NUM_DEVICES = 4
NUM_ENVS = NUM_DEVICES
STEPS_PER_CHUNK = 64
NUM_CHUNKS = 4
MODEL_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <body pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge"/></actuator>
</mujoco>
"""


@pytest.mark.integration
def test_four_gpu_warp_donation_matches_retained_buffers():
    devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    if len(devices) < NUM_DEVICES:
        pytest.skip("requires four CUDA GPUs and the Warp backend")
    mesh = jax.sharding.Mesh(np.asarray(devices[:NUM_DEVICES]), ("data",))
    cpu_model = mujoco.MjModel.from_xml_string(MODEL_XML)
    model = mjx.put_model(cpu_model, impl="warp")
    model = put_tree(model, jax.tree.map(lambda _: REPLICATED, model), mesh)
    template = mjx.make_data(cpu_model, impl="warp", naconmax=NUM_ENVS, njmax=NUM_ENVS)
    # Each leaf gets independent storage, as required by the donation contract.
    initial = jax.tree.map(lambda value: np.repeat(np.asarray(value)[None], NUM_ENVS, axis=0), template)
    initial = initial.replace(qpos=np.linspace(-0.2, 0.2, NUM_ENVS, dtype=np.float32)[:, None])
    initial_keys = np.asarray(jax.random.split(jax.random.PRNGKey(17), NUM_DEVICES))
    data_specs = jax.tree.map(lambda _: DATA, initial)

    def rollout(model, state, keys):
        state = jax.tree.map(lambda value: jnp.squeeze(value, axis=0), state)
        key = jnp.squeeze(keys, axis=0)

        def step(_, carry):
            state, key = carry
            key, action_key = jax.random.split(key)
            control = 0.05 * jax.random.normal(action_key, state.ctrl.shape)
            state = mjx.step(model, state.replace(ctrl=control))
            return state, key

        state, key = jax.lax.fori_loop(0, STEPS_PER_CHUNK, step, (state, key))
        return jax.tree.map(lambda value: value[None], state), key[None]

    def validate(model, state, keys):
        state, keys = rollout(model, state, keys)
        metrics = ValidationSummary(
            episode_count=jnp.asarray(state.qpos.shape[0], dtype=jnp.float32),
            mean_episode_return=jnp.mean(state.qpos),
        )
        return ValidationStepResult(add_validation_shard_axis(metrics), keys)

    metric_specs = jax.tree.map(lambda _: DATA, ValidationSummary())
    outputs = []
    disable_warp_graph_eviction()
    for donate in (False, True):
        state = put_tree(initial, data_specs, mesh)
        keys = put_tree(initial_keys, DATA, mesh)
        validation_keys = put_tree(initial_keys.copy(), DATA, mesh)
        train = compile_sharded(
            rollout,
            mesh=mesh,
            in_specs=(REPLICATED, data_specs, DATA),
            out_specs=(data_specs, DATA),
            donate_argnums=(1, 2) if donate else (),
        )
        validation = compile_sharded(
            validate,
            mesh=mesh,
            in_specs=(REPLICATED, data_specs, DATA),
            out_specs=ValidationStepResult(metric_specs, DATA),
            donate_argnums=(2,) if donate else (),
        )
        compile_start = monotonic()
        executable = train.lower(model, state, keys).compile()
        assert ("input_output_alias=" in executable.as_text()) == donate
        validation.lower(model, state, validation_keys).compile()
        print(f"donation={donate} compile_seconds={monotonic() - compile_start:.3f}", flush=True)
        validation_returns = []
        for chunk in range(NUM_CHUNKS):
            state, keys = train(model, state, keys)
            result = finish_validation(
                validation(model, state, validation_keys),
                None,
                evaluate_all=False,
                event_name=f"donation_{donate}_{chunk}",
            )
            validation_keys = result.rng
            assert float(result.metrics.episode_count) == NUM_ENVS
            assert np.isfinite(result.metrics.mean_episode_return)
            validation_returns.append(float(result.metrics.mean_episode_return))
            jax.block_until_ready((state, keys, validation_keys))
            clear_warp_graph_cache()
            if chunk == 1:
                # Exercise continuation with newly allocated restored state.
                state = put_tree(jax.tree.map(lambda value: np.array(value, copy=True), state), data_specs, mesh)
            print(f"donation={donate} chunk={chunk + 1} validation_completed", flush=True)
        outputs.append(
            (
                np.asarray(state.qpos),
                np.asarray(state.qvel),
                np.asarray(keys),
                np.asarray(validation_keys),
                np.asarray(validation_returns),
            )
        )
    for retained, donated in zip(*outputs, strict=True):
        if np.issubdtype(retained.dtype, np.integer):
            np.testing.assert_array_equal(donated, retained)
        else:
            np.testing.assert_allclose(donated, retained, rtol=1e-6, atol=1e-6)
