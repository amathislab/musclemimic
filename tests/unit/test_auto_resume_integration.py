"""Integration tests for auto-resume with real Orbax checkpoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.common.checkpoint_manager import (
    CheckpointMetadata,
    create_checkpoint_manager,
)
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.algorithms.common.optimizer import get_optimizer
from musclemimic.algorithms.ppo.checkpoint import (
    _replace_target_owned_runtime_leaves,
    load_checkpoint_for_resume,
)
from musclemimic.algorithms.ppo.learning import initialize_train_state
from musclemimic.runner.checkpointing import find_latest_checkpoint, resolve_checkpoint_dir


@pytest.mark.integration
def test_auto_resume_orbax_roundtrip(tmp_path):
    launch_dir = tmp_path / "launch"
    result_dir = tmp_path / "results" / "run1"
    launch_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)

    resolved_dir = resolve_checkpoint_dir(
        configured_ckpt_dir="checkpoints",
        launch_dir=str(launch_dir),
        result_dir=str(result_dir),
        experiment_id="exp123",
        auto_resume=True,
    )
    ckpt_dir = Path(resolved_dir)

    agent_conf = SimpleNamespace(config=OmegaConf.create({"experiment": {"num_envs": 1}}))
    train_state = SimpleNamespace(
        params={"w": jnp.array([1.0])},
        opt_state={"o": jnp.array([0.0])},
        step=jnp.array(1),
        run_stats={},
    )
    agent_state = SimpleNamespace(train_state=train_state)

    manager = create_checkpoint_manager(str(ckpt_dir), async_save=False)
    try:
        md1 = CheckpointMetadata(
            step=1,
            update_number=1,
            global_timestep=10,
            target_global_timestep=10,
            learning_rate=0.1,
            num_envs=1,
            num_steps=1,
            num_minibatches=1,
            update_epochs=1,
        )
        manager.save_checkpoint(1, agent_conf, agent_state, md1)

        md2 = CheckpointMetadata(
            step=2,
            update_number=2,
            global_timestep=20,
            target_global_timestep=20,
            learning_rate=0.1,
            num_envs=1,
            num_steps=1,
            num_minibatches=1,
            update_epochs=1,
            wandb_run_id="source-run",
        )
        manager.save_checkpoint(2, agent_conf, agent_state, md2)
    finally:
        manager.close()

    (ckpt_dir / "checkpoint_999").mkdir(parents=True)

    latest = find_latest_checkpoint(ckpt_dir)
    assert latest == str(ckpt_dir / "checkpoint_2")

    loader = create_checkpoint_manager(str(ckpt_dir), async_save=False)
    try:
        (_, _), metadata = loader.load_checkpoint(latest)
        assert metadata.update_number == 2
        assert metadata.global_timestep == 20
        assert metadata.wandb_run_id == "source-run"
    finally:
        loader.close()


def _assert_same_array_leaves(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for lhs, rhs in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        if hasattr(lhs, "shape"):
            assert lhs.shape == rhs.shape
            assert lhs.dtype == rhs.dtype
        if hasattr(lhs, "dtype") and lhs.dtype == jnp.bool_:
            assert bool(lhs) == bool(rhs)
        else:
            assert jnp.allclose(lhs, rhs, atol=0.0, rtol=0.0)


def test_runtime_target_merge_preserves_datawarp_treedef_metadata():
    from mujoco.mjx.warp.types import DataWarp

    def make_datawarp(static_size: int) -> DataWarp:
        values = {}
        for field in DataWarp.fields():
            if field.type is int:
                values[field.name] = static_size
            elif field.name.startswith("contact__"):
                values[field.name] = jnp.zeros((static_size,), dtype=jnp.float32)
            else:
                values[field.name] = jnp.zeros((1,), dtype=jnp.float32)
        return DataWarp(**values)

    restored = {"env_state": {"data": {"_impl": make_datawarp(8)}}}
    target = {"env_state": {"data": {"_impl": make_datawarp(2)}}}

    merged = _replace_target_owned_runtime_leaves(restored, target)
    merged_impl = merged["env_state"]["data"]["_impl"]
    target_impl = target["env_state"]["data"]["_impl"]

    assert jax.tree_util.tree_structure(merged_impl) == jax.tree_util.tree_structure(target_impl)
    assert merged_impl.naconmax == 2
    assert merged_impl.contact__dim.shape == (2,)
    assert merged_impl.efc__D.shape == (1,)


@pytest.mark.integration
def test_resume_loader_restores_typed_muon_optimizer_state(tmp_path):
    """Orbax resume must preserve optax/Muon state instead of re-initializing it."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    config = OmegaConf.create(
        {
            "experiment": {
                "lr": 4e-4,
                "anneal_lr": True,
                "lr_schedule_type": "linear",
                "warmup_steps": None,
                "min_lr_ratio": 0.1,
                "num_minibatches": 4,
                "update_epochs": 1,
                "num_updates": 20,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "optimizer_type": "muon",
                "muon_config": {
                    "beta": 0.95,
                    "nesterov": True,
                    "ns_steps": 5,
                    "adaptive": False,
                },
            }
        }
    )
    tx = get_optimizer(config.experiment)
    params = {
        "matrix": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) * 1e-3,
        "bias": jnp.ones((4,), dtype=jnp.float32),
    }
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, params)
    for _ in range(3):
        updates, opt_state = tx.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)

    train_state = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(3, dtype=jnp.int32),
        run_stats={},
    )
    agent_conf = SimpleNamespace(config=config, tx=tx, network=DummyNetwork())
    agent_state = SimpleNamespace(train_state=train_state)

    manager = create_checkpoint_manager(str(tmp_path), async_save=False)
    try:
        metadata = CheckpointMetadata(
            step=3,
            update_number=3,
            global_timestep=12,
            target_global_timestep=80,
            learning_rate=1e-4,
            num_envs=2,
            num_steps=2,
            num_minibatches=4,
            update_epochs=1,
        )
        checkpoint_path = manager.save_checkpoint(3, agent_conf, agent_state, metadata)
    finally:
        manager.close()

    loaded_state, resume_info = load_checkpoint_for_resume(checkpoint_path, agent_conf)
    loaded_opt_state = loaded_state.train_state.opt_state

    _assert_same_array_leaves(loaded_opt_state, opt_state)
    assert resume_info["update_number"] == 3
    assert config.experiment.resume_lr_override == pytest.approx(1e-4)
    with pytest.raises(ValueError, match="runtime_state"):
        load_checkpoint_for_resume(
            checkpoint_path,
            agent_conf,
            runtime_state_target={},
            require_runtime_state=True,
        )

    resumed = initialize_train_state(
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        optimizer=tx,
        agent_state=loaded_state,
        config=config.experiment,
        apply_resume_resets=False,
    )
    _assert_same_array_leaves(resumed.opt_state, opt_state)


@pytest.mark.integration
def test_resume_loader_restores_runtime_state(tmp_path):
    """Local auto-resume must restore runner runtime state, not rebuild it."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    config = OmegaConf.create(
        {
            "experiment": {
                "lr": 3e-4,
                "anneal_lr": False,
                "optimizer_type": "adamw",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            }
        }
    )
    tx = get_optimizer(config.experiment)
    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    opt_state = tx.init(params)
    runtime_state = {
        "env_state": {
            "obs_rms": jnp.array([10.0, 20.0], dtype=jnp.float32),
            "data": {"history": jnp.zeros((2, 0, 7), dtype=jnp.float32)},
        },
        "last_obs": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        "rng": jnp.array([123, 456], dtype=jnp.uint32),
        "lr": jnp.array(1.25e-4, dtype=jnp.float32),
        "val_rng": jnp.array([789, 101], dtype=jnp.uint32),
        "curriculum_state": {"threshold": jnp.array(0.42, dtype=jnp.float32)},
        "reward_curriculum_state": {"qvel_w_sum": jnp.array(0.15, dtype=jnp.float32)},
    }
    runtime_state_target = jax.tree_util.tree_map(jnp.zeros_like, runtime_state)

    train_state = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(9, dtype=jnp.int32),
        run_stats={},
    )
    agent_conf = SimpleNamespace(config=config, tx=tx, network=DummyNetwork())
    agent_state = SimpleNamespace(train_state=train_state, runtime_state=runtime_state)

    manager = create_checkpoint_manager(str(tmp_path), async_save=False)
    try:
        metadata = CheckpointMetadata(
            step=9,
            update_number=9,
            global_timestep=36,
            target_global_timestep=80,
            learning_rate=1.25e-4,
            num_envs=2,
            num_steps=2,
            num_minibatches=1,
            update_epochs=1,
        )
        checkpoint_path = manager.save_checkpoint(9, agent_conf, agent_state, metadata)
    finally:
        manager.close()

    loaded_state, resume_info = load_checkpoint_for_resume(
        checkpoint_path,
        agent_conf,
        runtime_state_target=runtime_state_target,
        require_runtime_state=True,
    )

    _assert_same_array_leaves(loaded_state.runtime_state, runtime_state)
    assert resume_info["update_number"] == 9
    assert config.experiment.resume_lr_override == pytest.approx(1.25e-4)


@pytest.mark.integration
def test_resume_loader_uses_checkpoint_shapes_for_typed_runtime_target(tmp_path):
    """A tiny typed target should restore with the checkpoint's saved array shapes."""

    @struct.dataclass
    class RuntimeEnvState:
        obs_rms: jax.Array

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    config = OmegaConf.create(
        {
            "experiment": {
                "lr": 3e-4,
                "anneal_lr": False,
                "optimizer_type": "adamw",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            }
        }
    )
    tx = get_optimizer(config.experiment)
    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    opt_state = tx.init(params)
    runtime_state = {
        "env_state": RuntimeEnvState(
            obs_rms=jnp.arange(12, dtype=jnp.float32).reshape(4, 3),
        )
    }
    runtime_state_target = {
        "env_state": RuntimeEnvState(
            obs_rms=jnp.zeros((1, 3), dtype=jnp.float32),
        )
    }

    train_state = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(5, dtype=jnp.int32),
        run_stats={},
    )
    agent_conf = SimpleNamespace(config=config, tx=tx, network=DummyNetwork())
    agent_state = SimpleNamespace(train_state=train_state, runtime_state=runtime_state)

    manager = create_checkpoint_manager(str(tmp_path), async_save=False)
    try:
        metadata = CheckpointMetadata(
            step=5,
            update_number=5,
            global_timestep=20,
            target_global_timestep=80,
            learning_rate=1.25e-4,
            num_envs=4,
            num_steps=2,
            num_minibatches=1,
            update_epochs=1,
        )
        checkpoint_path = manager.save_checkpoint(5, agent_conf, agent_state, metadata)
    finally:
        manager.close()

    loaded_state, _ = load_checkpoint_for_resume(
        checkpoint_path,
        agent_conf,
        runtime_state_target=runtime_state_target,
        require_runtime_state=True,
    )
    env_state = loaded_state.runtime_state["env_state"]

    assert isinstance(env_state, RuntimeEnvState)
    assert env_state.obs_rms.shape == (4, 3)
    np.testing.assert_array_equal(np.asarray(env_state.obs_rms), np.arange(12, dtype=np.float32).reshape(4, 3))


@pytest.mark.integration
def test_resume_loader_restores_sharded_zero_size_runtime_leaves(tmp_path):
    """Zero-size runtime leaves keep the checkpoint's sharded env batch shape."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    config = OmegaConf.create(
        {
            "experiment": {
                "lr": 3e-4,
                "anneal_lr": False,
                "optimizer_type": "adamw",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            }
        }
    )
    tx = get_optimizer(config.experiment)
    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    opt_state = tx.init(params)
    runtime_state = {
        "last_obs": jnp.zeros((4, 2, 3), dtype=jnp.float32),
        "env_state": {
            "data": {
                "history": jnp.zeros((4, 2, 0), dtype=jnp.float32),
            }
        },
    }
    runtime_state_target = {
        "last_obs": jnp.zeros((4, 1, 3), dtype=jnp.float32),
        "env_state": {
            "data": {
                "history": jnp.zeros((4, 1, 0), dtype=jnp.float32),
            }
        },
    }

    train_state = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(5, dtype=jnp.int32),
        run_stats={},
    )
    agent_conf = SimpleNamespace(config=config, tx=tx, network=DummyNetwork())
    agent_state = SimpleNamespace(train_state=train_state, runtime_state=runtime_state)

    manager = create_checkpoint_manager(str(tmp_path), async_save=False)
    try:
        metadata = CheckpointMetadata(
            step=5,
            update_number=5,
            global_timestep=40,
            target_global_timestep=80,
            learning_rate=1.25e-4,
            num_envs=8,
            num_steps=2,
            num_minibatches=1,
            update_epochs=1,
        )
        checkpoint_path = manager.save_checkpoint(5, agent_conf, agent_state, metadata)
    finally:
        manager.close()

    loaded_state, _ = load_checkpoint_for_resume(
        checkpoint_path,
        agent_conf,
        runtime_state_target=runtime_state_target,
        require_runtime_state=True,
    )

    assert loaded_state.runtime_state["last_obs"].shape == (4, 2, 3)
    assert loaded_state.runtime_state["env_state"]["data"]["history"].shape == (4, 2, 0)


@pytest.mark.integration
def test_resume_loader_uses_target_mjx_contact_buffers(tmp_path):
    """MJX contact buffers are scratch space and should match the current env target."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    config = OmegaConf.create(
        {
            "experiment": {
                "lr": 3e-4,
                "anneal_lr": False,
                "optimizer_type": "adamw",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            }
        }
    )
    tx = get_optimizer(config.experiment)
    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    opt_state = tx.init(params)
    runtime_state = {
        "last_obs": jnp.zeros((4, 2, 3), dtype=jnp.float32),
        "env_state": {
            "data": {
                "_impl": {
                    "contact__dim": jnp.ones((4, 8), dtype=jnp.int32),
                    "non_contact": jnp.arange(32, dtype=jnp.float32).reshape(4, 8),
                    "naconmax": 8,
                }
            }
        },
    }
    runtime_state_target = {
        "last_obs": jnp.zeros((4, 1, 3), dtype=jnp.float32),
        "env_state": {
            "data": {
                "_impl": {
                    "contact__dim": jnp.zeros((4, 2), dtype=jnp.int32),
                    "non_contact": jnp.zeros((4, 1), dtype=jnp.float32),
                    "naconmax": 2,
                }
            }
        },
    }

    train_state = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(5, dtype=jnp.int32),
        run_stats={},
    )
    agent_conf = SimpleNamespace(config=config, tx=tx, network=DummyNetwork())
    agent_state = SimpleNamespace(train_state=train_state, runtime_state=runtime_state)

    manager = create_checkpoint_manager(str(tmp_path), async_save=False)
    try:
        metadata = CheckpointMetadata(
            step=5,
            update_number=5,
            global_timestep=40,
            target_global_timestep=80,
            learning_rate=1.25e-4,
            num_envs=8,
            num_steps=2,
            num_minibatches=1,
            update_epochs=1,
        )
        checkpoint_path = manager.save_checkpoint(5, agent_conf, agent_state, metadata)
    finally:
        manager.close()

    loaded_state, _ = load_checkpoint_for_resume(
        checkpoint_path,
        agent_conf,
        runtime_state_target=runtime_state_target,
        require_runtime_state=True,
    )
    loaded_impl = loaded_state.runtime_state["env_state"]["data"]["_impl"]

    assert loaded_impl["contact__dim"].shape == (4, 2)
    np.testing.assert_array_equal(np.asarray(loaded_impl["contact__dim"]), np.zeros((4, 2), dtype=np.int32))
    assert loaded_impl["naconmax"] == 2
    assert loaded_impl["non_contact"].shape == (4, 8)
    np.testing.assert_array_equal(np.asarray(loaded_impl["non_contact"]), np.arange(32, dtype=np.float32).reshape(4, 8))


@pytest.mark.integration
def test_resume_loader_fills_new_runtime_state_fields(tmp_path):
    """Old runtime checkpoints should restore into the current runtime-state schema."""

    @struct.dataclass
    class OldRewardState:
        last_qvel: jax.Array
        last_action: jax.Array
        imitation_error_total: jax.Array

    @struct.dataclass
    class NewRewardState:
        last_qvel: jax.Array
        last_action: jax.Array
        imitation_error_total: jax.Array

    @struct.dataclass
    class RewardTerminationState:
        low_reward_steps: jax.Array

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    config = OmegaConf.create(
        {
            "experiment": {
                "lr": 3e-4,
                "anneal_lr": False,
                "optimizer_type": "adamw",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
            }
        }
    )
    tx = get_optimizer(config.experiment)
    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    opt_state = tx.init(params)
    runtime_state = {
        "env_state": {
            "additional_carry": {
                "reward_state": OldRewardState(
                    last_qvel=jnp.array([1.0, 2.0], dtype=jnp.float32),
                    last_action=jnp.array([0.5], dtype=jnp.float32),
                    imitation_error_total=jnp.array(0.25, dtype=jnp.float32),
                )
            }
        }
    }
    runtime_state_target = {
        "env_state": {
            "additional_carry": {
                "reward_state": NewRewardState(
                    last_qvel=jnp.zeros((2,), dtype=jnp.float32),
                    last_action=jnp.zeros((1,), dtype=jnp.float32),
                    imitation_error_total=jnp.array(0.0, dtype=jnp.float32),
                ),
                "reward_termination_state": RewardTerminationState(
                    low_reward_steps=jnp.array(0, dtype=jnp.int32),
                ),
            }
        }
    }

    train_state = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(5, dtype=jnp.int32),
        run_stats={},
    )
    agent_conf = SimpleNamespace(config=config, tx=tx, network=DummyNetwork())
    agent_state = SimpleNamespace(train_state=train_state, runtime_state=runtime_state)

    manager = create_checkpoint_manager(str(tmp_path), async_save=False)
    try:
        metadata = CheckpointMetadata(
            step=5,
            update_number=5,
            global_timestep=20,
            target_global_timestep=80,
            learning_rate=1.25e-4,
            num_envs=2,
            num_steps=2,
            num_minibatches=1,
            update_epochs=1,
        )
        checkpoint_path = manager.save_checkpoint(5, agent_conf, agent_state, metadata)
    finally:
        manager.close()

    loaded_state, _ = load_checkpoint_for_resume(
        checkpoint_path,
        agent_conf,
        runtime_state_target=runtime_state_target,
        require_runtime_state=True,
    )
    reward_state = loaded_state.runtime_state["env_state"]["additional_carry"]["reward_state"]
    reward_termination_state = loaded_state.runtime_state["env_state"]["additional_carry"]["reward_termination_state"]

    assert isinstance(reward_state, NewRewardState)
    np.testing.assert_array_equal(np.asarray(reward_state.last_qvel), np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(reward_state.last_action), np.array([0.5], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(reward_state.imitation_error_total), np.array(0.25, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(reward_termination_state.low_reward_steps), np.array(0, dtype=np.int32))
