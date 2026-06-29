"""Unit tests for musclemimic.rl_core.rollout_buffer."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from musclemimic.rl_core.rollout_buffer import RolloutBuffer, compute_gae, create_minibatches


class SimpleTransition(NamedTuple):
    """Minimal transition for testing."""

    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    done: jnp.ndarray
    absorbing: jnp.ndarray


class TestComputeGAE:
    """Tests for compute_gae function."""

    def test_basic_computation(self):
        """Test GAE against hand-computed values."""
        rewards = jnp.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        values = jnp.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])
        dones = jnp.zeros((3, 2))
        absorbing = jnp.zeros((3, 2))
        last_value = jnp.array([0.5, 1.0])

        advantages, returns = compute_gae(rewards, values, dones, absorbing, last_value, 0.99, 0.95)

        # Manual: delta = 1 + 0.99*0.5 - 0.5 = 0.995
        # Step 2: gae = 0.995
        # Step 1: gae = 0.995 + 0.99*0.95*0.995 = 1.9308
        # Step 0: gae = 0.995 + 0.99*0.95*1.9308 = 2.8109
        assert jnp.allclose(advantages[2, 0], 0.995, atol=1e-4)
        assert jnp.allclose(advantages[1, 0], 1.9308, atol=1e-3)
        assert jnp.allclose(advantages[0, 0], 2.8109, atol=1e-3)
        assert jnp.allclose(returns, advantages + values)

    def test_absorbing_stops_bootstrap(self):
        """Test that absorbing=1 prevents value bootstrapping."""
        rewards = jnp.array([[1.0], [1.0], [1.0]])
        values = jnp.array([[0.5], [0.5], [0.5]])
        dones = jnp.array([[0.0], [1.0], [0.0]])
        absorbing = jnp.array([[0.0], [1.0], [0.0]])
        last_value = jnp.array([0.5])

        advantages, _ = compute_gae(rewards, values, dones, absorbing, last_value, 0.99, 0.95)

        # Step 1 (absorbing=1): delta = 1 + 0 - 0.5 = 0.5
        assert jnp.allclose(advantages[1, 0], 0.5, atol=1e-4)

    def test_done_without_absorbing_bootstraps(self):
        """Test done=1 with absorbing=0 still bootstraps (time limit case)."""
        rewards = jnp.array([[1.0], [1.0], [1.0]])
        values = jnp.array([[0.5], [0.5], [0.5]])
        dones = jnp.array([[0.0], [1.0], [0.0]])
        absorbing = jnp.array([[0.0], [0.0], [0.0]])
        last_value = jnp.array([0.5])

        advantages, _ = compute_gae(rewards, values, dones, absorbing, last_value, 0.99, 0.95)

        # Step 1 (done=1, absorbing=0): delta = 1 + 0.99*0.5 - 0.5 = 0.995
        assert jnp.allclose(advantages[1, 0], 0.995, atol=1e-4)

    def test_bootstrap_mask_stops_success_terminal_bootstrap(self):
        """A successful terminal step can stop bootstrap without being absorbing."""
        rewards = jnp.array([[1.0], [1.0], [1.0]])
        values = jnp.array([[0.5], [0.5], [0.5]])
        dones = jnp.array([[0.0], [1.0], [0.0]])
        absorbing = jnp.array([[0.0], [0.0], [0.0]])
        bootstrap_mask = jnp.array([[1.0], [0.0], [1.0]])
        last_value = jnp.array([999.0])

        advantages, _ = compute_gae(
            rewards,
            values,
            dones,
            absorbing,
            last_value,
            0.99,
            0.95,
            bootstrap_mask=bootstrap_mask,
        )

        # Step 1 is done but not absorbing; bootstrap_mask makes it terminal-success.
        assert jnp.allclose(advantages[1, 0], 0.5, atol=1e-4)

    def test_shifted_values_bootstrap_from_next_rollout_value(self):
        """Bootstrap uses shifted rollout values and only last step uses last_value."""
        rewards = jnp.array([[1.0], [1.0], [1.0]])
        values = jnp.array([[0.5], [2.0], [4.0]])
        dones = jnp.zeros((3, 1))
        absorbing = jnp.zeros((3, 1))
        last_value = jnp.array([8.0])

        advantages, _ = compute_gae(
            rewards,
            values,
            dones,
            absorbing,
            last_value,
            0.99,
            0.95,
        )

        # Step 1 delta uses values[2], while the final step uses last_value.
        expected_step1_delta = 1.0 + 0.99 * 4.0 - 2.0
        expected_step2_adv = 1.0 + 0.99 * 8.0 - 4.0
        assert jnp.allclose(advantages[1, 0], expected_step1_delta + 0.99 * 0.95 * expected_step2_adv, atol=1e-4)
        assert jnp.allclose(advantages[2, 0], expected_step2_adv, atol=1e-4)

    def test_jit_compatible(self):
        """Test that compute_gae works under JIT."""

        @jax.jit
        def jitted_gae(r, v, d, a, lv):
            return compute_gae(r, v, d, a, lv, 0.99, 0.95)

        rewards = jnp.ones((10, 4))
        values = jnp.ones((10, 4)) * 0.5
        dones = jnp.zeros((10, 4))
        absorbing = jnp.zeros((10, 4))
        last_value = jnp.ones(4) * 0.5

        advantages, returns = jitted_gae(rewards, values, dones, absorbing, last_value)
        assert advantages.shape == (10, 4)
        assert returns.shape == (10, 4)

    def test_unroll_parameter_does_not_change_result(self):
        """Different scan unroll factors should produce identical output."""
        num_steps, num_envs = 6, 3
        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 5)

        rewards = jax.random.normal(keys[0], (num_steps, num_envs))
        values = jax.random.normal(keys[1], (num_steps, num_envs))
        dones = jax.random.bernoulli(keys[2], 0.2, (num_steps, num_envs)).astype(jnp.float32)
        absorbing = jax.random.bernoulli(keys[3], 0.1, (num_steps, num_envs)).astype(jnp.float32)
        last_value = jax.random.normal(keys[4], (num_envs,))

        adv_default, ret_default = compute_gae(
            rewards, values, dones, absorbing, last_value, gamma=0.99, gae_lambda=0.95, unroll=16
        )
        adv_unrolled, ret_unrolled = compute_gae(
            rewards, values, dones, absorbing, last_value, gamma=0.99, gae_lambda=0.95, unroll=1
        )

        assert jnp.allclose(adv_default, adv_unrolled, atol=1e-6)
        assert jnp.allclose(ret_default, ret_unrolled, atol=1e-6)

    def test_all_done_zero_bootstrap(self):
        """
        All steps are terminal AND absorbing.
        Ensures last_value is NOT used for bootstrapping
        and returns reduce to rewards only.
        Guards against leaking bootstrap values.
        """
        rewards = jnp.ones((4, 2))
        values = jnp.ones((4, 2)) * 0.5
        dones = jnp.ones((4, 2))
        absorbing = jnp.ones((4, 2))
        last_value = jnp.ones(2) * 999.0  # should be ignored

        _adv, ret = compute_gae(rewards, values, dones, absorbing, last_value, 0.99, 0.95)

        # returns should reduce to reward only
        assert jnp.allclose(ret, rewards, atol=1e-6)


class TestCreateMinibatches:
    """Tests for create_minibatches function."""

    def test_shapes(self):
        """Test output shapes are correct."""
        num_steps, num_envs, obs_dim = 20, 64, 32
        num_minibatches = 4
        batch_size = num_steps * num_envs
        mb_size = batch_size // num_minibatches

        traj = SimpleTransition(
            obs=jnp.zeros((num_steps, num_envs, obs_dim)),
            action=jnp.zeros((num_steps, num_envs, 8)),
            reward=jnp.zeros((num_steps, num_envs)),
            value=jnp.zeros((num_steps, num_envs)),
            done=jnp.zeros((num_steps, num_envs)),
            absorbing=jnp.zeros((num_steps, num_envs)),
        )
        advantages = jnp.zeros((num_steps, num_envs))
        targets = jnp.zeros((num_steps, num_envs))

        batched_traj, batched_adv, batched_tgt = create_minibatches(
            traj, advantages, targets, num_minibatches, jax.random.PRNGKey(0)
        )

        assert batched_traj.obs.shape == (num_minibatches, mb_size, obs_dim)
        assert batched_traj.action.shape == (num_minibatches, mb_size, 8)
        assert batched_traj.reward.shape == (num_minibatches, mb_size)
        assert batched_adv.shape == (num_minibatches, mb_size)
        assert batched_tgt.shape == (num_minibatches, mb_size)

    def test_no_data_loss(self):
        """Test all data is preserved after shuffling."""
        num_steps, num_envs = 10, 8
        batch_size = num_steps * num_envs

        # Use unique values to verify no loss
        rewards = jnp.arange(batch_size, dtype=jnp.float32).reshape(num_steps, num_envs)
        traj = SimpleTransition(
            obs=jnp.zeros((num_steps, num_envs, 4)),
            action=jnp.zeros((num_steps, num_envs, 2)),
            reward=rewards,
            value=jnp.zeros((num_steps, num_envs)),
            done=jnp.zeros((num_steps, num_envs)),
            absorbing=jnp.zeros((num_steps, num_envs)),
        )
        advantages = jnp.zeros((num_steps, num_envs))
        targets = jnp.zeros((num_steps, num_envs))

        batched_traj, _, _ = create_minibatches(traj, advantages, targets, 4, jax.random.PRNGKey(42))

        # Flatten and check all values present
        all_rewards = batched_traj.reward.reshape(-1)
        unique = jnp.unique(all_rewards)
        assert len(unique) == batch_size

    def test_raises_on_non_divisible(self):
        """Validate divisibility check for minibatches."""
        num_steps, num_envs = 5, 3  # 15 total samples
        traj = SimpleTransition(
            obs=jnp.zeros((num_steps, num_envs, 4)),
            action=jnp.zeros((num_steps, num_envs, 2)),
            reward=jnp.zeros((num_steps, num_envs)),
            value=jnp.zeros((num_steps, num_envs)),
            done=jnp.zeros((num_steps, num_envs)),
            absorbing=jnp.zeros((num_steps, num_envs)),
        )
        adv = jnp.zeros((num_steps, num_envs))
        tgt = jnp.zeros((num_steps, num_envs))
        with pytest.raises(ValueError):
            create_minibatches(traj, adv, tgt, num_minibatches=4, rng=jax.random.PRNGKey(0))

    def test_jit_compatible(self):
        """Test minibatch creation under JIT."""
        num_steps, num_envs = 10, 8

        traj = SimpleTransition(
            obs=jnp.zeros((num_steps, num_envs, 4)),
            action=jnp.zeros((num_steps, num_envs, 2)),
            reward=jnp.zeros((num_steps, num_envs)),
            value=jnp.zeros((num_steps, num_envs)),
            done=jnp.zeros((num_steps, num_envs)),
            absorbing=jnp.zeros((num_steps, num_envs)),
        )

        @jax.jit
        def jitted_minibatch(rng):
            adv = jnp.zeros((num_steps, num_envs))
            tgt = jnp.zeros((num_steps, num_envs))
            return create_minibatches(traj, adv, tgt, 4, rng)

        batched_traj, _, _ = jitted_minibatch(jax.random.PRNGKey(0))
        assert batched_traj.obs.shape == (4, 20, 4)

    def test_deterministic_given_same_rng(self):
        """
        Using the same RNG must produce identical minibatches.
        This guarantees reproducibility and debuggability.
        """
        num_steps, num_envs = 10, 8
        traj = SimpleTransition(
            obs=jnp.arange(num_steps * num_envs * 2).reshape(num_steps, num_envs, 2),
            action=jnp.zeros((num_steps, num_envs, 1)),
            reward=jnp.zeros((num_steps, num_envs)),
            value=jnp.zeros((num_steps, num_envs)),
            done=jnp.zeros((num_steps, num_envs)),
            absorbing=jnp.zeros((num_steps, num_envs)),
        )
        adv = jnp.zeros((num_steps, num_envs))
        tgt = jnp.zeros((num_steps, num_envs))
        rng = jax.random.PRNGKey(0)

        out1 = create_minibatches(traj, adv, tgt, num_minibatches=4, rng=rng)
        out2 = create_minibatches(traj, adv, tgt, num_minibatches=4, rng=rng)

        assert jnp.allclose(out1[0].obs, out2[0].obs)


class TestRolloutBuffer:
    """Tests for RolloutBuffer class."""

    def test_from_scan(self):
        """Test creating buffer from scan output."""
        num_steps, num_envs = 20, 64
        traj = SimpleTransition(
            obs=jnp.zeros((num_steps, num_envs, 32)),
            action=jnp.zeros((num_steps, num_envs, 8)),
            reward=jnp.zeros((num_steps, num_envs)),
            value=jnp.zeros((num_steps, num_envs)),
            done=jnp.zeros((num_steps, num_envs)),
            absorbing=jnp.zeros((num_steps, num_envs)),
        )

        buffer = RolloutBuffer.from_scan(traj)

        assert buffer.num_steps == num_steps
        assert buffer.num_envs == num_envs
        assert buffer.batch_size == num_steps * num_envs

    def test_compute_advantages(self):
        """Test buffer's compute_advantages method."""
        num_steps, num_envs = 5, 4
        traj = SimpleTransition(
            obs=jnp.zeros((num_steps, num_envs, 8)),
            action=jnp.zeros((num_steps, num_envs, 2)),
            reward=jnp.ones((num_steps, num_envs)),
            value=jnp.ones((num_steps, num_envs)) * 0.5,
            done=jnp.zeros((num_steps, num_envs)),
            absorbing=jnp.zeros((num_steps, num_envs)),
        )

        buffer = RolloutBuffer.from_scan(traj)
        last_value = jnp.ones(num_envs) * 0.5

        advantages, targets = buffer.compute_advantages(last_value, gamma=0.99, gae_lambda=0.95)

        assert advantages.shape == (num_steps, num_envs)
        assert targets.shape == (num_steps, num_envs)

    def test_get_minibatches(self):
        """Test buffer's get_minibatches method."""
        num_steps, num_envs = 20, 16
        traj = SimpleTransition(
            obs=jnp.zeros((num_steps, num_envs, 8)),
            action=jnp.zeros((num_steps, num_envs, 2)),
            reward=jnp.zeros((num_steps, num_envs)),
            value=jnp.zeros((num_steps, num_envs)),
            done=jnp.zeros((num_steps, num_envs)),
            absorbing=jnp.zeros((num_steps, num_envs)),
        )

        buffer = RolloutBuffer.from_scan(traj)
        advantages = jnp.zeros((num_steps, num_envs))
        targets = jnp.zeros((num_steps, num_envs))

        batched_traj, batched_adv, _batched_tgt = buffer.get_minibatches(
            advantages, targets, num_minibatches=8, rng=jax.random.PRNGKey(0)
        )

        mb_size = (num_steps * num_envs) // 8
        assert batched_traj.obs.shape == (8, mb_size, 8)
        assert batched_adv.shape == (8, mb_size)

    def test_allocate_and_add_transition(self):
        """Test writing a transition into an allocated buffer."""
        num_steps, num_envs = 3, 2
        transition = SimpleTransition(
            obs=jnp.ones((num_envs, 4)),
            action=jnp.ones((num_envs, 1)) * 2,
            reward=jnp.ones((num_envs,)),
            value=jnp.ones((num_envs,)) * 0.5,
            done=jnp.zeros((num_envs,)),
            absorbing=jnp.zeros((num_envs,)),
        )

        buffer = RolloutBuffer.allocate(num_steps, num_envs, transition)
        buffer = buffer.add_transition(transition, step=1)

        assert jnp.allclose(buffer.traj_batch.obs[1], transition.obs)
        assert jnp.allclose(buffer.traj_batch.reward[1], transition.reward)

    def test_add_transition_out_of_bounds(self):
        """
        Writing past allocated rollout length must error.
        Prevents silent memory corruption and off-by-one bugs.
        """
        num_steps, num_envs = 3, 2
        transition = SimpleTransition(
            obs=jnp.zeros((num_envs, 4)),
            action=jnp.zeros((num_envs, 1)),
            reward=jnp.zeros((num_envs,)),
            value=jnp.zeros((num_envs,)),
            done=jnp.zeros((num_envs,)),
            absorbing=jnp.zeros((num_envs,)),
        )

        buffer = RolloutBuffer.allocate(num_steps, num_envs, transition)

        with pytest.raises(IndexError):
            buffer.add_transition(transition, step=num_steps)

    def test_from_scan_matches_manual_add(self):
        """
        from_scan() and allocate()+add_transition()
        must produce identical buffers when used correctly.

        This test verifies that scan-based rollout collection
        and step-wise rollout collection are equivalent.
        """
        num_steps, num_envs, obs_dim = 4, 3, 2

        traj = SimpleTransition(
            obs=jnp.arange(num_steps * num_envs * obs_dim).reshape(num_steps, num_envs, obs_dim),
            action=jnp.zeros((num_steps, num_envs, 1)),
            reward=jnp.ones((num_steps, num_envs)),
            value=jnp.zeros((num_steps, num_envs)),
            done=jnp.zeros((num_steps, num_envs)),
            absorbing=jnp.zeros((num_steps, num_envs)),
        )

        # Buffer built from scan output
        buffer_scan = RolloutBuffer.from_scan(traj)

        # Allocate using a single-step transition prototype.
        step0 = SimpleTransition(
            obs=traj.obs[0],
            action=traj.action[0],
            reward=traj.reward[0],
            value=traj.value[0],
            done=traj.done[0],
            absorbing=traj.absorbing[0],
        )

        buffer_manual = RolloutBuffer.allocate(num_steps, num_envs, step0)

        # Write transitions step-by-step
        for t in range(num_steps):
            step_transition = SimpleTransition(
                obs=traj.obs[t],
                action=traj.action[t],
                reward=traj.reward[t],
                value=traj.value[t],
                done=traj.done[t],
                absorbing=traj.absorbing[t],
            )
            buffer_manual = buffer_manual.add_transition(step_transition, step=t)

        # Now shapes and values should match exactly
        assert buffer_scan.traj_batch.obs.shape == buffer_manual.traj_batch.obs.shape
        assert jnp.allclose(buffer_scan.traj_batch.obs, buffer_manual.traj_batch.obs)
        assert jnp.allclose(buffer_scan.traj_batch.reward, buffer_manual.traj_batch.reward)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
