from collections.abc import Callable, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, normal, orthogonal


class SoftMoELayer(nn.Module):
    """
    Soft Mixture of Experts layer that can replace a standard Dense layer.

    This implements a soft routing mechanism where the output is a weighted
    average of all expert outputs, avoiding discrete routing decisions that
    can cause training instabilities.

    Attributes:
        output_dim: Output dimension for each expert
        num_experts: Number of expert networks
        hidden_dim: Hidden dimension for each expert (if None, uses output_dim)
        expert_hidden_ratio: Ratio of expert hidden dim to output dim (default: 0.5)
        use_bias: Whether to use bias in expert networks
        activation: Activation function to use within experts
        temperature: Temperature for gating softmax (lower = sharper selection)
        load_balance_loss_weight: Weight for load balancing auxiliary loss
        gate_init_std: Standard deviation for gating network initialization
    """

    output_dim: int
    num_experts: int = 8
    hidden_dim: int | None = None
    expert_hidden_ratio: float = 0.5
    use_bias: bool = True
    activation: Callable | None = None
    temperature: float = 1.0
    load_balance_loss_weight: float = 0.01
    gate_init_std: float = 0.01

    @nn.compact
    def __call__(self, x, return_metrics=False, axis_name: str | None = None):
        batch_shape = x.shape[:-1]
        hidden_dim = self.hidden_dim or int(self.output_dim * self.expert_hidden_ratio)

        # Gating network: produces soft weights for each expert
        gate_logits = nn.Dense(
            self.num_experts, kernel_init=normal(stddev=self.gate_init_std), bias_init=constant(0.0), name="gate"
        )(x)

        # Apply temperature scaling and softmax for soft routing
        gate_logits = gate_logits / self.temperature
        gate_weights = nn.softmax(gate_logits, axis=-1)

        # Expert networks: each is a small MLP
        expert_outputs = []
        for i in range(self.num_experts):
            # Each expert is a 2-layer MLP
            expert_out = nn.Dense(
                hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=f"expert_{i}_hidden"
            )(x)

            if self.activation is not None:
                expert_out = self.activation(expert_out)

            expert_out = nn.Dense(
                self.output_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0) if self.use_bias else None,
                use_bias=self.use_bias,
                name=f"expert_{i}_output",
            )(expert_out)

            expert_outputs.append(expert_out)

        # Stack expert outputs: [batch..., num_experts, output_dim]
        expert_outputs = jnp.stack(expert_outputs, axis=-2)

        # Soft aggregation: weighted average of expert outputs
        # gate_weights: [batch..., num_experts]
        # expert_outputs: [batch..., num_experts, output_dim]
        gate_weights_expanded = jnp.expand_dims(gate_weights, axis=-1)
        output = jnp.sum(expert_outputs * gate_weights_expanded, axis=-2)

        if return_metrics:
            batch_axes = tuple(range(len(batch_shape)))
            importance = jnp.mean(gate_weights, axis=batch_axes)
            gate_entropy = -jnp.mean(jnp.sum(gate_weights * jnp.log(gate_weights + 1e-10), axis=-1))
            top2_usage = jnp.mean(jnp.sum(jnp.sort(gate_weights, axis=-1)[..., -2:], axis=-1))
            gate_weight_mean = jnp.mean(gate_weights)
            gate_weight_second_moment = jnp.mean(jnp.square(gate_weights))

            # The auxiliary loss is nonlinear in expert importance, so combine
            # replica means before squaring. Averaging local losses is not equal.
            if axis_name is not None:
                importance = jax.lax.pmean(importance, axis_name=axis_name)
                gate_entropy = jax.lax.pmean(gate_entropy, axis_name=axis_name)
                top2_usage = jax.lax.pmean(top2_usage, axis_name=axis_name)
                gate_weight_mean = jax.lax.pmean(gate_weight_mean, axis_name=axis_name)
                gate_weight_second_moment = jax.lax.pmean(
                    gate_weight_second_moment,
                    axis_name=axis_name,
                )

            # Load balancing loss encourages uniform usage
            # Scale by num_experts so minimum is ~1 regardless of expert count
            load_balance_loss = self.num_experts * jnp.sum(importance**2)

            # Expert utilization variance (higher = less uniform)
            expert_utilization_var = jnp.var(importance)
            gate_weight_std = jnp.sqrt(jnp.maximum(gate_weight_second_moment - jnp.square(gate_weight_mean), 0.0))

            metrics = {
                "load_balance_loss": load_balance_loss * self.load_balance_loss_weight,
                "gate_entropy": gate_entropy,
                "expert_importance": importance,
                "expert_utilization_var": expert_utilization_var,
                "top2_expert_usage": top2_usage,
                "gate_weights_mean": gate_weight_mean,
                "gate_weights_std": gate_weight_std,
            }
            return output, metrics

        return output


class SoftMoEActorCritic(nn.Module):
    """
    Actor-Critic network with Soft Mixture of Experts layers.

    This replaces standard Dense layers with SoftMoE layers at specified
    positions in the network architecture.

    Attributes:
        action_dim: Dimension of action space
        activation: Activation function name (e.g., "tanh", "relu")
        init_std: Initial standard deviation for action distribution
        learnable_std: Whether the action std is learnable
        hidden_layer_dims: Dimensions of hidden layers
        actor_obs_ind: Indices of observations to use for actor
        critic_obs_ind: Indices of observations to use for critic
        num_experts: Number of experts in MoE layers
        moe_at_layers: Which layers to apply MoE to (0-indexed hidden layers)
        apply_moe_to_output: Whether to apply MoE to output layers
        apply_moe_to: Whether to apply MoE to "actor", "critic", or "both"
        expert_hidden_ratio: Ratio of expert hidden dim to layer output dim
        temperature: Temperature for soft gating
        load_balance_loss_weight: Weight for load balancing loss
        gate_init_std: Standard deviation for gating network initialization
    """

    action_dim: int
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray | None = None
    critic_obs_ind: jnp.ndarray | None = None
    num_experts: int = 8
    moe_at_layers: Sequence[int] = (1,)  # Apply MoE at second hidden layer by default
    apply_moe_to_output: bool = False  # Explicit control for output layer MoE
    apply_moe_to: str = "both"  # "actor", "critic", or "both"
    expert_hidden_ratio: float = 0.5
    temperature: float = 1.0
    load_balance_loss_weight: float = 0.01
    gate_init_std: float = 0.01
    use_layernorm: bool = False
    layernorm_eps: float = 1e-5

    def setup(self):
        from musclemimic.algorithms.common.networks import RunningMeanStd, get_activation_fn

        self.activation_fn = get_activation_fn(self.activation)
        self.running_mean_std = RunningMeanStd()

    def _build_network(self, x, is_actor=True, return_metrics=False, axis_name: str | None = None):
        """Build either actor or critic network with optional MoE layers."""
        metrics = {}
        total_load_balance_loss = 0.0

        apply_moe = (is_actor and self.apply_moe_to in ["actor", "both"]) or (
            not is_actor and self.apply_moe_to in ["critic", "both"]
        )

        # Build hidden layers
        for i, dim in enumerate(self.hidden_layer_dims):
            if apply_moe and i in self.moe_at_layers:
                # Use MoE layer
                if return_metrics:
                    x, layer_metrics = SoftMoELayer(
                        output_dim=dim,
                        num_experts=self.num_experts,
                        expert_hidden_ratio=self.expert_hidden_ratio,
                        activation=self.activation_fn,
                        temperature=self.temperature,
                        load_balance_loss_weight=self.load_balance_loss_weight,
                        gate_init_std=self.gate_init_std,
                        name=f"{'actor' if is_actor else 'critic'}_moe_layer_{i}",
                    )(x, return_metrics=True, axis_name=axis_name)

                    metrics[f"layer_{i}_metrics"] = layer_metrics
                    total_load_balance_loss += layer_metrics["load_balance_loss"]
                else:
                    x = SoftMoELayer(
                        output_dim=dim,
                        num_experts=self.num_experts,
                        expert_hidden_ratio=self.expert_hidden_ratio,
                        activation=self.activation_fn,
                        temperature=self.temperature,
                        load_balance_loss_weight=self.load_balance_loss_weight,
                        gate_init_std=self.gate_init_std,
                        name=f"{'actor' if is_actor else 'critic'}_moe_layer_{i}",
                    )(x, return_metrics=False)
            else:
                # Use standard Dense layer
                x = nn.Dense(
                    dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f"{'actor' if is_actor else 'critic'}_dense_{i}",
                )(x)

            if self.use_layernorm:
                x = nn.LayerNorm(
                    epsilon=self.layernorm_eps,
                    name=f"{'actor' if is_actor else 'critic'}_ln_{i}",
                )(x)
            x = self.activation_fn(x)

        # Output layer
        if is_actor:
            output_dim = self.action_dim
        else:
            output_dim = 1

        # Check if output layer should use MoE
        if apply_moe and self.apply_moe_to_output:
            if return_metrics:
                x, layer_metrics = SoftMoELayer(
                    output_dim=output_dim,
                    num_experts=self.num_experts,
                    expert_hidden_ratio=self.expert_hidden_ratio,
                    activation=None,  # No activation for output layer
                    temperature=self.temperature,
                    load_balance_loss_weight=self.load_balance_loss_weight,
                    gate_init_std=self.gate_init_std,
                    name=f"{'actor' if is_actor else 'critic'}_moe_output",
                )(x, return_metrics=True, axis_name=axis_name)

                metrics["output_metrics"] = layer_metrics
                total_load_balance_loss += layer_metrics["load_balance_loss"]
            else:
                x = SoftMoELayer(
                    output_dim=output_dim,
                    num_experts=self.num_experts,
                    expert_hidden_ratio=self.expert_hidden_ratio,
                    activation=None,
                    temperature=self.temperature,
                    load_balance_loss_weight=self.load_balance_loss_weight,
                    gate_init_std=self.gate_init_std,
                    name=f"{'actor' if is_actor else 'critic'}_moe_output",
                )(x, return_metrics=False)
        else:
            x = nn.Dense(
                output_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
                name=f"{'actor' if is_actor else 'critic'}_output",
            )(x)

        if return_metrics:
            metrics["total_load_balance_loss"] = total_load_balance_loss
            return x, metrics
        return x

    @nn.compact
    def __call__(
        self,
        x,
        return_metrics: bool = False,
        actor_only: bool = False,
        axis_name: str | None = None,
    ):
        # Normalize input
        x = self.running_mean_std(x)

        all_metrics = {}

        # Build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        if return_metrics:
            actor_mean, actor_metrics = self._build_network(
                actor_x,
                is_actor=True,
                return_metrics=True,
                axis_name=axis_name,
            )
            all_metrics["actor"] = actor_metrics
        else:
            actor_mean = self._build_network(actor_x, is_actor=True, return_metrics=False)

        # Actor std
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)), (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        if actor_only:
            return pi

        # Build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        if return_metrics:
            critic_out, critic_metrics = self._build_network(
                critic_x,
                is_actor=False,
                return_metrics=True,
                axis_name=axis_name,
            )
            all_metrics["critic"] = critic_metrics
        else:
            critic_out = self._build_network(critic_x, is_actor=False, return_metrics=False)

        critic_value = jnp.squeeze(critic_out, axis=-1)

        if return_metrics:
            # Combine load balance losses
            total_loss = all_metrics.get("actor", {}).get("total_load_balance_loss", 0.0) + all_metrics.get(
                "critic", {}
            ).get("total_load_balance_loss", 0.0)
            all_metrics["total_moe_loss"] = total_loss
            return pi, critic_value, all_metrics

        return pi, critic_value

    def update_observation_stats(self, x, axis_name: str | None = None):
        """Update only the observation normalizer, without running the policy."""
        return self.running_mean_std(x, update_stats=True, axis_name=axis_name)
