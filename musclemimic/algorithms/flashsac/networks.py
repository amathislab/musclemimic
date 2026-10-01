"""Norm-constrained actor and distributional double critic."""

from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict


class UnitDense(nn.Module):
    """Dense layer projected to unit-norm output weight vectors."""

    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        kernel = self.param(
            "kernel",
            nn.initializers.orthogonal(1.0),
            (inputs.shape[-1], self.features),
        )
        outputs = inputs @ kernel
        if self.use_bias:
            outputs = outputs + self.param("bias", nn.initializers.zeros, (self.features,))
        return outputs


class UnitBatchNorm(nn.Module):
    """Batch normalization with jointly constrained scale and bias."""

    momentum: float = 0.01
    epsilon: float = 1e-5

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        *,
        training: bool,
        axis_name: str | None = None,
    ) -> jax.Array:
        feature_dim = inputs.shape[-1]
        running_mean = self.variable("batch_stats", "mean", jnp.zeros, (feature_dim,))
        running_var = self.variable("batch_stats", "var", jnp.ones, (feature_dim,))

        if training:
            axes = tuple(range(inputs.ndim - 1))
            sample_sum = jnp.sum(inputs, axis=axes)
            square_sum = jnp.sum(jnp.square(inputs), axis=axes)
            sample_count = jnp.asarray(
                math.prod(inputs.shape[:-1]),
                dtype=inputs.dtype,
            )
            if axis_name is not None:
                sample_sum = jax.lax.psum(sample_sum, axis_name)
                square_sum = jax.lax.psum(square_sum, axis_name)
                sample_count = jax.lax.psum(sample_count, axis_name)
            mean = sample_sum / sample_count
            variance = jnp.maximum(square_sum / sample_count - jnp.square(mean), 0.0)
            running_variance = jnp.where(
                sample_count > 1,
                variance * sample_count / jnp.maximum(sample_count - 1, 1),
                variance,
            )
            running_mean.value = (1.0 - self.momentum) * running_mean.value + self.momentum * mean
            running_var.value = (1.0 - self.momentum) * running_var.value + self.momentum * running_variance
        else:
            mean = running_mean.value
            variance = running_var.value

        affine = self.param(
            "affine",
            lambda _key, shape: jnp.stack((jnp.ones(shape[1]), jnp.zeros(shape[1]))),
            (2, feature_dim),
        )
        normalized = (inputs - mean) * jax.lax.rsqrt(variance + self.epsilon)
        return normalized * affine[0] + affine[1]


class UnitRMSNorm(nn.Module):
    """RMS normalization with a constrained learned scale."""

    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        scale = self.param("scale", nn.initializers.ones, (inputs.shape[-1],))
        rms = jnp.sqrt(jnp.mean(jnp.square(inputs), axis=-1, keepdims=True) + self.epsilon)
        return inputs / rms * scale


class FlashSACBlock(nn.Module):
    hidden_dim: int
    expansion: int = 4

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        *,
        training: bool,
        axis_name: str | None = None,
    ) -> jax.Array:
        residual = inputs
        hidden = UnitDense(self.hidden_dim * self.expansion, name="expand")(inputs)
        hidden = UnitBatchNorm(name="expand_norm")(
            hidden,
            training=training,
            axis_name=axis_name,
        )
        hidden = nn.relu(hidden)
        hidden = UnitDense(self.hidden_dim, name="contract")(hidden)
        hidden = UnitBatchNorm(name="contract_norm")(
            hidden,
            training=training,
            axis_name=axis_name,
        )
        return nn.relu(hidden) + residual


class FlashSACActor(nn.Module):
    action_dim: int
    hidden_dim: int = 128
    num_blocks: int = 2
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        *,
        training: bool,
        axis_name: str | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        hidden = UnitBatchNorm(name="observation_norm")(
            observations,
            training=training,
            axis_name=axis_name,
        )
        hidden = UnitDense(self.hidden_dim, name="embedder")(hidden)
        for block_index in range(self.num_blocks):
            hidden = FlashSACBlock(self.hidden_dim, name=f"block_{block_index}")(
                hidden,
                training=training,
                axis_name=axis_name,
            )
        hidden = UnitRMSNorm(name="post_norm")(hidden)

        mean = UnitDense(self.action_dim, use_bias=True, name="mean")(hidden)
        raw_log_std = UnitDense(self.action_dim, use_bias=True, name="log_std")(hidden)
        scale = 0.5 * (1.0 + jnp.tanh(raw_log_std))
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * scale
        return mean, jnp.exp(log_std)


class _CategoricalCritic(nn.Module):
    hidden_dim: int
    num_blocks: int
    num_bins: int
    min_value: float
    max_value: float

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        *,
        training: bool,
        axis_name: str | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        hidden = jnp.concatenate((observations, actions), axis=-1)
        hidden = UnitBatchNorm(name="input_norm")(
            hidden,
            training=training,
            axis_name=axis_name,
        )
        hidden = UnitDense(self.hidden_dim, name="embedder")(hidden)
        for block_index in range(self.num_blocks):
            hidden = FlashSACBlock(self.hidden_dim, name=f"block_{block_index}")(
                hidden,
                training=training,
                axis_name=axis_name,
            )
        hidden = UnitRMSNorm(name="post_norm")(hidden)
        logits = UnitDense(self.num_bins, use_bias=True, name="value")(hidden)
        log_probabilities = nn.log_softmax(logits, axis=-1)
        bins = jnp.linspace(self.min_value, self.max_value, self.num_bins)
        value = jnp.sum(jnp.exp(log_probabilities) * bins, axis=-1)
        return value, log_probabilities


class FlashSACDoubleCritic(nn.Module):
    """Two independent categorical critics for clipped double Q-learning."""

    hidden_dim: int = 256
    num_blocks: int = 2
    num_bins: int = 101
    min_value: float = -5.0
    max_value: float = 5.0

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        *,
        training: bool,
        axis_name: str | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        outputs = [
            _CategoricalCritic(
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_blocks,
                num_bins=self.num_bins,
                min_value=self.min_value,
                max_value=self.max_value,
                name=f"critic_{critic_index}",
            )(
                observations,
                actions,
                training=training,
                axis_name=axis_name,
            )
            for critic_index in range(2)
        ]
        values, log_probabilities = zip(*outputs, strict=True)
        return jnp.stack(values), jnp.stack(log_probabilities)


def normalize_parameters(params: Any) -> Any:
    """Project every constrained parameter to its prescribed norm."""
    was_frozen = isinstance(params, FrozenDict)
    flattened = flatten_dict(unfreeze(params))

    for path, value in flattened.items():
        parameter_name = path[-1]
        if parameter_name == "kernel":
            denominator = jnp.sqrt(jnp.sum(jnp.square(value), axis=0, keepdims=True) + 1e-8)
            flattened[path] = value / denominator
        elif parameter_name == "affine":
            feature_dim = value.shape[-1]
            denominator = jnp.sqrt(jnp.sum(jnp.square(value)) + 1e-8)
            flattened[path] = value * (math.sqrt(feature_dim) / denominator)
        elif parameter_name == "scale":
            denominator = jnp.sqrt(jnp.sum(jnp.square(value)) + 1e-8)
            flattened[path] = value * (math.sqrt(value.size) / denominator)

    normalized = unflatten_dict(flattened)
    return freeze(normalized) if was_frozen else normalized
