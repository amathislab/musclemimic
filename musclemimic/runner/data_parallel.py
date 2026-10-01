"""Shared runner utilities for one-axis JAX data parallelism."""

from __future__ import annotations

from typing import Any

import jax

from musclemimic.algorithms.common.distributed import DATA_AXIS_NAME

P = jax.sharding.PartitionSpec
DATA = P(DATA_AXIS_NAME)
REPLICATED = P()


def replicated_specs(tree: Any) -> Any:
    """Return a matching pytree whose array leaves are replicated."""
    return _spec_tree(tree, REPLICATED)


def data_specs(tree: Any) -> Any:
    """Return a matching pytree sharded along each leaf's first axis."""
    return _spec_tree(tree, DATA)


def trajectory_specs(trajectory_data: Any | None) -> Any | None:
    """Return the placement spec carried by each trajectory array."""
    if trajectory_data is None:
        return None
    return jax.tree.map(lambda value: value.sharding.spec, trajectory_data)


def require_explicit_trajectory_data(
    env: Any,
    trajectory_data: Any | None,
    *,
    label: str,
) -> None:
    """Reject trajectory environments whose runtime data was not provided."""
    handler = getattr(env, "th", None)
    if handler is not None and getattr(handler, "traj", None) is not None and trajectory_data is None:
        raise ValueError(f"{label} trajectory data must be provided as an explicit runtime input.")


def put_tree(tree: Any | None, specs: Any, mesh: jax.sharding.Mesh) -> Any | None:
    """Place a global pytree according to matching partition specs."""
    if tree is None:
        return None

    def put(value: Any, spec: jax.sharding.PartitionSpec) -> jax.Array:
        return jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))

    return jax.tree.map(put, tree, specs)


def compile_sharded(
    function: Any,
    *,
    mesh: jax.sharding.Mesh,
    in_specs: Any,
    out_specs: Any,
    donate_argnums: tuple[int, ...] = (),
) -> Any:
    """Compile one explicit data-parallel executable."""
    mapped = jax.shard_map(
        function,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=True,
    )
    return jax.jit(mapped, donate_argnums=donate_argnums)


def eval_shard_shape(
    function: Any,
    *,
    mesh: jax.sharding.Mesh,
    in_specs: Any,
    args: tuple[Any, ...],
    out_specs: Any = REPLICATED,
) -> Any:
    """Evaluate shard-local output shapes with mesh collectives in scope."""
    mapped = jax.shard_map(
        function,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    return jax.eval_shape(mapped, *args)


def global_data_shape(local_value: Any, num_devices: int) -> Any:
    """Build an abstract global input from one shard-local leading axis."""
    if local_value is None:
        return None
    if not local_value.shape:
        raise ValueError("A data-sharded value must have a leading axis.")
    return jax.ShapeDtypeStruct(
        (num_devices * local_value.shape[0], *local_value.shape[1:]),
        local_value.dtype,
    )


def _spec_tree(tree: Any, spec: jax.sharding.PartitionSpec) -> Any:
    return jax.tree.map(lambda _value: spec, tree)
