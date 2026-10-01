"""Standalone two-GPU correctness probe for MJX's plain-Warp graph mode.

Run this module from the exact installed MuscleMimic environment, passing the
full commit pinned by the parent project::

    python -m musclemimic.runner.warp_multi_gpu_probe --expected-commit <40-character SHA>

The probe is intentionally strict: it succeeds only with exactly two visible
JAX CUDA devices and exactly two visible Warp CUDA devices.  It runs one
minimal MJX simulation through ``pmap`` and ``lax.scan`` twice.  Each result
tree is fully blocked before clearing the plain-Warp graph cache, so the second
execution validates clean recapture rather than reusing the first capture.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from musclemimic.runner.warp_graph_cache import clear_warp_graph_cache

_AXIS_NAME = "warp_probe_devices"
_DEVICE_COUNT = 2
_SCAN_STEPS = 3
_NACONMAX = 4
_NJMAX = 4
_FULL_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")

_MODEL_XML = """\
<mujoco model="musclemimic_warp_multi_gpu_probe">
  <option timestep="0.01"/>
  <worldbody>
    <body name="probe_body" pos="0 0 1">
      <joint name="probe_free_joint" type="free"/>
      <geom name="probe_sphere" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class ProbeError(RuntimeError):
    """A fail-closed rejection from the two-GPU hardware probe."""


@dataclass(frozen=True)
class _ProbeSimulation:
    mj_model: Any
    mj_data: Any
    host_system: Any


def _device_identity(device: Any) -> tuple[str, int]:
    try:
        return str(device.platform), int(device.id)
    except (AttributeError, TypeError, ValueError) as error:
        raise ProbeError(f"Unrecognized JAX device descriptor: {device!r}.") from error


def _validate_device_inventory(jax_devices: Sequence[Any], warp_devices: Sequence[Any]) -> None:
    """Require exactly two distinct JAX GPUs and two Warp CUDA devices."""
    if len(jax_devices) != _DEVICE_COUNT:
        raise ProbeError(f"Expected exactly two visible JAX CUDA devices; found {len(jax_devices)}.")

    identities = tuple(_device_identity(device) for device in jax_devices)
    non_gpu = tuple(identity for identity in identities if identity[0] != "gpu")
    if non_gpu:
        raise ProbeError(f"Every JAX probe device must be a GPU; found {non_gpu!r}.")
    if len(set(identities)) != _DEVICE_COUNT:
        raise ProbeError(f"JAX probe devices must be distinct; found {identities!r}.")

    if len(warp_devices) != _DEVICE_COUNT:
        raise ProbeError(f"Expected exactly two visible Warp CUDA devices; found {len(warp_devices)}.")


def _discover_devices() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Discover the two CUDA inventories without allowing a CPU fallback."""
    try:
        jax_devices = tuple(jax.local_devices(backend="gpu"))
    except Exception as error:
        raise ProbeError("JAX could not initialize its CUDA backend; CPU fallback is not accepted.") from error

    try:
        import warp as wp

        if not wp.is_cuda_available():
            raise ProbeError("Warp reports that CUDA is unavailable; CPU fallback is not accepted.")
        warp_devices = tuple(wp.get_cuda_devices())
    except ProbeError:
        raise
    except Exception as error:
        raise ProbeError("Warp could not enumerate its CUDA devices.") from error

    _validate_device_inventory(jax_devices, warp_devices)
    return jax_devices, warp_devices


def _model_graph_mode(model: Any) -> Any | None:
    direct_mode = getattr(model, "graph_mode", None)
    if direct_mode is not None:
        return direct_mode
    options = getattr(model, "opt", None)
    implementation_options = getattr(options, "_impl", None)
    return getattr(implementation_options, "graph_mode", None)


def _require_plain_warp_graph_mode(model: Any) -> None:
    """Reject staged graph modes, whose capture state is not replica-local."""
    from mujoco.mjx.warp.types import GraphMode

    mode = _model_graph_mode(model)
    if mode is None:
        raise ProbeError("The probe model does not expose its MJX-Warp graph mode.")
    resolved_mode = int(np.asarray(mode))
    if resolved_mode != int(GraphMode.WARP):
        raise ProbeError(f"Expected GraphMode.WARP ({int(GraphMode.WARP)}); found graph mode {resolved_mode}.")


def _build_simulation() -> _ProbeSimulation:
    """Build a host-resident minimal MJX model configured for plain Warp."""
    from mujoco.mjx.warp.types import GraphMode

    mj_model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    mj_data = mujoco.MjData(mj_model)
    host_system = jax.device_get(mjx.put_model(mj_model, impl="warp", graph_mode=GraphMode.WARP))
    _require_plain_warp_graph_mode(host_system)
    return _ProbeSimulation(mj_model=mj_model, mj_data=mj_data, host_system=host_system)


def _make_pmapped_rollout(simulation: _ProbeSimulation, devices: Sequence[Any]) -> Callable[[Any], Mapping[str, Any]]:
    """Compile the minimal MJX rollout over the two explicitly selected GPUs."""

    def rollout(sentinel: jax.Array) -> Mapping[str, jax.Array]:
        axis_id = jax.lax.axis_index(_AXIS_NAME)
        system = jax.tree.map(jnp.asarray, simulation.host_system)
        data = mjx.put_data(
            simulation.mj_model,
            simulation.mj_data,
            impl="warp",
            naconmax=_NACONMAX,
            njmax=_NJMAX,
        )
        data = data.replace(qpos=data.qpos.at[0].set(sentinel))
        data = mjx.forward(system, data)
        initial_x = data.qpos[0]

        def step(data: Any, _: Any) -> tuple[Any, jax.Array]:
            data = mjx.step(system, data)
            state_vector = jnp.concatenate((data.qpos, data.qvel))
            return data, state_vector

        data, trajectory = jax.lax.scan(step, data, xs=None, length=_SCAN_STEPS)
        return {
            "axis_id": axis_id,
            "final_x": data.qpos[0],
            "initial_x": initial_x,
            "sentinel": sentinel,
            "trajectory": trajectory,
        }

    return jax.pmap(rollout, axis_name=_AXIS_NAME, devices=tuple(devices))


def _put_sentinel_shards(sentinels: np.ndarray, devices: Sequence[Any]) -> jax.Array:
    """Place one sentinel directly on each probe device with current JAX APIs."""
    values = np.asarray(sentinels, dtype=np.float32)
    if values.shape != (_DEVICE_COUNT,):
        raise ProbeError(f"Sentinels must have shape ({_DEVICE_COUNT},); found {values.shape!r}.")
    if len(devices) != _DEVICE_COUNT:
        raise ProbeError(f"Sentinel placement requires exactly {_DEVICE_COUNT} devices; found {len(devices)}.")

    mesh = jax.sharding.Mesh(np.asarray(devices, dtype=object), (_AXIS_NAME,))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_AXIS_NAME))
    return jax.device_put(values, sharding)


def _execute_twice(
    pmapped_rollout: Callable[[Any], Any],
    device_inputs: Sequence[Any],
    *,
    block_until_ready: Callable[[Any], Any],
    clear_graph_cache: Callable[[], None],
) -> tuple[Any, Any]:
    """Execute, fully block, and clear twice to validate clean recapture."""
    if len(device_inputs) != 2:
        raise ProbeError(f"The probe requires exactly two executions; received {len(device_inputs)}.")

    completed = []
    for device_input in device_inputs:
        result = pmapped_rollout(device_input)
        completed.append(block_until_ready(result))
        # The complete result pytree is quiescent here.  Clearing before this
        # block, or after a failed block, could free an in-flight capture.
        # Clearing now forces the next execution to recapture from scratch.
        clear_graph_cache()
    return completed[0], completed[1]


def _validate_device_shards(result: Mapping[str, Any], devices: Sequence[Any]) -> None:
    expected = {_device_identity(device) for device in devices}
    for name, value in result.items():
        _validate_array_shards(f"Result {name!r}", value, expected)


def _validate_array_shards(label: str, value: Any, expected: set[tuple[str, int]]) -> None:
    shards = getattr(value, "addressable_shards", None)
    if shards is None:
        raise ProbeError(f"{label} does not expose device shards.")
    actual = [_device_identity(shard.device) for shard in shards]
    if len(actual) != _DEVICE_COUNT or set(actual) != expected:
        raise ProbeError(
            f"{label} is not represented once on each probe GPU: expected {sorted(expected)!r}, found {actual!r}."
        )


def _to_host_result(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {name: np.asarray(jax.device_get(value)) for name, value in result.items()}


def _validate_host_result(result: Mapping[str, np.ndarray], expected_sentinels: np.ndarray) -> None:
    required = {"axis_id", "final_x", "initial_x", "sentinel", "trajectory"}
    missing = required.difference(result)
    if missing:
        raise ProbeError(f"Probe result is missing fields: {sorted(missing)!r}.")

    for name in sorted(required):
        value = np.asarray(result[name])
        if value.shape[0] != _DEVICE_COUNT:
            raise ProbeError(f"Result {name!r} has leading size {value.shape[0]}, expected {_DEVICE_COUNT}.")
        if not np.all(np.isfinite(value)):
            raise ProbeError(f"Result {name!r} contains a non-finite value.")

    np.testing.assert_array_equal(result["axis_id"], np.arange(_DEVICE_COUNT, dtype=np.int32))
    np.testing.assert_array_equal(result["sentinel"], expected_sentinels)
    np.testing.assert_allclose(result["initial_x"], expected_sentinels, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result["final_x"], expected_sentinels, rtol=0.0, atol=1e-6)

    trajectory = np.asarray(result["trajectory"])
    if trajectory.ndim != 3 or trajectory.shape[1] != _SCAN_STEPS:
        raise ProbeError(
            f"Result 'trajectory' has shape {trajectory.shape!r}; expected two replicas and {_SCAN_STEPS} scan steps."
        )
    expected_trajectory_x = np.broadcast_to(expected_sentinels[:, np.newaxis], (_DEVICE_COUNT, _SCAN_STEPS))
    np.testing.assert_allclose(trajectory[:, :, 0], expected_trajectory_x, rtol=0.0, atol=1e-6)


def _validate_executions(
    results: Sequence[Mapping[str, Any]],
    expected_sentinels: Sequence[np.ndarray],
    devices: Sequence[Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if len(results) != 2 or len(expected_sentinels) != 2:
        raise ProbeError("Exactly two completed executions are required for validation.")

    host_results = []
    for result, sentinels in zip(results, expected_sentinels, strict=True):
        _validate_device_shards(result, devices)
        host_result = _to_host_result(result)
        _validate_host_result(host_result, sentinels)
        host_results.append(host_result)

    first_sentinels = np.asarray(host_results[0]["sentinel"])
    second_sentinels = np.asarray(host_results[1]["sentinel"])
    if np.unique(first_sentinels).size != _DEVICE_COUNT or np.unique(second_sentinels).size != _DEVICE_COUNT:
        raise ProbeError("Each execution must use one distinct sentinel per replica.")
    if np.any(first_sentinels == second_sentinels):
        raise ProbeError("Every replica sentinel must change between the two executions.")
    return host_results[0], host_results[1]


def _device_report(device: Any) -> dict[str, Any]:
    platform, device_id = _device_identity(device)
    return {
        "device_kind": str(getattr(device, "device_kind", "unknown")),
        "id": device_id,
        "platform": platform,
    }


def _installed_provenance() -> dict[str, Any]:
    """Return the installed package location and PEP 610 source evidence."""
    import musclemimic

    package_file = getattr(musclemimic, "__file__", None)
    if package_file is None:
        raise ProbeError("The imported musclemimic package does not expose __file__ provenance.")

    distribution = metadata.distribution("musclemimic")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = None if direct_url_text is None else json.loads(direct_url_text)
    return {
        "direct_url": direct_url,
        "distribution_version": distribution.version,
        "musclemimic_file": str(Path(package_file).resolve()),
    }


def _validate_expected_commit(provenance: Mapping[str, Any], expected_commit: str) -> str:
    """Require PEP 610 evidence for the exact parent-project dependency pin."""
    normalized_expected = expected_commit.strip().lower()
    if _FULL_COMMIT_PATTERN.fullmatch(normalized_expected) is None:
        raise ProbeError("--expected-commit must be a full 40-character hexadecimal Git commit.")

    direct_url = provenance.get("direct_url")
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, Mapping) else None
    actual_commit = vcs_info.get("commit_id") if isinstance(vcs_info, Mapping) else None
    if not isinstance(actual_commit, str) or _FULL_COMMIT_PATTERN.fullmatch(actual_commit.lower()) is None:
        raise ProbeError("Installed MuscleMimic provenance does not contain a full PEP 610 VCS commit_id.")
    if actual_commit.lower() != normalized_expected:
        raise ProbeError(
            "Installed MuscleMimic commit does not match the required pin: "
            f"expected {normalized_expected}, found {actual_commit.lower()}."
        )
    return normalized_expected


def run_probe(*, expected_commit: str) -> dict[str, Any]:
    """Run the strict hardware probe and return its machine-readable evidence."""
    import jaxlib
    import warp as wp

    provenance = _installed_provenance()
    normalized_expected_commit = _validate_expected_commit(provenance, expected_commit)
    devices, warp_devices = _discover_devices()
    simulation = _build_simulation()
    rollout = _make_pmapped_rollout(simulation, devices)

    expected_sentinels = (
        np.asarray((101.0, 102.0), dtype=np.float32),
        np.asarray((201.0, 202.0), dtype=np.float32),
    )
    device_inputs = tuple(_put_sentinel_shards(sentinels, devices) for sentinels in expected_sentinels)
    expected_device_identities = {_device_identity(device) for device in devices}
    for execution, device_input in enumerate(device_inputs, start=1):
        _validate_array_shards(f"Execution {execution} input", device_input, expected_device_identities)
    results = _execute_twice(
        rollout,
        device_inputs,
        block_until_ready=jax.block_until_ready,
        clear_graph_cache=clear_warp_graph_cache,
    )
    host_results = _validate_executions(results, expected_sentinels, devices)

    return {
        "executions": [
            {
                "axis_ids": result["axis_id"].tolist(),
                "final_x": result["final_x"].tolist(),
                "sentinels": result["sentinel"].tolist(),
            }
            for result in host_results
        ],
        "graph_cache_clears_after_full_block": 2,
        "graph_mode": "WARP",
        "installed_provenance": provenance,
        "jax_devices": [_device_report(device) for device in devices],
        "required_musclemimic_commit": normalized_expected_commit,
        "scan_steps": _SCAN_STEPS,
        "status": "PASS",
        "versions": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "mujoco": mujoco.__version__,
            "warp": wp.__version__,
        },
        "warp_devices": [str(device) for device in warp_devices],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="full 40-character MuscleMimic Git commit pinned by the parent project",
    )
    args = parser.parse_args(argv)
    report = run_probe(expected_commit=args.expected_commit)
    print(f"MUSCLEMIMIC_WARP_MULTI_GPU_PROBE {json.dumps(report, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"MUSCLEMIMIC_WARP_MULTI_GPU_PROBE FAIL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
