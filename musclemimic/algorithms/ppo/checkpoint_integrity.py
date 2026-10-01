"""Fail-closed integrity checks for persisted PPO optimizer state."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


class PPOCheckpointIntegrityError(ValueError):
    """Raised when a PPO checkpoint is not safe to save or resume."""


def _children(value: Any) -> tuple[tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        return tuple((str(name), child) for name, child in value.items())

    field_names = getattr(value, "_fields", None)
    if field_names is not None:
        return tuple((str(name), getattr(value, name)) for name in field_names)

    if is_dataclass(value) and not isinstance(value, type):
        return tuple((field.name, getattr(value, field.name)) for field in fields(value))

    if isinstance(value, (list, tuple)):
        return tuple((f"[{index}]", child) for index, child in enumerate(value))

    return ()


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    yield path, value
    for name, child in _children(value):
        yield from _walk(child, (*path, name))


def _path_text(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"


def _scalar_array(value: Any, *, label: str, context: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PPOCheckpointIntegrityError(
            f"PPO checkpoint integrity check failed during {context}: {label} is not a readable scalar."
        ) from exc

    if array.shape != () or not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise PPOCheckpointIntegrityError(
            f"PPO checkpoint integrity check failed during {context}: "
            f"{label} must be a numeric scalar, got shape={array.shape}, dtype={array.dtype}."
        )
    return array


def _counter_value(value: Any, *, label: str, context: str) -> int:
    array = _scalar_array(value, label=label, context=context)
    if np.issubdtype(array.dtype, np.bool_) or not np.isfinite(array).item():
        raise PPOCheckpointIntegrityError(
            f"PPO checkpoint integrity check failed during {context}: {label} is not a finite integer counter."
        )

    scalar = array.item()
    if not np.issubdtype(array.dtype, np.integer) and float(scalar) != float(int(scalar)):
        raise PPOCheckpointIntegrityError(
            f"PPO checkpoint integrity check failed during {context}: {label} is not an integer counter."
        )
    counter = int(scalar)
    if counter < 0:
        raise PPOCheckpointIntegrityError(
            f"PPO checkpoint integrity check failed during {context}: {label} is negative ({counter})."
        )
    return counter


def _named_children(value: Any) -> dict[str, Any]:
    return dict(_children(value))


def _validate_finite_tree(value: Any, *, path: tuple[str, ...], context: str) -> None:
    """Reject nonfinite model, running-statistic, or optimizer leaves."""
    for leaf_path, leaf in _walk(value, path):
        if _children(leaf):
            continue
        try:
            array = np.asarray(leaf)
        except (TypeError, ValueError):
            continue
        if not np.issubdtype(array.dtype, np.inexact):
            continue
        if not np.isfinite(array).all():
            raise PPOCheckpointIntegrityError(
                f"PPO checkpoint integrity check failed during {context}: "
                f"{_path_text(leaf_path)} contains NaN or infinity."
            )


def _validate_apply_if_finite_state(value: Any, *, path: tuple[str, ...], context: str) -> None:
    children = _named_children(value)
    marker_names = {"last_finite", "notfinite_count", "total_notfinite"}
    if not marker_names.intersection(children):
        return

    missing = marker_names.difference(children)
    if missing:
        raise PPOCheckpointIntegrityError(
            f"PPO checkpoint integrity check failed during {context}: apply_if_finite state at "
            f"{_path_text(path)} is missing {sorted(missing)}."
        )

    last_finite = _scalar_array(
        children["last_finite"],
        label=_path_text((*path, "last_finite")),
        context=context,
    )
    if not np.issubdtype(last_finite.dtype, np.bool_):
        raise PPOCheckpointIntegrityError(
            f"PPO checkpoint integrity check failed during {context}: "
            f"{_path_text((*path, 'last_finite'))} must be boolean."
        )
    if not bool(last_finite.item()):
        raise PPOCheckpointIntegrityError(
            f"PPO checkpoint integrity check failed during {context}: {_path_text((*path, 'last_finite'))}=False."
        )

    for name in ("notfinite_count", "total_notfinite"):
        counter = _counter_value(
            children[name],
            label=_path_text((*path, name)),
            context=context,
        )
        if counter != 0:
            raise PPOCheckpointIntegrityError(
                f"PPO checkpoint integrity check failed during {context}: "
                f"{_path_text((*path, name))}={counter}; refusing optimizer state with rejected non-finite updates."
            )


def validate_ppo_checkpoint_train_state(train_state: Any, *, context: str) -> None:
    """Reject PPO state that records non-finite updates or stalled optimizer progress.

    Legacy optimizer trees without ``apply_if_finite`` fields or inner ``count``
    leaves remain loadable. When counts are present, the smallest inner counter
    must equal ``TrainState.step``. This also accepts the intentional state after
    resetting an LR schedule: its schedule count restarts with the outer step,
    while momentum counters can remain larger.
    """
    root = _named_children(train_state)
    if "step" not in root or "opt_state" not in root:
        return

    for name in ("params", "run_stats", "opt_state"):
        if name in root:
            _validate_finite_tree(root[name], path=(name,), context=context)

    outer_step = _counter_value(root["step"], label="step", context=context)
    opt_state = root["opt_state"]

    inner_counts: list[tuple[str, int]] = []
    for path, value in _walk(opt_state, ("opt_state",)):
        _validate_apply_if_finite_state(value, path=path, context=context)
        if path and path[-1] == "count":
            inner_counts.append(
                (
                    _path_text(path),
                    _counter_value(value, label=_path_text(path), context=context),
                )
            )

    if not inner_counts:
        return

    minimum_inner_step = min(value for _, value in inner_counts)
    if minimum_inner_step != outer_step:
        details = ", ".join(f"{path}={value}" for path, value in inner_counts)
        raise PPOCheckpointIntegrityError(
            f"PPO checkpoint integrity check failed during {context}: outer step={outer_step} "
            f"does not match the minimum inner optimizer count={minimum_inner_step} ({details})."
        )
