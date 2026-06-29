from __future__ import annotations

from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.initial_state_handler.traj_init_state import TrajInitialStateHandler


class PerturbedTrajInitialStateHandler(TrajInitialStateHandler):
    """Trajectory reset with small random qpos/qvel perturbations."""

    def __init__(
        self,
        env: Any,
        qpos_noise_std: float | list[float] = 0.0,
        qvel_noise_std: float | list[float] = 0.0,
        perturb_free_joint_pos: bool = False,
        perturb_free_joint_quat: bool = False,
        perturb_free_joint_vel: bool = False,
        perturb_ball_joints: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)

        model = env._model
        self._qpos_noise_std = self._as_scale_array(qpos_noise_std, model.nq, "qpos_noise_std")
        self._qvel_noise_std = self._as_scale_array(qvel_noise_std, model.nv, "qvel_noise_std")

        qpos_mask = np.zeros(model.nq, dtype=np.float32)
        qvel_mask = np.zeros(model.nv, dtype=np.float32)
        qpos_low = np.full(model.nq, -np.inf, dtype=np.float32)
        qpos_high = np.full(model.nq, np.inf, dtype=np.float32)
        quat_qpos_ids: list[list[int]] = []

        for jnt_id in range(model.njnt):
            jnt_type = model.jnt_type[jnt_id]
            qpos_adr = int(model.jnt_qposadr[jnt_id])
            qvel_adr = int(model.jnt_dofadr[jnt_id])

            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                if perturb_free_joint_pos:
                    qpos_mask[qpos_adr : qpos_adr + 3] = 1.0
                if perturb_free_joint_quat:
                    qpos_mask[qpos_adr + 3 : qpos_adr + 7] = 1.0
                    quat_qpos_ids.append(list(range(qpos_adr + 3, qpos_adr + 7)))
                if perturb_free_joint_vel:
                    qvel_mask[qvel_adr : qvel_adr + 6] = 1.0
                continue

            if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                if perturb_ball_joints:
                    qpos_mask[qpos_adr : qpos_adr + 4] = 1.0
                    qvel_mask[qvel_adr : qvel_adr + 3] = 1.0
                    quat_qpos_ids.append(list(range(qpos_adr, qpos_adr + 4)))
                continue

            qpos_mask[qpos_adr] = 1.0
            qvel_mask[qvel_adr] = 1.0
            if model.jnt_limited[jnt_id]:
                qpos_low[qpos_adr] = model.jnt_range[jnt_id, 0]
                qpos_high[qpos_adr] = model.jnt_range[jnt_id, 1]

        self._qpos_noise_std = self._qpos_noise_std * qpos_mask
        self._qvel_noise_std = self._qvel_noise_std * qvel_mask
        self._qpos_low = qpos_low
        self._qpos_high = qpos_high
        self._quat_qpos_ids = np.asarray(quat_qpos_ids, dtype=np.int32).reshape(-1, 4)

    @staticmethod
    def _as_scale_array(value: float | list[float], size: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return np.full(size, float(arr), dtype=np.float32)
        if arr.shape == (size,):
            return arr
        raise ValueError(f"{name} must be a scalar or length-{size} list, got shape {arr.shape}.")

    def reset(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> tuple[MjData | Data, Any]:
        data, carry = super().reset(env, model, data, carry, backend)

        if backend == np:
            qpos = np.asarray(data.qpos).copy()
            qvel = np.asarray(data.qvel).copy()
            if np.any(self._qpos_noise_std):
                qpos += np.random.normal(size=qpos.shape).astype(qpos.dtype) * self._qpos_noise_std
                qpos = np.clip(qpos, self._qpos_low, self._qpos_high)
                qpos = self._normalize_np_quats(qpos)
            if np.any(self._qvel_noise_std):
                qvel += np.random.normal(size=qvel.shape).astype(qvel.dtype) * self._qvel_noise_std
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            return data, carry

        key, qpos_key, qvel_key = jax.random.split(carry.key, 3)
        qpos = data.qpos
        qvel = data.qvel

        if np.any(self._qpos_noise_std):
            qpos_noise = jax.random.normal(qpos_key, shape=qpos.shape, dtype=qpos.dtype)
            qpos = qpos + qpos_noise * jnp.asarray(self._qpos_noise_std, dtype=qpos.dtype)
            qpos = jnp.clip(
                qpos,
                jnp.asarray(self._qpos_low, dtype=qpos.dtype),
                jnp.asarray(self._qpos_high, dtype=qpos.dtype),
            )
            qpos = self._normalize_jax_quats(qpos)

        if np.any(self._qvel_noise_std):
            qvel_noise = jax.random.normal(qvel_key, shape=qvel.shape, dtype=qvel.dtype)
            qvel = qvel + qvel_noise * jnp.asarray(self._qvel_noise_std, dtype=qvel.dtype)

        data = data.replace(qpos=qpos, qvel=qvel)
        return data, carry.replace(key=key)

    def _normalize_np_quats(self, qpos: np.ndarray) -> np.ndarray:
        if self._quat_qpos_ids.size == 0:
            return qpos
        quat = qpos[self._quat_qpos_ids]
        norm = np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
        qpos[self._quat_qpos_ids] = quat / norm
        return qpos

    def _normalize_jax_quats(self, qpos: jax.Array) -> jax.Array:
        if self._quat_qpos_ids.size == 0:
            return qpos
        quat_ids = jnp.asarray(self._quat_qpos_ids)
        quat = qpos[quat_ids]
        norm = jnp.maximum(jnp.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
        return qpos.at[quat_ids].set(quat / norm)


PerturbedTrajInitialStateHandler.register()
