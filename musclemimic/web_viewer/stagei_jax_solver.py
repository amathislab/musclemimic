"""JAX exact-Jacobian joint dogleg solver for C3D -> SMPL-X Stage I."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree


class JaxStageIUnsupportedError(RuntimeError):
    """Raised when the active surface model cannot be exported to the JAX LBS path."""


@dataclass(frozen=True)
class JaxStageIJointResult:
    betas: np.ndarray
    markers_latent: np.ndarray
    ref_poses: np.ndarray
    ref_trans: np.ndarray
    terms: dict[str, float]
    nfev: int
    nit: int
    cost: float
    success: bool
    message: str

def fit_stagei_joint_dogleg(
    *,
    smpl,
    marker_model,
    ref_observed: np.ndarray,
    ref_availability: np.ndarray,
    ref_poses: np.ndarray,
    ref_trans: np.ndarray,
    base_betas: np.ndarray,
    base_markers_latent: np.ndarray,
    weights,
    coeff_type_weights: np.ndarray,
    free_pose_indices: np.ndarray,
    body_end: int,
    pose_prior=None,
    head_marker_corr=None,
    maxiter: int = 25,
) -> JaxStageIJointResult:
    """Optimize the full MoSh++ Stage-I variable block with JAX Jacobians."""

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_default_matmul_precision", "highest")

    from .c3d.optim import _minimize_dogleg_dense

    lbs = _extract_lbs_arrays(smpl)
    pose_dim = int(ref_poses.shape[1])
    if pose_dim != lbs["pose_dim"]:
        raise JaxStageIUnsupportedError(
            f"JAX joint Stage-I solver requires pose_dim={lbs['pose_dim']}, got {pose_dim}."
        )

    n_ref = int(ref_poses.shape[0])
    n_markers = int(base_markers_latent.shape[0])
    free_pose_indices = np.asarray(free_pose_indices, dtype=np.int32)
    n_free_pose = int(free_pose_indices.shape[0])
    ref_observed_padded, ref_ids, ref_mask = _pack_observations(ref_observed, ref_availability)

    constants = {
        "v_template": jnp.asarray(lbs["v_template"]),
        "shapedirs": jnp.asarray(lbs["shapedirs"]),
        "posedirs": jnp.asarray(lbs["posedirs"]),
        "j_regressor": jnp.asarray(lbs["j_regressor"]),
        "lbs_weights": jnp.asarray(lbs["lbs_weights"]),
        "parents": jnp.asarray(lbs["parents"]),
        "pose_dim": jnp.asarray(np.asarray([lbs["pose_dim"]], dtype=np.int32)),
        "full_pose_dim": jnp.asarray(np.asarray([lbs["full_pose_dim"]], dtype=np.int32)),
        "pose_body_dof": jnp.asarray(np.asarray([lbs["pose_body_dof"]], dtype=np.int32)),
        "pose_hand_dof": jnp.asarray(np.asarray([lbs["pose_hand_dof"]], dtype=np.int32)),
        "selected_components": jnp.asarray(lbs["selected_components"]),
        "hands_mean": jnp.asarray(lbs["hands_mean"]),
        "pose_mean": jnp.asarray(lbs["pose_mean"]),
        "ref_pose_template": jnp.asarray(ref_poses.astype(np.float32)),
        "free_pose_indices": jnp.asarray(free_pose_indices),
        "ref_observed": jnp.asarray(ref_observed_padded),
        "ref_ids": jnp.asarray(ref_ids),
        "ref_mask": jnp.asarray(ref_mask),
        "init_coeffs": jnp.asarray(marker_model.initial_coeffs.astype(np.float32)),
        "init_frame_vids": jnp.asarray(marker_model.frame_vids.astype(np.int32)),
        "desired_distances": jnp.asarray(marker_model.desired_distances.astype(np.float32)),
        "coeff_type_weights": jnp.asarray(coeff_type_weights.astype(np.float32)),
        "body_end": jnp.asarray(np.asarray([int(body_end)], dtype=np.int32)),
    }
    constants.update({key: jnp.asarray(value) for key, value in _head_corr_constants(head_marker_corr, n_markers).items()})
    constants.update({key: jnp.asarray(value) for key, value in _pose_prior_constants(pose_prior).items()})
    constants["head_init_weight"] = jnp.asarray(
        np.asarray([float(weights.init_by_type.get("body", 300.0 * weights.anneal_factor))], dtype=np.float32)
    )

    x0 = _pack_joint_absolute(ref_trans, base_markers_latent, ref_poses, base_betas, free_pose_indices)
    blocks_fn, canonical_fn = _build_joint_residual_blocks(
        constants,
        n_ref=n_ref,
        n_markers=n_markers,
        n_free_pose=n_free_pose,
        data_weight=float(weights.data),
        pose_weight=float(weights.pose_body),
        beta_weight=float(weights.betas),
        surf_weight=float(weights.surf),
        has_head_corr=head_marker_corr is not None,
        has_pose_prior=pose_prior is not None and hasattr(pose_prior, "residuals"),
    )

    def flat_residual(x, frame_vids, candidate_faces):
        blocks = blocks_fn(x, frame_vids, candidate_faces)
        return jnp.concatenate(blocks)

    canonical_jit = jax.jit(canonical_fn)
    residual_jit = jax.jit(flat_residual)
    term_blocks_jit = jax.jit(blocks_fn)
    n_vars = int(x0.shape[0])
    jacobian_jit = None
    jvp_chunk_jit = None
    jvp_basis_eye = None
    jvp_chunk_size = 0
    if jax.default_backend() == "gpu":
        import os

        jvp_chunk_size = max(1, int(os.environ.get("MUSCLEMIMIC_JAX_STAGEI_JVP_CHUNK", "32")))
        jvp_basis_eye = np.eye(n_vars, dtype=np.float32)

        def jvp_chunk(x, frame_vids, candidate_faces, basis):
            def residual_for_x(y):
                return flat_residual(y, frame_vids, candidate_faces)

            def tangent_jvp(tangent):
                return jax.jvp(residual_for_x, (x,), (tangent,))[1]

            return jax.vmap(tangent_jvp)(basis)

        jvp_chunk_jit = jax.jit(jvp_chunk)
    else:
        jacobian_jit = jax.jit(jax.jacfwd(flat_residual, argnums=0))

    def frame_constants(x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        canonical = np.asarray(canonical_jit(jnp.asarray(x_np, dtype=jnp.float32)))
        _, markers_latent, _, _ = _unpack_joint_absolute_np(
            x_np,
            n_ref=n_ref,
            n_markers=n_markers,
            n_free_pose=n_free_pose,
            pose_dim=pose_dim,
            free_pose_indices=free_pose_indices,
            ref_pose_template=ref_poses,
        )
        frame_vids = marker_model.compute_dynamic_frame_vids(canonical, markers_latent).astype(np.int32)
        candidate_faces = _surface_candidate_faces(marker_model, canonical, markers_latent, frame_vids)
        return frame_vids, candidate_faces

    # Compile once with the initial frame assignments. Values may change between
    # dogleg evaluations, but shapes stay fixed so JAX reuses the executable.
    init_frame_vids, init_candidate_faces = frame_constants(x0)
    residual_jit(
        jnp.asarray(x0, dtype=jnp.float32),
        jnp.asarray(init_frame_vids),
        jnp.asarray(init_candidate_faces),
    ).block_until_ready()
    if jvp_chunk_jit is not None:
        jvp_chunk_jit(
            jnp.asarray(x0, dtype=jnp.float32),
            jnp.asarray(init_frame_vids),
            jnp.asarray(init_candidate_faces),
            jnp.asarray(jvp_basis_eye[:jvp_chunk_size]),
        ).block_until_ready()
    else:
        jacobian_jit(
            jnp.asarray(x0, dtype=jnp.float32),
            jnp.asarray(init_frame_vids),
            jnp.asarray(init_candidate_faces),
        ).block_until_ready()

    def residual_np(x_np: np.ndarray) -> np.ndarray:
        frame_vids, candidate_faces = frame_constants(x_np)
        values = np.asarray(
            residual_jit(
                jnp.asarray(x_np, dtype=jnp.float32),
                jnp.asarray(frame_vids),
                jnp.asarray(candidate_faces),
            )
        )
        return np.nan_to_num(values, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

    def residual_and_jacobian_np(x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame_vids, candidate_faces = frame_constants(x_np)
        x_j = jnp.asarray(x_np, dtype=jnp.float32)
        frame_vids_j = jnp.asarray(frame_vids)
        candidate_faces_j = jnp.asarray(candidate_faces)
        residual = np.asarray(residual_jit(x_j, frame_vids_j, candidate_faces_j))
        if jvp_chunk_jit is not None:
            jacobian_chunks = []
            for start in range(0, n_vars, jvp_chunk_size):
                stop = min(start + jvp_chunk_size, n_vars)
                basis = jvp_basis_eye[start:stop]
                valid = int(basis.shape[0])
                if valid < jvp_chunk_size:
                    padded = np.zeros((jvp_chunk_size, n_vars), dtype=np.float32)
                    padded[:valid] = basis
                    basis = padded
                chunk = np.asarray(jvp_chunk_jit(x_j, frame_vids_j, candidate_faces_j, jnp.asarray(basis)))
                jacobian_chunks.append(chunk[:valid].T)
            jacobian = np.concatenate(jacobian_chunks, axis=1)
        else:
            jacobian = np.asarray(jacobian_jit(x_j, frame_vids_j, candidate_faces_j))
        return (
            np.nan_to_num(residual, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64),
            np.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64),
        )

    result = _minimize_dogleg_dense(
        residual_np=residual_np,
        residual_and_jacobian_np=residual_and_jacobian_np,
        x0=x0,
        maxiter=max(1, int(maxiter)),
        e_3=1e-3,
        delta_0=0.5,
    )

    final_frame_vids, final_candidate_faces = frame_constants(result.x)
    term_blocks = term_blocks_jit(
        jnp.asarray(result.x, dtype=jnp.float32),
        jnp.asarray(final_frame_vids),
        jnp.asarray(final_candidate_faces),
    )
    term_names = ("data", "pose", "beta", "init", "surf")
    terms = {
        name: float(np.sum(np.asarray(block, dtype=np.float64) ** 2))
        for name, block in zip(term_names, term_blocks, strict=True)
    }
    betas, markers_latent, out_ref_poses, out_ref_trans = _unpack_joint_absolute_np(
        result.x,
        n_ref=n_ref,
        n_markers=n_markers,
        n_free_pose=n_free_pose,
        pose_dim=pose_dim,
        free_pose_indices=free_pose_indices,
        ref_pose_template=ref_poses,
    )
    return JaxStageIJointResult(
        betas=betas,
        markers_latent=markers_latent,
        ref_poses=out_ref_poses,
        ref_trans=out_ref_trans,
        terms=terms,
        nfev=int(result.nfev),
        nit=int(result.nit),
        cost=float(result.cost),
        success=bool(result.success),
        message=str(result.message),
    )


def _extract_lbs_arrays(smpl) -> dict[str, np.ndarray]:
    model = getattr(smpl, "model", None)
    if model is None:
        model = smpl
    required = ("v_template", "shapedirs", "posedirs", "J_regressor", "parents", "lbs_weights")
    if not all(hasattr(model, name) for name in required):
        raise JaxStageIUnsupportedError(
            "JAX Stage-I solver requires a surface model exposing raw LBS tensors "
            "(v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights)."
        )

    def as_np(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    parents = as_np(model.parents).astype(np.int32).copy()
    parents[0] = 0
    posedirs = as_np(model.posedirs).astype(np.float32)
    if posedirs.ndim != 2:
        raise JaxStageIUnsupportedError(f"Expected posedirs as (P, V*3), got {posedirs.shape}.")
    pose_feature_dim = posedirs.shape[0]
    n_joints = pose_feature_dim // 9 + 1
    shapedirs = as_np(model.shapedirs).astype(np.float32)
    if shapedirs.shape[-1] != 16:
        raise JaxStageIUnsupportedError(
            f"JAX Stage-I solver currently optimizes 16 betas, but the model exposes {shapedirs.shape[-1]}."
        )
    pose_dim = int(getattr(model, "pose_dim", n_joints * 3))
    full_pose_dim = int(getattr(model, "full_pose_dim", n_joints * 3))
    pose_mean = as_np(getattr(model, "pose_mean", np.zeros(full_pose_dim, dtype=np.float32))).astype(np.float32).reshape(-1)
    if pose_mean.shape[0] != full_pose_dim:
        raise JaxStageIUnsupportedError(f"Expected pose_mean length {full_pose_dim}, got {pose_mean.shape[0]}.")

    return {
        "v_template": as_np(model.v_template).astype(np.float32),
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "j_regressor": as_np(model.J_regressor).astype(np.float32),
        "parents": parents[:n_joints],
        "lbs_weights": as_np(model.lbs_weights).astype(np.float32)[:, :n_joints],
        "pose_dim": pose_dim,
        "full_pose_dim": full_pose_dim,
        "pose_body_dof": int(getattr(model, "pose_body_dof", n_joints * 3)),
        "pose_hand_dof": int(getattr(model, "pose_hand_dof", 0)),
        "selected_components": as_np(getattr(model, "selected_components", np.zeros((0, 0), dtype=np.float32))).astype(np.float32),
        "hands_mean": as_np(getattr(model, "hands_mean", np.zeros(0, dtype=np.float32))).astype(np.float32),
        "pose_mean": pose_mean,
    }


def _pack_observations(ref_observed: np.ndarray, ref_availability: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_ref, _n_markers, _ = ref_observed.shape
    max_obs = int(ref_availability.sum(axis=1).max())
    observed = np.zeros((n_ref, max_obs, 3), dtype=np.float32)
    ids = np.zeros((n_ref, max_obs), dtype=np.int32)
    mask = np.zeros((n_ref, max_obs), dtype=np.float32)
    for ref_idx in range(n_ref):
        visible = np.flatnonzero(ref_availability[ref_idx])
        observed[ref_idx, : len(visible)] = ref_observed[ref_idx, visible]
        ids[ref_idx, : len(visible)] = visible.astype(np.int32)
        mask[ref_idx, : len(visible)] = 1.0
    return observed, ids, mask


def _surface_candidate_faces(marker_model, canonical_verts: np.ndarray, markers_latent: np.ndarray, frame_vids: np.ndarray) -> np.ndarray:
    if marker_model.vertex_faces is None:
        raise JaxStageIUnsupportedError("JAX Stage-I solver requires marker_model.vertex_faces for surface residuals.")

    k = min(8, int(canonical_verts.shape[0]))
    _, knn_vids = KDTree(canonical_verts).query(markers_latent, k=k)
    if knn_vids.ndim == 1:
        knn_vids = knn_vids[:, None]
    nearest_vids = np.concatenate([frame_vids, knn_vids.astype(np.int32)], axis=1)
    candidate_faces = marker_model.vertex_faces[nearest_vids]
    return candidate_faces.reshape(candidate_faces.shape[0], -1, 3).astype(np.int32)


def _head_corr_constants(head_marker_corr, n_markers: int) -> dict[str, np.ndarray]:
    if head_marker_corr is None:
        return {
            "head_indices": np.zeros(0, dtype=np.int32),
            "non_head_mask": np.ones(n_markers, dtype=np.float32),
            "head_corr": np.zeros((0, 0), dtype=np.float32),
        }

    head_indices = np.asarray(head_marker_corr.marker_indices, dtype=np.int32)
    non_head_mask = np.ones(n_markers, dtype=np.float32)
    non_head_mask[head_indices] = 0.0
    return {
        "head_indices": head_indices,
        "non_head_mask": non_head_mask,
        "head_corr": np.asarray(head_marker_corr.corr, dtype=np.float32),
    }


def _pose_prior_constants(pose_prior) -> dict[str, np.ndarray]:
    if pose_prior is None or not hasattr(pose_prior, "residuals"):
        return {
            "pose_prior_means": np.zeros((0, 0), dtype=np.float32),
            "pose_prior_chols": np.zeros((0, 0, 0), dtype=np.float32),
            "pose_prior_neg_log_weights": np.zeros(0, dtype=np.float32),
            "pose_prior_npose": np.asarray([0], dtype=np.int32),
        }

    def as_np(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    return {
        "pose_prior_means": as_np(pose_prior.means).astype(np.float32),
        "pose_prior_chols": as_np(pose_prior.chols).astype(np.float32),
        "pose_prior_neg_log_weights": as_np(pose_prior.neg_log_weights).astype(np.float32),
        "pose_prior_npose": np.asarray([int(pose_prior.npose)], dtype=np.int32),
    }


def _pack_joint_absolute(
    ref_trans: np.ndarray,
    markers_latent: np.ndarray,
    ref_poses: np.ndarray,
    betas: np.ndarray,
    free_pose_indices: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            ref_trans.reshape(-1),
            markers_latent.reshape(-1),
            ref_poses[:, free_pose_indices].reshape(-1),
            betas.reshape(-1),
        ]
    ).astype(np.float64)


def _unpack_joint_absolute_np(
    x: np.ndarray,
    *,
    n_ref: int,
    n_markers: int,
    n_free_pose: int,
    pose_dim: int,
    free_pose_indices: np.ndarray,
    ref_pose_template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cursor = 0
    ref_trans = x[cursor : cursor + n_ref * 3].reshape(n_ref, 3).astype(np.float32)
    cursor += n_ref * 3
    markers_latent = x[cursor : cursor + n_markers * 3].reshape(n_markers, 3).astype(np.float32)
    cursor += n_markers * 3
    ref_poses = np.asarray(ref_pose_template, dtype=np.float32).copy().reshape(n_ref, pose_dim)
    ref_poses[:, free_pose_indices] = x[cursor : cursor + n_ref * n_free_pose].reshape(
        n_ref, n_free_pose
    ).astype(np.float32)
    cursor += n_ref * n_free_pose
    betas = x[cursor : cursor + 16].reshape(1, 16).astype(np.float32)
    return betas, markers_latent, ref_poses, ref_trans


def _build_joint_residual_blocks(
    constants,
    *,
    n_ref: int,
    n_markers: int,
    n_free_pose: int,
    data_weight: float,
    pose_weight: float,
    beta_weight: float,
    surf_weight: float,
    has_head_corr: bool,
    has_pose_prior: bool,
):
    import jax
    import jax.numpy as jnp

    parents_static = np.asarray(constants["parents"], dtype=np.int32)
    n_joints_static = int(parents_static.shape[0])
    pose_dim_static = int(np.asarray(constants["pose_dim"])[0])
    full_pose_dim_static = int(np.asarray(constants["full_pose_dim"])[0])
    pose_body_dof_static = int(np.asarray(constants["pose_body_dof"])[0])
    pose_hand_dof_static = int(np.asarray(constants["pose_hand_dof"])[0])
    body_end_static = int(np.asarray(constants["body_end"])[0])
    npose_static = int(np.asarray(constants["pose_prior_npose"])[0])

    def unpack(x):
        cursor = 0
        ref_trans = x[cursor : cursor + n_ref * 3].reshape(n_ref, 3)
        cursor += n_ref * 3
        markers_latent = x[cursor : cursor + n_markers * 3].reshape(n_markers, 3)
        cursor += n_markers * 3
        ref_poses = constants["ref_pose_template"]
        ref_pose_values = x[cursor : cursor + n_ref * n_free_pose].reshape(n_ref, n_free_pose)
        ref_poses = ref_poses.at[:, constants["free_pose_indices"]].set(ref_pose_values)
        cursor += n_ref * n_free_pose
        betas = x[cursor : cursor + 16].reshape(1, 16).reshape(-1)
        return betas, markers_latent, ref_poses, ref_trans

    def rodrigues(rotvec):
        theta_sq = jnp.sum(rotvec * rotvec, axis=-1, keepdims=True)
        theta = jnp.sqrt(theta_sq)
        x, y, z = rotvec[..., 0], rotvec[..., 1], rotvec[..., 2]
        zeros = jnp.zeros_like(x)
        skew = jnp.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1).reshape(
            (*rotvec.shape[:-1], 3, 3)
        )
        ident = jnp.eye(3, dtype=rotvec.dtype)
        small = theta_sq < 1e-8
        theta_safe = jnp.where(small, jnp.ones_like(theta), theta)
        theta_sq_safe = jnp.where(small, jnp.ones_like(theta_sq), theta_sq)
        a = jnp.where(small, 1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0, jnp.sin(theta) / theta_safe)
        b = jnp.where(
            small,
            0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
            (1.0 - jnp.cos(theta)) / theta_sq_safe,
        )
        return ident + a[..., None] * skew + b[..., None] * (skew @ skew)

    def to_full_pose(pose):
        if pose_dim_static == full_pose_dim_static:
            full_pose = pose
        else:
            body_pose = pose[:, :pose_body_dof_static]
            hand_coeffs = pose[:, pose_body_dof_static : pose_body_dof_static + pose_hand_dof_static]
            hand_pose = constants["hands_mean"] + hand_coeffs @ constants["selected_components"]
            full_pose = jnp.concatenate([body_pose, hand_pose], axis=-1)
        return full_pose + constants["pose_mean"]

    def transform_mat(rot, trans):
        batch = rot.shape[0]
        bottom = jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0], dtype=rot.dtype), (batch, 1)).reshape(batch, 1, 4)
        return jnp.concatenate([jnp.concatenate([rot, trans[..., None]], axis=2), bottom], axis=1)

    def lbs(betas, pose, trans):
        pose = to_full_pose(pose)
        batch = pose.shape[0]
        v_shaped = constants["v_template"][None] + jnp.einsum("l,vcl->vc", betas, constants["shapedirs"])[None]
        v_shaped = jnp.repeat(v_shaped, batch, axis=0)
        joints = jnp.einsum("jv,bvc->bjc", constants["j_regressor"], v_shaped)

        rot_mats = rodrigues(pose.reshape(batch, -1, 3))
        ident = jnp.eye(3, dtype=pose.dtype)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch, -1)
        pose_offsets = (pose_feature @ constants["posedirs"]).reshape(batch, -1, 3)
        v_posed = v_shaped + pose_offsets

        parents = constants["parents"]
        rel_joints = joints.at[:, 1:].set(joints[:, 1:] - joints[:, parents[1:]])
        transforms = [transform_mat(rot_mats[:, 0], rel_joints[:, 0])]
        for joint_idx in range(1, n_joints_static):
            transforms.append(
                transforms[int(parents_static[joint_idx])]
                @ transform_mat(rot_mats[:, joint_idx], rel_joints[:, joint_idx])
            )
        transforms = jnp.stack(transforms, axis=1)

        joints_h = jnp.concatenate([joints, jnp.zeros((batch, joints.shape[1], 1), dtype=joints.dtype)], axis=2)[..., None]
        init_bone = transforms @ joints_h
        transforms = transforms - jnp.concatenate(
            [jnp.zeros((batch, joints.shape[1], 4, 3), dtype=transforms.dtype), init_bone],
            axis=3,
        )
        blended = jnp.einsum("vj,bjkl->bvkl", constants["lbs_weights"], transforms)
        verts_h = jnp.concatenate([v_posed, jnp.ones((batch, v_posed.shape[1], 1), dtype=v_posed.dtype)], axis=2)[..., None]
        return (blended @ verts_h)[:, :, :3, 0] + trans[:, None, :]

    def local_frame(verts, frame_vids):
        v0 = verts[frame_vids[:, 0]]
        v1 = verts[frame_vids[:, 1]]
        v2 = verts[frame_vids[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        t1 = e1 / (jnp.linalg.norm(e1, axis=-1, keepdims=True) + 1e-8)
        normal = jnp.cross(e1, e2)
        normal = normal / (jnp.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
        t2 = jnp.cross(normal, t1)
        t2 = t2 / (jnp.linalg.norm(t2, axis=-1, keepdims=True) + 1e-8)
        return v0, t1, t2, normal

    def latents_to_coeffs(canonical_verts, markers_latent, frame_vids):
        v0, t1, t2, normal = local_frame(canonical_verts, frame_vids)
        diff = markers_latent - v0
        return jnp.stack(
            [jnp.sum(diff * t1, axis=-1), jnp.sum(diff * t2, axis=-1), jnp.sum(diff * normal, axis=-1)],
            axis=-1,
        )

    def reconstruct(verts, coeffs, frame_vids):
        v0, t1, t2, normal = local_frame(verts, frame_vids)
        return v0 + coeffs[:, 0:1] * t1 + coeffs[:, 1:2] * t2 + coeffs[:, 2:3] * normal

    def point_to_segment_distance_sq(point, a, b, valid):
        ab = b - a
        denom = jnp.sum(ab * ab, axis=-1, keepdims=True)
        t = jnp.sum((point - a) * ab, axis=-1, keepdims=True) / jnp.maximum(denom, 1e-12)
        t = jnp.clip(t, 0.0, 1.0)
        closest = a + t * ab
        dist_sq = jnp.sum((point - closest) ** 2, axis=-1)
        return jnp.where(valid, dist_sq, jnp.full_like(dist_sq, 1e20))

    def point_to_triangle_distance(points, verts, candidate_faces):
        valid = jnp.all(candidate_faces >= 0, axis=-1)
        safe_faces = jnp.maximum(candidate_faces, 0)
        tri = verts[safe_faces]
        point = points[:, None, :]
        a = tri[:, :, 0, :]
        b = tri[:, :, 1, :]
        c = tri[:, :, 2, :]

        ab = b - a
        ac = c - a
        normal = jnp.cross(ab, ac)
        normal_norm = jnp.linalg.norm(normal, axis=-1, keepdims=True)
        normal_unit = normal / jnp.maximum(normal_norm, 1e-12)
        plane_offset = jnp.sum((point - a) * normal_unit, axis=-1, keepdims=True)
        projected = point - plane_offset * normal_unit

        v0 = ab
        v1 = ac
        v2 = projected - a
        d00 = jnp.sum(v0 * v0, axis=-1)
        d01 = jnp.sum(v0 * v1, axis=-1)
        d11 = jnp.sum(v1 * v1, axis=-1)
        d20 = jnp.sum(v2 * v0, axis=-1)
        d21 = jnp.sum(v2 * v1, axis=-1)
        denom = d00 * d11 - d01 * d01
        denom_safe = jnp.maximum(denom, 1e-12)
        bary_v = (d11 * d20 - d01 * d21) / denom_safe
        bary_w = (d00 * d21 - d01 * d20) / denom_safe
        bary_u = 1.0 - bary_v - bary_w
        inside = valid & (denom > 1e-12) & (bary_u >= 0.0) & (bary_v >= 0.0) & (bary_w >= 0.0)
        plane_dist_sq = jnp.sum((point - projected) ** 2, axis=-1)
        best = jnp.where(inside, plane_dist_sq, jnp.full_like(plane_dist_sq, 1e20))
        best = jnp.minimum(best, point_to_segment_distance_sq(point, a, b, valid))
        best = jnp.minimum(best, point_to_segment_distance_sq(point, b, c, valid))
        best = jnp.minimum(best, point_to_segment_distance_sq(point, c, a, valid))
        return jnp.sqrt(jnp.maximum(jnp.min(best, axis=1), 0.0) + 1e-12)

    def pose_prior_residual(body_pose):
        if not has_pose_prior:
            return body_pose.reshape(-1) * pose_weight
        body_pose = body_pose[:, :npose_static]
        diff = body_pose[:, None, :] - constants["pose_prior_means"][None, :, :]
        loglikes = jnp.sqrt(jnp.asarray(0.5, dtype=body_pose.dtype)) * jnp.einsum(
            "bkd,kde->bke", diff, constants["pose_prior_chols"]
        )
        component_costs = jnp.sum(loglikes * loglikes, axis=-1) + constants["pose_prior_neg_log_weights"][None, :]
        component_idx = jnp.argmin(component_costs, axis=1)
        selected = jnp.take_along_axis(loglikes, component_idx[:, None, None], axis=1)[:, 0, :]
        const = jnp.sqrt(jnp.maximum(constants["pose_prior_neg_log_weights"][component_idx], 1e-30))[:, None]
        return jnp.concatenate([selected, const], axis=1).reshape(-1) * pose_weight

    def canonical_verts_from_x(x):
        betas, _, _, _ = unpack(x)
        return lbs(
            betas,
            jnp.zeros((1, pose_dim_static), dtype=x.dtype),
            jnp.zeros((1, 3), dtype=x.dtype),
        )[0]

    def blocks(x, frame_vids, candidate_faces):
        betas, markers_latent, ref_poses, ref_trans = unpack(x)
        canonical_verts = lbs(
            betas,
            jnp.zeros((1, pose_dim_static), dtype=x.dtype),
            jnp.zeros((1, 3), dtype=x.dtype),
        )[0]
        marker_coeffs = latents_to_coeffs(canonical_verts, markers_latent, frame_vids)

        posed_verts = lbs(betas, ref_poses, ref_trans)
        pred_all = jax.vmap(lambda verts: reconstruct(verts, marker_coeffs, frame_vids))(posed_verts)
        pred_visible = jnp.take_along_axis(pred_all, constants["ref_ids"][..., None], axis=1)
        data_res = ((pred_visible - constants["ref_observed"]) * constants["ref_mask"][..., None]).reshape(-1) * data_weight

        pose_res = pose_prior_residual(ref_poses[:, 3:body_end_static])
        beta_res = betas.reshape(-1) * beta_weight

        init_markers = reconstruct(canonical_verts, constants["init_coeffs"], constants["init_frame_vids"])
        init_loss = markers_latent - init_markers
        if has_head_corr:
            non_head = (init_loss * constants["non_head_mask"][:, None] * constants["coeff_type_weights"][:, None]).reshape(-1)
            head_loss = init_loss[constants["head_indices"]]
            head_res = (constants["head_corr"] @ head_loss * constants["head_init_weight"][0]).reshape(-1)
            init_res = jnp.concatenate([non_head, head_res])
        else:
            init_res = (init_loss * constants["coeff_type_weights"][:, None]).reshape(-1)

        surface_distances = point_to_triangle_distance(markers_latent, canonical_verts, candidate_faces)
        surf_res = (surface_distances - constants["desired_distances"]) * surf_weight
        return data_res, pose_res, beta_res, init_res, surf_res

    return blocks, canonical_verts_from_x
