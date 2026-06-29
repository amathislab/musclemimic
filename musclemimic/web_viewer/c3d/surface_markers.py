from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from .markers import MarkerLayout, _smplx_no_eyeball_vids


@dataclass(frozen=True)
class SurfaceMarkerModel:
    vids: np.ndarray
    frame_vids: np.ndarray
    initial_coeffs: np.ndarray
    initial_latents: np.ndarray
    desired_distances: np.ndarray
    marker_types: tuple[str, ...]
    labels: tuple[str, ...] = ()
    vertex_faces: np.ndarray | None = None

    @staticmethod
    def from_layout(
        smpl_model,
        marker_layout: MarkerLayout,
        betas=None,
        pose_dim: int = 156,
        device: str = "cpu",
    ) -> SurfaceMarkerModel:
        import torch

        if betas is None:
            betas = torch.zeros(1, 16, device=device, dtype=torch.float32)

        with torch.no_grad():
            verts, _ = smpl_model.get_joints_verts(
                torch.zeros(1, pose_dim, device=device, dtype=torch.float32),
                th_betas=betas,
            )
            verts_np = verts[0].detach().cpu().numpy()

        faces = smpl_model.faces.astype(np.int64)
        vert_normals = _compute_vertex_normals_from_mesh(faces, verts_np)
        vertex_faces = _compute_vertex_faces_padded(faces, len(verts_np))

        labels = marker_layout.labels
        vids = np.asarray([marker_layout.marker_vids[label] for label in labels], dtype=np.int64)
        initial_latents = np.zeros((len(labels), 3), dtype=np.float32)
        desired_distances = np.zeros(len(labels), dtype=np.float32)
        marker_types: list[str] = []

        for idx, label in enumerate(labels):
            vid = vids[idx]
            marker_type = marker_layout.marker_type[label]
            desired_distances[idx] = marker_layout.m2b_distance[marker_type]

            latent = verts_np[vid] + vert_normals[vid] * desired_distances[idx]
            initial_latents[idx] = latent.astype(np.float32)
            marker_types.append(marker_type)

        frame_vids = SurfaceMarkerModel.compute_dynamic_frame_vids(verts_np, initial_latents)
        initial_coeffs = _latents_to_coeffs_np(verts_np, initial_latents, frame_vids)

        return SurfaceMarkerModel(
            vids=vids,
            frame_vids=frame_vids,
            initial_coeffs=initial_coeffs,
            initial_latents=initial_latents,
            desired_distances=desired_distances,
            marker_types=tuple(marker_types),
            labels=tuple(labels),
            vertex_faces=vertex_faces,
        )

    def latents_to_coeffs(self, canonical_verts, markers_latent, frame_vids_override=None):
        import torch

        squeeze = canonical_verts.dim() == 2 and markers_latent.dim() == 2
        if canonical_verts.dim() == 2:
            canonical_verts = canonical_verts.unsqueeze(0)

        markers_latent = markers_latent.to(canonical_verts.device)
        if markers_latent.dim() == 2:
            markers_latent = markers_latent.unsqueeze(0).expand(canonical_verts.shape[0], -1, -1)

        if frame_vids_override is None:
            frame_vids_t = torch.as_tensor(self.frame_vids, dtype=torch.long, device=canonical_verts.device)
        else:
            frame_vids_t = frame_vids_override

        v0 = canonical_verts[:, frame_vids_t[:, 0]]
        v1 = canonical_verts[:, frame_vids_t[:, 1]]
        v2 = canonical_verts[:, frame_vids_t[:, 2]]

        t1, t2, n = _build_local_frame_torch(v0, v1, v2)
        diff = markers_latent - v0
        # Coefficients are tangent, tangent, normal; coeff[..., 2] is the surface-normal offset.
        coeffs = torch.cat(
            [
                (diff * t1).sum(dim=-1, keepdim=True),
                (diff * t2).sum(dim=-1, keepdim=True),
                (diff * n).sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        return coeffs.squeeze(0) if squeeze else coeffs

    def reconstruct(self, posed_verts, coeffs=None, markers_latent=None, canonical_verts=None, frame_vids_override=None):
        import torch

        squeeze = posed_verts.dim() == 2
        if squeeze:
            posed_verts = posed_verts.unsqueeze(0)

        if markers_latent is not None:
            if canonical_verts is None:
                canonical_verts = posed_verts
            coeffs_t = self.latents_to_coeffs(canonical_verts, markers_latent, frame_vids_override=frame_vids_override)
        else:
            coeffs_t = (
                torch.as_tensor(self.initial_coeffs, dtype=torch.float32, device=posed_verts.device)
                if coeffs is None
                else coeffs.to(posed_verts.device)
            )
        if coeffs_t.dim() == 2:
            coeffs_t = coeffs_t.unsqueeze(0).expand(posed_verts.shape[0], -1, -1)

        if frame_vids_override is None:
            frame_vids_t = torch.as_tensor(self.frame_vids, dtype=torch.long, device=posed_verts.device)
        else:
            frame_vids_t = frame_vids_override

        v0 = posed_verts[:, frame_vids_t[:, 0]]
        v1 = posed_verts[:, frame_vids_t[:, 1]]
        v2 = posed_verts[:, frame_vids_t[:, 2]]

        t1, t2, n = _build_local_frame_torch(v0, v1, v2)
        markers = v0 + coeffs_t[:, :, 0:1] * t1 + coeffs_t[:, :, 1:2] * t2 + coeffs_t[:, :, 2:3] * n
        return markers.squeeze(0) if squeeze else markers

    @staticmethod
    def compute_dynamic_frame_vids(
        canonical_verts_np: np.ndarray,
        markers_latent_np: np.ndarray,
        k: int = 8,
    ) -> np.ndarray:
        """Compute the 3-closest-vertex triangle for each marker latent.

        Mirrors MoSh++ chumpy `transformed_lm.TransformedCoeffs.on_changed`: for
        every marker, run kNN against the current canonical mesh, pick the
        nearest vertex as v0, then advance through the remaining neighbors
        until a non-collinear triple is found.

        Returns shape (M, 3) int64. Non-differentiable; gradient flows
        through the chosen verts' coordinates, not through the indices.
        """
        search_vids = _smplx_no_eyeball_vids(canonical_verts_np.shape[0])
        search_verts = canonical_verts_np if search_vids is None else canonical_verts_np[search_vids]
        tree = KDTree(search_verts)
        k = min(k, search_verts.shape[0])
        _, nn = tree.query(markers_latent_np, k=k)
        if nn.ndim == 1:
            nn = nn[:, None]
        if search_vids is not None:
            nn = search_vids[nn]
        n_markers = markers_latent_np.shape[0]
        out = np.zeros((n_markers, 3), dtype=np.int64)
        for i in range(n_markers):
            nbrs = [int(x) for x in np.atleast_1d(nn[i])]
            v0 = canonical_verts_np[nbrs[0]]
            v1 = canonical_verts_np[nbrs[1]] if len(nbrs) > 1 else canonical_verts_np[nbrs[0]]
            e1 = v1 - v0
            chosen2 = nbrs[1] if len(nbrs) > 1 else nbrs[0]
            chosen3 = nbrs[2] if len(nbrs) > 2 else nbrs[0]
            for cand in nbrs[2:]:
                e2 = canonical_verts_np[cand] - v0
                if np.linalg.norm(np.cross(e1, e2)) > 1e-10:
                    chosen3 = cand
                    break
            out[i, 0] = nbrs[0]
            out[i, 1] = chosen2
            out[i, 2] = chosen3

        return out


def _build_local_frame_np(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1 = v1 - v0
    e2 = v2 - v0
    t1 = e1 / (np.linalg.norm(e1) + 1e-8)
    n = np.cross(e1, e2)
    n = n / (np.linalg.norm(n) + 1e-8)
    t2 = np.cross(n, t1)
    t2 = t2 / (np.linalg.norm(t2) + 1e-8)
    return t1.astype(np.float32), t2.astype(np.float32), n.astype(np.float32)


def _latents_to_coeffs_np(verts_np: np.ndarray, markers_latent_np: np.ndarray, frame_vids_np: np.ndarray) -> np.ndarray:
    coeffs = np.zeros((markers_latent_np.shape[0], 3), dtype=np.float32)
    for idx, (v0_id, v1_id, v2_id) in enumerate(frame_vids_np):
        v0 = verts_np[v0_id]
        v1 = verts_np[v1_id]
        v2 = verts_np[v2_id]
        t1, t2, n = _build_local_frame_np(v0, v1, v2)
        diff = markers_latent_np[idx] - v0
        coeffs[idx] = np.array([np.dot(diff, t1), np.dot(diff, t2), np.dot(diff, n)], dtype=np.float32)
    return coeffs


def _build_local_frame_torch(v0, v1, v2):
    """Build a right-handed orthonormal frame at v0 from triangle (v0, v1, v2).

    `t1` points from the nearest vertex `v0` toward the second chosen vertex
    `v1`. `n` is the raw triangle normal from `cross(v1 - v0, v2 - v0)`.
    `t2` is the in-plane direction perpendicular to `t1`, completing the
    tangent plane as `(t1, t2)`.

    We mirror MoSh++ `transformed_lm` for nearest-frame selection and use the
    same raw normal `cross(e1, e2)`. Coefficients are ordered as
    `(tangent, tangent, normal)` so the third coefficient remains the
    marker-to-surface offset. MoSh++ coefficient arrays use a different axis
    order and need remapping before direct reuse here.
    """
    import torch

    e1 = v1 - v0
    e2 = v2 - v0
    t1 = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-8)
    n = torch.cross(e1, e2, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
    t2 = torch.cross(n, t1, dim=-1)
    t2 = t2 / (t2.norm(dim=-1, keepdim=True) + 1e-8)
    return t1, t2, n


def _surface_distances_to_mesh(
    marker_model: SurfaceMarkerModel,
    canonical_verts,
    markers_latent,
    *,
    nearest_vids,
    fallback_coeffs=None,
):
    import torch

    if marker_model.vertex_faces is None:
        if fallback_coeffs is None:
            raise ValueError("Surface distance fallback requires local marker coeffs.")
        return fallback_coeffs[:, 2].abs()

    vertex_faces = torch.as_tensor(marker_model.vertex_faces, dtype=torch.long, device=canonical_verts.device)
    with torch.no_grad():
        verts_np = canonical_verts.detach().cpu().numpy()
        markers_np = markers_latent.detach().cpu().numpy()
        k = min(8, verts_np.shape[0])
        _, knn_vids_np = KDTree(verts_np).query(markers_np, k=k)
        if knn_vids_np.ndim == 1:
            knn_vids_np = knn_vids_np[:, None]
    knn_vids = torch.as_tensor(knn_vids_np, dtype=torch.long, device=canonical_verts.device)
    if nearest_vids is not None:
        if nearest_vids.dim() == 1:
            nearest_vids = nearest_vids[:, None]
        nearest_vids = torch.cat([nearest_vids.to(canonical_verts.device), knn_vids], dim=1)
    else:
        nearest_vids = knn_vids
    candidate_faces = vertex_faces[nearest_vids]
    if candidate_faces.dim() == 4:
        candidate_faces = candidate_faces.reshape(candidate_faces.shape[0], -1, candidate_faces.shape[-1])
    return _point_to_triangle_distance_torch(markers_latent, canonical_verts, candidate_faces)


def _point_to_triangle_distance_torch(points, verts, candidate_faces):
    import torch

    valid = (candidate_faces >= 0).all(dim=-1)
    safe_faces = candidate_faces.clamp_min(0)
    tri = verts[safe_faces]
    p = points[:, None, :]
    a = tri[:, :, 0, :]
    b = tri[:, :, 1, :]
    c = tri[:, :, 2, :]

    ab = b - a
    ac = c - a
    n = torch.cross(ab, ac, dim=-1)
    n_norm = n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n_unit = n / n_norm
    plane_offset = ((p - a) * n_unit).sum(dim=-1, keepdim=True)
    q = p - plane_offset * n_unit

    v0 = ab
    v1 = ac
    v2 = q - a
    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0).sum(dim=-1)
    d21 = (v2 * v1).sum(dim=-1)
    denom = d00 * d11 - d01 * d01
    denom_safe = denom.clamp_min(1e-12)
    bary_v = (d11 * d20 - d01 * d21) / denom_safe
    bary_w = (d00 * d21 - d01 * d20) / denom_safe
    bary_u = 1.0 - bary_v - bary_w
    inside = (
        valid
        & (denom > 1e-12)
        & (bary_u >= 0.0)
        & (bary_v >= 0.0)
        & (bary_w >= 0.0)
    )
    plane_d2 = (p - q).pow(2).sum(dim=-1)
    inf = torch.full_like(plane_d2, float("inf"))
    best_d2 = torch.where(inside, plane_d2, inf)

    best_d2 = torch.minimum(best_d2, _point_to_segment_distance_sq_torch(p, a, b, valid))
    best_d2 = torch.minimum(best_d2, _point_to_segment_distance_sq_torch(p, b, c, valid))
    best_d2 = torch.minimum(best_d2, _point_to_segment_distance_sq_torch(p, c, a, valid))
    min_d2 = best_d2.min(dim=1).values
    return torch.sqrt(min_d2.clamp_min(0.0) + 1e-12)


def _point_to_segment_distance_sq_torch(p, a, b, valid):
    import torch

    ab = b - a
    denom = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    t = ((p - a) * ab).sum(dim=-1, keepdim=True) / denom
    t = t.clamp(0.0, 1.0)
    closest = a + t * ab
    d2 = (p - closest).pow(2).sum(dim=-1)
    return torch.where(valid, d2, torch.full_like(d2, float("inf")))


def _compute_vertex_normals_from_mesh(faces: np.ndarray, verts_np: np.ndarray) -> np.ndarray:
    v0 = verts_np[faces[:, 0]]
    v1 = verts_np[faces[:, 1]]
    v2 = verts_np[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vert_normals = np.zeros_like(verts_np)
    np.add.at(vert_normals, faces[:, 0], face_normals)
    np.add.at(vert_normals, faces[:, 1], face_normals)
    np.add.at(vert_normals, faces[:, 2], face_normals)
    return vert_normals / (np.linalg.norm(vert_normals, axis=-1, keepdims=True) + 1e-8)


def _compute_vertex_faces_padded(faces: np.ndarray, n_verts: int) -> np.ndarray:
    vertex_faces: list[list[int]] = [[] for _ in range(n_verts)]
    for face_idx, (a, b, c) in enumerate(faces):
        vertex_faces[int(a)].append(face_idx)
        vertex_faces[int(b)].append(face_idx)
        vertex_faces[int(c)].append(face_idx)
    max_faces = max((len(items) for items in vertex_faces), default=0)
    out = np.full((n_verts, max_faces, 3), -1, dtype=np.int64)
    for vid, items in enumerate(vertex_faces):
        if items:
            out[vid, : len(items)] = faces[items]
    return out


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    sign_mat = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    rot = vt.T @ sign_mat @ u.T
    trans = tgt_mean - rot @ src_mean
    return rot.astype(np.float32), trans.astype(np.float32)
