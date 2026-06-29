from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class MoshGmmPosePrior:
    """Torch version of MoSh++'s MaxMixtureComplete body-pose prior."""

    def __init__(self, prior_path: str | Path, *, device: str = "cpu", npose: int = 63):
        import torch

        prior_path = Path(prior_path)
        if not prior_path.exists():
            raise FileNotFoundError(f"MoSh++ pose body prior not found: {prior_path}")
        with prior_path.open("rb") as handle:
            gmm = pickle.load(handle, encoding="latin-1")

        covars = np.asarray(gmm["covars"], dtype=np.float64)[:, :npose, :npose]
        means = np.asarray(gmm["means"], dtype=np.float64)[:, :npose]
        weights = np.asarray(gmm["weights"], dtype=np.float64)

        precs = np.asarray([np.linalg.inv(cov) for cov in covars], dtype=np.float64)
        chols = np.asarray([np.linalg.cholesky(prec) for prec in precs], dtype=np.float64)
        sqrdets = np.asarray([np.sqrt(np.linalg.det(cov)) for cov in covars], dtype=np.float64)
        const = (2.0 * np.pi) ** (npose / 2.0)
        weights = weights / (const * (sqrdets / sqrdets.min()))
        weights = np.maximum(weights, 1e-300)

        self.means = torch.as_tensor(means, dtype=torch.float32, device=device)
        self.chols = torch.as_tensor(chols, dtype=torch.float32, device=device)
        self.neg_log_weights = torch.as_tensor(-np.log(weights), dtype=torch.float32, device=device)
        self.npose = int(npose)

    def __call__(self, body_pose):
        residuals = self.residuals(body_pose)
        return residuals.pow(2).sum()

    def residuals(self, body_pose):
        import torch

        if body_pose.dim() > 2:
            body_pose = body_pose.reshape(-1, body_pose.shape[-1])
        body_pose = body_pose[:, : self.npose]
        diff = body_pose[:, None, :] - self.means[None, :, :]
        loglikes = np.sqrt(0.5) * torch.einsum("bkd,kde->bke", diff, self.chols)
        component_costs = loglikes.pow(2).sum(dim=-1) + self.neg_log_weights[None, :]
        component_idx = component_costs.argmin(dim=1)
        batch_idx = torch.arange(body_pose.shape[0], device=body_pose.device)
        selected = loglikes[batch_idx, component_idx]
        const = torch.sqrt(self.neg_log_weights[component_idx].clamp_min(1e-30)).reshape(-1, 1)
        return torch.cat([selected, const], dim=1).reshape(-1)
