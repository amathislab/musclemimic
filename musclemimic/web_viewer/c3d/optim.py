from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _DoglegResult:
    x: np.ndarray
    cost: float
    nit: int
    nfev: int
    success: bool
    message: str


def _minimize_dogleg_dense(
    *,
    residual_np,
    residual_and_jacobian_np,
    x0: np.ndarray,
    maxiter: int,
    e_1: float = 1e-15,
    e_2: float = 1e-15,
    e_3: float = 1e-3,
    delta_0: float = 0.5,
) -> _DoglegResult:
    """Dense Powell dogleg matching chumpy's Stage-I optimizer flow."""

    x = np.asarray(x0, dtype=np.float64).copy()
    delta = float(delta_0)
    r, jac = residual_and_jacobian_np(x)
    nfev = 1
    cost = float(np.dot(r, r))
    success = False
    message = "maxiter reached"

    def normal_terms(jacobian: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jt = jacobian.T
        return jt @ jacobian, jt @ (-residual)

    def solve_normal(a_mat: np.ndarray, g_vec: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(a_mat, g_vec)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(a_mat, g_vec, rcond=None)[0]

    a_mat, g_vec = normal_terms(jac, r)
    if np.linalg.norm(g_vec, ord=np.inf) < e_1:
        return _DoglegResult(x=x, cost=cost, nit=0, nfev=nfev, success=True, message="gradient below threshold")

    nit = 0
    while nit < maxiter:
        nit += 1
        jg = jac @ g_vec
        denom = float(np.dot(jg, jg))
        if denom <= 1e-30:
            d_sd = g_vec.copy()
        else:
            d_sd = (float(np.dot(g_vec, g_vec)) / denom) * g_vec

        d_gn = solve_normal(a_mat, g_vec)
        while True:
            norm_sd = float(np.linalg.norm(d_sd))
            if norm_sd >= delta:
                step = (delta / max(norm_sd, 1e-30)) * d_sd
            else:
                norm_gn = float(np.linalg.norm(d_gn))
                if norm_gn <= delta:
                    step = d_gn
                    if delta <= 0.0:
                        delta = norm_gn
                else:
                    diff = d_gn - d_sd
                    a = float(np.dot(diff, diff))
                    b = 2.0 * float(np.dot(d_sd, diff))
                    c = float(np.dot(d_sd, d_sd)) - delta * delta
                    disc = max(0.0, b * b - 4.0 * a * c)
                    tau = 0.0 if a <= 1e-30 else (-b + np.sqrt(disc)) / (2.0 * a)
                    step = d_sd + tau * diff

            step_norm = float(np.linalg.norm(step))
            if step_norm <= e_2 * max(float(np.linalg.norm(x)), 1.0):
                message = "step below threshold"
                success = True
                return _DoglegResult(x=x, cost=cost, nit=nit, nfev=nfev, success=success, message=message)

            trial_x = x + step
            trial_r = residual_np(trial_x)
            nfev += 1
            trial_cost = float(np.dot(trial_r, trial_r))
            actual_improvement = cost - trial_cost
            predicted_improvement = float(2.0 * np.dot(g_vec, step) - np.dot(step, a_mat @ step))
            rho = actual_improvement / predicted_improvement if predicted_improvement > 0.0 else -np.inf

            if rho > 0.9:
                delta = max(delta, 2.5 * step_norm)
            elif rho < 0.05:
                delta *= 0.25

            if actual_improvement > 0.0 and rho > 0.0:
                improvement_ratio = actual_improvement / max(cost, 1e-30)
                x = trial_x
                r = trial_r
                cost = trial_cost
                if improvement_ratio < e_3:
                    message = "improvement below threshold"
                    success = True
                    return _DoglegResult(x=x, cost=cost, nit=nit, nfev=nfev, success=success, message=message)
                r, jac = residual_and_jacobian_np(x)
                nfev += 1
                cost = float(np.dot(r, r))
                a_mat, g_vec = normal_terms(jac, r)
                if np.linalg.norm(g_vec, ord=np.inf) < e_1:
                    message = "gradient below threshold"
                    success = True
                    return _DoglegResult(x=x, cost=cost, nit=nit, nfev=nfev, success=success, message=message)
                break

            if delta <= e_2 * max(float(np.linalg.norm(x)), 1.0):
                message = "trust region below threshold"
                success = True
                return _DoglegResult(x=x, cost=cost, nit=nit, nfev=nfev, success=success, message=message)

    return _DoglegResult(x=x, cost=cost, nit=nit, nfev=nfev, success=success, message=message)
