"""Pre-jit pure-NumPy reference implementations of the PDE kernels.

These are the pure-NumPy acceleration baselines: the
benchmark script times the Numba-jitted kernels against these, and tests
assert the jitted results match them within numerical tolerance. Not used
by the engine itself.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def thomas_ref(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Tridiagonal solve via scipy's banded solver (the natural pre-jit call)."""
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def march_explicit_ref(
    u: np.ndarray,
    lo: float,
    di: float,
    up: float,
    dt: float,
    v_lo: np.ndarray,
    v_hi: np.ndarray,
    barrier_idx: int,
) -> np.ndarray:
    """Vectorized NumPy explicit march (edges + barrier re-enforced per step)."""
    for i in range(len(v_lo)):
        out = u.copy()
        out[1:-1] = u[1:-1] + dt * (lo * u[:-2] + di * u[1:-1] + up * u[2:])
        out[0] = v_lo[i]
        out[-1] = v_hi[i]
        if barrier_idx >= 0:
            out[barrier_idx] = 0.0
        u = out
    return u


def march_theta_ref(
    u: np.ndarray,
    mat_lo: np.ndarray,
    mat_di: np.ndarray,
    mat_up: np.ndarray,
    lo_c: float,
    di_c: float,
    up_c: float,
    dt: float,
    theta: float,
    v_lo: np.ndarray,
    v_hi: np.ndarray,
    barrier_idx: int,
) -> np.ndarray:
    """Vectorized NumPy implicit-family march (implicit/CN/single-factor ADI)."""
    for i in range(len(v_lo)):
        rhs = u[1:-1] + (1.0 - theta) * dt * (
            lo_c * u[:-2] + di_c * u[1:-1] + up_c * u[2:]
        )
        rhs[0] += theta * dt * lo_c * v_lo[i]
        rhs[-1] += theta * dt * up_c * v_hi[i]
        out = u.copy()
        out[1:-1] = thomas_ref(mat_lo, mat_di, mat_up, rhs)
        out[0] = v_lo[i]
        out[-1] = v_hi[i]
        if barrier_idx >= 0:
            out[barrier_idx] = 0.0
        u = out
    return u


def psor_ref(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 20_000,
) -> np.ndarray:
    """Projected SOR without jit — same iteration as pricing.pde.solvers.psor."""
    n = len(diag)
    u = obstacle.copy()
    omega = 1.2
    for _ in range(max_iter):
        delta_max = 0.0
        for j in range(n):
            if j == 0:
                s = upper[0] * u[1]
            elif j == n - 1:
                s = lower[j] * u[j - 1]
            else:
                s = lower[j] * u[j - 1] + upper[j] * u[j + 1]
            upd = max((rhs[j] - s) / diag[j], obstacle[j])
            delta_max = max(delta_max, abs(upd - u[j]))
            u[j] = u[j] + omega * (upd - u[j])
        if delta_max < tol:
            break
    return u
