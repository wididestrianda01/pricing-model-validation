"""Tridiagonal and projected solvers for the finite-difference PDE engine.

Both kernels are Numba-jitted (ADR-0001): they are the PDE engine's inner
loops, called once or many times per time step.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def thomas(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system.

    lower[j] multiplies x[j-1], upper[j] multiplies x[j+1];
    lower[0] and upper[-1] are ignored.
    """
    n = len(diag)
    cp = np.empty(n)
    dp = np.empty(n)
    beta = diag[0]
    cp[0] = upper[0] / beta
    dp[0] = rhs[0] / beta
    for j in range(1, n):
        beta = diag[j] - lower[j] * cp[j - 1]
        cp[j] = upper[j] / beta
        dp[j] = (rhs[j] - lower[j] * dp[j - 1]) / beta
    x = np.empty(n)
    x[n - 1] = dp[n - 1]
    for j in range(n - 2, -1, -1):
        x[j] = dp[j] - cp[j] * x[j + 1]
    return x


@njit(cache=True)
def psor(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 20_000,
) -> np.ndarray:
    """Projected successive over-relaxation for the American LCP:
        solve L u = rhs subject to u >= obstacle componentwise.

    Starts from the obstacle (immediate exercise), sweeps with relaxation
    omega = 1.2, stops when the sup-norm update falls under tol.
    """
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
            upd = (rhs[j] - s) / diag[j]
            upd = max(upd, obstacle[j])
            delta_max = max(delta_max, abs(upd - u[j]))
            u[j] = u[j] + omega * (upd - u[j])
        if delta_max < tol:
            break
    return u


@njit(cache=True)
def march_explicit(
    u: np.ndarray,
    lo: float,
    di: float,
    up: float,
    dt: float,
    v_lo: np.ndarray,
    v_hi: np.ndarray,
    barrier_idx: int,
) -> np.ndarray:
    """Explicit scheme marched over every step; Dirichlet edges and the
    absorbing barrier node re-enforced after each step (mutates u)."""
    n = len(u)
    tmp = np.empty(n - 2)
    for i in range(len(v_lo)):
        for j in range(n - 2):
            tmp[j] = u[j + 1] + dt * (lo * u[j] + di * u[j + 1] + up * u[j + 2])
        u[0] = v_lo[i]
        u[n - 1] = v_hi[i]
        for j in range(n - 2):
            u[j + 1] = tmp[j]
        if barrier_idx >= 0:
            u[barrier_idx] = 0.0
    return u


@njit(cache=True)
def march_theta(
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
    """Implicit-family scheme ((I - theta*dt*A) u_new = (I + (1-theta)*dt*A) u)
    marched over every step via the Thomas solve; covers implicit (theta=1),
    Crank-Nicolson (theta=1/2), and the single-factor Douglas/ADI sweep
    (algebraically CN). Mutates u."""
    n = len(u)
    rhs = np.empty(n - 2)
    c = (1.0 - theta) * dt
    for i in range(len(v_lo)):
        for j in range(n - 2):
            rhs[j] = u[j + 1] + c * (lo_c * u[j] + di_c * u[j + 1] + up_c * u[j + 2])
        rhs[0] += theta * dt * lo_c * v_lo[i]
        rhs[-1] += theta * dt * up_c * v_hi[i]
        x = thomas(mat_lo, mat_di, mat_up, rhs)
        u[0] = v_lo[i]
        u[n - 1] = v_hi[i]
        for j in range(n - 2):
            u[j + 1] = x[j]
        if barrier_idx >= 0:
            u[barrier_idx] = 0.0
    return u
