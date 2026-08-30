"""Monte Carlo engine: seeded GBM path simulation, European pricing, convergence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from pricing.benchmarks._validate import check_option_type, check_positive

SCHEMES = ("euler", "milstein", "exact")


@njit(cache=True)
def _terminal_euler(s0: float, r: float, vol: float, dt: float, z: np.ndarray) -> np.ndarray:
    n_paths, n_steps = z.shape
    out = np.empty(n_paths, dtype=np.float64)
    drift = r * dt
    vol_dt = vol * np.sqrt(dt)
    for i in range(n_paths):
        s = s0
        for j in range(n_steps):
            s *= 1.0 + drift + vol_dt * z[i, j]
        out[i] = s
    return out


@njit(cache=True)
def _terminal_milstein(s0: float, r: float, vol: float, dt: float, z: np.ndarray) -> np.ndarray:
    n_paths, n_steps = z.shape
    out = np.empty(n_paths, dtype=np.float64)
    drift = r * dt
    vol_dt = vol * np.sqrt(dt)
    half_vol2_dt = 0.5 * vol * vol * dt
    for i in range(n_paths):
        s = s0
        for j in range(n_steps):
            zz = z[i, j]
            s *= 1.0 + drift + vol_dt * zz + half_vol2_dt * (zz * zz - 1.0)
        out[i] = s
    return out


@njit(cache=True)
def _terminal_exact(s0: float, r: float, vol: float, dt: float, z: np.ndarray) -> np.ndarray:
    """Exact GBM solution S_T = S_0 exp((r - 0.5*vol^2)*T + vol*W_T), driven by the same z."""
    n_paths, n_steps = z.shape
    out = np.empty(n_paths, dtype=np.float64)
    tte = dt * n_steps
    drift = (r - 0.5 * vol * vol) * tte
    vol_dt = vol * np.sqrt(dt)
    for i in range(n_paths):
        w = 0.0
        for j in range(n_steps):
            w += z[i, j]
        out[i] = s0 * np.exp(drift + vol_dt * w)
    return out


def standard_normals(n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    """Seeded standard-normal draws of shape (n_paths, n_steps)."""
    if n_paths < 1 or n_steps < 1:
        raise ValueError(f"n_paths and n_steps must be >= 1, got {n_paths}, {n_steps}")
    return np.random.default_rng(seed).standard_normal((n_paths, n_steps))


def simulate_terminal(spot: float, tte: float, vol: float, rate: float, z: np.ndarray, scheme: str = "euler") -> np.ndarray:
    """Terminal asset values for each path, given standard-normal increments z.

    z has shape (n_paths, n_steps); the time step is tte / n_steps. "exact" jumps
    straight to the closed-form GBM terminal (no time discretization error).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    check_positive(spot=spot)
    if vol < 0.0:
        raise ValueError(f"vol must be >= 0, got {vol}")
    n_steps = z.shape[1]
    dt = tte / n_steps
    if scheme == "euler":
        return _terminal_euler(spot, rate, vol, dt, z)
    if scheme == "milstein":
        return _terminal_milstein(spot, rate, vol, dt, z)
    return _terminal_exact(spot, rate, vol, dt, z)


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with uncertainty.

    `variance`: sample variance of the per-path estimator — except for Sobol
    QMC, where it is the across-replication variance of replication prices.
    """
    price: float
    std_error: float
    variance: float
    variance_ratio: float
    n_paths: int
    scheme: str


def discounted_payoff(s_t: np.ndarray, strike: float, rate: float, tte: float, option_type: str) -> np.ndarray:
    """Discounted terminal payoff for a European call/put."""
    check_positive(strike=strike)
    disc = np.exp(-rate * tte)
    if option_type == "call":
        return disc * np.maximum(s_t - strike, 0.0)
    return disc * np.maximum(strike - s_t, 0.0)


def summarize(est: np.ndarray, n_paths: int, scheme: str, variance_ratio: float = 1.0) -> MCResult:
    """Build an MCResult from an estimator sample (mean, variance, standard error)."""
    price = float(est.mean())
    variance = float(est.var(ddof=1)) if n_paths > 1 else 0.0
    std_error = float(np.sqrt(variance / n_paths))
    return MCResult(price, std_error, variance, variance_ratio, n_paths, scheme)


def path_payoffs(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str,
    z: np.ndarray,
    scheme: str,
) -> np.ndarray:
    """Discounted terminal payoff per path, given standard-normal increments z."""
    s_t = simulate_terminal(spot, tte, vol, rate, z, scheme)
    return discounted_payoff(s_t, strike, rate, tte, option_type)


def terminal_mean(spot: float, tte: float, rate: float, scheme: str, n_steps: int) -> float:
    """Expected terminal asset value E[S_T] for the scheme.

    Euler and Milstein both step S -> S (1 + r dt + ...), so E[S_T] = S0 (1 + r dt)^n;
    the exact jump has E[S_T] = S0 e^{rT}.
    """
    if scheme == "exact":
        return float(spot * np.exp(rate * tte))
    return float(spot * (1.0 + rate * tte / n_steps) ** n_steps)


def mc_european(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 1,
    seed: int = 0,
    scheme: str = "euler",
) -> MCResult:
    """Price a European call/put by Monte Carlo under geometric Brownian motion.

    Returns the discounted-payoff mean with its standard error and per-draw variance.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    z = standard_normals(n_paths, n_steps, seed)
    disc = path_payoffs(spot, strike, tte, vol, rate, option_type, z, scheme)
    return summarize(disc, n_paths, scheme)
