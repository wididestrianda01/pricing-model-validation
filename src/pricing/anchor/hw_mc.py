"""Longstaff-Schwartz Monte Carlo pricing of Bermudan swaptions under Hull-White.

Simulates the short rate r_t = r0 + x_t where x follows an Ornstein-Uhlenbeck
process exactly (dx = -a*x*dt + sigma*dW), accumulating pathwise discount
factors along each trajectory. At every Bermudan exercise date the holder
exercises into the underlying swap when its model value exceeds the fitted
continuation value. Once exercised, the path credits ``swap_pv(r_tk)``
discounted to today along its own discount factor — equivalent to entering
the swap and carrying it to maturity, since the same model prices both legs.

The regression fit introduces a low bias (continuation values are underfit on
finite samples); magnitude is documented in the anchor validation memo.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pricing.anchor.instrument import (
    BermudanSwaption,
    HullWhiteParams,
    schedule_grid,
    swap_pv,
)
from pricing.monte_carlo.engine import MCResult, summarize


@njit(cache=True)
def _ou_paths(phi: np.ndarray, vol: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Exact centered-OU transitions: x[:, j+1] = phi[j]*x[:, j] + vol[j]*z."""
    n_paths, n_steps = z.shape
    x = np.empty((n_paths, n_steps + 1))
    for i in range(n_paths):
        x[i, 0] = 0.0
        for j in range(n_steps):
            x[i, j + 1] = phi[j] * x[i, j] + vol[j] * z[i, j]
    return x

def simulate_hw_paths(
    params: HullWhiteParams,
    times: np.ndarray,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-OU short-rate paths and pathwise discount factors on ``times``.

    Returns ``(r, disc)`` with shapes ``(n_paths, len(times))``; ``disc[:, k]``
    discounts a unit paid at ``times[k]`` back to 0 along the path.
    """
    dt = np.diff(times)
    phi = np.exp(-params.a * dt)
    vol = params.sigma * np.sqrt((1.0 - np.exp(-2.0 * params.a * dt)) / (2.0 * params.a))
    # Brigo-Mercurio decomposition: r_t = x_t + alpha(t) where x is a CENTERED
    # Ornstein-Uhlenbeck process (dx = -a*x*dt + sigma*dW, exact steps) and
    # alpha(t) = f(0,t) + sigma^2/(2a^2)*(1-exp(-a t))^2 fits the initial curve.
    alpha = (
        params.r0
        + params.sigma**2 / (2.0 * params.a**2)
        * (1.0 - np.exp(-params.a * times)) ** 2
    )
    z = np.random.default_rng(seed).standard_normal((n_paths, len(dt)))
    x = _ou_paths(phi, vol, z)
    r = alpha + x
    log_disc = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(-r[:, :-1] * dt, axis=1)], axis=1
    )
    return r, np.exp(log_disc)


def lsm_bermudan_swaption(
    swaption: BermudanSwaption,
    params: HullWhiteParams,
    *,
    n_paths: int = 100_000,
    substeps_per_year: int = 52,
    basis_degree: int = 3,
    seed: int = 0,
) -> MCResult:
    """Bermudan swaption price by Longstaff-Schwartz regression on OU paths."""
    times, idx = schedule_grid(
        swaption.exercise_times, swaption.pay_times, substeps_per_year
    )
    r, disc = simulate_hw_paths(params, times, n_paths, seed)

    ex_nodes = [idx[float(t)] for t in swaption.exercise_times]
    pv = np.zeros(n_paths)
    payoff_node = np.full(n_paths, -1)

    for node in reversed(ex_nodes):
        t_k = float(times[node])
        intrinsic = np.maximum(swap_pv(swaption, params, t_k, r[:, node]), 0.0)
        itm = intrinsic > 0.0
        if payoff_node.max() < 0:
            y = np.zeros(n_paths)
        else:
            y = pv * disc[:, node] / disc[np.arange(n_paths), payoff_node]
        ri = r[itm, node]
        basis = np.vander((ri - ri.mean()) / (ri.std() + 1e-12), basis_degree + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(basis, y[itm], rcond=None)
        continuation = basis @ coef
        do_ex = np.zeros(n_paths, dtype=bool)
        do_ex[itm] = intrinsic[itm] > continuation
        pv[do_ex] = intrinsic[do_ex]
        payoff_node[do_ex] = node

    est = pv * disc[np.arange(n_paths), np.maximum(payoff_node, 0)]
    est[payoff_node < 0] = 0.0
    return summarize(est, n_paths, scheme="ls")
