"""Longstaff-Schwartz least-squares Monte Carlo for American/Bermudan exercise."""

from __future__ import annotations

import numpy as np

from pricing.benchmarks._validate import check_option_type, check_positive
from pricing.monte_carlo.engine import MCResult, summarize


def _payoff(s: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(s - strike, 0.0)
    return np.maximum(strike - s, 0.0)


def _design(s: np.ndarray, k: float, degree: int) -> np.ndarray:
    """Polynomial basis in moneyness s/K: columns 1, x, ..., x^degree."""
    x = s / k
    return np.vander(x, N=degree + 1, increasing=True)


def lsm_american(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    *,
    n_paths: int = 100_000,
    n_steps: int = 50,
    seed: int = 0,
    basis_degree: int = 3,
) -> MCResult:
    """American price by Longstaff-Schwartz regression on GBM paths.

    Exact-GBM path simulation; continuation values fitted by OLS on in-the-money
    paths at each exercise date using a polynomial basis in moneyness. The
    regression fit introduces a low bias (continuation values are underfit on
    finite samples); magnitude documented in the validation report.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    check_positive(spot=spot, strike=strike)
    if tte <= 0.0:
        raise ValueError("tte must be > 0")

    dt = tte / n_steps
    drift = (rate - 0.5 * vol**2) * dt
    diff = vol * np.sqrt(dt)
    z = np.random.default_rng(seed).standard_normal((n_paths, n_steps))
    log_s = np.empty((n_paths, n_steps + 1))
    log_s[:, 0] = np.log(spot)
    log_s[:, 1:] = np.log(spot) + np.cumsum(drift + diff * z, axis=1)
    s = np.exp(log_s)

    cashflow = _payoff(s[:, -1], strike, option_type)
    tau_idx = np.full(n_paths, n_steps)  # exercise step of each path's cashflow

    for t in range(n_steps - 1, 0, -1):
        y = cashflow * np.exp(-rate * dt * (tau_idx - t))  # valued at time t
        intrinsic = _payoff(s[:, t], strike, option_type)
        itm = intrinsic > 0.0
        if not itm.any():
            continue
        basis = _design(s[itm, t], strike, basis_degree)
        coef, *_ = np.linalg.lstsq(basis, y[itm], rcond=None)
        continuation = basis @ coef
        exercise = np.zeros(n_paths, dtype=bool)
        exercise[itm] = intrinsic[itm] > continuation
        cashflow[exercise] = intrinsic[exercise]
        tau_idx[exercise] = t

    est = cashflow * np.exp(-rate * dt * tau_idx)
    return summarize(est, n_paths, scheme="ls")
