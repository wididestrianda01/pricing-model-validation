"""Scrambled Sobol quasi-Monte Carlo as an alternative RNG for the engine."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats.qmc import Sobol

from pricing.benchmarks._validate import check_option_type
from pricing.monte_carlo.engine import MCResult, path_payoffs


def sobol_normals(n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    """Scrambled Sobol low-discrepancy points mapped to standard normals.

    Scrambling (fixed by seed) keeps the estimate deterministic while remaining
    unbiased, so the error is estimable across independent replications. The point
    count is rounded up to a power of two to preserve the sequence's balance; the
    terminal (n_steps=1) use case is power-of-two by construction.
    """
    n = n_paths * n_steps
    n_balanced = 1 << (n - 1).bit_length()
    sampler = Sobol(d=1, scramble=True, seed=seed)
    u = sampler.random(n_balanced).ravel()[:n]
    return ndtri(u).reshape(n_paths, n_steps)


def mc_european_sobol(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
    n_paths: int = 2**16,
    n_steps: int = 1,
    seed: int = 0,
    scheme: str = "exact",
    n_replications: int = 16,
) -> MCResult:
    """European price by scrambled Sobol QMC.

    The price is the mean over `n_replications` independent scrambled seeds, so
    the standard error is estimated from the replication spread (scrambling keeps
    the error estimable, unlike plain low-discrepancy sequences). Note that
    `MCResult.variance` here is the across-replication price variance, not
    the per-path estimator variance returned by the other estimators.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    prices = np.empty(n_replications, dtype=np.float64)
    for r in range(n_replications):
        z = sobol_normals(n_paths, n_steps, seed=seed + r)
        prices[r] = path_payoffs(spot, strike, tte, vol, rate, option_type, z, scheme).mean()
    price = float(prices.mean())
    variance = float(prices.var(ddof=1)) if n_replications > 1 else 0.0
    std_error = float(np.sqrt(variance / n_replications))
    return MCResult(price, std_error, variance, 1.0, n_paths, f"{scheme}+sobol")
