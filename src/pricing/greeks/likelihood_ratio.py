"""Likelihood-ratio (score-function) Greeks for the exact-GBM lognormal model."""

from __future__ import annotations

import numpy as np

from pricing.benchmarks._validate import check_option_type
from pricing.monte_carlo import simulate_terminal, standard_normals
from pricing.monte_carlo.engine import discounted_payoff


def lr_scores(z: np.ndarray, spot: float, vol: float, tte: float) -> dict[str, np.ndarray]:
    """Per-path likelihood-ratio scores for spot (delta, gamma) and vol (vega).

    `z` has shape (n_paths, n_steps); the effective terminal standard normal is
    `z.sum(axis=1) / sqrt(n_steps)`. Valid for the exact-GBM lognormal terminal,
    and payoff-agnostic — so it also works on discontinuous payoffs.
    """
    n_steps = z.shape[1]
    big_z = z.sum(axis=1) / np.sqrt(n_steps)
    sqrt_t = np.sqrt(tte)
    delta = big_z / (spot * vol * sqrt_t)
    gamma = (big_z**2 - big_z * vol * sqrt_t - 1.0) / (spot**2 * vol**2 * tte)
    vega = (big_z**2 - 1.0) / vol - sqrt_t * big_z
    return {"delta": delta, "gamma": gamma, "vega": vega}


def lr_greeks(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 1,
    seed: int = 0,
) -> dict[str, float]:
    """Likelihood-ratio delta, gamma, and vega for a European call/put.

    The score function is valid on discontinuous payoffs too (unlike pathwise),
    at the cost of higher variance. Uses exact-GBM simulation.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    z = standard_normals(n_paths, n_steps, seed)
    s_t = simulate_terminal(spot, tte, vol, rate, z, "exact")
    payoff = discounted_payoff(s_t, strike, rate, tte, option_type)
    scores = lr_scores(z, spot, vol, tte)
    return {key: float(np.mean(payoff * scores[key])) for key in ("delta", "gamma", "vega")}
