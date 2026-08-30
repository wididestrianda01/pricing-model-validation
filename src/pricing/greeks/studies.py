"""Greek validation studies: bump bias-variance tradeoff and cross-method agreement."""

from __future__ import annotations

import numpy as np

from pricing.benchmarks import black_scholes_greeks
from pricing.greeks.finite_difference import bump_greeks
from pricing.greeks.likelihood_ratio import lr_greeks
from pricing.greeks.pathwise import pathwise_greeks


def bias_variance_sweep(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
    *,
    h_list: list[float],
    n_paths: int,
    n_replications: int,
    base_seed: int = 0,
    scheme: str = "exact",
    difference: str = "central",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bias and variance of a bump delta across a bump sweep.

    `difference` selects central (bias ~ h^2) or forward (bias ~ h). Each bump
    size is estimated `n_replications` times at independent seeds; bias is the
    mean estimate minus the closed-form delta, variance is the spread across
    replications. As the bump shrinks, bias falls but variance rises (the
    bump-difference noise scales as 1/h^2), so a finite optimum exists.
    """
    delta_true = black_scholes_greeks(spot, strike, tte, vol, rate, option_type)["delta"]
    hs = np.asarray(h_list, dtype=np.float64)
    biases = np.empty_like(hs)
    variances = np.empty_like(hs)
    for i, h in enumerate(hs):
        ests = np.empty(n_replications)
        for r in range(n_replications):
            ests[r] = bump_greeks(
                spot, strike, tte, vol, rate, option_type,
                n_paths=n_paths, n_steps=1, seed=base_seed + r,
                scheme=scheme, h_spot=float(h), common_random=False,
                difference=difference,
            )["delta"]
        biases[i] = ests.mean() - delta_true
        variances[i] = ests.var(ddof=1)

    return hs, biases, variances


def cross_method_greeks(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
    *,
    n_paths: int,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Closed-form, bump, pathwise, and likelihood-ratio Greeks side by side.

    The single seam every estimator validates against: a function taking market
    data + instrument spec and returning Greeks, cross-checked against the
    closed form and against each other.
    """
    return {
        "closed_form": black_scholes_greeks(spot, strike, tte, vol, rate, option_type),
        "bump": bump_greeks(spot, strike, tte, vol, rate, option_type, n_paths=n_paths, seed=seed),
        "pathwise": pathwise_greeks(spot, strike, tte, vol, rate, option_type, n_paths=n_paths, seed=seed),
        "likelihood_ratio": lr_greeks(spot, strike, tte, vol, rate, option_type, n_paths=n_paths, seed=seed),
    }
