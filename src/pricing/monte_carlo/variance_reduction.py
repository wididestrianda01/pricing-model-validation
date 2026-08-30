"""Variance reduction: antithetic variates, control variate, importance sampling."""

from __future__ import annotations

import numpy as np

from pricing.benchmarks._validate import check_option_type
from pricing.monte_carlo.engine import (
    MCResult,
    discounted_payoff,
    path_payoffs,
    simulate_terminal,
    standard_normals,
    summarize,
    terminal_mean,
)


def mc_european_antithetic(
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
    """European price using antithetic pairs: average payoff(z) and payoff(-z).

    n_paths counts pairs (2*n_paths paths drawn). variance_ratio is the
    cost-normalized reduction: naive variance over the paired estimator variance,
    divided by 2 (each pair costs two paths), so > 1 when the pairing helps.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    z = standard_normals(n_paths, n_steps, seed)
    p_plus = path_payoffs(spot, strike, tte, vol, rate, option_type, z, scheme)
    p_minus = path_payoffs(spot, strike, tte, vol, rate, option_type, -z, scheme)
    est = 0.5 * (p_plus + p_minus)
    ratio = float(p_plus.var(ddof=1) / (2.0 * est.var(ddof=1)))
    return summarize(est, n_paths, f"{scheme}+antithetic", ratio)


def mc_european_control_variate(
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
    """European price using the discounted terminal asset as a control variate.

    The control has known expectation E[e^{-rT} S_T] = E[S_T] e^{-rT}, where
    E[S_T] is scheme-dependent (S0 e^{rT} for exact, S0 (1 + r dt)^n for
    Euler/Milstein). Subtracting b * (control - E[control]) keeps the estimator
    unbiased for any b; b is estimated from the sample. variance_ratio is the
    naive estimator variance over the controlled variance (> 1 when it helps).
    """
    option_type = check_option_type(option_type, ("call", "put"))
    z = standard_normals(n_paths, n_steps, seed)
    s_t = simulate_terminal(spot, tte, vol, rate, z, scheme)
    y = discounted_payoff(s_t, strike, rate, tte, option_type)
    control = np.exp(-rate * tte) * s_t
    control_mean = np.exp(-rate * tte) * terminal_mean(spot, tte, rate, scheme, n_steps)
    b = np.cov(y, control, ddof=1)[0, 1] / np.var(control, ddof=1)
    est = y - b * (control - control_mean)
    ratio = float(y.var(ddof=1) / est.var(ddof=1))
    return summarize(est, n_paths, f"{scheme}+control", ratio)


def _strike_shift(spot: float, strike: float, tte: float, vol: float, rate: float) -> float:
    """Drift shift that centers the proposal distribution at the strike."""
    return float((np.log(strike / spot) - (rate - 0.5 * vol**2) * tte) / (vol * np.sqrt(tte)))


def mc_european_importance_sampling(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 0,
    scheme: str = "exact",
    shift: float | None = None,
) -> MCResult:
    """European price under a drift-shifted proposal, reweighted by likelihood ratio.

    The proposal shifts the terminal normal by `shift` (default: centered at the
    strike, so out-of-the-money instruments waste fewer paths), and each payoff is
    reweighted by exp(-shift*xi - shift^2/2), keeping the estimator unbiased.
    variance_ratio compares the unreweighted (original-measure) estimator variance
    to the reweighted variance (> 1 when importance sampling reduces variance).
    """
    option_type = check_option_type(option_type, ("call", "put"))
    if shift is None:
        shift = _strike_shift(spot, strike, tte, vol, rate)
    xi = standard_normals(n_paths, 1, seed)
    z = xi + shift
    y = path_payoffs(spot, strike, tte, vol, rate, option_type, z, scheme)
    weight = np.exp(-shift * xi[:, 0] - 0.5 * shift**2)
    est = y * weight
    y_naive = path_payoffs(spot, strike, tte, vol, rate, option_type, xi, scheme)
    ratio = float(y_naive.var(ddof=1) / est.var(ddof=1))
    return summarize(est, n_paths, f"{scheme}+importance-sampling", ratio)
