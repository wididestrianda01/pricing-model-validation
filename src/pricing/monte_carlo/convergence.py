"""Convergence studies: weak/strong order of the schemes and SE scaling."""

from __future__ import annotations

import numpy as np

from pricing.benchmarks import black_scholes
from pricing.monte_carlo.engine import mc_european, simulate_terminal, standard_normals


def _loglog_order(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def weak_convergence(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str,
    n_paths: int,
    n_steps_list: list[int],
    seed: int,
    scheme: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Weak (price) convergence order: how the discretization bias falls with dt."""
    bs = black_scholes(spot, strike, tte, vol, rate, option_type)
    dts = np.asarray([tte / n for n in n_steps_list])
    errors = np.empty(len(n_steps_list))
    for k, n in enumerate(n_steps_list):
        r = mc_european(spot, strike, tte, vol, rate, option_type, n_paths, n, seed, scheme)
        errors[k] = abs(r.price - bs)
    return _loglog_order(dts, errors), dts, errors


def strong_convergence(
    spot: float,
    tte: float,
    vol: float,
    rate: float,
    n_paths: int,
    n_steps_list: list[int],
    seed: int,
    scheme: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Strong (pathwise) convergence order against the exact GBM terminal, same z."""
    dts = np.asarray([tte / n for n in n_steps_list])
    errors = np.empty(len(n_steps_list))
    for k, n in enumerate(n_steps_list):
        z = standard_normals(n_paths, n, seed)
        s_scheme = simulate_terminal(spot, tte, vol, rate, z, scheme)
        s_exact = simulate_terminal(spot, tte, vol, rate, z, "exact")
        errors[k] = float(np.mean(np.abs(s_scheme - s_exact)))
    return _loglog_order(dts, errors), dts, errors


def se_scaling(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str,
    n_paths_list: list[int],
    seed: int,
    scheme: str = "exact",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Standard-error vs path count: the slope should be -0.5 (1/sqrt(N) decay)."""
    ns = np.asarray(n_paths_list, dtype=np.float64)
    ses = np.empty(len(n_paths_list))
    for k, n in enumerate(n_paths_list):
        r = mc_european(spot, strike, tte, vol, rate, option_type, n, 1, seed, scheme)
        ses[k] = r.std_error
    return _loglog_order(ns, ses), ns, ses
