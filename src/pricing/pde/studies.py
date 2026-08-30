"""Grid-convergence studies and Richardson extrapolation for the PDE engine."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pricing.benchmarks import black_scholes
from pricing.pde.schemes import fd_american, fd_price


def _loglog_order(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def reference_price(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    payoff: str = "vanilla",
) -> float:
    """Closed form for the supported European payoffs (vanilla, digital).

    Vanilla delegates to the shared benchmarks closed form; only the digital
    branch is implemented here (the repo has no other digital source).
    """
    if payoff == "vanilla":
        return black_scholes(spot, strike, tte, vol, rate, option_type)
    if tte <= 0.0:
        raise ValueError("tte must be > 0")
    x = np.log(spot / strike)
    sqrt_t = np.sqrt(tte)
    mu = rate - 0.5 * vol**2
    d2 = (x + mu * tte) / (vol * sqrt_t)
    disc = np.exp(-rate * tte)
    p = norm.cdf(d2) if option_type == "call" else norm.cdf(-d2)
    return float(disc * p)


def richardson_extrapolate(v_coarse: float, v_fine: float, order: float) -> float:
    """Extrapolate two halved-grid values to the continuum limit given the order."""
    ratio = 2.0**order
    return float((ratio * v_fine - v_coarse) / (ratio - 1.0))


def _sweep(
    sizes: tuple[int, ...],
    price_at: callable,
    ref: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Shared convergence sweep: price at each grid size, error vs ref,
    observed log-log slope against a step-size axis (coarsest = 1)."""
    axis = np.empty(len(sizes))
    errs = np.empty(len(sizes))
    for k, s in enumerate(sizes):
        axis[k] = sizes[-1] / s  # proportional to the step being halved
        errs[k] = abs(price_at(s) - ref)
    return _loglog_order(axis, errs), axis, errs


def spatial_order(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    *,
    scheme: str = "cn",
    payoff: str = "vanilla",
    barrier: float | None = None,
    n_space_list: tuple[int, ...] = (50, 100, 200, 400),
    n_time: int = 400,
    reference: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Observed order of convergence as dx halves with time error negligible.

    reference overrides the closed-form target (barrier options pass a
    QuantLib or extrapolated value here).
    """
    ref = (
        reference_price(spot, strike, tte, vol, rate, option_type, payoff)
        if reference is None
        else reference
    )
    return _sweep(
        n_space_list,
        lambda n: fd_price(
            spot,
            strike,
            tte,
            vol,
            rate,
            option_type,
            scheme=scheme,
            payoff=payoff,
            barrier=barrier,
            n_space=n,
            n_time=n_time,
        ),
        ref,
    )


def temporal_order(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    *,
    scheme: str = "cn",
    payoff: str = "vanilla",
    n_time_list: tuple[int, ...] = (25, 50, 100, 200),
    n_space: int = 400,
    reference: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Observed order of convergence as dt halves with space error negligible."""
    ref = (
        reference_price(spot, strike, tte, vol, rate, option_type, payoff)
        if reference is None
        else reference
    )
    return _sweep(
        n_time_list,
        lambda n: fd_price(
            spot,
            strike,
            tte,
            vol,
            rate,
            option_type,
            scheme=scheme,
            payoff=payoff,
            n_space=n_space,
            n_time=n,
        ),
        ref,
    )


def early_exercise_order(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    *,
    n_space_list: tuple[int, ...] = (50, 100, 200),
    n_time: int = 400,
    reference: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Convergence of the PSOR American price as the space grid refines.

    The free boundary's kink degrades the observed order relative to smooth
    European data -- that degradation is the delivered result, not a failure.
    """
    ref = (
        reference
        if reference is not None
        else fd_american(
            spot,
            strike,
            tte,
            vol,
            rate,
            option_type,
            n_space=n_space_list[-1] * 2,
            n_time=n_time,
        )
    )
    return _sweep(
        n_space_list,
        lambda n: fd_american(
            spot, strike, tte, vol, rate, option_type, n_space=n, n_time=n_time
        ),
        ref,
    )
