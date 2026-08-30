"""Heston calibration to a cross-expiry smile, plus an Euler-MC cross-check pricer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from pricing.benchmarks.heston import heston
from pricing.calibration.implied_vol import implied_vol


@dataclass(frozen=True)
class HestonFit:
    """Calibrated Heston parameters plus fit quality."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    price_rmse: float  # option-price RMSE in currency units
    iv_rmse: float  # implied-vol RMSE in vol points (at the fitted optimum)
    feller_violation: float  # 2*kappa*theta - xi^2; negative means violated
    converged: bool


HESTON_X0 = np.array([0.04, 1.5, 0.04, 0.4, -0.3])  # v0, kappa, theta, xi, rho
HESTON_LOWER = np.array([1e-4, 0.05, 1e-4, 0.05, -0.999])
HESTON_UPPER = np.array([1.0, 20.0, 1.0, 8.0, 0.999])


def calibrate_heston(
    spot: float,
    strikes: np.ndarray,
    ttes: np.ndarray,
    prices: np.ndarray,
    rate: float,
    dividend: float = 0.0,
) -> HestonFit:
    """Fit (v0, kappa, theta, xi, rho) to European option prices.

    ``strikes``, ``ttes`` and ``prices`` are same-length quote arrays; several
    maturities pin down theta and kappa. The objective is price-space RMSE via
    the closed-form characteristic-function pricer; IV RMSE is reported at the
    optimum. Start point deterministic.
    """
    strikes = np.asarray(strikes, dtype=float)
    ttes = np.asarray(ttes, dtype=float)
    prices = np.asarray(prices, dtype=float)
    if not strikes.shape == ttes.shape == prices.shape or strikes.ndim != 1:
        raise ValueError("strikes, ttes and prices must be same-length 1-D arrays")
    if len(strikes) < 5:
        raise ValueError("need at least 5 quotes to identify 5 parameters")
    x0, lower, upper = HESTON_X0, HESTON_LOWER, HESTON_UPPER

    def model_prices(x: np.ndarray) -> np.ndarray:
        v0, kappa, theta, xi, rho = x
        out = np.empty(len(strikes))
        for i in range(len(strikes)):
            out[i] = heston(spot, strikes[i], ttes[i], v0, kappa, theta, xi, rho,
                            rate, dividend)
        return out

    def residual(x: np.ndarray) -> np.ndarray:
        return model_prices(x) - prices

    res = least_squares(residual, x0, bounds=(lower, upper), xtol=1e-10, ftol=1e-10)
    v0, kappa, theta, xi, rho = res.x

    final_prices = model_prices(res.x)
    fitted_ivs = []
    target_ivs = []
    for i in range(len(strikes)):
        opt_type = "call" if strikes[i] >= spot else "put"
        fitted_ivs.append(implied_vol(final_prices[i], spot, strikes[i], ttes[i],
                                      rate, opt_type, dividend))
        target_ivs.append(implied_vol(prices[i], spot, strikes[i], ttes[i], rate,
                                      opt_type, dividend))
    return HestonFit(
        v0=float(v0),
        kappa=float(kappa),
        theta=float(theta),
        xi=float(xi),
        rho=float(rho),
        price_rmse=float(np.sqrt(np.mean((final_prices - prices) ** 2))),
        iv_rmse=float(np.sqrt(np.mean((np.asarray(fitted_ivs) - np.asarray(target_ivs)) ** 2))),
        feller_violation=2.0 * kappa * theta - xi**2,
        converged=bool(res.success),
    )


@njit(cache=True)
def _heston_euler_paths(
    spot: float,
    rate: float,
    dividend: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    tte: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-truncation Euler paths; returns terminal log-spot and variance."""
    dt = tte / n_steps
    log_s = np.full(n_paths, np.log(spot))
    v = np.full(n_paths, v0)
    np.random.seed(seed)
    for step in range(n_steps):
        z_s = np.random.standard_normal(n_paths)
        z_v = rho * z_s + np.sqrt(1.0 - rho**2) * np.random.standard_normal(n_paths)
        v_pos = np.maximum(v, 0.0)
        log_s += (rate - dividend - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z_s
        v += kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z_v
    return log_s, v


def heston_mc_price(
    spot: float,
    strike: float,
    tte: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    rate: float,
    dividend: float = 0.0,
    option_type: str = "call",
    *,
    n_steps: int = 64,
    n_paths: int = 50_000,
    seed: int = 7,
) -> tuple[float, float]:
    """Independent Euler full-truncation MC price with its standard error."""
    log_s, _ = _heston_euler_paths(
        spot, rate, dividend, v0, kappa, theta, xi, rho, tte, n_steps, n_paths, seed
    )
    s_t = np.exp(log_s)
    payoff = np.maximum(s_t - strike, 0.0) if option_type == "call" else np.maximum(strike - s_t, 0.0)
    disc = np.exp(-rate * tte)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return float(price), float(stderr)
