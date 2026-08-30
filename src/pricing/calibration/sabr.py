"""SABR smile: Hagan (2002) lognormal implied volatility and least-squares calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def sabr_vol(
    forward: float,
    strikes: np.ndarray,
    tte: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Hagan et al. (2002) SABR implied lognormal volatility, vectorised in ``strikes``.

    alpha: initial vol scale. beta: CEV exponent (fixed at calibration time).
    rho: spot/vol correlation. nu: vol-of-vol. tte: time to expiry in years.
    """
    if forward <= 0.0:
        raise ValueError("forward must be > 0")
    if tte <= 0.0:
        raise ValueError("tte must be > 0")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    if abs(rho) >= 1.0:
        raise ValueError("rho must lie in (-1, 1)")
    if alpha <= 0.0 or nu < 0.0:
        raise ValueError("alpha must be > 0 and nu must be >= 0")

    k = np.asarray(strikes, dtype=float)
    if np.any(k <= 0.0):
        raise ValueError("strikes must be > 0")

    f_pow = forward ** (1.0 - beta)
    log_fk = np.log(forward / k)
    z = nu / alpha * f_pow * log_fk
    # x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho)); z/x -> 1 as z -> 0
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z**2)
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))
    z_over_x = np.empty_like(z)
    small = np.abs(z) < 1e-12
    z_over_x[~small] = z[~small] / x_z[~small]
    z_over_x[small] = 1.0 - 0.5 * rho * z[small]

    correction = (
        ((1.0 - beta) ** 2 / 24.0) * alpha**2 / f_pow**2
        + rho * beta * nu * alpha / (4.0 * f_pow)
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    )
    return alpha / f_pow * z_over_x * (1.0 + correction * tte)


@dataclass(frozen=True)
class SABRFit:
    """Calibrated SABR parameters plus fit quality."""

    alpha: float
    beta: float
    rho: float
    nu: float
    atm_vol: float
    rmse: float  # implied-vol RMSE, vol points
    converged: bool


def calibrate_sabr(
    strikes: np.ndarray,
    iv: np.ndarray,
    forward: float,
    tte: float,
    *,
    beta: float = 1.0,
    weights: np.ndarray | None = None,
) -> SABRFit:
    """Fit SABR (alpha, rho, nu) to a quoted smile at fixed ``beta``.

    Starting alpha is backed out from the ATM-implied approximation; start
    point is deterministic so refits are reproducible.
    """
    k = np.asarray(strikes, dtype=float)
    target = np.asarray(iv, dtype=float)
    if k.shape != target.shape or k.ndim != 1:
        raise ValueError("strikes and iv must be same-length 1-D arrays")
    if len(k) < 3:
        raise ValueError("need at least 3 smile points")

    atm_idx = int(np.argmin(np.abs(k - forward)))
    atm_vol = float(target[atm_idx])
    alpha0 = max(atm_vol * forward ** (1.0 - beta), 1e-6)
    x0 = np.array([alpha0, -0.3, 1.0])
    lower = np.array([1e-8, -0.999, 0.0])
    upper = np.array([10.0, 0.999, 50.0])
    w = np.ones_like(target) if weights is None else np.asarray(weights, dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        model = sabr_vol(forward, k, tte, x[0], beta, x[1], x[2])
        return w * (model - target)

    res = least_squares(residual, x0, bounds=(lower, upper), xtol=1e-12, ftol=1e-12)
    model = sabr_vol(forward, k, tte, res.x[0], beta, res.x[1], res.x[2])
    return SABRFit(
        alpha=float(res.x[0]),
        beta=float(beta),
        rho=float(res.x[1]),
        nu=float(res.x[2]),
        atm_vol=atm_vol,
        rmse=float(np.sqrt(np.mean((model - target) ** 2))),
        converged=bool(res.success),
    )
