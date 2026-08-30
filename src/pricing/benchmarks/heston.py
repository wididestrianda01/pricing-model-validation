"""Closed-form Heston European option price via characteristic-function integration."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from pricing.benchmarks._validate import check_option_type, check_positive


def _char(u: complex, t: float, kappa: float, theta: float, xi: float, rho: float,
          v0: float, r: float, q: float, s0: float) -> complex:
    """Heston log-price characteristic function (Albrecher et al. 'Little Trap')."""
    x = np.log(s0) + (r - q) * t
    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi**2 * (1j * u + u**2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)
    e_dt = np.exp(-d * t)
    C = (kappa * theta / xi**2) * ((kappa - rho * xi * 1j * u - d) * t
                                   - 2.0 * np.log((1.0 - g * e_dt) / (1.0 - g)))
    D = (kappa - rho * xi * 1j * u - d) * (1.0 - e_dt) / (xi**2 * (1.0 - g * e_dt))
    return np.exp(C + D * v0 + 1j * u * x)


def heston(
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
    u_max: float = 100.0,
) -> float:
    """Price a European call/put under Heston stochastic volatility.

    v0: initial variance. kappa: mean reversion speed. theta: long-run variance.
    xi: vol-of-vol. rho: spot/variance correlation. rate: risk-free rate.
    dividend: continuous dividend yield. option_type: "call" | "put".
    u_max: characteristic-function integration cutoff; raise it for very long
    maturities or extreme vol-of-vol if quad reports poor convergence.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    check_positive(spot=spot, strike=strike, kappa=kappa)
    if min(v0, theta, xi) < 0.0:
        raise ValueError("v0, theta, xi must be >= 0")
    if abs(rho) > 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    log_k = np.log(strike)
    forward = spot * np.exp((rate - dividend) * tte)

    def p1_integrand(u: float) -> float:
        phi = _char(u - 1j, tte, kappa, theta, xi, rho, v0, rate, dividend, spot)
        return float(np.real(np.exp(-1j * u * log_k) * phi / (1j * u * forward)))

    def p2_integrand(u: float) -> float:
        phi = _char(u, tte, kappa, theta, xi, rho, v0, rate, dividend, spot)
        return float(np.real(np.exp(-1j * u * log_k) * phi / (1j * u)))

    p1 = 0.5 + (1.0 / np.pi) * quad(p1_integrand, 1e-10, u_max, limit=200)[0]
    p2 = 0.5 + (1.0 / np.pi) * quad(p2_integrand, 1e-10, u_max, limit=200)[0]

    call = spot * np.exp(-dividend * tte) * p1 - strike * np.exp(-rate * tte) * p2
    if option_type == "call":
        return float(call)
    return float(call - spot * np.exp(-dividend * tte) + strike * np.exp(-rate * tte))
