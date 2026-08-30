"""Closed-form Black-Scholes European option price."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pricing.benchmarks._validate import check_option_type, check_positive, option_sign


def black_scholes(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
) -> float:
    """Price a European call/put under Black-Scholes (continuous, no dividend).

    tte: time to expiry in years. vol: annualized lognormal volatility.
    rate: continuously compounded risk-free rate. option_type: "call" | "put".
    """
    option_type = check_option_type(option_type, ("call", "put"))
    check_positive(spot=spot, strike=strike)
    if vol < 0.0:
        raise ValueError(f"vol must be >= 0, got {vol}")
    if tte <= 0.0:
        disc = strike * np.exp(-rate * tte)
        return float(max(0.0, (spot - disc) if option_type == "call" else (disc - spot)))

    sqrt_t = np.sqrt(tte)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * tte) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    if option_type == "call":
        return float(spot * norm.cdf(d1) - strike * np.exp(-rate * tte) * norm.cdf(d2))
    return float(strike * np.exp(-rate * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def black_scholes_greeks(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Closed-form Black-Scholes Greeks: delta, gamma, vega, theta, rho.

    vega is dV/dσ per unit volatility (not per 1%); theta is ∂V/∂t per year.
    Undefined at expiry or zero volatility.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    if tte <= 0.0:
        raise ValueError("Greeks are undefined at expiry (tte <= 0)")
    if vol <= 0.0:
        raise ValueError("Greeks are undefined at zero volatility")

    sqrt_t = np.sqrt(tte)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * tte) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    pdf = norm.pdf(d1)  # N'(d1)
    disc = np.exp(-rate * tte)
    sign = option_sign(option_type)

    delta = norm.cdf(d1) - (0.0 if option_type == "call" else 1.0)
    gamma = pdf / (spot * vol * sqrt_t)
    vega = spot * pdf * sqrt_t
    theta = -spot * pdf * vol / (2.0 * sqrt_t) - sign * rate * strike * disc * norm.cdf(sign * d2)
    rho = sign * strike * tte * disc * norm.cdf(sign * d2)
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }
