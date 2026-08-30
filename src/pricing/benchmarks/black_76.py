"""Closed-form Black-76 European option on a forward (caplet/swaption)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pricing.benchmarks._validate import check_option_type, check_positive


def black_76(
    forward: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
) -> float:
    """Price a European call/put on a forward (Black model, 1976).

    forward: the forward price (or forward rate) at expiry. rate: the
    discounting rate from today to expiry. option_type: "call" | "put".
    """
    option_type = check_option_type(option_type, ("call", "put"))
    check_positive(forward=forward, strike=strike)
    if vol < 0.0:
        raise ValueError(f"vol must be >= 0, got {vol}")
    if tte <= 0.0:
        disc = np.exp(-rate * tte)
        return float(max(0.0, disc * (forward - strike) if option_type == "call" else disc * (strike - forward)))
    if vol <= 0.0:
        disc = np.exp(-rate * tte)
        return float(max(0.0, disc * (forward - strike) if option_type == "call" else disc * (strike - forward)))

    sqrt_t = np.sqrt(tte)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * tte) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = np.exp(-rate * tte)

    if option_type == "call":
        return float(disc * (forward * norm.cdf(d1) - strike * norm.cdf(d2)))
    return float(disc * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1)))
