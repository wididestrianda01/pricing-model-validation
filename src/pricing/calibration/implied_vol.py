"""Black-Scholes implied volatility via bracketed root-finding."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from pricing.benchmarks.black_scholes import black_scholes


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    tte: float,
    rate: float,
    option_type: str = "call",
    dividend: float = 0.0,
) -> float:
    """Implied lognormal volatility of a European option price.

    Raises ValueError if the price violates static no-arbitrage bounds
    (below intrinsic, above super-replication) or no sign change exists
    within [1e-8, 10] — i.e. the quote is inconsistent with any BS price.
    """
    if tte <= 0.0:
        raise ValueError("tte must be > 0")
    disc_r = np.exp(-rate * tte)
    disc_q = np.exp(-dividend * tte)
    if option_type == "call":
        lower = max(0.0, spot * disc_q - strike * disc_r)
        upper = spot * disc_q
    elif option_type == "put":
        lower = max(0.0, strike * disc_r - spot * disc_q)
        upper = strike * disc_r
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    if not lower <= price <= upper:
        raise ValueError(
            f"price {price:.6g} outside no-arbitrage bounds [{lower:.6g}, {upper:.6g}]"
        )

    def objective(vol: float) -> float:
        return black_scholes(spot, strike, tte, vol, rate, option_type) - price

    lo, hi = 1e-8, 10.0
    try:
        return float(brentq(objective, lo, hi, xtol=1e-12, rtol=1e-14))
    except ValueError as exc:
        raise ValueError(
            f"no implied vol in [{lo}, {hi}] for price {price:.6g}"
        ) from exc
