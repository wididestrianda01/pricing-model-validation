"""Pathwise (payoff-derivative) Greeks from exact-GBM simulated paths."""

from __future__ import annotations

import numpy as np

from pricing.benchmarks._validate import check_option_type
from pricing.monte_carlo import simulate_terminal, standard_normals


def pathwise_greeks(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 1,
    seed: int = 0,
) -> dict[str, float]:
    """Pathwise delta and vega for a European call/put under exact GBM.

    Differentiates the payoff along each simulated path. Valid where the payoff
    is Lipschitz (smooth call/put). Gamma is omitted: the second derivative of a
    vanilla payoff is distributional, so gamma is delivered by the
    likelihood-ratio and bump estimators instead.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    z = standard_normals(n_paths, n_steps, seed)
    s_t = simulate_terminal(spot, tte, vol, rate, z, "exact")
    w_t = np.sqrt(tte / n_steps) * z.sum(axis=1)  # Brownian terminal, same z as s_t
    disc = np.exp(-rate * tte)
    sign = 1.0 if option_type == "call" else -1.0
    itm = (s_t > strike) if option_type == "call" else (s_t < strike)

    d_spot = s_t / spot                    # dS_T / dS_0
    d_vol = s_t * (w_t - vol * tte)        # dS_T / dσ

    delta = disc * np.mean(sign * itm * d_spot)
    vega = disc * np.mean(sign * itm * d_vol)
    return {"delta": float(delta), "vega": float(vega)}
