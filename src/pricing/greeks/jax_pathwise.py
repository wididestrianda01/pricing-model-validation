"""Pathwise Greeks via JAX automatic differentiation (ADR 0001: JAX only here)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pricing.benchmarks._validate import check_option_type


def jax_pathwise_greeks(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 0,
) -> dict[str, float]:
    """Pathwise delta and vega by JAX automatic differentiation.

    The price is a JAX function of spot and volatility over one Monte Carlo
    batch; `jax.grad` differentiates through the exact-GBM terminal and the
    payoff, recovering the pathwise delta and vega without hand-derived
    derivatives. Gamma is omitted — the second derivative of a vanilla payoff
    is distributional, so autodiff through `max` yields zero there, matching
    the manual pathwise scope.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    z = jax.random.normal(jax.random.PRNGKey(seed), (n_paths,))

    def price(s0: float, sigma: float) -> float:
        s_t = s0 * jnp.exp((rate - 0.5 * sigma**2) * tte + sigma * jnp.sqrt(tte) * z)
        if option_type == "call":
            payoff = jnp.maximum(s_t - strike, 0.0)
        else:
            payoff = jnp.maximum(strike - s_t, 0.0)
        return jnp.exp(-rate * tte) * jnp.mean(payoff)

    delta = float(jax.grad(price, argnums=0)(spot, vol))
    vega = float(jax.grad(price, argnums=1)(spot, vol))
    return {"delta": delta, "vega": vega}
