"""Finite-difference (bump-and-reprice) Greeks via the Monte Carlo pricing seam."""

from __future__ import annotations

from pricing.benchmarks._validate import check_option_type, check_positive
from pricing.monte_carlo import mc_european

_DIFFERENCES = ("central", "forward")


def bump_greeks(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 1,
    seed: int = 0,
    scheme: str = "exact",
    h_spot: float | None = None,
    h_vol: float | None = None,
    h_rate: float | None = None,
    h_tte: float | None = None,
    common_random: bool = True,
    difference: str = "central",
) -> dict[str, float]:
    """Delta, gamma, vega, theta, and rho by bump-and-reprice through the MC seam.

    `difference="central"` (default) uses symmetric bumps (bias ~ h^2) and
    returns all five Greeks; `difference="forward"` uses one-sided bumps
    (bias ~ h) and returns delta and vega only. With `common_random` (default)
    all reprices share one seed so the difference cancels most Monte Carlo
    noise; set it False to reprice each bump at an independent seed — the
    textbook setup where the estimator variance scales as 1/h^2 (used by the
    bias-variance study).
    Default bumps: h_spot = 1% of spot, h_vol = 1 vol point, h_rate = 1 bp,
    h_tte = 1 day (1/365). theta = -dV/dtte and rho = dV/dr, per year.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    if difference not in _DIFFERENCES:
        raise ValueError(f"difference must be one of {_DIFFERENCES}, got {difference!r}")
    h_s = spot * 0.01 if h_spot is None else h_spot
    h_v = 0.01 if h_vol is None else h_vol
    h_r = 1e-4 if h_rate is None else h_rate
    h_t = 1.0 / 365.0 if h_tte is None else h_tte
    check_positive(h_spot=h_s, h_vol=h_v, h_rate=h_r, h_tte=h_t)

    def price(s: float, v: float, r: float, t: float, seed_offset: int) -> float:
        return mc_european(s, strike, t, v, r, option_type, n_paths, n_steps, seed + seed_offset, scheme).price

    seed_stride = 0 if common_random else 1
    base = price(spot, vol, rate, tte, 0)
    up_s = price(spot + h_s, vol, rate, tte, 1 * seed_stride)

    if difference == "central":
        dn_s = price(spot - h_s, vol, rate, tte, 2 * seed_stride)
        up_v = price(spot, vol + h_v, rate, tte, 3 * seed_stride)
        dn_v = price(spot, vol - h_v, rate, tte, 4 * seed_stride)
        up_r = price(spot, vol, rate + h_r, tte, 5 * seed_stride)
        dn_r = price(spot, vol, rate - h_r, tte, 6 * seed_stride)
        up_t = price(spot, vol, rate, tte + h_t, 7 * seed_stride)
        dn_t = price(spot, vol, rate, tte - h_t, 8 * seed_stride)
        return {
            "delta": (up_s - dn_s) / (2.0 * h_s),
            "gamma": (up_s - 2.0 * base + dn_s) / (h_s * h_s),
            "vega": (up_v - dn_v) / (2.0 * h_v),
            "rho": (up_r - dn_r) / (2.0 * h_r),
            "theta": -(up_t - dn_t) / (2.0 * h_t),
        }

    up_v = price(spot, vol + h_v, rate, tte, 3 * seed_stride)
    return {
        "delta": (up_s - base) / h_s,
        "vega": (up_v - base) / h_v,
    }
