"""FRTB PLA consistency for the Bermudan swaption hedge leg.

Builds risk-theoretical P&L (RTPL) from first-order Greeks — curve delta
(bump-and-reprice on the flat-curve level) and vega (short-rate volatility) —
over a deterministic scenario grid, and compares it against hypothetical P&L
(HPL) from full revaluation. The FRTB PLA consistency bar is Spearman rank
correlation > 0.80 and Kolmogorov-Smirnov distance < 0.09 between the two P&L
distributions (CRR3/FRTB Reg 2024/1623; see research note 03).

The default pricing engine for bumps is the PDE pricer (deterministic,
fast); passing ``engine="lsm"`` recomputes everything by Monte Carlo so the
two routes can be cross-checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from pricing.anchor import BermudanSwaption, HullWhiteParams


def _price(
    swaption: BermudanSwaption,
    params: HullWhiteParams,
    engine: str,
    n_paths: int,
) -> float:
    if engine == "pde":
        from pricing.anchor.hw_pde import pde_bermudan_swaption

        return pde_bermudan_swaption(swaption, params)
    if engine == "lsm":
        from pricing.anchor.hw_mc import lsm_bermudan_swaption

        return float(lsm_bermudan_swaption(swaption, params, n_paths=n_paths).price)
    raise ValueError(f"engine must be 'pde' or 'lsm', got {engine!r}")


def swaption_greeks(
    swaption: BermudanSwaption,
    params: HullWhiteParams,
    *,
    engine: str = "pde",
    bump_rate: float = 1e-4,
    bump_vol: float = 1e-4,
    n_paths: int = 200_000,
) -> dict[str, float]:
    """Central-difference delta (per unit rate shift) and vega (per unit vol)."""
    up_r = HullWhiteParams(params.r0 + bump_rate, params.a, params.sigma)
    dn_r = HullWhiteParams(params.r0 - bump_rate, params.a, params.sigma)
    up_v = HullWhiteParams(params.r0, params.a, params.sigma + bump_vol)
    dn_v = HullWhiteParams(params.r0, params.a, max(params.sigma - bump_vol, 1e-6))
    delta = (_price(swaption, up_r, engine, n_paths) - _price(swaption, dn_r, engine, n_paths)) / (
        2.0 * bump_rate
    )
    vega = (_price(swaption, up_v, engine, n_paths) - _price(swaption, dn_v, engine, n_paths)) / (
        2.0 * bump_vol
    )
    return {"delta": delta, "vega": vega}


@dataclass(frozen=True)
class PLAResult:
    """PLA consistency outcome over the scenario grid."""

    spearman: float
    ks_statistic: float
    ks_pvalue: float
    hpl: np.ndarray
    rtpl: np.ndarray


def pla_consistency(
    swaption: BermudanSwaption,
    params: HullWhiteParams,
    *,
    engine: str = "pde",
    rate_shifts: np.ndarray | None = None,
    vol_shifts: np.ndarray | None = None,
    bump_rate: float = 1e-4,
    bump_vol: float = 1e-4,
    n_paths: int = 200_000,
) -> PLAResult:
    """Run the RTPL-vs-HPL comparison over the crossed scenario grid.

    Defaults: ten symmetric parallel curve shifts (±10bp .. ±50bp, one-day
    scale) crossed with four vol shifts (±2.5bp, ±5bp).
    """
    if rate_shifts is None:
        rate_shifts = np.array(
            [-0.005, -0.004, -0.003, -0.002, -0.001, 0.001, 0.002, 0.003, 0.004, 0.005]
        )
    if vol_shifts is None:
        vol_shifts = np.array([-0.0005, -0.00025, 0.00025, 0.0005])
    greeks = swaption_greeks(
        swaption, params, engine=engine, bump_rate=bump_rate, bump_vol=bump_vol, n_paths=n_paths
    )
    base = _price(swaption, params, engine, n_paths)

    hpl_rows: list[float] = []
    rtpl_rows: list[float] = []
    for ds in rate_shifts:
        for dv in vol_shifts:
            bumped = HullWhiteParams(params.r0 + ds, params.a, params.sigma + dv)
            hpl_rows.append(_price(swaption, bumped, engine, n_paths) - base)
            rtpl_rows.append(greeks["delta"] * ds + greeks["vega"] * dv)

    hpl = np.array(hpl_rows)
    rtpl = np.array(rtpl_rows)
    rho, _ = stats.spearmanr(rtpl, hpl)
    ks = stats.ks_2samp(rtpl, hpl)
    return PLAResult(
        spearman=float(rho),
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        hpl=hpl,
        rtpl=rtpl,
    )
