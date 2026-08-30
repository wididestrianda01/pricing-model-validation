"""Uniform log-space grid for the finite-difference engine.

The Black-Scholes PDE has constant coefficients in x = ln(S):
    dV/dt = 0.5*vol^2 * V_xx + (rate - vol^2/2) * V_x - rate * V
so the grid lives in log-spot and every node sees the same stencil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Uniform log-spot grid: n_space interior nodes plus two boundary nodes."""

    x: np.ndarray  # all nodes, length n_space + 2
    dx: float


def build_grid(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    n_space: int,
    barrier: float | None = None,
    width_sigmas: float = 5.0,
) -> Grid:
    """Log-grid wide enough that truncation error is negligible, aligned so a
    barrier lands exactly on a node (absorbing boundary must sit on the grid).

    n_space: number of interior intervals; the returned x has n_space + 2 nodes.
    """
    lo = min(spot, strike)
    hi = max(spot, strike)
    if barrier is not None:
        lo = min(lo, barrier)
        hi = max(hi, barrier)
    pad = width_sigmas * vol * np.sqrt(tte) + 1.0
    x_lo = np.log(lo) - pad
    x_hi = np.log(hi) + pad
    dx = (x_hi - x_lo) / n_space
    if barrier is not None:
        # shift the window so the barrier falls on a node
        n_below = int(np.floor((np.log(barrier) - x_lo) / dx))
        x_lo = np.log(barrier) - n_below * dx
    x = x_lo + dx * np.arange(n_space + 2)
    return Grid(x=x, dx=float(dx))


def payoff_initial(
    grid: Grid, strike: float, payoff: str, option_type: str
) -> np.ndarray:
    s = np.exp(grid.x)
    if payoff == "vanilla":
        return (
            np.maximum(s - strike, 0.0)
            if option_type == "call"
            else np.maximum(strike - s, 0.0)
        )
    if payoff == "digital":
        return (
            (s > strike).astype(float)
            if option_type == "call"
            else (s < strike).astype(float)
        )
    raise ValueError(f"payoff must be 'vanilla' or 'digital', got {payoff!r}")


def boundary_value(
    x_edge: float,
    strike: float,
    rate: float,
    tau: float,
    edge: str,
    option_type: str,
    payoff: str = "vanilla",
) -> float:
    """Dirichlet value at a domain edge after tau years past expiry.

    Vanilla: deep-OTM edge -> 0, deep-ITM edge -> discounted intrinsic.
    Digital: the deep-ITM side pays the discounted unit.
    """
    disc = np.exp(-rate * tau)
    itm = (edge == "hi") if option_type == "call" else (edge == "lo")
    if not itm:
        return 0.0
    if payoff == "digital":
        return float(disc)
    intrinsic = (
        max(0.0, np.exp(x_edge) - strike)
        if option_type == "call"
        else max(0.0, strike - np.exp(x_edge))
    )
    return float(intrinsic) * disc
