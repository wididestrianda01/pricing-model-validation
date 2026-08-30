"""Bermudan swaption instrument spec and Hull-White market data.

Shared seam for the anchor's two pricing engines (LSM Monte Carlo and
finite-difference PDE). The model is one-factor Hull-White on a flat initial
zero curve at continuously compounded ``r0`` (matching the closed-form
benchmarks in :mod:`pricing.benchmarks.hull_white`), so Jamshidian and
QuantLib prices are directly comparable.
"""

from __future__ import annotations

import numpy as np

from pricing.benchmarks._validate import check_positive


class HullWhiteParams:
    """One-factor Hull-White parameters on a flat initial curve.

    Attributes:
        r0: flat initial short rate (continuously compounded).
        a: mean-reversion speed (> 0).
        sigma: short-rate volatility (> 0).
    """

    __slots__ = ("a", "r0", "sigma")

    def __init__(self, r0: float, a: float, sigma: float) -> None:
        check_positive(r0=r0)
        if a <= 0.0:
            raise ValueError("a must be > 0")
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")
        self.r0 = r0
        self.a = a
        self.sigma = sigma


class BermudanSwaption:
    """Payer/receiver Bermudan swaption on a fixed-vs-floating swap schedule.

    Attributes:
        strike_rate: fixed rate K of the underlying swap.
        payer: True for a payer swaption (exercise into paying K, receiving float).
        taus: accrual fractions of each swap period, shape (m,), all > 0.
        pay_times: payment times of each period (years from today), shape (m,),
            strictly increasing; the swap starts at pay_times[0] - taus[0].
        exercise_times: Bermudan exercise dates (years from today), shape (n,),
            strictly increasing; exercise_times[-1] must equal that swap start.
    """

    __slots__ = ("exercise_times", "pay_times", "payer", "strike_rate", "taus")

    def __init__(
        self,
        strike_rate: float,
        taus: np.ndarray,
        pay_times: np.ndarray,
        exercise_times: np.ndarray,
        *,
        payer: bool = True,
    ) -> None:
        check_positive(strike_rate=strike_rate)
        taus = np.asarray(taus, dtype=float)
        pay_times = np.asarray(pay_times, dtype=float)
        exercise_times = np.asarray(exercise_times, dtype=float)
        if taus.ndim != 1 or taus.size == 0 or not np.all(taus > 0):
            raise ValueError("taus must be a non-empty positive vector")
        if pay_times.shape != taus.shape or not np.all(np.diff(pay_times) > 0):
            raise ValueError("pay_times must be strictly increasing, same length as taus")
        if exercise_times.ndim != 1 or exercise_times.size == 0 or not np.all(
            np.diff(exercise_times) > 0
        ):
            raise ValueError("exercise_times must be strictly increasing and non-empty")
        if not np.isclose(exercise_times[-1], pay_times[0] - taus[0]):
            raise ValueError("last exercise date must equal the swap start")
        self.strike_rate = strike_rate
        self.payer = payer
        self.taus = taus
        self.pay_times = pay_times
        self.exercise_times = exercise_times



def zcb(params: HullWhiteParams, t: float, T: float, r):
    """Model zero-coupon bond price P(t,T) given short rate(s) ``r`` at time t."""
    b = (1.0 - np.exp(-params.a * (T - t))) / params.a
    a_disc = np.exp(
        -params.r0 * (T - t)
        + b * params.r0
        - (params.sigma**2 / (4.0 * params.a))
        * (1.0 - np.exp(-2.0 * params.a * t))
        * b**2
    )
    return a_disc * np.exp(-b * np.asarray(r, dtype=float))



def swap_pv(swaption: BermudanSwaption, params: HullWhiteParams, t: float, r):
    """Model price at time t of the underlying swap given short rate(s) ``r``.

    Positive when exercising is favourable for the holder's option side.
    """
    r_arr = np.atleast_1d(np.asarray(r, dtype=float))
    bonds = np.stack(
        [np.broadcast_to(zcb(params, t, float(T), r_arr), r_arr.shape) for T in swaption.pay_times]
    )
    annuity = np.tensordot(swaption.taus, bonds, axes=(0, 0))
    value = 1.0 - bonds[-1] - swaption.strike_rate * annuity
    if not swaption.payer:
        value = -value
    return value if np.ndim(r) else value[0]


def schedule_grid(
    exercise_times: np.ndarray, pay_times: np.ndarray, steps_per_year: int
) -> tuple[np.ndarray, dict[float, int]]:
    """Uniform-per-segment time grid hitting every exercise/payment date.

    Walks the distinct key dates (exercise ∪ payment), giving each segment
    ``ceil(span * steps_per_year)`` equal substeps. Returns ``(times,
    key_nodes)``: every grid point strictly after 0, and the node index of
    each key date. Shared by the MC and PDE engines so both walk identical
    time grids.
    """
    keys = np.unique(np.concatenate((exercise_times, pay_times)))
    pieces: list[np.ndarray] = []
    key_nodes: dict[float, int] = {}
    prev = 0.0
    total = 0
    for key in keys:
        span = float(key) - prev
        n = max(1, int(np.ceil(span * steps_per_year)))
        pieces.append(prev + np.arange(1, n + 1) * (span / n))
        total += n
        key_nodes[float(key)] = total - 1
        prev = float(key)
    return np.concatenate(pieces), key_nodes
