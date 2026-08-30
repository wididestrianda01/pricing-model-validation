"""Finite-difference PDE pricer for the Bermudan swaption."""

from __future__ import annotations

import numpy as np
import pytest

from pricing.anchor import BermudanSwaption, HullWhiteParams
from pricing.anchor.hw_mc import lsm_bermudan_swaption
from pricing.anchor.hw_pde import pde_bermudan_swaption
from pricing.benchmarks.hull_white import hw_jamshidian_swaption

PARAMS = HullWhiteParams(r0=0.03, a=0.05, sigma=0.01)
STRIKE = 0.032
TAUS = np.full(10, 0.5)
PAY_TIMES = 5.0 + np.arange(1, 11) * 0.5
EURO_EX = np.array([5.0])
BERMUDAN_EX = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def _swaption(exercise_times: np.ndarray) -> BermudanSwaption:
    return BermudanSwaption(
        strike_rate=STRIKE,
        taus=TAUS,
        pay_times=PAY_TIMES,
        exercise_times=exercise_times,
    )


def test_european_converges_to_jamshidian() -> None:
    jam = hw_jamshidian_swaption(
        expiry=5.0,
        tenors=list(np.arange(5.5, 10.01, 0.5)),
        strike_rate=STRIKE,
        a=PARAMS.a,
        sigma=PARAMS.sigma,
        r0=PARAMS.r0,
    )
    prices = [
        pde_bermudan_swaption(_swaption(EURO_EX), PARAMS, n_space=n)
        for n in (101, 201, 401)
    ]
    # successive refinements close the gap monotonically
    gaps = [abs(p - jam) for p in prices]
    assert gaps[1] < gaps[0]
    assert gaps[2] < gaps[1]
    assert prices[-1] == pytest.approx(jam, rel=2e-3)


def test_bermudan_agrees_with_lsm() -> None:
    pde = pde_bermudan_swaption(_swaption(BERMUDAN_EX), PARAMS)
    lsm = lsm_bermudan_swaption(_swaption(BERMUDAN_EX), PARAMS, n_paths=400_000, seed=7)
    assert pde >= lsm.price - 4 * lsm.std_error  # LSM low bias keeps it below
    assert pde == pytest.approx(lsm.price, rel=0.01)


def test_bermudan_below_jamshidian_upper_bound() -> None:
    upper = sum(
        hw_jamshidian_swaption(
            expiry=float(t),
            tenors=list(np.arange(t + 0.5, PAY_TIMES[-1] + 0.01, 0.5)),
            strike_rate=STRIKE,
            a=PARAMS.a,
            sigma=PARAMS.sigma,
            r0=PARAMS.r0,
        )
        for t in BERMUDAN_EX
    )
    pde = pde_bermudan_swaption(_swaption(BERMUDAN_EX), PARAMS)
    assert pde < upper


def test_receiver_symmetry() -> None:
    """Receiver PDE price equals payer flipped through forward-swap parity."""
    payer = pde_bermudan_swaption(_swaption(EURO_EX), PARAMS)
    receiver = pde_bermudan_swaption(
        BermudanSwaption(
            strike_rate=STRIKE,
            taus=TAUS,
            pay_times=PAY_TIMES,
            exercise_times=EURO_EX,
            payer=False,
        ),
        PARAMS,
    )
    forward_swap_pv = float(
        np.exp(-PARAMS.r0 * 5.0)
        - np.exp(-PARAMS.r0 * PAY_TIMES[-1])
        - STRIKE * np.sum(TAUS * np.exp(-PARAMS.r0 * PAY_TIMES))
    )
    assert payer - receiver == pytest.approx(forward_swap_pv, abs=5e-4)


def test_deterministic() -> None:
    s = _swaption(BERMUDAN_EX)
    assert pde_bermudan_swaption(s, PARAMS) == pde_bermudan_swaption(s, PARAMS)
