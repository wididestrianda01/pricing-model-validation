"""Hull-White Monte Carlo + Bermudan swaption LSM pricer."""

from __future__ import annotations

import numpy as np
import pytest

from pricing.anchor import BermudanSwaption, HullWhiteParams
from pricing.anchor.hw_mc import lsm_bermudan_swaption, simulate_hw_paths
from pricing.anchor.instrument import schedule_grid, zcb
from pricing.benchmarks.hull_white import hw_jamshidian_swaption

PARAMS = HullWhiteParams(r0=0.03, a=0.05, sigma=0.01)
TAUS = np.full(10, 0.5)
PAY_TIMES = 5.0 + np.arange(1, 11) * 0.5
STRIKE = 0.032
EURO_EX = np.array([5.0])
BERMUDAN_EX = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def _swaption(exercise_times: np.ndarray) -> BermudanSwaption:
    return BermudanSwaption(
        strike_rate=STRIKE,
        taus=TAUS,
        pay_times=PAY_TIMES,
        exercise_times=exercise_times,
    )


def test_curve_repricing() -> None:
    """Simulated discount factors reprice the flat initial curve."""
    times, _ = schedule_grid(BERMUDAN_EX, PAY_TIMES, 52)
    _, disc = simulate_hw_paths(PARAMS, times, n_paths=50_000, seed=3)
    for t in (1.0, 5.0, 8.0):
        node = round(t * 52)
        mc = disc[:, node].mean()
        closed = float(zcb(PARAMS, 0.0, t, PARAMS.r0))
        tol = 4 * disc[:, node].std() / np.sqrt(50_000)
        assert abs(mc - closed) < tol


def test_european_matches_jamshidian() -> None:
    jam = hw_jamshidian_swaption(
        expiry=5.0,
        tenors=[5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        strike_rate=STRIKE,
        a=PARAMS.a,
        sigma=PARAMS.sigma,
        r0=PARAMS.r0,
    )
    res = lsm_bermudan_swaption(_swaption(EURO_EX), PARAMS, n_paths=200_000, seed=7)
    assert res.price == pytest.approx(jam, rel=0.01)


def test_european_matches_quantlib() -> None:
    pytest.importorskip("QuantLib")
    from pricing.benchmarks.quantlib_benchmarks import ql_jamshidian_swaption

    ref = ql_jamshidian_swaption(
        expiry=5.0,
        tenors=[6.0, 7.0, 8.0, 9.0, 10.0],
        strike_rate=0.035,
        a=PARAMS.a,
        sigma=PARAMS.sigma,
        r0=PARAMS.r0,
    )
    annual = BermudanSwaption(
        strike_rate=0.035,
        taus=np.full(5, 1.0),
        pay_times=np.arange(6, 11) * 1.0,
        exercise_times=np.array([5.0]),
    )
    res = lsm_bermudan_swaption(annual, PARAMS, n_paths=200_000, seed=7)
    assert res.price == pytest.approx(ref, rel=0.01)


def test_bermudan_between_european_and_upper_bound() -> None:
    euro = lsm_bermudan_swaption(_swaption(EURO_EX), PARAMS, n_paths=200_000, seed=7).price
    berm = lsm_bermudan_swaption(_swaption(BERMUDAN_EX), PARAMS, n_paths=200_000, seed=7)
    upper = sum(
        hw_jamshidian_swaption(
            expiry=t,
            tenors=list(np.arange(t + 0.5, PAY_TIMES[-1] + 0.01, 0.5)),
            strike_rate=STRIKE,
            a=PARAMS.a,
            sigma=PARAMS.sigma,
            r0=PARAMS.r0,
        )
        for t in BERMUDAN_EX
    )
    assert berm.price >= euro - 4 * berm.std_error
    assert berm.price <= upper + 4 * berm.std_error


def test_deterministic_given_seed() -> None:
    s = _swaption(BERMUDAN_EX)
    a = lsm_bermudan_swaption(s, PARAMS, n_paths=10_000, seed=11).price
    b = lsm_bermudan_swaption(s, PARAMS, n_paths=10_000, seed=11).price
    assert a == b


def test_receiver_parity() -> None:
    """Payer - receiver equals the forward swap PV (put-call parity)."""
    payer = lsm_bermudan_swaption(
        _swaption(EURO_EX), PARAMS, n_paths=100_000, seed=5
    ).price
    s_rec = BermudanSwaption(
        strike_rate=STRIKE,
        taus=TAUS,
        pay_times=PAY_TIMES,
        exercise_times=EURO_EX,
        payer=False,
    )
    receiver = lsm_bermudan_swaption(s_rec, PARAMS, n_paths=100_000, seed=5).price
    forward_swap_pv = float(
        np.exp(-PARAMS.r0 * 5.0)
        - np.exp(-PARAMS.r0 * PAY_TIMES[-1])
        - STRIKE * np.sum(TAUS * np.exp(-PARAMS.r0 * PAY_TIMES))
    )
    assert payer - receiver == pytest.approx(forward_swap_pv, abs=4e-4)
