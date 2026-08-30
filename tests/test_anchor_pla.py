"""Greeks and FRTB PLA consistency for the hedge leg."""

from __future__ import annotations

import numpy as np
import pytest

from pricing.anchor import BermudanSwaption, HullWhiteParams
from pricing.anchor.pla import pla_consistency, swaption_greeks

PARAMS = HullWhiteParams(r0=0.03, a=0.05, sigma=0.01)
SWAPTION = BermudanSwaption(
    strike_rate=0.032,
    taus=np.full(10, 0.5),
    pay_times=5.0 + np.arange(1, 11) * 0.5,
    exercise_times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
)


def test_payer_greek_signs() -> None:
    greeks = swaption_greeks(SWAPTION, PARAMS)
    assert greeks["delta"] > 0  # payer gains when rates rise
    assert greeks["vega"] > 0  # more vol -> more option value


def test_engines_agree_on_greeks() -> None:
    pde = swaption_greeks(SWAPTION, PARAMS, engine="pde")
    lsm = swaption_greeks(SWAPTION, PARAMS, engine="lsm", n_paths=400_000)
    assert pde["delta"] == pytest.approx(lsm["delta"], rel=0.05)
    assert pde["vega"] == pytest.approx(lsm["vega"], rel=0.15)


def test_pla_passes_frtb_bar() -> None:
    res = pla_consistency(SWAPTION, PARAMS)
    assert res.spearman > 0.80
    assert res.ks_statistic < 0.09
    assert len(res.hpl) == len(res.rtpl) == 40  # 10 rate x 4 vol shifts


def test_rtpl_linear_in_shifts() -> None:
    """RTPL is exactly delta*ds + vega*dv by construction."""
    res = pla_consistency(SWAPTION, PARAMS)
    greeks = swaption_greeks(SWAPTION, PARAMS)
    rate_shifts = [-0.005, -0.004, -0.003, -0.002, -0.001, 0.001, 0.002, 0.003, 0.004, 0.005]
    ds = np.array([s for s in rate_shifts for _ in range(4)])
    dv = np.tile([-0.0005, -0.00025, 0.00025, 0.0005], 10)
    expected = greeks["delta"] * ds + greeks["vega"] * dv
    np.testing.assert_allclose(res.rtpl, expected)


def test_reproducible() -> None:
    a = pla_consistency(SWAPTION, PARAMS)
    b = pla_consistency(SWAPTION, PARAMS)
    assert a.spearman == b.spearman
    assert np.array_equal(a.hpl, b.hpl)
