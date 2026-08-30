"""Tests for Heston calibration and the MC cross-check."""

import numpy as np
import pytest

from pricing.benchmarks.heston import heston
from pricing.calibration.heston_fit import calibrate_heston, heston_mc_price

SPOT = 4500.0
RATE = 0.045
TRUE = {"v0": 0.035, "kappa": 2.0, "theta": 0.05, "xi": 0.55, "rho": -0.6}


def _surface(true=TRUE, maturities=(0.25, 0.5), moneyness=(0.85, 0.95, 1.0, 1.05, 1.15)):
    strikes, ttes, prices = [], [], []
    for tte in maturities:
        for m in moneyness:
            strike = SPOT * m
            strikes.append(strike)
            ttes.append(tte)
            prices.append(heston(SPOT, strike, tte, true["v0"], true["kappa"],
                                 true["theta"], true["xi"], true["rho"], RATE))
    return np.array(strikes), np.array(ttes), np.array(prices)


class TestCalibrateHeston:
    def test_recovers_ground_truth(self):
        strikes, ttes, prices = _surface()
        fit = calibrate_heston(SPOT, strikes, ttes, prices, RATE)
        assert fit.converged
        assert fit.v0 == pytest.approx(TRUE["v0"], abs=5e-3)
        assert fit.kappa == pytest.approx(TRUE["kappa"], rel=0.1)
        assert fit.theta == pytest.approx(TRUE["theta"], abs=5e-3)
        assert fit.xi == pytest.approx(TRUE["xi"], rel=0.15)
        assert fit.rho == pytest.approx(TRUE["rho"], abs=0.05)
        assert fit.price_rmse < 1e-6
        assert fit.iv_rmse < 1e-6

    def test_fit_is_deterministic(self):
        strikes, ttes, prices = _surface()
        a = calibrate_heston(SPOT, strikes, ttes, prices, RATE)
        b = calibrate_heston(SPOT, strikes, ttes, prices, RATE)
        assert a == b

    def test_feller_flag_reports_true_state(self):
        # TRUE params violate Feller (2*kappa*theta - xi^2 < 0); flag must say so.
        strikes, ttes, prices = _surface()
        fit = calibrate_heston(SPOT, strikes, ttes, prices, RATE)
        assert fit.feller_violation == pytest.approx(
            2 * TRUE["kappa"] * TRUE["theta"] - TRUE["xi"] ** 2, abs=1e-2
        )

    def test_too_few_quotes_rejected(self):
        strikes, ttes, prices = _surface()
        with pytest.raises(ValueError, match="5 quotes"):
            calibrate_heston(SPOT, strikes[:4], ttes[:4], prices[:4], RATE)

    def test_shape_mismatch_rejected(self):
        strikes, ttes, prices = _surface()
        with pytest.raises(ValueError, match="same-length"):
            calibrate_heston(SPOT, strikes[:-1], ttes, prices, RATE)


class TestHestonMcCrossCheck:
    def test_mc_agrees_with_closed_form_within_error(self):
        strike = SPOT * 1.02
        mc, stderr = heston_mc_price(SPOT, strike, 0.5, n_paths=100_000,
                                     **TRUE, rate=RATE)
        closed = heston(SPOT, strike, 0.5, TRUE["v0"], TRUE["kappa"], TRUE["theta"],
                        TRUE["xi"], TRUE["rho"], RATE)
        assert mc == pytest.approx(closed, abs=4.0 * stderr)

    def test_put_parity_holds_in_mc(self):
        # C - P = DF(F - K) must hold for MC call/put pairs of the same paths.
        strike = SPOT
        kw = {"n_paths": 100_000, "seed": 11}
        call, _ = heston_mc_price(SPOT, strike, 0.5, option_type="call", **TRUE, rate=RATE, **kw)
        put, _ = heston_mc_price(SPOT, strike, 0.5, option_type="put", **TRUE, rate=RATE, **kw)
        forward = SPOT * np.exp(RATE * 0.5)
        assert call - put == pytest.approx(np.exp(-RATE * 0.5) * (forward - strike), rel=5e-3)

    def test_mc_is_seeded_reproducible(self):
        a = heston_mc_price(SPOT, SPOT, 0.25, n_paths=20_000, **TRUE, rate=RATE)
        b = heston_mc_price(SPOT, SPOT, 0.25, n_paths=20_000, **TRUE, rate=RATE)
        assert a == b
