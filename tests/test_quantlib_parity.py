"""QuantLib parity: the from-scratch benchmarks must agree with QuantLib."""

import pytest

from pricing.benchmarks import (
    black_76,
    black_scholes,
    hw_caplet,
    hw_jamshidian_swaption,
    hw_zcb_option,
)
from pricing.benchmarks.quantlib_benchmarks import (
    ql_black_76,
    ql_black_scholes,
    ql_hw_caplet,
    ql_hw_zcb_option,
    ql_jamshidian_swaption,
)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black_scholes_matches_quantlib(option_type):
    spot, strike, tte, vol, rate = 100.0, 105.0, 1.0, 0.2, 0.05
    mine = black_scholes(spot, strike, tte, vol, rate, option_type)
    ref = ql_black_scholes(spot, strike, tte, vol, rate, option_type)
    assert mine == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black_76_matches_quantlib(option_type):
    forward, strike, tte, vol, rate = 100.0, 95.0, 1.0, 0.3, 0.02
    mine = black_76(forward, strike, tte, vol, rate, option_type)
    ref = ql_black_76(forward, strike, tte, vol, rate, option_type)
    assert mine == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_hw_zcb_option_matches_quantlib(option_type):
    expiry, maturity, a, sigma, r0, strike = 1.0, 3.0, 0.1, 0.01, 0.03, 0.95
    mine = hw_zcb_option(0.0, expiry, maturity, r0, a, sigma, r0, strike, option_type)
    ref = ql_hw_zcb_option(expiry, maturity, a, sigma, r0, strike, option_type)
    assert mine == pytest.approx(ref, abs=1e-12)


def test_hw_caplet_matches_quantlib():
    expiry, maturity, a, sigma, r0, strike_rate = 1.0, 1.5, 0.1, 0.01, 0.03, 0.04
    mine = hw_caplet(0.0, expiry, maturity, r0, a, sigma, r0, strike_rate)
    ref = ql_hw_caplet(expiry, maturity, a, sigma, r0, strike_rate)
    assert mine == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("option_type", ["payer", "receiver"])
def test_jamshidian_swaption_matches_quantlib(option_type):
    expiry, tenors, strike_rate, a, sigma, r0 = 1.0, [2.0, 3.0, 4.0, 5.0, 6.0], 0.03, 0.1, 0.01, 0.02
    mine = hw_jamshidian_swaption(expiry, tenors, strike_rate, a, sigma, r0, option_type)
    ref = ql_jamshidian_swaption(expiry, tenors, strike_rate, a, sigma, r0, option_type)
    # QuantLib uses actual/365 day counts (leap years) while the closed form
    # uses exact-year time; residual is ~5e-6 relative.
    assert mine == pytest.approx(ref, rel=1e-4)



def test_black_scholes_matches_quantlib_at_non_whole_year_expiry():
    # The seam must be exact for any whole-day maturity, not just whole years.
    tte = 91 / 365
    mine = black_scholes(100.0, 105.0, tte, 0.2, 0.05, "call")
    ref = ql_black_scholes(100.0, 105.0, tte, 0.2, 0.05, "call")
    assert mine == pytest.approx(ref, abs=1e-11)


def test_ql_bs_wrapper_rejects_unrepresentable_maturity():
    # Rounding used to silently price a shifted maturity (~0.1% off on 3M ATM).
    with pytest.raises(ValueError):
        ql_black_scholes(100.0, 105.0, 0.25, 0.2, 0.05)


def test_ql_swaption_wrapper_rejects_bad_schedule():
    with pytest.raises(ValueError):  # tenor not a whole day count
        ql_jamshidian_swaption(1.0, [2.0, 3.7], 0.03, 0.1, 0.01, 0.02)
    with pytest.raises(ValueError):  # tenor spacing not annual
        ql_jamshidian_swaption(1.0, [2.5, 3.5], 0.03, 0.1, 0.01, 0.02)
