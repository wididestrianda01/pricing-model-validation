"""American exercise via projected SOR."""

from pricing.benchmarks import black_scholes
from pricing.benchmarks.quantlib_benchmarks import ql_american
from pricing.pde import fd_american

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 100.0, 1.0, 0.2, 0.05


def _psor() -> float:
    return fd_american(SPOT, STRIKE, TTE, VOL, RATE, "put", n_space=400, n_time=250)


def test_american_put_matches_quantlib_reference():
    ref = ql_american(SPOT, STRIKE, TTE, VOL, RATE, "put")
    assert abs(_psor() - ref) < 0.02


def test_early_exercise_premium_is_positive():
    euro = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "put")
    assert _psor() > euro


def test_american_price_respects_intrinsic_value():
    p = _psor()
    assert p >= max(STRIKE - SPOT, 0.0)


def test_deep_itm_american_put_tends_to_discounted_strike():
    # deep ITM: immediate exercise optimal -> price ~ K*e^{-r*tau} path from spot
    p = fd_american(60.0, 100.0, 1.0, 0.2, 0.05, "put", n_space=400, n_time=250)
    assert abs(p - (100.0 - 60.0)) < 1.5  # bounded above by strike; near intrinsic
    assert p <= 40.0 + 1e-9
