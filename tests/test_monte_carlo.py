"""Monte Carlo engine spine: seam, determinism, convergence to the benchmark."""

import numpy as np
import pytest

from pricing.benchmarks import black_scholes
from pricing.benchmarks.quantlib_benchmarks import ql_black_scholes
from pricing.monte_carlo import mc_european

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 0.5, 0.2, 0.03


def test_exact_scheme_matches_black_scholes_within_se():
    bs = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    r = mc_european(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2_000_000, n_steps=1, seed=0, scheme="exact")
    assert abs(r.price - bs) < 4 * r.std_error


def test_exact_scheme_matches_quantlib_within_se():
    # QuantLib is the independent challenger benchmark (ADR-0001); the engine must
    # agree with it directly, not only transitively via the closed form.
    # Whole-day expiry: QuantLib dates resolve whole Act/365 days only.
    tte = 183 / 365
    ref = ql_black_scholes(SPOT, STRIKE, tte, VOL, RATE, "call")
    r = mc_european(SPOT, STRIKE, tte, VOL, RATE, "call", n_paths=2_000_000, n_steps=1, seed=0, scheme="exact")
    assert abs(r.price - ref) < 4 * r.std_error


def test_euler_and_milstein_approach_black_scholes():
    bs = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    for scheme in ("euler", "milstein"):
        # Bias is O(dt) and oscillates around the strike (kink); a loose bound
        # separates a working scheme from a broken one. Weak order is tested
        # rigorously in test_convergence.py.
        r = mc_european(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=1_000_000, n_steps=256, seed=0, scheme=scheme)
        assert abs(r.price - bs) < 1e-2


def test_seed_determinism():
    a = mc_european(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=10_000, n_steps=4, seed=42, scheme="euler")
    b = mc_european(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=10_000, n_steps=4, seed=42, scheme="euler")
    assert a.price == b.price


def test_put_call_parity_under_mc():
    parity = SPOT - STRIKE * np.exp(-RATE * TTE)
    call = mc_european(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2_000_000, n_steps=1, seed=1, scheme="exact")
    put = mc_european(SPOT, STRIKE, TTE, VOL, RATE, "put", n_paths=2_000_000, n_steps=1, seed=1, scheme="exact")
    assert abs((call.price - put.price) - parity) < 4 * (call.std_error + put.std_error)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        mc_european(SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="bogus")


def test_unknown_option_type_raises():
    with pytest.raises(ValueError):
        mc_european(SPOT, STRIKE, TTE, VOL, RATE, "straddle", scheme="exact")



def test_mc_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        mc_european(0.0, STRIKE, TTE, VOL, RATE, "call")
    with pytest.raises(ValueError):
        mc_european(SPOT, STRIKE, TTE, -0.2, RATE, "call")
