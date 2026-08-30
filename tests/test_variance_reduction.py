"""Variance reduction: antithetic, control variate, importance sampling."""

import pytest

from pricing.benchmarks import black_scholes
from pricing.monte_carlo import (
    mc_european,
    mc_european_antithetic,
    mc_european_control_variate,
    mc_european_importance_sampling,
)

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 0.5, 0.2, 0.03

def test_antithetic_unbiased_and_reduces_variance():
    bs = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    anti = mc_european_antithetic(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=100_000, n_steps=1, seed=0, scheme="exact")
    assert abs(anti.price - bs) < 4 * anti.std_error
    assert anti.variance_ratio > 1.0


def test_control_variate_unbiased_and_reduces_variance():
    bs = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    cv = mc_european_control_variate(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=200_000, n_steps=1, seed=0, scheme="exact")
    assert abs(cv.price - bs) < 4 * cv.std_error
    assert cv.variance_ratio > 1.0


def test_importance_sampling_unbiased_and_reduces_variance():
    # Deep out-of-the-money call: naive Monte Carlo wastes most paths.
    strike = 130.0
    bs = black_scholes(SPOT, strike, TTE, VOL, RATE, "call")
    ism = mc_european_importance_sampling(SPOT, strike, TTE, VOL, RATE, "call", n_paths=400_000, seed=0, scheme="exact")
    assert abs(ism.price - bs) < 4 * ism.std_error
    assert ism.variance_ratio > 1.0


def test_importance_sampling_zero_shift_reduces_to_plain_mc():
    a = mc_european_importance_sampling(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=100_000, seed=0, shift=0.0, scheme="exact")
    b = mc_european(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=100_000, n_steps=1, seed=0, scheme="exact")
    assert a.price == pytest.approx(b.price, abs=1e-12)
