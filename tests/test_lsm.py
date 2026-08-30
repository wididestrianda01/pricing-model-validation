"""Longstaff-Schwartz LSM cross-checked against PSOR-PDE and QuantLib."""

from pricing.benchmarks.quantlib_benchmarks import ql_american
from pricing.pde import fd_american, lsm_american

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 100.0, 1.0, 0.2, 0.05


def _lsm(n_paths=200_000, n_steps=50, seed=7):
    return lsm_american(
        SPOT, STRIKE, TTE, VOL, RATE, "put", n_paths=n_paths, n_steps=n_steps, seed=seed
    )


def test_lsm_matches_quantlib_within_noise():
    res = _lsm()
    ref = ql_american(SPOT, STRIKE, TTE, VOL, RATE, "put")
    assert abs(res.price - ref) < 4 * res.std_error + 0.02


def test_lsm_agrees_with_psor_pde():
    """Two structurally independent methods for the same contract."""
    res = _lsm()
    psor = fd_american(SPOT, STRIKE, TTE, VOL, RATE, "put", n_space=400, n_time=250)
    assert abs(res.price - psor) < 0.05


def test_lsm_standard_error_decays_with_path_count():
    se_small = _lsm(n_paths=20_000, n_steps=25).std_error
    se_large = _lsm(n_paths=160_000, n_steps=25).std_error
    ratio = se_small / se_large
    assert 2.0 < ratio < 3.2  # ~sqrt(8) = 2.83


def test_lsm_price_at_least_european():
    from pricing.benchmarks import black_scholes

    res = _lsm()
    floor = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "put")
    # low regression bias is small relative to the exercise premium here
    assert res.price > floor - 0.02


def test_lsm_regression_bias_bounded_below_reference_plus_noise():
    """Low-bias discipline: the LSM estimator must not sit above the binomial
    reference by more than Monte-Carlo noise (regression bias is downward)."""
    res = _lsm()
    ref = ql_american(SPOT, STRIKE, TTE, VOL, RATE, "put")
    assert res.price < ref + 3 * res.std_error
