"""Greeks engine: closed-form benchmark + estimators + validation studies."""

import numpy as np
import pytest
from scipy.stats import norm

from pricing.benchmarks import black_scholes, black_scholes_greeks
from pricing.benchmarks.quantlib_benchmarks import ql_black_scholes_greeks
from pricing.greeks.finite_difference import bump_greeks
from pricing.greeks.jax_pathwise import jax_pathwise_greeks
from pricing.greeks.likelihood_ratio import lr_greeks, lr_scores
from pricing.greeks.pathwise import pathwise_greeks
from pricing.greeks.studies import bias_variance_sweep, cross_method_greeks
from pricing.monte_carlo import simulate_terminal, standard_normals

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 1.0, 0.2, 0.05


# --- Closed-form Black-Scholes Greek benchmark --------------------------------

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_closed_form_greeks_match_quantlib(option_type):
    mine = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, option_type)
    ref = ql_black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, option_type)
    for key in ("delta", "gamma", "vega", "rho"):
        assert mine[key] == pytest.approx(ref[key], abs=1e-10)


def test_closed_form_delta_put_call_parity():
    call = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call")
    put = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, "put")
    assert call["delta"] - put["delta"] == pytest.approx(1.0, abs=1e-12)
    assert call["gamma"] == pytest.approx(put["gamma"], abs=1e-12)
    assert call["vega"] == pytest.approx(put["vega"], abs=1e-12)


def test_closed_form_theta_satisfies_bs_pde():
    # theta + r*S*delta + 0.5*vol^2*S^2*gamma == r*V  (Black-Scholes PDE).
    g = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call")
    v = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    lhs = g["theta"] + RATE * SPOT * g["delta"] + 0.5 * VOL**2 * SPOT**2 * g["gamma"]
    assert lhs == pytest.approx(RATE * v, abs=1e-8)


def test_closed_form_greeks_undefined_at_expiry_and_zero_vol():
    with pytest.raises(ValueError):
        black_scholes_greeks(SPOT, STRIKE, 0.0, VOL, RATE, "call")
    with pytest.raises(ValueError):
        black_scholes_greeks(SPOT, STRIKE, TTE, 0.0, RATE, "call")


# --- Finite-difference bump Greeks -------------------------------------------

def test_bump_greeks_match_closed_form():
    ref = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call")
    g = bump_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=1_000_000, seed=0)
    assert g["delta"] == pytest.approx(ref["delta"], abs=5e-3)
    assert g["gamma"] == pytest.approx(ref["gamma"], abs=1e-3)
    assert g["vega"] == pytest.approx(ref["vega"], abs=0.5)


# --- Pathwise Greeks ----------------------------------------------------------

def test_pathwise_greeks_match_closed_form():
    ref = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call")
    g = pathwise_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2_000_000, seed=0)
    assert g["delta"] == pytest.approx(ref["delta"], abs=5e-3)
    assert g["vega"] == pytest.approx(ref["vega"], abs=0.5)


def test_pathwise_greeks_match_closed_form_put():
    ref = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, "put")
    g = pathwise_greeks(SPOT, STRIKE, TTE, VOL, RATE, "put", n_paths=2_000_000, seed=0)
    assert g["delta"] == pytest.approx(ref["delta"], abs=5e-3)
    assert g["vega"] == pytest.approx(ref["vega"], abs=0.5)


# --- Likelihood-ratio Greeks --------------------------------------------------

def test_lr_greeks_match_closed_form():
    ref = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call")
    g = lr_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2_000_000, seed=0)
    assert g["delta"] == pytest.approx(ref["delta"], abs=5e-3)
    assert g["gamma"] == pytest.approx(ref["gamma"], abs=2e-3)
    assert g["vega"] == pytest.approx(ref["vega"], abs=0.6)


def _digital_call_delta(spot: float, strike: float, tte: float, vol: float, rate: float) -> float:
    """Analytic delta of a digital call e^{-rT} * 1{S_T > K}."""
    d2 = (np.log(spot / strike) + (rate - 0.5 * vol**2) * tte) / (vol * np.sqrt(tte))
    return np.exp(-rate * tte) * norm.pdf(d2) / (spot * vol * np.sqrt(tte))


def test_lr_recovers_digital_delta_where_pathwise_gives_zero():
    # Pathwise differentiates the payoff; a digital's indicator has zero
    # derivative a.e., so a pathwise digital delta is identically zero while the
    # true delta is positive. LR's score function recovers it.
    true_delta = _digital_call_delta(SPOT, STRIKE, TTE, VOL, RATE)
    assert true_delta > 0.0  # pathwise misses this entirely (would return 0)

    z = standard_normals(500_000, 1, 0)
    s_t = simulate_terminal(SPOT, TTE, VOL, RATE, z, "exact")
    disc_payoff = np.exp(-RATE * TTE) * (s_t > STRIKE)
    lr_delta = float(np.mean(disc_payoff * lr_scores(z, SPOT, VOL, TTE)["delta"]))
    assert lr_delta == pytest.approx(true_delta, abs=5e-3)


def test_lr_higher_variance_than_pathwise():
    # LR's score is payoff-agnostic but higher-variance than pathwise on a smooth
    # payoff; report the ratio so the tradeoff is measured, not asserted.
    z = standard_normals(200_000, 1, 0)
    s_t = simulate_terminal(SPOT, TTE, VOL, RATE, z, "exact")
    payoff = np.exp(-RATE * TTE) * np.maximum(s_t - STRIKE, 0.0)
    scores = lr_scores(z, SPOT, VOL, TTE)

    lr_delta_est = payoff * scores["delta"]
    # pathwise delta estimator per path: e^{-rT} * 1{S_T>K} * S_T/S0
    pw_delta_est = np.exp(-RATE * TTE) * (s_t > STRIKE) * (s_t / SPOT)

    assert lr_delta_est.var(ddof=1) > pw_delta_est.var(ddof=1)


# --- Bias-variance tradeoff study --------------------------------------------

def test_bias_variance_tradeoff_direction():
    _hs, biases, variances = bias_variance_sweep(
        SPOT, STRIKE, TTE, VOL, RATE, "call",
        h_list=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        n_paths=50_000, n_replications=30, base_seed=0,
    )
    # variance falls as the bump grows (the difference noise is scaled by 1/h)
    assert variances[0] > variances[-1]
    # bias grows with the bump in the bias-dominated (large-h) regime (h^2)
    assert abs(biases[-1]) > abs(biases[-2]) > abs(biases[-3])


def test_optimal_bump_shrinks_with_path_count():
    def optimal_h(n_paths: int) -> float:
        hs, biases, variances = bias_variance_sweep(
            SPOT, STRIKE, TTE, VOL, RATE, "call",
            h_list=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            n_paths=n_paths, n_replications=30, base_seed=0,
        )
        rmse = np.sqrt(biases**2 + variances)
        return float(hs[int(np.argmin(rmse))])

    # more paths shrink the optimal bump: variance scales 1/N, bias does not
    assert optimal_h(800_000) < optimal_h(50_000)


# --- Cross-method validation --------------------------------------------------

def test_cross_method_greeks_agree():
    ref = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call")
    c = cross_method_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2_000_000, seed=0)

    for key, tol in (("delta", 5e-3), ("vega", 0.6)):
        for method in ("bump", "pathwise", "likelihood_ratio"):
            assert c[method][key] == pytest.approx(ref[key], abs=tol)

    # gamma is delivered by bump and LR (pathwise gamma is distributional)
    assert c["bump"]["gamma"] == pytest.approx(ref["gamma"], abs=2e-3)
    assert c["likelihood_ratio"]["gamma"] == pytest.approx(ref["gamma"], abs=2e-3)


def test_greeks_consistent_with_price():
    # A delta from any estimator must predict the closed-form price change
    # across a small spot move (Greek-price consistency, the seam PLA builds on).
    d_spot = 0.1
    v0 = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    v1 = black_scholes(SPOT + d_spot, STRIKE, TTE, VOL, RATE, "call")
    c = cross_method_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2_000_000, seed=0)

    for method in ("bump", "pathwise", "likelihood_ratio"):
        predicted = c[method]["delta"] * d_spot
        assert predicted == pytest.approx(v1 - v0, abs=2e-3)



# --- JAX autodiff pathwise Greeks --------------------------------------------

def test_jax_pathwise_matches_manual_and_closed_form():
    ref = black_scholes_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call")
    manual = pathwise_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2_000_000, seed=0)
    g = jax_pathwise_greeks(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2_000_000, seed=0)

    assert g["delta"] == pytest.approx(ref["delta"], abs=5e-3)
    assert g["vega"] == pytest.approx(ref["vega"], abs=0.5)
    # autodiff must agree with the hand-derived pathwise estimator, not only the
    # closed form — the two independent implementations corroborate each other
    assert g["delta"] == pytest.approx(manual["delta"], abs=3e-3)
    assert g["vega"] == pytest.approx(manual["vega"], abs=0.4)


def test_central_difference_has_higher_order_than_forward():
    # Central differences raise the bias order to h^2, so at the same bump the
    # central bias is far smaller than the one-sided (forward) bias ~ h.
    h_list = [4.0, 8.0, 16.0, 32.0]
    _, fwd_bias, _ = bias_variance_sweep(
        SPOT, STRIKE, TTE, VOL, RATE, "call",
        h_list=h_list, n_paths=800_000, n_replications=30, base_seed=0,
        difference="forward",
    )
    _, ctr_bias, _ = bias_variance_sweep(
        SPOT, STRIKE, TTE, VOL, RATE, "call",
        h_list=h_list, n_paths=800_000, n_replications=30, base_seed=0,
        difference="central",
    )

    # forward bias ~ h (first order), not h^2
    fwd_slope = np.polyfit(np.log(h_list), np.log(np.abs(fwd_bias)), 1)[0]
    assert 0.7 < fwd_slope < 1.6
    # at the largest bump, central is far more accurate (h^2 vs h)
    assert abs(ctr_bias[-1]) < abs(fwd_bias[-1]) / 3.0
