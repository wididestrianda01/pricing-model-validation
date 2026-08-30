"""Self-consistency tests for the closed-form benchmark module.

Each test asserts an external, independently-derived property — parity, a
limit, or a hand-calculated textbook value — never a re-run of the same
formula the implementation uses.
"""

import numpy as np
import pytest

from pricing.benchmarks import (
    black_76,
    black_scholes,
    heston,
    hw_caplet,
    hw_jamshidian_swaption,
    hw_zcb,
    hw_zcb_option,
)

# --- Black-Scholes -----------------------------------------------------------

def test_black_scholes_textbook_call_and_put():
    # Hull (Options, Futures & Other Derivatives): S=42, K=40, T=0.5, sigma=0.2, r=0.1.
    call = black_scholes(42.0, 40.0, 0.5, 0.2, 0.1, "call")
    put = black_scholes(42.0, 40.0, 0.5, 0.2, 0.1, "put")
    assert call == pytest.approx(4.759422392871532, abs=1e-12)
    assert put == pytest.approx(0.8085993729000922, abs=1e-12)


def test_black_scholes_put_call_parity():
    s, k, t, sig, r = 100.0, 105.0, 0.75, 0.25, 0.03
    call = black_scholes(s, k, t, sig, r, "call")
    put = black_scholes(s, k, t, sig, r, "put")
    assert call - put == pytest.approx(s - k * np.exp(-r * t), abs=1e-12)


    s, k, t, r = 100.0, 40.0, 0.5, 0.05
    deep_call = black_scholes(s, k, t, 0.2, r, "call")
    assert deep_call == pytest.approx(s - k * np.exp(-r * t), rel=1e-8)
    # Deep OTM put (strike far below spot) is essentially worthless.
    assert black_scholes(s, 40.0, t, 0.2, r, "put") < 1e-6


def test_black_scholes_zero_vol_is_intrinsic():
    s, k, t, r = 100.0, 90.0, 1.0, 0.04
    intrinsic = s - k * np.exp(-r * t)
    assert black_scholes(s, k, t, 0.0, r, "call") == pytest.approx(intrinsic)
    assert black_scholes(s, k, t, 0.0, r, "put") == pytest.approx(0.0)


# --- Black-76 ----------------------------------------------------------------

def test_black_76_matches_black_scholes_at_forward():
    # On a deterministic rate curve Black-76(F=Se^{rT}) == Black-Scholes(S).
    s, k, t, sig, r = 100.0, 110.0, 0.5, 0.2, 0.04
    forward = s * np.exp(r * t)
    assert black_76(forward, k, t, sig, r, "call") == pytest.approx(
        black_scholes(s, k, t, sig, r, "call"), abs=1e-12
    )
    assert black_76(forward, k, t, sig, r, "put") == pytest.approx(
        black_scholes(s, k, t, sig, r, "put"), abs=1e-12
    )


def test_black_76_put_call_parity():
    f, k, t, sig, r = 100.0, 95.0, 1.0, 0.3, 0.02
    call = black_76(f, k, t, sig, r, "call")
    put = black_76(f, k, t, sig, r, "put")
    assert call - put == pytest.approx(np.exp(-r * t) * (f - k), abs=1e-12)


# --- Hull-White --------------------------------------------------------------

@pytest.mark.parametrize("a,sigma", [(0.1, 0.01), (0.5, 0.02), (0.05, 0.03)])
def test_hw_zcb_recovers_flat_curve_at_time_zero(a, sigma):
    # P(0,T) must equal the market curve e^{-r0 T} for any a, sigma.
    r0 = 0.03
    for T in [0.5, 1.0, 3.0, 10.0]:
        assert hw_zcb(0.0, T, r0, a, sigma, r0) == pytest.approx(np.exp(-r0 * T), abs=1e-12)


def test_hw_zcb_option_put_call_parity():
    t, T, S, r, a, sigma, r0 = 0.0, 1.0, 3.0, 0.03, 0.1, 0.01, 0.03
    strike = 0.95
    call = hw_zcb_option(t, T, S, r, a, sigma, r0, strike, "call")
    put = hw_zcb_option(t, T, S, r, a, sigma, r0, strike, "put")
    p_tT = hw_zcb(t, T, r, a, sigma, r0)
    p_tS = hw_zcb(t, S, r, a, sigma, r0)
    assert call - put == pytest.approx(p_tS - strike * p_tT, abs=1e-12)


def test_hw_caplet_equals_bond_put_identity():
    # Caplet = (1 + tau*K) * Put(bond P(expiry,maturity), strike 1/(1+tau*K)).
    expiry, maturity, r, a, sigma, r0, K = 1.0, 1.5, 0.03, 0.1, 0.01, 0.03, 0.04
    tau = maturity - expiry
    bond_strike = 1.0 / (1.0 + tau * K)
    expected = (1.0 + tau * K) * hw_zcb_option(0.0, expiry, maturity, r, a, sigma, r0, bond_strike, "put")
    assert hw_caplet(0.0, expiry, maturity, r, a, sigma, r0, K) == pytest.approx(expected, abs=1e-12)


def test_hw_jamshidian_payer_nonnegative_and_parity():
    expiry, tenors, K, a, sigma, r0 = 1.0, [1.5, 2.0, 2.5, 3.0, 3.5], 0.03, 0.1, 0.01, 0.02
    payer = hw_jamshidian_swaption(expiry, tenors, K, a, sigma, r0, "payer")
    receiver = hw_jamshidian_swaption(expiry, tenors, K, a, sigma, r0, "receiver")
    assert payer > 0.0
    assert receiver > 0.0

    tau = np.diff(np.concatenate(([expiry], tenors)))
    tenors = np.asarray(tenors, dtype=float)
    forward_swap_pv = (
        np.exp(-r0 * expiry) - np.exp(-r0 * tenors[-1])
        - K * np.sum(tau * np.exp(-r0 * tenors))
    )
    assert payer - receiver == pytest.approx(forward_swap_pv, abs=1e-12)


# --- Heston ------------------------------------------------------------------

def test_heston_collapses_to_black_scholes_when_vol_of_vol_vanishes():
    # As xi -> 0 with v0 = theta, variance stays at v0 and Heston == BS(sigma=sqrt(v0)).
    v0 = theta = 0.04  # sigma = 0.2
    bs = black_scholes(100.0, 100.0, 1.0, 0.2, 0.05, "call")
    hv = heston(100.0, 100.0, 1.0, v0=v0, kappa=2.0, theta=theta, xi=1e-4, rho=-0.7, rate=0.05)
    assert hv == pytest.approx(bs, abs=1e-3)


def test_heston_put_call_parity():
    s, k, t, r, q = 100.0, 105.0, 0.5, 0.03, 0.01
    call = heston(s, k, t, v0=0.05, kappa=1.5, theta=0.04, xi=0.3, rho=-0.5, rate=r, dividend=q, option_type="call")
    put = heston(s, k, t, v0=0.05, kappa=1.5, theta=0.04, xi=0.3, rho=-0.5, rate=r, dividend=q, option_type="put")
    assert call - put == pytest.approx(s * np.exp(-q * t) - k * np.exp(-r * t), abs=1e-6)


def test_heston_price_nonnegative():
    price = heston(100.0, 90.0, 1.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.2, rho=-0.7, rate=0.02)
    assert price > 0.0


def test_unknown_option_type_raises():
    with pytest.raises(ValueError):
        black_scholes(100.0, 100.0, 1.0, 0.2, 0.05, "calll")
    with pytest.raises(ValueError):
        black_76(100.0, 100.0, 1.0, 0.2, 0.05, "PUT")
    with pytest.raises(ValueError):
        hw_jamshidian_swaption(1.0, [2.0, 3.0], 0.03, 0.1, 0.01, 0.02, "payre")


@pytest.mark.parametrize("spot,strike", [(-100.0, 105.0), (100.0, -105.0)])
def test_black_scholes_rejects_nonpositive_spot_strike(spot, strike):
    with pytest.raises(ValueError):
        black_scholes(spot, strike, 1.0, 0.2, 0.05)


def test_hw_zcb_option_at_expiry_is_intrinsic():
    # T == t collapses the bond-option variance to zero: deterministic exchange.
    t, S, r, a, sigma, r0, K = 1.0, 3.0, 0.03, 0.1, 0.01, 0.03, 0.95
    call = hw_zcb_option(t, t, S, r, a, sigma, r0, K, "call")
    put = hw_zcb_option(t, t, S, r, a, sigma, r0, K, "put")
    intrinsic_call = hw_zcb(t, S, r, a, sigma, r0) - K * hw_zcb(t, t, r, a, sigma, r0)
    assert call == pytest.approx(max(intrinsic_call, 0.0), abs=1e-15)
    assert put == pytest.approx(max(-intrinsic_call, 0.0), abs=1e-15)


def test_black_scholes_rejects_negative_vol():
    with pytest.raises(ValueError):
        black_scholes(100.0, 105.0, 1.0, -0.2, 0.05)


@pytest.mark.parametrize("rho", [-1.5, 1.5])
def test_heston_rejects_invalid_inputs(rho):
    with pytest.raises(ValueError):
        heston(100.0, 100.0, 1.0, v0=-0.04, kappa=2.0, theta=0.04, xi=0.2, rho=rho, rate=0.03)


def test_heston_u_max_extends_integration_range():
    # Default cutoff and an explicitly larger one agree on a benign case.
    a = heston(100.0, 90.0, 5.0, v0=0.09, kappa=1.0, theta=0.06, xi=1.0, rho=-0.9, rate=0.03)
    b = heston(100.0, 90.0, 5.0, v0=0.09, kappa=1.0, theta=0.06, xi=1.0, rho=-0.9, rate=0.03, u_max=400.0)
    assert a == pytest.approx(b, rel=1e-8)
