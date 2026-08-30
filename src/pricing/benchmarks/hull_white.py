"""Closed-form one-factor Hull-White bond / swaplet / swaption prices.

Assumes a flat initial zero curve at continuously compounded rate ``r0`` so the
instantaneous forward f(0,t) = r0, giving clean, hand-checkable closed forms.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from pricing.benchmarks._validate import check_option_type, check_positive


def _b(t: float, T: float, a: float) -> float:
    return (1.0 - np.exp(-a * (T - t))) / a


def hw_zcb(
    t: float,
    T: float,
    r: float,
    a: float,
    sigma: float,
    r0: float,
) -> float:
    """Zero-coupon bond price P(t,T) under Hull-White (flat initial curve r0).

    t: valuation time. T: maturity. r: short rate at t. a: mean reversion.
    sigma: short-rate volatility. r0: flat initial curve rate.
    """
    check_positive(a=a)
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    B = _b(t, T, a)
    a_disc = np.exp(-r0 * (T - t) + B * r0 - (sigma**2 / (4.0 * a)) * (1.0 - np.exp(-2.0 * a * t)) * B**2)
    return float(a_disc * np.exp(-B * r))


def hw_zcb_option(
    t: float,
    T: float,
    S: float,
    r: float,
    a: float,
    sigma: float,
    r0: float,
    strike: float,
    option_type: str = "call",
) -> float:
    """European option at t, expiring T, on a zero-coupon bond maturing S."""
    option_type = check_option_type(option_type, ("call", "put"))
    p_tT = hw_zcb(t, T, r, a, sigma, r0)
    p_tS = hw_zcb(t, S, r, a, sigma, r0)
    B_TS = _b(T, S, a)
    v = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * (T - t))) / (2.0 * a)) * B_TS
    if v == 0.0:
        # Option is at expiry (T == t) or mean reversion has collapsed the
        # variance: the bond exchange is deterministic, return intrinsic.
        intrinsic_call = p_tS - strike * p_tT
        if option_type == "call":
            return float(max(intrinsic_call, 0.0))
        return float(max(-intrinsic_call, 0.0))
    d1 = (np.log(p_tS / (strike * p_tT)) + 0.5 * v**2) / v
    d2 = d1 - v

    if option_type == "call":
        return float(p_tS * norm.cdf(d1) - strike * p_tT * norm.cdf(d2))
    return float(strike * p_tT * norm.cdf(-d2) - p_tS * norm.cdf(-d1))


def hw_caplet(
    t: float,
    expiry: float,
    maturity: float,
    r: float,
    a: float,
    sigma: float,
    r0: float,
    strike_rate: float,
) -> float:
    """Caplet paying tau*(L(expiry,maturity)-K)^+ at maturity (a single swaplet).

    expiry: rate reset date T_{i-1}. maturity: payment date T_i.
    strike_rate: cap rate K. tau = maturity - expiry.
    """
    tau = maturity - expiry
    bond_strike = 1.0 / (1.0 + tau * strike_rate)
    put = hw_zcb_option(t, expiry, maturity, r, a, sigma, r0, bond_strike, "put")
    return float((1.0 + tau * strike_rate) * put)


def _swap_pv(r: float, tau: np.ndarray, A: np.ndarray, B: np.ndarray, strike_rate: float) -> float:
    """Value of a payer swap (pay fixed K, receive floating) at expiry, given r."""
    bonds = A * np.exp(-B * r)
    # Floating leg telescopes to 1 - P(expiry, Tn).
    return float(1.0 - bonds[-1] - strike_rate * np.sum(tau * bonds))


def hw_jamshidian_swaption(
    expiry: float,
    tenors: list[float],
    strike_rate: float,
    a: float,
    sigma: float,
    r0: float,
    option_type: str = "payer",
) -> float:
    """Price a swaption by Jamshidian's decomposition into bond options.

    expiry: option expiry T0. tenors: reset/payment dates [T1, ..., Tn] with
    T0 < T1 < ... < Tn, equally spaced (tau constant). strike_rate: fixed leg
    rate K. option_type: "payer" (pay fixed) | "receiver" (receive fixed).
    """
    option_type = check_option_type(option_type, ("payer", "receiver"))
    check_positive(a=a)
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    tenors = np.asarray(tenors, dtype=float)
    tau = np.diff(np.concatenate(([expiry], tenors)))
    Tn = tenors[-1]

    # Bond coefficients P(T0, Ti) = A_i * exp(-B_i * r(T0)); A, B independent of r.
    B = np.array([_b(expiry, Ti, a) for Ti in tenors])
    A = np.array([
        np.exp(-r0 * (Ti - expiry) + _b(expiry, Ti, a) * r0
               - (sigma**2 / (4.0 * a)) * (1.0 - np.exp(-2.0 * a * expiry)) * _b(expiry, Ti, a) ** 2)
        for Ti in tenors
    ])

    # Strike rate r* where the swap has zero value.
    r_star = brentq(lambda r: _swap_pv(r, tau, A, B, strike_rate), -10.0, 10.0)
    K_i = A * np.exp(-B * r_star)  # bond prices at r*
    # Payer swaption = K * sum(tau_i * Put(bond Ti, strike Ki)) + Put(bond Tn, strike Kn).
    payer = 0.0
    for i, (Ti, ki) in enumerate(zip(tenors, K_i)):
        put = hw_zcb_option(0.0, expiry, Ti, r0, a, sigma, r0, ki, "put")
        payer += (tau[i] * strike_rate) * put
    payer += hw_zcb_option(0.0, expiry, Tn, r0, a, sigma, r0, K_i[-1], "put")

    if option_type == "payer":
        return float(payer)
    # Receiver swaption via put-call parity: payer - receiver = forward swap PV.
    # Forward payer swap PV (flat curve) = P(0,T0) - P(0,Tn) - K * sum(tau_i * P(0,T_i)).
    forward_swap_pv = (
        np.exp(-r0 * expiry) - np.exp(-r0 * Tn)
        - strike_rate * np.sum(tau * np.exp(-r0 * tenors))
    )
    return float(payer - forward_swap_pv)
