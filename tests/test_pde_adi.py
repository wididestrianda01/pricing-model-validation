"""Douglas ADI with Rannacher startup: 1-D reduction and oscillation control."""

import numpy as np

from pricing.pde import fd_price, fd_solution

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 0.25, 0.2, 0.03


def test_douglas_agrees_with_crank_nicolson_on_smooth_vanilla():
    p_dg = fd_price(
        SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="douglas", n_space=200, n_time=100
    )
    p_cn = fd_price(
        SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="cn", n_space=200, n_time=100
    )
    # single factor: Douglas splitting reduces to CN; the Rannacher ramp's
    # footprint on smooth data is negligible
    assert abs(p_dg - p_cn) < 1e-3


def test_rannacher_ramp_removes_cn_oscillations_on_digital():
    # fine space, very coarse time: dt >> dx^2/vol^2 is where plain CN rings
    ns, nt = 200, 4

    def sign_changes(scheme):
        grid, u = fd_solution(
            SPOT,
            STRIKE,
            TTE,
            VOL,
            RATE,
            "call",
            scheme=scheme,
            payoff="digital",
            n_space=ns,
            n_time=nt,
        )
        i = int(np.searchsorted(grid.x, np.log(STRIKE)))
        d = np.diff(u[i - 4 : i + 6])
        return int(np.sum(np.diff(np.sign(d)) != 0))

    assert sign_changes("cn") > 0  # oscillation present without the ramp
    assert sign_changes("douglas") == 0  # Rannacher startup removes it


def test_rannacher_improves_digital_accuracy():
    ns, nt = 200, 4
    from scipy.stats import norm

    d2 = (np.log(SPOT / STRIKE) + (RATE - 0.5 * VOL**2) * TTE) / (VOL * np.sqrt(TTE))
    exact = np.exp(-RATE * TTE) * norm.cdf(d2)
    err = {}
    for scheme in ("cn", "douglas"):
        p = fd_price(
            SPOT,
            STRIKE,
            TTE,
            VOL,
            RATE,
            "call",
            scheme=scheme,
            payoff="digital",
            n_space=ns,
            n_time=nt,
        )
        err[scheme] = abs(p - exact)
    assert err["douglas"] < err["cn"]
