"""Grid-convergence orders, Richardson extrapolation, degradation cases."""

import numpy as np

from pricing.benchmarks import black_scholes
from pricing.benchmarks.quantlib_benchmarks import ql_american, ql_barrier
from pricing.pde import (
    early_exercise_order,
    fd_price,
    richardson_extrapolate,
    spatial_order,
    temporal_order,
)
from pricing.pde.schemes import fd_american_solution

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 0.5, 0.2, 0.03
BARRIER_TTE = 73 / 365  # whole Act/365 days so the QuantLib seam matches exactly


def test_vanilla_spatial_order_is_second():
    order, _, _ = spatial_order(SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="cn")
    assert 1.8 < order < 2.9


def test_vanilla_temporal_orders_match_scheme_theory():
    ot, _, _ = temporal_order(
        SPOT,
        STRIKE,
        TTE,
        VOL,
        RATE,
        "call",
        scheme="cn",
        n_space=2000,
        n_time_list=(5, 10, 20, 40),
    )
    assert 1.7 < ot < 2.6
    ot, _, _ = temporal_order(
        SPOT,
        STRIKE,
        TTE,
        VOL,
        RATE,
        "call",
        scheme="implicit",
        n_space=2000,
        n_time_list=(5, 10, 20, 40),
    )
    assert 0.7 < ot < 1.3


def test_richardson_extrapolation_shrinks_error():
    bs = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    coarse = fd_price(
        SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="cn", n_space=50, n_time=800
    )
    fine = fd_price(
        SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="cn", n_space=100, n_time=1600
    )
    extrap = richardson_extrapolate(coarse, fine, 2.0)
    assert abs(extrap - bs) < abs(fine - bs) < abs(coarse - bs)


def test_digital_initial_data_degrades_observed_order():
    smooth, _, _ = spatial_order(SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="douglas")
    kinked, _, _ = spatial_order(
        SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="douglas", payoff="digital"
    )
    assert kinked < smooth - 0.5  # documented degradation, not hidden


def test_barrier_absorbing_boundary_prices_against_quantlib():
    barrier_vol = 0.25
    ref = ql_barrier(SPOT, 105.0, BARRIER_TTE, barrier_vol, RATE, barrier=90.0)
    errs = []
    for n in (100, 200, 400):
        p = fd_price(
            SPOT,
            105.0,
            BARRIER_TTE,
            barrier_vol,
            RATE,
            "call",
            scheme="cn",
            n_space=n,
            n_time=2 * n,
            barrier=90.0,
        )
        errs.append(abs(p - ref))
    assert errs == sorted(errs, reverse=True)  # monotone refinement
    assert errs[-1] < 0.005


def test_early_exercise_boundary_stable_under_refinement():
    def s_star(n_space):
        grid, u = fd_american_solution(
            100.0, 100.0, 1.0, 0.2, 0.05, "put", n_space=n_space, n_time=250
        )
        obstacle = np.maximum(100.0 - np.exp(grid.x), 0.0)
        cont = u - obstacle > 1e-9
        return float(np.exp(grid.x[int(np.argmin(cont))]))  # first continuation node

    coarse = s_star(150)
    fine = s_star(600)
    assert abs(coarse - fine) < 0.05 * 100.0  # free boundary stable under refinement


def test_early_exercise_case_converges_with_degraded_order():
    ref = ql_american(100.0, 100.0, 1.0, 0.2, 0.05, "put")
    order, _, _ = early_exercise_order(
        100.0,
        100.0,
        1.0,
        0.2,
        0.05,
        "put",
        n_space_list=(50, 100, 200),
        n_time=250,
        reference=ref,
    )
    # free-boundary kink degrades the order below the smooth European value,
    # measured on the same scheme (implicit) and grid ladder
    smooth, _, _ = spatial_order(
        100.0, 100.0, 1.0, 0.2, 0.05, "put",
        scheme="implicit", n_space_list=(50, 100, 200),
    )
    assert 1.2 < order
    assert order < smooth
