"""PDE schemes: explicit CFL stability, implicit/Crank-Nicolson accuracy."""

import numpy as np
import pytest

from pricing.benchmarks import black_scholes
from pricing.pde import cfl_dt, fd_price
from pricing.pde.grid import build_grid

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 0.5, 0.2, 0.03


def _bs_call() -> float:
    return black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")


def _safe_n_time() -> int:
    grid = build_grid(SPOT, STRIKE, TTE, VOL, 400)
    return int(1.5 * TTE / cfl_dt(VOL, RATE, grid.dx))


def test_explicit_matches_closed_form_inside_stability_region():
    p = fd_price(
        SPOT,
        STRIKE,
        TTE,
        VOL,
        RATE,
        "call",
        scheme="explicit",
        n_space=400,
        n_time=_safe_n_time(),
    )
    assert abs(p - _bs_call()) < 0.02


def test_explicit_run_at_stability_boundary_stays_bounded():
    # just inside the CFL bound: weights barely non-negative, no blow-up
    grid = build_grid(SPOT, STRIKE, TTE, VOL, 400)
    n_edge = int(1.02 * TTE / cfl_dt(VOL, RATE, grid.dx))
    p = fd_price(
        SPOT,
        STRIKE,
        TTE,
        VOL,
        RATE,
        "call",
        scheme="explicit",
        n_space=400,
        n_time=n_edge,
    )
    assert np.isfinite(p)
    assert 0.0 <= p <= 2.0 * _bs_call()


def test_explicit_blows_up_outside_stability_region():
    # 50 steps is far above the CFL bound -> the scheme must visibly explode
    p = fd_price(
        SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="explicit", n_space=400, n_time=50
    )
    assert not np.isfinite(p) or abs(p) > 1e6


def test_cfl_dt_shrinks_as_grid_refines():
    coarse = cfl_dt(VOL, RATE, build_grid(SPOT, STRIKE, TTE, VOL, 100).dx)
    fine = cfl_dt(VOL, RATE, build_grid(SPOT, STRIKE, TTE, VOL, 800).dx)
    assert fine < coarse / 50  # dt_max ~ dx^2


def test_implicit_and_cn_match_closed_form():
    bs = _bs_call()
    for scheme in ("implicit", "cn"):
        p = fd_price(
            SPOT, STRIKE, TTE, VOL, RATE, "call", scheme=scheme, n_space=300, n_time=300
        )
        assert abs(p - bs) < 0.01, scheme


def test_implicit_unconditionally_stable_where_explicit_blows_up():
    # 50 steps destabilises explicit (test above); implicit must stay sane
    p = fd_price(
        SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="implicit", n_space=400, n_time=50
    )
    assert abs(p - _bs_call()) < 0.02


def test_put_matches_closed_form():
    bs_put = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "put")
    p = fd_price(
        SPOT, STRIKE, TTE, VOL, RATE, "put", scheme="cn", n_space=300, n_time=300
    )
    assert abs(p - bs_put) < 0.01


def test_put_call_parity_holds_on_pde_prices():
    call = fd_price(
        SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="cn", n_space=300, n_time=300
    )
    put = fd_price(
        SPOT, STRIKE, TTE, VOL, RATE, "put", scheme="cn", n_space=300, n_time=300
    )
    parity = SPOT - STRIKE * np.exp(-RATE * TTE)
    assert abs((call - put) - parity) < 0.01


def test_invalid_scheme_rejected():
    with pytest.raises(ValueError, match="scheme"):
        fd_price(SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="leapfrog")
