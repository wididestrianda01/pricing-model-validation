"""Jitted kernels reproduce the pure-NumPy baseline results."""

import numpy as np

from pricing.pde._numpy_ref import march_explicit_ref, march_theta_ref
from pricing.pde.solvers import march_explicit, march_theta

N_SPACE, N_TIME = 120, 60
DT = 0.25 / N_TIME


def _problem():
    rng = np.random.default_rng(3)
    lo_c, di_c, up_c = -0.6, -2.4, 1.5
    mats = (
        np.full(N_SPACE, -DT * lo_c),
        np.full(N_SPACE, 1.0 - DT * di_c),
        np.full(N_SPACE, -DT * up_c),
    )
    v_lo = np.linspace(0.0, 0.05, N_TIME)
    v_hi = np.linspace(60.0, 45.0, N_TIME)
    u0 = rng.uniform(0.0, 55.0, N_SPACE + 2)
    return lo_c, di_c, up_c, mats, v_lo, v_hi, u0


def test_march_explicit_matches_numpy_reference():
    args = _problem()
    got = march_explicit(args[-1].copy(), args[0], args[1], args[2], DT, args[4], args[5], -1)
    ref = march_explicit_ref(args[-1].copy(), args[0], args[1], args[2], DT, args[4], args[5], -1)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_march_theta_matches_numpy_reference():
    lo_c, di_c, up_c, mats, v_lo, v_hi, u0 = _problem()
    for theta in (1.0, 0.5):
        got = march_theta(
            u0.copy(), *mats, lo_c, di_c, up_c, DT, theta, v_lo, v_hi, -1
        )
        ref = march_theta_ref(
            u0.copy(), *mats, lo_c, di_c, up_c, DT, theta, v_lo, v_hi, -1
        )
        assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)

def test_barrier_zeroing_identical_in_both_paths():
    args = _problem()
    barrier_idx = N_SPACE // 2
    got = march_explicit(args[-1].copy(), args[0], args[1], args[2], DT, args[4], args[5], barrier_idx)
    ref = march_explicit_ref(
        args[-1].copy(), args[0], args[1], args[2], DT, args[4], args[5], barrier_idx
    )
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)
    assert got[barrier_idx] == 0.0
