"""Measured Numba-jit vs pure-NumPy speedup per PDE kernel.

Run: uv run python scripts/bench_pde_kernels.py

Times each jitted kernel (pricing.pde.solvers) against its pure-NumPy
reference (pricing.pde._numpy_ref) on a representative production-size
problem, reports the speedup ratio per kernel, and asserts numerical
agreement. Barrier handling is part of the march kernels.
"""

import time

import numpy as np

from pricing.pde._numpy_ref import (
    march_explicit_ref,
    march_theta_ref,
    psor_ref,
    thomas_ref,
)
from pricing.pde.solvers import march_explicit, march_theta, psor, thomas


def _best_of(fn, *args, repeats: int = 3):
    """Best-of-N wall time; march kernels mutate u in place, so callers pass
    a factory producing fresh arguments per repeat."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        call_args = [a() if callable(a) else a for a in args]
        t0 = time.perf_counter()
        result = fn(*call_args)
        best = min(best, time.perf_counter() - t0)
    return best, result


def main() -> None:
    n = 400
    n_steps = 400
    dt = 0.5 / n_steps
    rng = np.random.default_rng(0)
    lo_c, di_c, up_c = -0.5, -2.5, 1.4  # representative stencil weights
    v_lo = np.full(n_steps, 0.01)
    v_hi = np.full(n_steps, 50.0)
    u0 = rng.uniform(0.0, 40.0, n + 2)
    mat_implicit = (
        np.full(n, -dt * lo_c),
        np.full(n, 1.0 - dt * di_c),
        np.full(n, -dt * up_c),
    )
    rows = []

    # tridiagonal solve
    rhs = u0[1:-1].copy()
    t_jit, x_jit = _best_of(thomas, mat_implicit[0], mat_implicit[1], mat_implicit[2], rhs)
    t_ref, x_ref = _best_of(
        thomas_ref, mat_implicit[0], mat_implicit[1], mat_implicit[2], rhs
    )
    assert np.allclose(x_jit, x_ref, atol=1e-12)
    rows.append(("thomas solve", t_jit, t_ref))

    # explicit march (barrier handling included via barrier_idx; -1 = none here,
    # exercised with an active barrier in tests/test_pde_accel.py)
    t_jit, x_jit = _best_of(
        march_explicit, lambda: u0.copy(), lo_c, di_c, up_c, dt, v_lo, v_hi, -1
    )
    t_ref, x_ref = _best_of(
        march_explicit_ref, lambda: u0.copy(), lo_c, di_c, up_c, dt, v_lo, v_hi, -1
    )
    assert np.allclose(x_jit, x_ref, rtol=1e-12, atol=1e-12)
    rows.append(("explicit march", t_jit, t_ref))

    # implicit/CN/Douglas(ADI single-factor) march
    t_jit, x_jit = _best_of(
        march_theta,
        lambda: u0.copy(),
        *mat_implicit,
        lo_c,
        di_c,
        up_c,
        dt,
        1.0,
        v_lo,
        v_hi,
        -1,
    )
    t_ref, x_ref = _best_of(
        march_theta_ref,
        lambda: u0.copy(),
        *mat_implicit,
        lo_c,
        di_c,
        up_c,
        dt,
        1.0,
        v_lo,
        v_hi,
        -1,
    )
    assert np.allclose(x_jit, x_ref, rtol=1e-10, atol=1e-10)
    rows.append(("implicit/CN/ADI march", t_jit, t_ref))

    # projected SOR
    obstacle = np.minimum(u0[1:-1], 20.0)
    rhs = u0[1:-1].copy()
    t_jit, x_jit = _best_of(psor, *mat_implicit, rhs, obstacle)
    t_ref, x_ref = _best_of(psor_ref, *mat_implicit, rhs, obstacle)
    assert np.allclose(x_jit, x_ref, atol=1e-8)
    rows.append(("PSOR iteration", t_jit, t_ref))

    print(f"{'kernel':<24}{'jit (s)':>12}{'numpy (s)':>12}{'speedup':>10}")
    for name, tj, tr in rows:
        print(f"{name:<24}{tj:>12.4f}{tr:>12.4f}{tr / tj:>9.1f}x")


if __name__ == "__main__":
    main()
