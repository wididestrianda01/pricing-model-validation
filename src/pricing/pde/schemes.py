"""Finite-difference Black-Scholes engine in log-spot space.

Schemes: explicit, fully implicit, Crank-Nicolson, and Douglas ADI with a
Rannacher startup ramp. American exercise via projected SOR on the implicit
scheme. Barrier options via an absorbing Dirichlet node.

All schemes march tau (time since expiry) forward from the terminal payoff.
The per-step kernels live in pricing.pde.solvers and are Numba-jitted
(ADR-0001).
"""

from __future__ import annotations

import numpy as np

from pricing.benchmarks._validate import check_option_type, check_positive
from pricing.pde.grid import Grid, boundary_value, build_grid, payoff_initial
from pricing.pde.solvers import march_explicit, march_theta, psor

SCHEMES = ("explicit", "implicit", "cn", "douglas")
PAYOFFS = ("vanilla", "digital")


def cfl_dt(vol: float, rate: float, dx: float) -> float:
    """Largest stable explicit time step: dt <= 1 / (vol^2/dx^2 + rate).

    Von Neumann analysis: for a mode exp(i*m*x) the explicit amplification
    factor is g = 1 + dt*(di + 2*sqrt(lo*up)*cos(m*dx)), with lo, di, up the
    stencil weights of _coeffs. |g| <= 1 for every m holds iff
      (a) every stencil weight is non-negative:
          dx <= vol^2 / |rate - vol^2/2|  (positivity condition), and
      (b) 1 + dt*di >= 0.
    For rate > 0, di < 0 and |di + 2*sqrt(lo*up)| is maximal as m -> 0, so
    the zero-frequency mode binds and (b) gives the time-step restriction
    below; given (a), it is also sufficient.
    """
    return 1.0 / (vol**2 / dx**2 + rate)


def _coeffs(vol: float, rate: float, dx: float) -> tuple[float, float, float]:
    """Stencil coefficients of A = 0.5*vol^2 D2 + (rate - vol^2/2) D1 - rate."""
    mu = rate - 0.5 * vol**2
    lo = 0.5 * vol**2 / dx**2 - mu / (2.0 * dx)
    di = -(vol**2) / dx**2 - rate
    up = 0.5 * vol**2 / dx**2 + mu / (2.0 * dx)
    return lo, di, up


def _edge_values(
    grid: Grid,
    strike: float,
    rate: float,
    tau: float,
    option_type: str,
    payoff: str,
) -> tuple[float, float]:
    """Dirichlet values at the domain edges after tau years past expiry."""
    return (
        boundary_value(grid.x[0], strike, rate, tau, "lo", option_type, payoff),
        boundary_value(grid.x[-1], strike, rate, tau, "hi", option_type, payoff),
    )


def _edge_series(
    grid: Grid,
    strike: float,
    rate: float,
    i0: int,
    n_steps: int,
    step_size: float,
    option_type: str,
    payoff: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute Dirichlet edge values for every step of a march."""
    v_lo = np.empty(n_steps)
    v_hi = np.empty(n_steps)
    for i in range(n_steps):
        tau = (i0 + i + 1) * step_size
        v_lo[i], v_hi[i] = _edge_values(grid, strike, rate, tau, option_type, payoff)
    return v_lo, v_hi


def fd_solution(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    *,
    scheme: str = "cn",
    payoff: str = "vanilla",
    n_space: int = 400,
    n_time: int = 400,
    barrier: float | None = None,
) -> tuple[Grid, np.ndarray]:
    """Full nodal solution (grid, V) of the European/barrier problem.

    scheme: "explicit" | "implicit" | "cn" | "douglas". The douglas scheme
    includes a Rannacher startup ramp (four half-size implicit steps) that
    damps Crank-Nicolson oscillations on non-smooth payoffs.
    """
    option_type = check_option_type(option_type, ("call", "put"))
    check_positive(spot=spot, strike=strike)
    if payoff not in PAYOFFS:
        raise ValueError(f"payoff must be one of {PAYOFFS}, got {payoff!r}")
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if tte <= 0.0:
        raise ValueError("tte must be > 0")
    if barrier is not None and (barrier <= 0.0 or barrier == spot):
        raise ValueError("barrier must be positive and different from spot")

    grid = build_grid(spot, strike, tte, vol, n_space, barrier)
    u = payoff_initial(grid, strike, payoff, option_type)

    # knock-out: absorbing Dirichlet zero on the aligned barrier node.
    # The grid builder places the barrier exactly on a node so this is exact.
    barrier_idx = (
        int(np.argmin(np.abs(grid.x - np.log(barrier)))) if barrier is not None else -1
    )
    if barrier_idx >= 0:
        if barrier < spot:  # down-and-out: zero at/below the barrier node
            u[: barrier_idx + 1] = 0.0
        else:  # up-and-out: zero at/above it
            u[barrier_idx:] = 0.0
    if scheme == "douglas" and n_time < 4:
        raise ValueError(
            "douglas/Rannacher needs n_time >= 4 (ramp consumes two full steps)"
        )

    coeffs = _coeffs(vol, rate, grid.dx)
    lo_c, di_c, up_c = coeffs
    dt = tte / n_time
    n = n_space

    def tridiag(step_dt: float, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constant factors of (I - theta*step_dt*A), built once per problem."""
        w = theta * step_dt
        return (
            np.full(n, -w * lo_c),
            np.full(n, 1.0 - w * di_c),
            np.full(n, -w * up_c),
        )

    mat_implicit = tridiag(dt, 1.0)
    mat_cn = tridiag(dt, 0.5)
    mat_ramp_half = tridiag(dt / 2.0, 1.0)  # half-size steps for the ramp

    if scheme == "explicit":
        v_lo, v_hi = _edge_series(
            grid, strike, rate, 0, n_time, dt, option_type, payoff
        )
        u = march_explicit(u, lo_c, di_c, up_c, dt, v_lo, v_hi, barrier_idx)
    elif scheme == "implicit":
        v_lo, v_hi = _edge_series(
            grid, strike, rate, 0, n_time, dt, option_type, payoff
        )
        u = march_theta(
            u, *mat_implicit, lo_c, di_c, up_c, dt, 1.0, v_lo, v_hi, barrier_idx
        )
    elif scheme == "cn":
        v_lo, v_hi = _edge_series(
            grid, strike, rate, 0, n_time, dt, option_type, payoff
        )
        u = march_theta(u, *mat_cn, lo_c, di_c, up_c, dt, 0.5, v_lo, v_hi, barrier_idx)
    else:  # douglas with Rannacher startup ramp
        # first two full steps replaced by four half-size FULLY IMPLICIT steps,
        # killing CN's spurious oscillations on kinked data; then plain CN.
        v_lo, v_hi = _edge_series(grid, strike, rate, 0, 4, dt / 2.0, option_type, payoff)
        u = march_theta(
            u, *mat_ramp_half, lo_c, di_c, up_c, dt / 2.0, 1.0, v_lo, v_hi, barrier_idx
        )
        v_lo, v_hi = _edge_series(
            grid, strike, rate, 2, n_time - 2, dt, option_type, payoff
        )
        u = march_theta(u, *mat_cn, lo_c, di_c, up_c, dt, 0.5, v_lo, v_hi, barrier_idx)
    return grid, u


def fd_price(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    *,
    scheme: str = "cn",
    payoff: str = "vanilla",
    n_space: int = 400,
    n_time: int = 400,
    barrier: float | None = None,
) -> float:
    """European (optionally continuous knock-out barrier) price by finite differences."""
    grid, u = fd_solution(
        spot,
        strike,
        tte,
        vol,
        rate,
        option_type,
        scheme=scheme,
        payoff=payoff,
        n_space=n_space,
        n_time=n_time,
        barrier=barrier,
    )
    return float(np.interp(np.log(spot), grid.x, u))


def fd_american_solution(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    *,
    n_space: int = 400,
    n_time: int = 400,
    tol: float = 1e-10,
) -> tuple[Grid, np.ndarray]:
    """Full nodal American solution (grid, V) via projected SOR."""
    option_type = check_option_type(option_type, ("call", "put"))
    check_positive(spot=spot, strike=strike)
    if tte <= 0.0:
        raise ValueError("tte must be > 0")

    grid = build_grid(spot, strike, tte, vol, n_space)
    u = payoff_initial(grid, strike, "vanilla", option_type)
    obstacle = u.copy()  # immediate-exercise value on every node
    lo_c, di_c, up_c = _coeffs(vol, rate, grid.dx)
    dt = tte / n_time
    n = n_space

    mat_lo = np.full(n, -dt * lo_c)
    mat_up = np.full(n, -dt * up_c)
    mat_di = np.full(n, 1.0 - dt * di_c)
    for i in range(n_time):
        tau = (i + 1) * dt
        v_lo = boundary_value(grid.x[0], strike, rate, tau, "lo", option_type)
        v_hi = boundary_value(grid.x[-1], strike, rate, tau, "hi", option_type)
        # (I - dt*A) u_new = u_old, boundary contributions folded into rhs
        rhs = u[1:-1].copy()
        rhs[0] += dt * lo_c * v_lo
        rhs[-1] += dt * up_c * v_hi
        u[1:-1] = psor(mat_lo, mat_di, mat_up, rhs, obstacle[1:-1], tol=tol)
        u[0] = max(v_lo, obstacle[0])
        u[-1] = max(v_hi, obstacle[-1])
    return grid, u


def fd_american(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    *,
    n_space: int = 400,
    n_time: int = 400,
    tol: float = 1e-10,
) -> float:
    """American price via projected SOR on the fully implicit scheme."""
    grid, u = fd_american_solution(
        spot,
        strike,
        tte,
        vol,
        rate,
        option_type,
        n_space=n_space,
        n_time=n_time,
        tol=tol,
    )
    return float(np.interp(np.log(spot), grid.x, u))
