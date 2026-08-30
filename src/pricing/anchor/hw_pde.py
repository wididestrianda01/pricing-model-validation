"""Finite-difference PDE pricer for Bermudan swaptions under Hull-White.

Works in the Brigo-Mercurio state variable x = r - alpha(t), where x follows a
centered Ornstein-Uhlenbeck process (dx = -a*x*dt + sigma*dW) and
alpha(t) = f(0,t) + sigma^2/(2a^2)*(1 - exp(-a t))^2 fits the flat initial
curve. By Feynman-Kac the value function satisfies

    marched backward from V(T_last_payment) = 0 with spatial generator

        L = (-a*x)*d/dx + 0.5*sigma^2*d2/dx2 - (x + alpha(t)),

    i.e. dV/dt + L V = 0. Reversing time via tau = T - t turns this into
    dV/dtau = L V, which the theta-scheme integrates forward in tau:
    (I - theta*dt*L) V^{n+1} = (I + (1-theta)*dt*L) V^n.
At every Bermudan exercise date the solution is projected onto immediate
exercise,
V <- max(V, swap_pv(x + alpha(t_k))). Crank-Nicolson with Rannacher startup
(two fully-implicit steps after each projection) damps the exercise kinks.
Boundaries are linearly extrapolated; the default domain spans +-8 stationary
standard deviations of x, wide enough that extrapolation error sits below
grid tolerance.
"""

from __future__ import annotations

import numpy as np

from pricing.anchor.instrument import (
    BermudanSwaption,
    HullWhiteParams,
    schedule_grid,
    swap_pv,
)
from pricing.pde.solvers import thomas


def alpha_shift(params: HullWhiteParams, t):
    """Deterministic curve-fit shift alpha(t) of the BM decomposition."""
    t_arr = np.asarray(t, dtype=float)
    return params.r0 + params.sigma**2 / (2.0 * params.a**2) * (
        1.0 - np.exp(-params.a * t_arr)
    ) ** 2


def pde_bermudan_swaption(
    swaption: BermudanSwaption,
    params: HullWhiteParams,
    *,
    n_space: int = 401,
    steps_per_year: int = 52,
    x_width_std: float = 8.0,
) -> float:
    """Bermudan swaption price by Crank-Nicolson FD on the HW short-rate PDE."""
    x_std = params.sigma / np.sqrt(2.0 * params.a)
    x_max = x_width_std * x_std
    x = np.linspace(-x_max, x_max, n_space)
    dx = x[1] - x[0]
    diff = 0.5 * params.sigma**2

    times, key_nodes = schedule_grid(
        swaption.exercise_times, swaption.pay_times, steps_per_year
    )
    exercises = {round(float(t), 12) for t in swaption.exercise_times}
    ex_nodes = {
        node for key, node in key_nodes.items() if round(key, 12) in exercises
    }
    grid_t = np.concatenate((times, [float(swaption.pay_times[-1])]))

    def apply_A(v: np.ndarray, rate: np.ndarray) -> np.ndarray:
        """Action of A = mu*d/dx + diff*d2/dx2 - rate, mu = -a*x."""
        out = np.zeros_like(v)
        vi = v[1:-1]
        conv = -params.a * x[1:-1]
        out[1:-1] = (
            conv * (v[2:] - v[:-2]) / (2.0 * dx)
            + diff * (v[2:] - 2.0 * vi + v[:-2]) / dx**2
            - rate[1:-1] * vi
        )
        return out

    v = np.zeros(n_space)  # terminal condition at the last payment date
    implicit_run = 0

    for k in range(len(times) - 2, -1, -1):
        dt = grid_t[k + 1] - grid_t[k]
        t_target = float(grid_t[k])
        rate = x + float(alpha_shift(params, t_target + dt / 2.0))
        theta = 1.0 if implicit_run < 2 else 0.5
        th_dt = theta * dt

        # Rows of (I - theta*dt*A), A's stencil per apply_A.
        conv = -params.a * x[1:-1]
        lower = th_dt * (+conv / (2.0 * dx) - diff / dx**2)
        upper = th_dt * (-conv / (2.0 * dx) - diff / dx**2)
        diag = 1.0 + th_dt * (2.0 * diff / dx**2 + rate[1:-1])

        rhs = v[1:-1] + (1.0 - theta) * dt * apply_A(v, rate)[1:-1]
        rhs[0] -= lower[0] * v[0]
        rhs[-1] -= upper[-1] * v[-1]

        v_new = np.empty_like(v)
        v_new[0] = v[0]
        v_new[-1] = v[-1]
        v_new[1:-1] = thomas(lower, diag, upper, rhs)
        v_new[0] = 2.0 * v_new[1] - v_new[2]
        v_new[-1] = 2.0 * v_new[-2] - v_new[-3]
        v = v_new
        implicit_run += 1

        if k in ex_nodes:
            exercise_value = swap_pv(
                swaption, params, t_target, x + float(alpha_shift(params, t_target))
            )
            v = np.maximum(v, exercise_value)
            implicit_run = 0

    return float(v[n_space // 2])
