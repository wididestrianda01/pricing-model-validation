# 05. PDE: finite differences, stability, early exercise

This chapter introduces the partial differential equation behind a price, the
finite-difference schemes that solve it, and how the project measures their
accuracy and their limits.

## Feynman-Kac: expectation and PDE are one object

The expectation form of a price and a partial differential equation are two
views of the same function. Feynman-Kac states that the risk-neutral
expectation solves the Black-Scholes PDE

```
dV/dt + 0.5 sigma^2 S^2 V_SS + r S V_S - r V = 0
```

with terminal condition `V(T, S) = payoff(S)`. The Monte Carlo engine
integrates the expectation; the PDE engine solves the differential equation.
That the two agree is itself a validation argument, because the methods share
nothing but the model.

> **Key takeaways.** Monte Carlo and finite differences compute the same
> number by different routes. Their agreement is independent corroboration,
> not a coincidence of shared code.

## Log-space: constant coefficients

In `x = ln S`, the Black-Scholes operator has constant coefficients:

```
dV/dt = 0.5 sigma^2 V_xx + (r - 0.5 sigma^2) V_x - r V
```

Constant coefficients mean one stencil serves every grid node, which the grid
module exploits (`src/pricing/pde/grid.py` builds a uniform log-space grid).
It also places a barrier exactly on a node when one is present, which the
absorbing boundary condition requires.

> **Key takeaways.** Working in log space turns the PDE into a constant-
> coefficient equation, so a single finite-difference stencil applies
> everywhere. This is a transform, not an approximation.

## The theta scheme

Discretizing space with a three-point stencil and time with a weighted scheme
gives a family indexed by `theta`:

```
(I - theta dt A) u_new = (I + (1 - theta) dt A) u
```

where `A` is the discrete spatial operator. `theta = 0` is explicit,
`theta = 1` is fully implicit, and `theta = 0.5` is Crank-Nicolson. The scheme
is implemented as `march_explicit` and `march_theta` in
`src/pricing/pde/solvers.py`, with the tridiagonal solve by the Thomas
algorithm (`thomas`).

- **Explicit** needs no matrix solve but is only conditionally stable: the
  time step must satisfy the CFL bound `dt <= 1/(sigma^2/dx^2 + r)`.
- **Implicit** is unconditionally stable and first order in time.
- **Crank-Nicolson** is second order in time and unconditionally stable.

Measured temporal orders: Crank-Nicolson `2.16`, implicit `0.98`, both against
theory (2 and 1).

> **Key takeaways.** The theta parameter trades solve cost against accuracy.
> Crank-Nicolson is the workhorse: second order and stable, at the cost of one
> tridiagonal solve per step.

## Convergence and Richardson extrapolation

The project measures the spatial order of Crank-Nicolson by refining the grid
and fitting the error slope. Measured spatial order `2.62`, and halving the
grid reduced the error by a factor of 8, consistent with second order.
Richardson extrapolation combines two grid levels to cancel the leading error
term, another confirmation of the order.

> **Key takeaways.** Measuring the order, not just the error, tells you how
> fast refinement pays off. A factor-of-8 error drop on halving confirms
> second order.

## Early exercise: American options

An American option can be exercised at any time before expiry, which turns the
PDE into an inequality: the value is never below the intrinsic payoff. The
project solves this two ways.

**Projected SOR** (`psor` in `src/pricing/pde/solvers.py`, driven by
`fd_american` in `src/pricing/pde/schemes.py`) enforces the constraint at each
step of the fully implicit solve, projecting the value up to intrinsic
wherever it dips below.

**Longstaff-Schwartz** (`lsm_american` in `src/pricing/pde/lsm.py`) works by
Monte Carlo: at each exercise date it regresses the future cashflow on a
polynomial basis in moneyness and exercises when intrinsic exceeds the fitted
continuation value.

Both recover the American price, and each is benchmarked against the other and
against a QuantLib binomial tree.

> **Key takeaways.** Early exercise turns pricing into a free-boundary
> problem. PSOR solves the constrained PDE; Longstaff-Schwartz solves it by
> regression on simulated paths. The two are independent and must agree.

## Order degradation on kinked payoffs

A kink in the payoff or the early-exercise boundary destroys smoothness, and
the measured convergence order drops exactly as the theory predicts. The
project records these degradations rather than hiding them:

| Problem | Measured order | Cause |
| --- | --- | --- |
| Vanilla (smooth) | 2.62 | reference second order |
| Digital payoff | 0.80 | payoff discontinuity |
| Knock-out barrier | 1.63 | barrier discontinuity |
| American (PSOR) | 1.96 | free-boundary kink |

These are recorded as `info` rows, not failures: the degradation is the
delivered result, and it tells the reader where grid refinement must
compensate.

> **Key takeaways.** Non-smooth payoffs degrade the convergence order. A
> validation report states this degradation; it does not suppress it. The
> order you get tells you how hard a problem is.

## Implementation map

- `src/pricing/pde/grid.py`: `build_grid`, `Grid`, `payoff_initial`,
  `boundary_value`.
- `src/pricing/pde/schemes.py`: `fd_solution`, `fd_price`, `fd_american`.
- `src/pricing/pde/solvers.py`: `thomas`, `psor`, `march_explicit`,
  `march_theta`.
- `src/pricing/pde/lsm.py`: `lsm_american`.
- `src/pricing/pde/studies.py`: the convergence studies.

## Decisions and rationale

- **Crank-Nicolson as the workhorse.** Second order in time and
  unconditionally stable, for the price of a tridiagonal solve. The Thomas
  algorithm makes that solve linear in the grid size, so the scheme is cheap
  enough to use everywhere.
- **Rannacher startup for early exercise.** Crank-Nicolson oscillates after a
  kink; a few fully implicit steps damp the oscillation. The anchor PDE uses
  two fully implicit steps after each exercise projection for this reason.
- **Projected SOR for the American constraint.** SOR is the standard iterative
  solve for the constrained problem and maps directly onto the
  complementarity condition. Its degraded order (1.96) is the expected price
  of the free boundary.
- **Record degradation, do not hide it.** A reader who sees only clean orders
  cannot trust the clean cases. Reporting the digital, barrier, and American
  orders together makes the smooth-case results credible.
