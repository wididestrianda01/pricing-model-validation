# 06. The anchor: a Bermudan swaption under Hull-White

This chapter follows one instrument through the full price-hedge-validate
workflow. It is the project's anchor: the single object carried end to end to
demonstrate the discipline a model-validation reviewer wants to see.

## Why one instrument, and why this one

A validation reviewer reads one complete controlled workflow, not a method
sampler. The project therefore anchors on a single Bermudan swaption on a
one-factor Hull-White model (ADR-0002). A Bermudan swaption is the canonical
rates object: it combines an interest-rate derivative with early exercise, so
it exercises every numerical tool the project built (Monte Carlo, PDE, Greeks,
calibration to a curve), and it is a real traded object at rates desks.

> **Key takeaways.** The anchor is the "whole animal": one instrument that
> forces every engine to work together on the same problem, so the agreement
> between engines means something.

## The Hull-White model

Hull-White models the short rate `r_t` as a mean-reverting Gaussian process:

```
dr_t = (theta(t) - a r_t) dt + sigma dW_t
```

where `a` is the mean-reversion speed and `sigma` the volatility. The mean
reversion pulls the rate back toward a time-dependent level `theta(t)` fitted
to the initial curve. For a flat curve, `theta` is constant and the model has
closed forms (chapter 02).

The anchor uses parameters `r0 = 0.03`, `a = 0.05`, `sigma = 0.01` on a flat
initial curve. The flat-curve assumption is a documented limitation: a real
term structure would need a bootstrapped `theta(t)`.

> **Key takeaways.** Hull-White is a mean-reverting Gaussian short-rate model.
> Its one factor is enough to price the anchor, and its flat-curve closed forms
> are the analytic benchmarks.

## The instrument

The anchor is a payer Bermudan swaption (`BermudanSwaption` in
`src/pricing/anchor/instrument.py`): the right to enter a payer swap at any of
five exercise dates. Strike 3.2%, ten semiannual payment periods starting at
T = 5.0 (payments 5.5 through 10.0), exercisable at T = 1, 2, 3, 4, 5.

> **Key takeaways.** The instrument spec is the shared seam both engines
> consume. Both price the *same* contract, so their difference is engine
> error, not specification drift.

## Engine 1: Longstaff-Schwartz Monte Carlo

`lsm_bermudan_swaption` in `src/pricing/anchor/hw_mc.py` simulates the short
rate. It uses the exact Ornstein-Uhlenbeck transition for the centered
process `x = r - alpha(t)`, so no discretization error enters the state. At
each exercise date it fits a continuation value by ordinary least squares on
in-the-money paths using a cubic polynomial in the short rate, and exercises
when the swap's value exceeds the fitted continuation.

The regression underfits the continuation value on finite samples, so the
estimator is biased low. This is the standard Longstaff-Schwartz property and
is recorded as a finding, not a defect.

> **Key takeaways.** LSM prices early exercise by regressing future cashflow
> on the state. The fit is biased low, so LSM is treated as a lower bound and
> reported with its standard error.

## Engine 2: Crank-Nicolson PDE

`pde_bermudan_swaption` in `src/pricing/anchor/hw_pde.py` solves the short-rate
PDE in the Brigo-Mercurio state variable `x = r - alpha(t)`, where `x` follows
a centered Ornstein-Uhlenbeck process. The value marches backward with
Crank-Nicolson, and at each exercise date it projects onto immediate exercise.
Two fully implicit Rannacher steps after each projection damp the kink. The
grid uses 401 nodes, 52 steps per year, and a domain of plus or minus eight
stationary standard deviations with linear extrapolation.

> **Key takeaways.** The PDE engine trades Monte Carlo noise for grid error,
> and the Rannacher steps keep Crank-Nicolson from oscillating after the
> exercise projections. It is deterministic: same grid, same price.

## The agreement

The two engines agree on the Bermudan price:

| Engine | Price |
| --- | --- |
| LSM Monte Carlo, 300k paths | 0.105548 (SE 0.0001) |
| Crank-Nicolson PDE | 0.10578 |
| Relative difference | 0.22% |

On the European special case, LSM matches Jamshidian's closed form to `0.08%`
(0.024928 vs 0.024909). The Bermudan price exceeds the European price by the
exercise premium, and both sit below the sum of per-date Jamshidian prices,
the no-early-exercise upper bound.

> **Key takeaways.** 0.22% agreement between engines that share only the
> instrument spec is the headline result. The European case ties both to the
> analytic benchmark, closing the loop.

## Hedging and the FRTB PLA test

A price is only half the story; a model must also hedge. `pla_consistency` in
`src/pricing/anchor/pla.py` builds two P&L series:

- **Risk-theoretical P&L (RTPL)**: the P&L implied by delta and vega over a
  scenario grid (10 rate shifts crossed with 4 vol shifts, 40 scenarios).
- **Hypothetical P&L (HPL)**: full revaluation over the same grid.

If the Greeks are right, RTPL tracks HPL. The FRTB consistency test requires
Spearman rank correlation above `0.80` and Kolmogorov-Smirnov distance below
`0.09`. Measured: Spearman `1.000`, KS `0.075`. The hedge leg passes, with the
KS margin recorded as thin because first-order RTPL omits gamma.

> **Key takeaways.** The PLA test asks "do my Greeks explain my P&L." Spearman
> and KS are the two statistics that answer it, and the project passes both,
> narrowly on KS.

## Findings

The anchor carries three documented findings (from `docs/anchor-validation.md`):

- LSM is biased low through regression underfit; treat PDE as reference.
- PLA KS passes with limited headroom; the 40-scenario grid is coarse.
- The flat-curve assumption limits the anchor to a demonstration of method,
  not of a live curve.

Each finding has a named remediation, and none blocks the intended use.

> **Key takeaways.** Findings are not failures. They are quantified
> limitations with owners and remediations, and their presence is what makes
> the rest of the report credible.

## Decisions and rationale

- **Bermudan swaption, not an equity exotic.** An equity autocallable would
  tie the real-data calibration into the anchor, but it is path-dependent and
  scope-heavy. A Bermudan swaption exercises Monte Carlo, PDE, curve, and early
  exercise together, and it is the canonical traded rates object.
- **Two independent engines.** LSM is Monte Carlo and stochastic; the PDE is
  deterministic and grid-based. They share nothing but the instrument spec, so
  agreement is evidence, not tautology.
- **Flat curve, documented.** The flat-curve Hull-White has hand-checkable
  closed forms, which the anchor needs for its benchmarks. The cost is a
  documented limitation, accepted for this project.
