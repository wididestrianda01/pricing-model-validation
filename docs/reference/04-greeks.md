# 04. Greeks: sensitivities and how to estimate them

This chapter covers the sensitivities of a price to its inputs, why a desk
needs them, and the four routes the project uses to compute them, plus the
trade each route makes.

## What Greeks are

A Greek is a partial derivative of the option price with respect to one input:

| Greek | Definition | Meaning |
| --- | --- | --- |
| Delta | `dV/dS` | sensitivity to the underlying |
| Gamma | `d^2V/dS^2` | how fast delta itself moves |
| Vega | `dV/dsigma` | sensitivity to volatility |
| Theta | `dV/dt` | time decay |
| Rho | `dV/dr` | sensitivity to the rate |

A desk hedges with Greeks: it buys or sells the underlying and other
instruments to cancel the portfolio's sensitivity to each risk factor. If the
Greeks are wrong, the hedge is wrong, and P&L that should be flat moves.

> **Key takeaways.** Greeks are partial derivatives. They drive hedging, so
> their accuracy is a first-order risk concern, not a cosmetic one.

## The closed-form reference

For Black-Scholes, every Greek has a closed form, implemented in
`src/pricing/benchmarks/black_scholes.py` as `black_scholes_greeks`. These are
the reference the numerical estimators are measured against. The project
reports relative error against them; the bars are 5% for delta and 15% for
vega, reflecting that vega estimators are noisier.

> **Key takeaways.** The closed-form Greeks are the benchmark. Delta and vega
> get different tolerance bars because their estimators have different
> variance.

## Route 1: bump-and-reprice

`bump_greeks` in `src/pricing/greeks/finite_difference.py` perturbs an input,
reprices, and forms a finite difference. A central difference
`(V(S+h) - V(S-h)) / (2h)` has error of order `h^2`; a forward difference has
error of order `h`. The estimator defaults to central differences with common
random numbers, meaning every reprice shares one seed so Monte Carlo noise
cancels in the difference.

The measured delta error is `0.000694` and vega `0.001698`, both well inside
their bars.

> **Key takeaways.** Bump-and-reprice is the workhorse: simple, robust, and
> model-agnostic. Its two knobs are the bump size and whether reprices share
> random numbers. Sharing them cancels most Monte Carlo noise in the
> difference.

## Route 2: pathwise

`pathwise_greeks` in `src/pricing/greeks/pathwise.py` differentiates inside the
expectation. By the chain rule, for a smooth payoff,

```
dV/dS = E[ exp(-rT) (d payoff/dS_T) (dS_T/dS) ]
```

with `dS_T/dS = S_T/S` under GBM. The estimator differentiates the payoff
along each simulated path. It is low-variance but only valid where the payoff
is differentiable; gamma is omitted because the second derivative of a vanilla
payoff is distributional, not a function.

Measured: delta error `0.001114`, vega `0.001699`.

> **Key takeaways.** Pathwise moves the derivative inside the expectation and
> differentiates the payoff. It is cheap and clean but breaks on kinked or
> discontinuous payoffs, and it cannot produce gamma for vanilla options.

## Route 3: likelihood ratio

`lr_greeks` in `src/pricing/greeks/likelihood_ratio.py` differentiates the
*density*, not the payoff:

```
dV/dS = E[ payoff * d(log density)/dS ]
```

The score function `d(log density)/dS` does not involve the payoff, so the
method works on discontinuous payoffs that pathwise cannot touch. The cost is
higher variance: the score multiplies the full payoff. This is why likelihood
ratio vega (`0.004342`, the worst in the suite) is noisier than the others.

> **Key takeaways.** Likelihood ratio is payoff-agnostic and handles
> discontinuities, at the price of variance. It is the fallback when pathwise
> does not apply.

## Route 4: JAX automatic differentiation

`jax_pathwise_greeks` in `src/pricing/greeks/jax_pathwise.py` writes the price
as a differentiable JAX function and calls `jax.grad`. Autodiff recovers the
pathwise delta and vega with no hand-derived derivative, differentiating
through the exact-GBM terminal and the payoff. Its scope is deliberately
narrow (ADR-0001): JAX earns its place only here, where autodiff through the
Monte Carlo graph is a genuine advantage.

> **Key takeaways.** Autodiff removes the hand-derivation step and is a useful
> cross-check on the manual pathwise code. The project confines JAX to this
> one use rather than adopting it as the general engine.

## Bias-variance tradeoff of the bump

A larger bump reduces the variance of the finite-difference ratio but
increases the Taylor bias (`h^2` terms). The project sweeps the bump size and
locates the minimum root-mean-square error. The sweep found the optimum at the
edge of the tested range (`optimal_bump_h = 4.0`, RMSE `0.006272`), which the
findings ledger records as a sign that the range, not the true optimum, was
measured.

> **Key takeaways.** The bump size is a bias-variance knob: bigger bump, less
> variance, more bias. The sweep quantifies the trade rather than guessing it.

## Implementation map

- `src/pricing/greeks/finite_difference.py`: `bump_greeks`.
- `src/pricing/greeks/pathwise.py`: `pathwise_greeks`.
- `src/pricing/greeks/likelihood_ratio.py`: `lr_greeks`, `lr_scores`.
- `src/pricing/greeks/jax_pathwise.py`: `jax_pathwise_greeks`.

## Decisions and rationale

- **Compute all four routes on the same seam.** Each route fails differently:
  bump needs a smooth price, pathwise needs a smooth payoff, likelihood ratio
  needs the density, autodiff needs a differentiable graph. Running all four
  and cross-checking them against the closed form is itself the validation
  argument; no single route is trusted on its own.
- **Omit gamma from pathwise and autodiff.** The second derivative of a
  vanilla payoff is distributional, so differentiating the payoff twice yields
  zero away from the strike. Gamma is delivered by bump and likelihood ratio
  instead, where it is well defined.
- **Scope JAX to pathwise autodiff only.** JAX is not the engine; it is a
  cross-check for one estimator. Confining it avoids a second, parallel
  numerical stack for no benefit elsewhere.
