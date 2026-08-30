# Domain language: pricing and model validation

A compact numerical core for derivatives pricing and independent model validation: Monte Carlo, finite-difference PDEs, and Greeks, anchored by one Bermudan swaption, validated against independent benchmarks, and documented as an SR 26-2 validation memo.

## Language

### Instruments & pricing

**Price**:
The discounted risk-neutral expectation of an instrument's payoff, produced by a pricing function.
_Avoid_: Value, NPV

**Payoff**:
The terminal cashflow an instrument pays as a function of the underlying's path.
_Avoid_: Payout

**Greeks**:
The sensitivity of a price to a model or market input (delta, gamma, vega, theta, rho).
_Avoid_: Sensitivities

**Anchor**:
The single instrument — a Bermudan swaption on Hull-White — carried through the full price → hedge → validate → memo workflow.

**Volatility surface**:
Implied volatility quoted across strikes and maturities — the target of SABR/Heston calibration.
_Avoid_: Vol curve, smile (a smile is a single-maturity slice)

**Calibration**:
Choosing a model's parameters so its prices fit a volatility surface.
_Avoid_: Fitting, tuning

### Numerical methods

**Discretization**:
A finite-step approximation of a stochastic or PDE process (Euler, Milstein, Crank–Nicolson).

**Weak convergence**:
The order at which the expected price error falls with step size — the notion pricing cares about.

**Strong convergence**:
The order at which the pathwise error falls with step size.

**Variance reduction**:
Techniques (antithetic variates, control variates, importance sampling) that cut Monte Carlo variance without biasing the estimate.

**Quasi-Monte Carlo (QMC)**:
Low-discrepancy Sobol sequences replacing pseudo-random draws, scrambled so error is still estimable.

### Validation

**Benchmark**:
A known-good reference price — a closed form or QuantLib — that a numerical method must converge to.
_Avoid_: Ground truth, oracle

**Challenger model**:
An independent second implementation of the same (or an alternative) theory, used to corroborate the primary model.

**Independent reimplementation**:
Recomputing a price by a different method (e.g. PDE vs Monte Carlo) to anchor the validation argument.

**Validation finding**:
A recorded gap with severity (green/amber/red), owner, evidence, limitation, and remediation, driving an approve / approve-with-limitations / reject outcome.

**PLA (profit & loss attribution)**:
The FRTB test that risk-theoretical P&L (built from Greeks) tracks hypothetical P&L (full revaluation) — Spearman > 0.80, KS < 0.09.

### Data & reproducibility

**Raw data**:
Fetched market data (option chains, rate series) as downloaded, kept in a gitignored local location — never committed.

**Frozen snapshot**:
The small processed copy of market data, committed to the repo for reproducibility.
_Avoid_: Dataset, dump

**Run manifest**:
The record of source release, as-of dates, checksums, seeds, and parameters that makes a run deterministic and replayable.
