# 03. Monte Carlo: simulation, convergence, variance reduction

This chapter introduces the random process behind equity pricing, the schemes
that simulate it, the way their error is measured, and the techniques that
shrink that error. It is the first place stochastic calculus appears.

## Brownian motion and geometric Brownian motion

A **Brownian motion** `W_t` is a continuous random process with independent,
normally distributed increments: `W_{t+h} - W_t` is normal with mean 0 and
variance `h`. It is the noise source behind most continuous-time finance.

**Geometric Brownian motion** (GBM) models the underlying price:

```
dS_t = mu S_t dt + sigma S_t dW_t
```

The `mu S_t dt` term is drift; the `sigma S_t dW_t` term is random fluctuation
proportional to the current price, which keeps the price positive. Under the
risk-neutral measure, `mu` is replaced by the risk-free rate `r`.

**Ito's lemma** is the chain rule for functions of a random process. Applied
to `ln S_t` it shows that the log price has a constant drift and volatility,
so the terminal price is lognormal:

```
S_T = S_0 exp((r - 0.5 sigma^2) T + sigma W_T)
```

The `-0.5 sigma^2` term appears because the quadratic variation of the
Brownian path is nonzero: `(dW)^2 = dt`, which Ito's lemma accounts for.

> **Key takeaways.** GBM is the model the equity engine simulates. Ito's lemma
> is the rule that turns `dS = mu S dt + sigma S dW` into the lognormal
> terminal above, including the `-0.5 sigma^2` correction.

## Discretization schemes

A computer cannot simulate a continuous path. It steps in increments `dt` and
approximates. Three schemes are implemented in
`src/pricing/monte_carlo/engine.py`:

- **Euler** (`_terminal_euler`): `S_{t+dt} = S_t (1 + r dt + sigma sqrt(dt) Z)`.
  First order, simplest.
- **Milstein** (`_terminal_milstein`): adds a second-order correction
  `0.5 sigma^2 (Z^2 - 1) dt` to reduce pathwise error.
- **Exact** (`_terminal_exact`): samples the lognormal terminal directly,
  with no discretization error at all, driven by the same normal draws.

The exact scheme is the reference the other two are measured against. The
point of keeping Euler and Milstein is to demonstrate, not to avoid, their
error.

> **Key takeaways.** Euler is first order, Milstein improves the strong error,
> and exact GBM has no time-step error. The engine ships all three so the
> error of the first two is visible and measured.

## Weak and strong convergence

Two distinct notions of convergence matter:

- **Weak convergence** is about prices: how the error in the *expected* payoff
  falls with `dt`. Pricing cares about weak convergence.
- **Strong convergence** is about paths: how the error in the *simulated path
  itself* falls with `dt`, against the exact terminal using the same draws.

The project measures both. Weak order is estimated by
`weak_convergence` in `src/pricing/monte_carlo/convergence.py`, strong order by
`strong_convergence`. The measured values:

| Quantity | Euler | Milstein | Theory |
| --- | --- | --- | --- |
| Weak order | 1.29 | 1.43 | ~1 (higher on kinked payoffs) |
| Strong order | 0.47 | 0.92 | 0.5 / 1.0 |

The strong orders match theory (0.5 and 1.0). The weak orders sit above 1
because the call payoff has a kink at the strike, and a kink can make the weak
error decay faster than the textbook smooth-payoff rate. Both facts are
recorded as acceptance bars.

> **Key takeaways.** Weak convergence governs prices; strong convergence
> governs paths. Milstein buys a better strong order (0.92 vs 0.47) at almost
> no extra cost, which is why it exists alongside Euler.

## Standard error scaling

The central limit theorem predicts Monte Carlo error falls as `1/sqrt(N)`. The
project verifies this by measuring the standard error across path counts from
10k to 160k and fitting the log-log slope. The measured slope is `-0.49556`,
against a bar of `[-0.6, -0.4]`.

> **Key takeaways.** The engine's error does exactly what the central limit
> theorem says. `se_scaling_slope = -0.50` is the evidence that the estimator
> is behaving as an average of independent draws, not hiding a bias.

## Variance reduction

Halving the standard error by quadrupling paths is expensive. Variance
reduction cuts the variance at fixed path count. Three methods are
implemented in `src/pricing/monte_carlo/variance_reduction.py`:

- **Antithetic variates** pair each draw `Z` with `-Z`. The two payoffs are
  negatively correlated, so their average has lower variance.
- **Control variate** subtracts a known-mean quantity (the discounted terminal
  asset, whose expectation is the spot) from the payoff, removing its noise.
- **Importance sampling** shifts the drift so paths land where the payoff is
  nonzero, then reweights by the likelihood ratio.

Each is tested for unbiasedness (estimate within four standard errors of the
Black-Scholes price) and for its variance ratio against plain Monte Carlo:

| Method | Variance ratio | Note |
| --- | --- | --- |
| Antithetic | 1.39 | cheap, always available |
| Control variate | 3.33 | removes the discount noise |
| Importance sampling | 41.7 | deep out-of-the-money call, strike 130 |

Importance sampling dominates because on a deep out-of-the-money call, plain
simulation wastes most paths on a zero payoff; shifting the drift concentrates
paths where the payoff lives.

> **Key takeaways.** Variance reduction is about placing paths where the
> payoff is. The methods are cheap (antithetic) to dramatic (importance
> sampling at 41.7x), and each must still be unbiased: the project verifies
> the estimate stays within four standard errors of the closed form.

## Quasi-Monte Carlo

Pseudo-random draws cluster and leave gaps. A low-discrepancy Sobol sequence
fills the space more evenly, reducing error faster than `1/sqrt(N)` for smooth
integrands. The project uses **scrambled** Sobol (`src/pricing/monte_carlo/qmc.py`):
scrambling keeps the sequence deterministic and unbiased, and it makes the
error estimable across independent replications, which a plain low-discrepancy
sequence does not.

> **Key takeaways.** QMC trades random points for evenly spread points. The
> scramble is what keeps the error measurable; without it, the estimator has
> no honest standard error.

## Implementation map

- `src/pricing/monte_carlo/engine.py`: `simulate_terminal`, `mc_european`,
  `standard_normals`, the three terminal schemes, and `MCResult`.
- `src/pricing/monte_carlo/convergence.py`: `weak_convergence`,
  `strong_convergence`, `se_scaling`.
- `src/pricing/monte_carlo/variance_reduction.py`: the three variance-reduced
  estimators.
- `src/pricing/monte_carlo/qmc.py`: `sobol_normals`, `mc_european_sobol`.

## Decisions and rationale

- **Ship Euler, Milstein, and exact together.** A pricing team does not need
  Euler once exact GBM exists, but a teaching and validation artifact does:
  the value is in measuring the error, not in avoiding it. The exact scheme
  also gives strong convergence a reference.
- **Measure both weak and strong order.** Weak order is what pricing needs;
  strong order is what the scheme's path accuracy needs. Reporting both
  separates "does the price converge" from "does the path converge," which are
  different questions with different answers.
- **Pick antithetic, control, and importance sampling.** They span the three
  standard mechanisms: symmetry, known means, and proposal reweighting. Each
  demonstrates a different way to cut variance, and importance sampling
  demonstrates the one that matters most in practice for out-of-the-money
  tails.
- **Use scrambled Sobol rather than plain Sobol.** Plain low-discrepancy
  sequences cannot report a standard error; scrambling restores error
  estimation while keeping the even coverage.
