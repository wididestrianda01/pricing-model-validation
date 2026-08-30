# 00. Foundations: probability and calculus

This chapter introduces the two bodies of mathematics the rest of the project
rests on. Nothing here is stochastic calculus; that begins in
[03-monte-carlo.md](03-monte-carlo.md) when the code first simulates a random
process. The goal is to make the later chapters read as extensions of ideas
already in place.

## Random variables and expectation

A random variable assigns a number to each outcome of a random experiment. Its
distribution describes how likely each value is.

- **Expectation** `E[X]` is the probability-weighted average of the values
  `X` can take.
- **Variance** `Var(X) = E[(X - E[X])^2]` measures how far values spread from
  the mean.
- **Standard deviation** is the square root of the variance. It carries the
  same units as `X`, so it is easier to interpret.

> **Key takeaways.** Expectation is a weighted average. Variance and standard
> deviation measure spread. Every Monte Carlo result in this project is an
> estimate of an expectation, plus an estimate of the error in that estimate,
> and the error is expressed through the variance.

## The normal distribution

The standard normal has mean 0 and variance 1. Its density is

```
phi(x) = (1 / sqrt(2 pi)) exp(-x^2 / 2)
```

and its cumulative distribution function `N(x)` gives the probability that a
standard normal draw is at most `x`. A general normal random variable with
mean `mu` and standard deviation `sigma` is obtained by scaling and shifting a
standard normal: `X = mu + sigma Z`.

> **Key takeaways.** `N(x)` and `phi(x)` appear throughout the closed forms.
> `N(d1)` and `N(d2)` in Black-Scholes are probabilities under two different
> measures, and `phi(d1)` appears in gamma and vega.

## The law of large numbers and the central limit theorem

The **law of large numbers** states that the sample mean of independent draws
converges to the true expectation as the sample size grows.

The **central limit theorem** goes further: the error of the sample mean is
itself approximately normal, with standard deviation `sigma / sqrt(N)`, where
`sigma` is the per-draw standard deviation and `N` is the number of draws.

This is the whole reason Monte Carlo works and the whole reason its error is
controllable:

```
standard error = sigma / sqrt(N)
```

To halve the error, quadruple the number of paths. The project measures this
exact slope and records it as an acceptance bar.

> **Key takeaways.** Monte Carlo error falls as `1 / sqrt(N)`. The project
> asserts this in `se_scaling_slope`, which must land near `-0.5`. This one
> number justifies the entire engine.

## Sample statistics and standard error

Given `N` draws with sample mean `m`, the sample variance is

```
s^2 = (1 / (N - 1)) sum (x_i - m)^2
```

and the standard error of the mean is `s / sqrt(N)`. The standard error, not
the sample variance, is what a reported price must carry. A price without an
error bar is a claim, not a measurement.

> **Key takeaways.** Every `MCResult` in this project stores the price, the
> standard error, and the variance. A validation reviewer reads all three.

## Derivatives, partial derivatives, and the chain rule

The derivative of a function measures how much the output changes per unit
change in an input. A partial derivative does the same when the function has
several inputs and only one moves. The chain rule composes derivatives:

```
d/dx f(g(x)) = f'(g(x)) g'(x)
```

These are the tools behind every sensitivity in this project. Delta is the
partial derivative of price with respect to spot. Vega is the partial
derivative with respect to volatility. Chapter
[04-greeks.md](04-greeks.md) computes them three different ways, and the chain
rule is the reason the pathwise estimator is even possible.

## Taylor expansion

A smooth function is approximated locally by its tangent line, and better by a
parabola:

```
f(x + h) = f(x) + f'(x) h + (1/2) f''(x) h^2 + O(h^3)
```

This expansion is the origin of two ideas used later. First, a central
difference `(f(x+h) - f(x-h)) / (2h)` approximates `f'(x)` with error of order
`h^2`, while a forward difference has error of order `h`. Second, a
discretization scheme that is first order in the step `dt` has a price error
that falls linearly as `dt` shrinks; a second-order scheme falls quadratically.

> **Key takeaways.** The order of an approximation decides how fast error
> falls as the grid or step refines. Convergence orders are the acceptance
> bars in the Monte Carlo and PDE chapters, and the bump-size sweep in the
> Greeks chapter is a direct consequence of the Taylor error terms.

## Continuous compounding and the discount factor

A unit of cash paid at time `T` is worth `exp(-r T)` today, where `r` is the
continuously compounded rate. The exponential is used throughout because it is
the only convention where the discount factor is smooth in both time and rate.

> **Key takeaways.** `exp(-r T)` is the discount factor. Every price in this
> project multiplies a future payoff by its discount factor.

## The lognormal terminal distribution

If a price process is geometric Brownian motion (defined properly in
[03-monte-carlo.md](03-monte-carlo.md)), its terminal value is lognormal:

```
S_T = S_0 exp((r - 0.5 sigma^2) T + sigma W_T)
```

where `W_T` is normal with mean 0 and variance `T`. The log of `S_T / S_0` is
normal, which is what "lognormal" means. The `-0.5 sigma^2` term is the
convexity correction: geometric returns drift below arithmetic returns by half
the squared volatility.

> **Key takeaways.** The exact-GBM terminal above is the reference the
> Monte Carlo engine's Euler and Milstein schemes are measured against. When a
> scheme can sample this terminal exactly, it is called the exact scheme.
