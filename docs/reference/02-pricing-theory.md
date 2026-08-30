# 02. Pricing theory: options, no-arbitrage, and closed forms

This chapter builds the conceptual core the rest of the project prices
against. It defines an option, explains risk-neutral pricing, and gives the
closed-form prices that serve as benchmarks.

## What an option is

A European call gives its holder the right, not the obligation, to buy the
underlying at a fixed strike price on a fixed expiry. Its payoff at expiry is

```
payoff = max(S_T - K, 0)
```

where `S_T` is the underlying price at expiry and `K` is the strike. A put
pays `max(K - S_T, 0)`. The option costs money today because the payoff is
never negative and is sometimes positive. The central question of derivatives
pricing is: what is the fair price today of a contract that pays this in the
future?

> **Key takeaways.** An option is a right, not an obligation. Its payoff is
> one-sided and convex. Pricing is the problem of assigning today's value to a
> future, uncertain, one-sided payoff.

## No-arbitrage and replication

The fair price is the cost of a trading strategy that exactly reproduces the
payoff in every possible future state. If a cheaper strategy replicated the
payoff, a trader could buy the strategy and sell the option, lock in a profit
with no risk, and earn an arbitrage. Markets eliminate such opportunities, so
the option price must equal the cheapest replicating cost.

Black and Scholes showed that, when the underlying follows geometric Brownian
motion and trading is continuous, the payoff of a European option can be
replicated exactly by a continuously rebalanced position in the underlying and
a cash account. The price falls out of this replication argument.

> **Key takeaways.** No-arbitrage means "equal payoffs have equal prices." The
> option's price is the cost of replicating it. This one idea underlies every
> price in the project.

## Risk-neutral pricing

Replication yields a shortcut. After accounting for the replicating portfolio,
the option price is an expectation of the discounted payoff under a modified
probability measure, called the risk-neutral measure, where the underlying
drifts at the risk-free rate instead of its real-world drift:

```
V_0 = E_Q [ exp(-r T) payoff(S_T) ]
```

Under this measure, the discounted underlying is a martingale: its expected
future value, discounted, equals its value today. The real-world drift drops
out entirely. This is why option prices do not depend on an investor's view of
how fast the underlying grows.

> **Key takeaways.** Risk-neutral pricing replaces "what will the asset do"
> with "what is the discounted expected payoff under the measure where the
> asset grows at the risk-free rate." The drift disappears; only the
> volatility and the discount rate remain.

## The Black-Scholes formula

For a European call on a non-dividend asset, the expectation above evaluates
to a closed form:

```
C = S_0 N(d1) - K exp(-r T) N(d2)

d1 = (ln(S_0/K) + (r + 0.5 sigma^2) T) / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
```

`N` is the standard normal CDF. The two terms can be read as "the probability
of ending in the money, weighted by the discounted amounts." The closed-form
Greeks (delta, gamma, vega, theta, rho) follow by differentiation and appear
in `src/pricing/benchmarks/black_scholes.py`.

At the canonical market data used by the equity experiments (spot 100, strike
105, time 0.5, vol 0.2, rate 0.03), the call is worth 1.86809. This is the
number every Monte Carlo and Greek estimator must converge to.

> **Key takeaways.** Black-Scholes is the benchmark for the equity side. The
> canonical call is 1.86809, and it anchors every convergence test in chapters
> 03 and 04.

## Black-76: options on forwards

Rates instruments are options on rates, which are forwards, not spot assets.
Black-76 prices a European option on a forward `F` with the same functional
form, discounting at expiry. It is the workhorse for caplets and swaptions
where the underlying is an interest rate. `src/pricing/benchmarks/black_76.py`
implements it.

> **Key takeaways.** Black-Scholes prices options on spot; Black-76 prices
> options on forwards. Interest-rate derivatives use Black-76 because their
> underlying is a forward rate, not a traded spot asset.

## Hull-White: bond options, caplets, Jamshidian

Hull-White is a one-factor short-rate model (defined fully in
[06-anchor.md](06-anchor.md)). For a flat initial curve it has clean
closed forms for three objects:

- **Zero-coupon bond price** `P(t,T)` as a function of the short rate.
- **Bond option** and **caplet**, via a normal-based formula.
- **Swaption**, via Jamshidian's decomposition, which prices a swaption as a
  portfolio of bond options.

Jamshidian's insight is that in a one-factor model, the exercise decision at
the swaption's expiry depends on the short rate monotonically, so the
swaption payoff can be written as a sum of options on the individual bond
payments, each struck at the bond price that makes its payment exactly at the
money. These closed forms live in `src/pricing/benchmarks/hull_white.py`.

> **Key takeaways.** Hull-White gives closed forms for bonds, caplets, and
> swaptions. Jamshidian's decomposition turns a swaption into a portfolio of
> bond options, which is the analytic benchmark the anchor's numerical engines
> must match.

## Heston: a closed form with stochastic volatility

Heston models the variance as a mean-reverting random process rather than a
constant. Its European price has no elementary formula but does have a
semi-closed form through the characteristic function, evaluated by numerical
integration. `src/pricing/benchmarks/heston.py` implements this with the
"little trap" form of the characteristic function (Albrecher et al.), which is
numerically stable where the naive form is not.

> **Key takeaways.** Stochastic-volatility models lose the elementary closed
> form but keep a semi-closed form through the characteristic function. The
> Heston benchmark is what the calibration in
> [07-calibration.md](07-calibration.md) fits to.

## Benchmarks as a testing seam

Every closed form above exists to be a **benchmark**: a known-good reference
that a numerical method must converge to. The project pairs each closed form
with an independent implementation in QuantLib-Python
(`src/pricing/benchmarks/quantlib_benchmarks.py`), so a closed form is never
validated only against itself. At the tested quote, the closed-form
Black-Scholes and QuantLib's analytic engine agree to zero difference, which
is the first hard acceptance bar in the evidence table.

> **Key takeaways.** A benchmark is only convincing if it is independent. The
> project's rule is that no engine is validated against a variant of itself;
> each from-scratch method is checked against a closed form or QuantLib, which
> share no derivation and no code with it.
