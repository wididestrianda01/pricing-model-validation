# 07. Calibration: fitting models to market prices

This chapter covers how model parameters are chosen so that model prices match
observed market prices, and how the project proves its fitters before trusting
them on real data.

## Implied volatility

A market option price can be inverted to a Black-Scholes volatility: the
number `sigma` that makes the closed-form price equal the observed price.
`implied_vol` in `src/pricing/calibration/implied_vol.py` does this by
bracketed root finding (`brentq`), and it rejects quotes that violate static
no-arbitrage bounds (below intrinsic, above super-replication).

> **Key takeaways.** Implied vol is the volatility the market is pricing in,
> not a forecast. It is a quote transformed into a common unit, and it is only
> meaningful when the quote respects no-arbitrage bounds.

## The volatility surface

Quoting every option's implied vol reveals a structure: volatility varies with
strike (the smile or skew) and with maturity (the term structure). A
**volatility surface** is this two-dimensional object. A model that fits the
surface can price options consistently across strikes and maturities, which
Black-Scholes with one constant vol cannot.

> **Key takeaways.** The smile is evidence that Black-Scholes is wrong in a
> specific, systematic way. Calibration is the act of fitting a richer model
> to the surface.

## SABR

SABR (Hagan et al., 2002) models the forward and its volatility jointly and
gives an explicit formula for the implied volatility smile:

```
dF = sigma F^beta dW1
dsigma = nu sigma dW2,   corr(dW1, dW2) = rho
```

Four parameters shape the smile: `alpha` (level), `beta` (the elasticity of
volatility to the forward), `rho` (the skew), and `nu` (vol-of-vol, the smile
curvature). `sabr_vol` in `src/pricing/calibration/sabr.py` implements the
Hagan formula; `calibrate_sabr` fits `(alpha, rho, nu)` to a quoted smile at
fixed `beta`.

> **Key takeaways.** SABR gives a closed-form smile controlled by four
> parameters. `rho` sets the skew, `nu` the curvature, `alpha` the level, and
> `beta` is usually fixed at a conventional value. This is the industry's
> standard smile model.

## Heston

Heston models the variance as a mean-reverting square-root process (chapter
02 gives the closed form). Its five parameters are `v0` (initial variance),
`kappa` (mean-reversion speed), `theta` (long-run variance), `xi` (vol-of-vol),
and `rho` (spot-variance correlation). `calibrate_heston` in
`src/pricing/calibration/heston_fit.py` fits all five across a full expiry
surface, with a bounded least-squares objective and a fixed, deterministic
start.

Heston also carries a constraint called the **Feller condition**:
`2 kappa theta >= xi^2`. When it holds, the variance process never reaches
zero. The fitter flags a violation but does not force it, which the findings
ledger records.

> **Key takeaways.** Heston fits the whole surface with one parameter set, at
> the cost of a slower semi-closed price and a positivity condition (Feller)
> that real data may violate. The fitter surfaces the violation rather than
> hiding it.

## Synthetic first, then real

The project proves each fitter on a **synthetic** surface with known
parameters before applying it to real data. This is the decisive step: on a
synthetic surface the true parameters are known, so recovery can be measured
exactly. Measured synthetic recovery is at machine precision (parameter errors
below 1e-7, implied-vol RMSE below 1e-12).

Only after the fitter is proven does it touch the committed SPX snapshot.

> **Key takeaways.** Testing on synthetic data first converts "the fit looks
> plausible" into "the fit recovers the truth." A fitter that cannot recover
> known parameters cannot be trusted on unknown ones.

## The real-snapshot pipeline

`extract_slices` in `src/pricing/calibration/snapshot.py` turns a raw option
chain into clean per-expiry slices through a chain of filters:

- Spread filter: `(ask - bid) / mid <= 0.10`.
- Liquidity filter: volume or open interest at least 5 (the snapshot's open
  interest is mostly zero, so volume carries the signal).
- Moneyness band: strike/forward in `[0.80, 1.20]`.
- Implied-vol bounds `[0.03, 3.0]`.

Forwards come from put-call parity against the frozen FRED curve
(`parity_forward` takes the median of the parity estimates), not from an
assumed rate. SABR is then fit per selected tenor (30d, 90d, 1y) and Heston
across the same slices.

Real-snapshot fit quality: SABR mean slice RMSE `1.44` volatility points,
Heston cross-slice IV RMSE `1.94` volatility points. These are ordinary
magnitudes for single-day index fits given bid-ask noise in the underlying
quotes.

> **Key takeaways.** Real-data fitting is dominated by quote noise, and the
> honest measure of fit is RMSE in volatility points, reported per model.
> Heston's higher RMSE than SABR is expected: one parameter set across all
> tenors trades fit quality for consistency.

## The multi-date backtest

A single snapshot says nothing about stability. `run_spy_backtest` in
`src/pricing/calibration/backtest.py` repeats the SABR fit at 84 month-ends
from 2019 through 2025 on the SPY chain, producing a parameter trajectory.
The trajectory shows a visible volatility spike through the COVID period, and
a mean fit RMSE of `0.55` volatility points, which is the kind of stability a
reviewer wants to see across a regime change.

> **Key takeaways.** A backtest answers "is the fit stable over time." The
> parameter trajectory is the evidence; the mean RMSE across 84 dates is the
> summary statistic.

## Implementation map

- `src/pricing/calibration/implied_vol.py`: `implied_vol`.
- `src/pricing/calibration/sabr.py`: `sabr_vol`, `calibrate_sabr`, `SABRFit`.
- `src/pricing/calibration/heston_fit.py`: `calibrate_heston`, `HestonFit`,
  `heston_mc_price`.
- `src/pricing/calibration/snapshot.py`: `extract_slices`, `parity_forward`,
  `fit_snapshot`.
- `src/pricing/calibration/backtest.py`: `run_spy_backtest`.

## Decisions and rationale

- **SABR for slices, Heston for the surface.** SABR fits a single maturity
  smile with a closed form; Heston fits the whole surface with one parameter
  set. Using both covers the two standard calibration targets, and each cross
  checks the other's implied-vol output.
- **Synthetic recovery before real fitting.** A fitter is only trustworthy
  after it recovers known truth. Synthetic data makes that test exact, which
  real data never can.
- **Forwards from put-call parity, not an assumed rate.** Assuming a flat rate
  injects error into every forward and therefore every fitted smile. Parity
  against the actual curve removes that assumption.
- **Filter the chain before fitting.** Spread, liquidity, and moneyness cuts
  remove the quotes that are noise rather than information. The cuts are
  documented so the fitting set is reproducible.
