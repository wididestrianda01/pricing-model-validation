# 01. Data: sources, pipeline, and reproducibility

This chapter covers where the project's data comes from, how it is stored,
and why the storage rules matter more than the data itself.

## Why data matters in validation

A model-validation finding is only as strong as the data behind it. If a
reviewer cannot reproduce the input, the output is an anecdote. For this
reason the project treats data provenance as a first-class concern: every
dataset carries a source, an as-of date, and a checksum, and no step after the
initial fetch touches the network.

## The datasets

| Dataset | Content | Source | Use |
| --- | --- | --- | --- |
| `SPY_options.parquet` | SPY option chain, 2008-2025, ~24M rows | lambdaclass mirror of philippdubach/options-data | SABR backtest |
| `SPY_underlying.parquet` | SPY underlying prices, 1999-2025 | same | backtest dates |
| `SOFR`, `DGS1MO`...`DGS30` | USD discount/rate curve | FRED | curve and discounting |
| `ESTR.csv` | Euro short-term rate | ECB Data Portal | curve toolkit (unused by anchor) |
| `STIBOR3M.csv` | SEK STIBOR 3M | Riksbank | curve toolkit (unused) |
| `SPX_snapshot.csv` | SPX option chain, one point in time | yfinance | real-snapshot SABR/Heston fit |

The USD and EUR and SEK rate series exist for curve-toolkit completeness. Only
the USD curve is consumed by the anchor. The EUR and SEK series are fetched
because a rates toolkit is expected to handle them, not because an instrument
needs them. Historical index-option and FX-option surfaces are deliberately
out of scope because they are paywall-gated and would break offline
reproducibility.

> **Key takeaways.** The project separates "data fetched" from "data used."
> The anchor consumes a USD curve; the EUR and SEK series are toolkit
> completeness. Scope discipline shows in the data list: nothing is fetched
> that would require a login at runtime.

## Raw data stays out of the repo

Raw market data is large and login-gated. The SPY backtest dataset is about
600 MB. Committing it would bloat the repository and complicate
reproducibility, so the project adopted a split (ADR-0003):

- **Raw data** lives in `data/raw/`, is gitignored, and is fetched once.
- **Frozen snapshots** of the small processed inputs live in `data/processed/`
  and are committed.

The rationale is a trade between two requirements. Reproducibility wants the
exact inputs committed; repository hygiene wants large binary blobs out. The
split satisfies both: the large raw files stay local, while the small derived
inputs that the experiments actually consume are committed and checksummed.
Nothing after the initial fetch touches the network.

> **Key takeaways.** Raw is gitignored; processed is committed. This is the
> standard division a reproducibility-minded team draws between "what we
> downloaded" and "what we actually consume."

## The manifest and checksums

`data/manifest.json` records every dataset with its source release, as-of
date, and SHA-256 checksum. On each run, the validation pipeline re-computes
the checksums of all committed processed artifacts. A mismatch sets
`processed_checksums_ok = 0`, which fails the corresponding acceptance bar.

This closes the loop: an edited artifact cannot silently change a reported
number. The run either verifies the inputs or fails loudly.

> **Key takeaways.** Checksums turn "trust me" into "verify me." The pipeline
> re-verifies them on every run, so a corrupted or tampered input fails the
> run rather than drifting the results.

## The discount curve

The calibration work needs a discount factor at arbitrary times. `load_zero_curve`
in `src/pricing/calibration/rates.py` builds one from the frozen FRED series:

- Tenors: SOFR (overnight), DGS3MO, DGS6MO, DGS1, DGS2.
- Continuously compounded zero rates, linearly interpolated in time, flat
  beyond the end points.
- Each series uses its latest observation on or before the as-of date.

The result is a function `discount_factor(tte) = exp(-zero_rate(tte) * tte)`.
Forwards for the SPX snapshot are then bootstrapped from put-call parity
against this curve, rather than assumed from a single rate.

> **Key takeaways.** A discount curve is a set of zero rates across tenors,
> interpolated. The calibration code does not assume a flat rate; it reads the
> curve and derives forwards from parity.

## The pipeline end to end

1. `scripts/fetch_data.py` downloads raw data once (idempotent; cached files
   are skipped).
2. `scripts/calibrate_snapshot.py` and `scripts/run_spy_backtest.py` turn raw
   data into the committed processed snapshots.
3. `scripts/run_all.py` runs every experiment offline against the committed
   snapshots and re-verifies their checksums.

Steps 1 and 2 are run rarely; step 3 is run constantly. The division means the
nine-second validation run needs no network and no login.

> **Key takeaways.** The pipeline has a one-way dependency: raw is fetched
> once, processed snapshots are derived and committed, and the validation run
> consumes only the committed snapshots. This is what "offline and
> deterministic" means in practice.

## Decisions and rationale

- **Raw out of repo, snapshot in.** ADR-0003. The alternative, committing raw
  data, costs 600 MB and pollutes history; the alternative, re-fetching on
  demand, breaks offline guarantees and is non-deterministic. The split is the
  only option that preserves both.
- **Checksums in the manifest, verified each run.** A reproducibility claim is
  worth nothing if an artifact can change out of band. Verifying on every run
  makes the claim mechanical.
- **Curve from FRED, not a flat assumption.** A flat rate is a convenience,
  not a market observation. Reading the actual curve removes an assumption the
  anchor otherwise would carry into its benchmarks.
