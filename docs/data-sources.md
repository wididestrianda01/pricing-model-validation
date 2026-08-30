# Data sources

Every dataset this project consumes is fetched once into `data/raw/` (gitignored,
per ADR-0003) and recorded in `data/manifest.json` with its source release, as-of
date, and SHA-256 checksum. A later step commits a small frozen snapshot of the
*processed* inputs for offline reproducibility; nothing after that touches the network.

Reproduce the fetch (idempotent — cached files are skipped):

```bash
uv sync --group fetch
uv run --group fetch python scripts/fetch_data.py
```

| Dataset | What it is | Source | License / access |
|---|---|---|---|
| `SPY_options.parquet` | SPY option chain, 2008–2025, ~24M rows (multi-date backtest) | `lambdaclass/options_portfolio_backtester` release `data-v1` | MIT, no login |
| `SPY_underlying.parquet` | SPY underlying prices, 1999–2025 | same release | MIT, no login |
| `SOFR`, `DGS1MO`…`DGS30` | USD discount/rate curve (overnight + Treasury CMT 1M–30Y) | FRED (St. Louis Fed) `fredgraph.csv` | free, keyless |
| `ESTR.csv` | Euro short-term rate, daily, 2019-10 onward | ECB Data Portal (SDMX REST) | free, keyless |
| `STIBOR3M.csv` | SEK STIBOR 3M, 1987-01–2020-07 | Riksbank SWEA v1 API | free, keyless (IP rate-limited) |
| `SPX_snapshot.csv` | SPX option chain snapshot, all listed expiries | yfinance (Yahoo Finance) `^SPX` | free, no key |

## Provenance notes

**SPY options (lambdaclass mirror).** The original `philippdubach/options-data`
dataset (MIT, 104 symbols, 2008–2025) is no longer reachable — its GitHub repos and
Cloudflare R2 static host return 404 as of 2026-08. `lambdaclass/options_portfolio_backtester`
re-published the same data as GitHub release assets under tag `data-v1`, including
`SPY_underlying.parquet`, which fixes the empty-`underlying.parquet` bug noted for
SPY/QQQ/IWM in the original release. SPY is American-style; implied volatility from
Black–Scholes is approximate (acceptable for OTM slices — documented under
calibration).

**Rates are USD + EUR + SEK short rates only.** The project has no EUR/SEK
instrument (anchor is a USD swaption; calibration is SPX/SPY), so the EUR/SEK series
are fetched for curve-toolkit completeness, not consumed by the anchor. Historical
*index-option* (SPX/OMXS30/EURO STOXX 50) and *FX-option* volatility surfaces are
deliberately out of scope — they are license/paywall-gated and would break
reproducibility.

**STIBOR is closed; SWESTR is not in the public API.** STIBOR was discontinued in
2020 and replaced by SWESTR (the transaction-based Swedish short rate). The public
Riksbank SWEA v1 API exposes the closed STIBOR series (`SEDP3MSTIBORDELAYC`,
1987-01–2020-07) but does not expose SWESTR, so the fetched SEK series ends in 2020.

**SPX snapshot is a point-in-time cross-section.** The yfinance pull is taken once
and pinned by its manifest checksum; the raw CSV is gitignored. The frozen snapshot
committed to the repo is derived from this (or from a SPY slice of the historical
dataset).

**Committed processed snapshots.** The calibration stage commits four derived
artifacts under `data/processed/` for offline
reproducibility: `spx_calibration_slices.csv` (filtered SPX slices with own
implied vols and parity forwards), `spx_calibration_results.json` (fitted SABR
per tenor and surface-wide Heston, with the raw-file checksum recorded),
`spy_sabr_backtest.csv` (multi-date SPY SABR parameter table), and
`spy_sabr_backtest_summary.json` (stability summary over the backtest table). Each is
registered in `data/manifest.json`; regenerate with
`uv run python scripts/calibrate_snapshot.py` and
`uv run python scripts/run_spy_backtest.py`. Filter cuts are documented in the
results JSON. The snapshot's `openInterest` column is mostly zero in the
yfinance pull, so liquidity filtering leans on volume; forwards come from
put-call parity against the frozen FRED curve rather than an assumed rate.
