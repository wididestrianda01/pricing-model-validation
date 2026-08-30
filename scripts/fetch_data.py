"""Fetch raw market data into `data/raw/` and record a run manifest.

Sources (all free, no-login, recorded in `data/manifest.json` with a SHA-256
checksum so the frozen inputs can be verified offline):

- SPY options chain + underlying — lambdaclass mirror of `philippdubach/options-data` (MIT)
- USD discount/rate curve — FRED (SOFR + Treasury constant-maturity yields)
- EUR short rate — ECB ESTR (daily)
- SEK short rate — Riksbank STIBOR 3M (closed 2020; SWESTR not in the public API)
- SPX options snapshot — yfinance (current chain, pinned as-of)

Idempotent: existing files are skipped, so a partial re-run does not
re-download the ~600 MB SPY parquet.

Run with the fetch dependency group:
    uv run --group fetch python scripts/fetch_data.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

OPTIONS_DIR = Path("data/raw/options")
RATES_DIR = Path("data/raw/rates")

# GitHub release assets mirroring the MIT-licensed philippdubach/options-data
# dataset (original repo + static host are no longer reachable; lambdaclass
# re-published the same data as release assets).
SPY_RELEASE = "https://github.com/lambdaclass/options_portfolio_backtester/releases/download/data-v1"
SPY_OPTIONS = (
    f"{SPY_RELEASE}/SPY_options.parquet",
    "SPY options chain, 2008-2025, 24M rows (multi-date backtest source)",
)
SPY_UNDERLYING = (
    f"{SPY_RELEASE}/SPY_underlying.parquet",
    "SPY underlying prices, 1999-2025 (fixes the empty underlying.parquet bug in the original dataset)",
)

# USD discount / rate curve — free, keyless via FRED fredgraph.csv.
FRED_SERIES = {
    "SOFR": "Secured Overnight Financing Rate (overnight)",
    "DGS1MO": "Treasury constant maturity 1 month",
    "DGS3MO": "Treasury constant maturity 3 month",
    "DGS6MO": "Treasury constant maturity 6 month",
    "DGS1": "Treasury constant maturity 1 year",
    "DGS2": "Treasury constant maturity 2 year",
    "DGS5": "Treasury constant maturity 5 year",
    "DGS10": "Treasury constant maturity 10 year",
    "DGS30": "Treasury constant maturity 30 year",
}

# EUR short rate — ECB ESTR, keyless SDMX REST API.
ECB_ESTR = (
    "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata",
    "Euro short-term rate (ESTR), daily, 2019-10 onward",
)

# SEK short rate — Riksbank SWEA v1 API, keyless (rate-limited per IP).
RIKSBANK_STIBOR3M = (
    "SEDP3MSTIBORDELAYC",
    "STIBOR 3 Months (SEK), 1987-01 to 2020-07 — series closed when STIBOR was "
    "replaced by SWESTR, which the public SWEA v1 API does not expose",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  (cached) {dest}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {dest}  ({url})", flush=True)
    with urllib.request.urlopen(url) as src, dest.open("wb") as out:
        total = 0
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if total % (64 << 20) == 0:
                print(f"     {total / (1 << 20):.0f} MiB", flush=True)
    print(f"     done: {total / (1 << 20):.1f} MiB", flush=True)


def fetch_json(url: str) -> object:
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def record(manifest: dict, name: str, description: str, source: str,
           source_release: str, path: Path) -> None:
    manifest["datasets"].append({
        "name": name,
        "description": description,
        "source": source,
        "source_release": source_release,
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    })

def fetch_spx_snapshot() -> tuple[Path, str]:
    """Fetch the current SPX option chain via yfinance, tagged with the as-of date."""
    import pandas as pd
    import yfinance as yf

    ticker = yf.Ticker("^SPX")
    frames = []
    for exp in ticker.options:
        chain = ticker.option_chain(exp)
        calls = chain.calls.copy()
        calls["option_type"] = "call"
        calls["expiry"] = exp
        puts = chain.puts.copy()
        puts["option_type"] = "put"
        puts["expiry"] = exp
        frames.append(pd.concat([calls, puts], ignore_index=True))
    df = pd.concat(frames, ignore_index=True)
    dest = OPTIONS_DIR / "SPX_snapshot.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest, yf.__version__


def main() -> int:
    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": date.today().isoformat(),
        "datasets": [],
    }

    # Multi-date SPY options + underlying.
    for url, desc, name in (
        (*SPY_OPTIONS, "SPY_options.parquet"),
        (*SPY_UNDERLYING, "SPY_underlying.parquet"),
    ):
        dest = OPTIONS_DIR / name
        download(url, dest)
        record(manifest, dest.stem, desc, url,
               "lambdaclass/options_portfolio_backtester data-v1 (mirror of philippdubach/options-data, MIT)",
               dest)

    # USD discount / rate curve.
    for series, desc in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        dest = RATES_DIR / f"{series}.csv"
        download(url, dest)
        record(manifest, series, desc, url,
               "FRED (St. Louis Fed), full history, keyless fredgraph.csv", dest)

    # EUR short rate (ECB ESTR).
    url, desc = ECB_ESTR
    dest = RATES_DIR / "ESTR.csv"
    download(url, dest)
    record(manifest, "ESTR", desc, url, "ECB Data Portal (SDMX REST), keyless", dest)

    # SEK short rate (Riksbank STIBOR 3M).
    series, desc = RIKSBANK_STIBOR3M
    url = f"https://api.riksbank.se/swea/v1/Observations/{series}/1987-01-01/2020-12-31"
    dest = RATES_DIR / "STIBOR3M.csv"
    if not dest.exists():
        print(f"  -> {dest}  ({url})", flush=True)
        obs = fetch_json(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "value"])
            for row in obs:
                writer.writerow([row["date"], row["value"]])
    else:
        print(f"  (cached) {dest}", flush=True)
    record(manifest, "STIBOR3M", desc, url, "Riksbank SWEA v1 API, keyless", dest)

    # SPX options snapshot (yfinance).
    spx_dest, yf_version = fetch_spx_snapshot()
    record(manifest, "SPX_snapshot", "SPX option chain snapshot (all expiries, current)",
           "yfinance (Yahoo Finance) Ticker('^SPX')",
           f"yfinance {yf_version}", spx_dest)

    manifest_path = Path("data/manifest.json")
    # Wall-clock fields make every run differ; rewrite only if the dataset
    # records changed, so re-running the idempotent fetch keeps the tree clean.
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            old = None
        if old is not None and old.get("datasets") == manifest["datasets"]:
            print("\nManifest datasets unchanged; not rewritten")
            return 0
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
