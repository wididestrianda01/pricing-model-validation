"""Multi-date SPY SABR calibration backtest over the frozen historical chain.

For each monthly observation date the pipeline extracts one ~30-day tenor
slice (same filtering/IV/parity machinery as the snapshot pipeline), fits
SABR, and collects parameters plus fit quality into a stability table.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from pricing.calibration.rates import load_zero_curve
from pricing.calibration.sabr import SABRFit, calibrate_sabr
from pricing.calibration.snapshot import (
    TARGET_TTES,
    extract_slices,
    select_slice_expiries,
)

COLUMN_RENAME = {"type": "option_type", "open_interest": "openInterest"}
MIN_QUOTES = 5


def _observation_dates(underlying: pd.DataFrame, start: str | pd.Timestamp,
                       end: str | pd.Timestamp) -> list[pd.Timestamp]:
    """Last trading day of each month in [start, end], from the underlying table."""
    u = underlying.copy()
    u["date"] = pd.to_datetime(u["date"])
    u = u[(u["date"] >= pd.Timestamp(start)) & (u["date"] <= pd.Timestamp(end))]
    return list(u.groupby(u["date"].dt.to_period("M"))["date"].max())


def _choose_expiry(dataset, obs: pd.Timestamp, *,
                   target_days: int = 30, min_days: int = 15,
                   max_days: int = 90) -> pd.Timestamp | None:
    """Closest listed expiry to the target tenor tradable on this date."""
    tbl = dataset.to_table(
        filter=(ds.field("date") == pc.scalar(obs))
        & (ds.field("expiration") > pc.scalar(obs + pd.Timedelta(days=min_days))),
        columns=["expiration"],
    )
    if tbl.num_rows == 0:
        return None
    expirations = pd.DatetimeIndex(pd.Series(tbl.column("expiration").to_pylist()).unique())
    tte_days = np.asarray((expirations - obs).days)
    ok = tte_days <= max_days
    if not ok.any():
        return None
    return expirations[ok][np.abs(tte_days[ok] - target_days).argmin()]


def load_date_slice(dataset, obs: pd.Timestamp, expiry: pd.Timestamp,
                    rates_dir: Path | str) -> pd.DataFrame:
    """Extract the filtered calibration slice for one observation date + expiry."""
    chain = dataset.to_table(
        filter=(ds.field("date") == pc.scalar(obs))
        & (ds.field("expiration") == pc.scalar(expiry)),
        columns=["strike", "type", "bid", "ask", "volume", "open_interest"],
    ).to_pandas().rename(columns=COLUMN_RENAME)
    if chain.empty:
        return pd.DataFrame()
    chain["expiry"] = expiry.strftime("%Y-%m-%d")
    discount_fn = load_zero_curve(rates_dir, obs)
    return extract_slices(chain, obs, discount_fn)


def fit_backtest_date(slices: pd.DataFrame) -> dict | None:
    """SABR-fit the chosen tenor slice; returns the result row or None."""
    if slices.empty:
        return None
    label = select_slice_expiries(slices, TARGET_TTES)["30d"]
    grp = slices[slices["expiry"] == label]
    if len(grp) < MIN_QUOTES:
        return None
    fit: SABRFit = calibrate_sabr(
        grp["strike"].to_numpy(), grp["iv"].to_numpy(),
        float(grp["forward"].iloc[0]), float(grp["tte"].iloc[0]),
    )
    return {
        "expiry": label,
        "forward": float(grp["forward"].iloc[0]),
        "discount": float(grp["discount"].iloc[0]),
        "n_quotes": len(grp),
        **asdict(fit),
    }


def run_spy_backtest(options_parquet: Path | str, underlying_parquet: Path | str,
                     rates_dir: Path | str, *, start: str, end: str) -> pd.DataFrame:
    """Run the SABR backtest over month-end dates in [start, end]."""
    dataset = ds.dataset(str(options_parquet))
    underlying = pd.read_parquet(underlying_parquet, columns=["date"])
    rows = []
    for obs in _observation_dates(underlying, start, end):
        expiry = _choose_expiry(dataset, obs)
        if expiry is None:
            continue
        slices = load_date_slice(dataset, obs, expiry, rates_dir)
        row = fit_backtest_date(slices)
        if row is not None:
            row["obs_date"] = obs.date().isoformat()
            rows.append(row)
    return pd.DataFrame(rows)
