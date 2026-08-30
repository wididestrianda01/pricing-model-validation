"""Run the multi-date SPY SABR calibration backtest and commit the result.

    uv run python scripts/run_spy_backtest.py [start] [end]

Defaults to 2019-01 through 2025-12 month-ends. Reads only frozen raw data;
writes the committed artifacts data/processed/spy_sabr_backtest.csv and
spy_sabr_backtest_summary.json, registered in data/manifest.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from manifest import make_entry, register_processed, sha256

from pricing.calibration.backtest import run_spy_backtest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2019-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2025-12"

    res = run_spy_backtest(
        ROOT / "data" / "raw" / "options" / "SPY_options.parquet",
        ROOT / "data" / "raw" / "options" / "SPY_underlying.parquet",
        ROOT / "data" / "raw" / "rates",
        start=start,
        end=end,
    )
    out_dir = ROOT / "data" / "processed"
    table_path = out_dir / "spy_sabr_backtest.csv"
    res.to_csv(table_path, index=False)

    print(f"wrote {table_path.relative_to(ROOT)} ({len(res)} dates)")
    if res.empty:
        return
    key_cols = ["obs_date", "expiry", "n_quotes", "atm_vol", "alpha", "rho",
                "nu", "rmse"]
    summary = {
        "n_dates": len(res),
        "first_obs": res["obs_date"].iloc[0],
        "last_obs": res["obs_date"].iloc[-1],
        "stability": res[key_cols].describe().loc[["mean", "std", "min", "max"]]
        .round(4).to_dict(),
        "backtest_sha256": sha256(table_path),
    }
    summary_path = out_dir / "spy_sabr_backtest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path.read_text())

    manifest_path = ROOT / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw = {d["name"]: d for d in manifest["datasets"]}
    provenance = (
        f"derived from SPY_options/SPY_underlying as-of {summary['last_obs']}"
    )
    register_processed(manifest_path, [
        make_entry(
            "processed/spy_sabr_backtest",
            "Multi-date SPY SABR parameter trajectory + fit-quality table",
            table_path, ROOT,
            source=provenance,
            source_release=raw["SPY_options"]["source_release"],
        ),
        make_entry(
            "processed/spy_sabr_backtest_summary",
            "Stability summary (describe stats) over the SABR backtest table",
            summary_path, ROOT,
            source=provenance,
            source_release=raw["SPY_options"]["source_release"],
        ),
    ])


if __name__ == "__main__":
    main()
