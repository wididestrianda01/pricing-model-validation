"""Regenerate the committed SPX calibration snapshot from frozen raw data.

Reads only gitignored raw inputs under data/raw/ and rewrites the committed
processed artifacts under data/processed/. Run once per data refresh:

    uv run python scripts/calibrate_snapshot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from manifest import make_entry, register_processed, sha256

from pricing.calibration import snapshot as snap
from pricing.calibration.snapshot import run_snapshot_calibration

ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2026-08-24"  # manifest as-of of the frozen SPX_snapshot.csv

def main() -> None:
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    slices, fits = run_snapshot_calibration(
        ROOT / "data" / "raw" / "options" / "SPX_snapshot.csv",
        ROOT / "data" / "raw" / "rates",
        as_of=AS_OF,
    )

    slices_path = out_dir / "spx_calibration_slices.csv"
    slices.to_csv(slices_path, index=False)

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    raw_entry = next(d for d in manifest["datasets"] if d["name"] == "SPX_snapshot")
    results = {
        "as_of": AS_OF,
        "source": {
            "dataset": "SPX_snapshot",
            "provider": raw_entry["source"],
            "raw_sha256": raw_entry["sha256"],
            "processed_slices_sha256": sha256(slices_path),
            "filter_cuts": {
                "max_rel_spread": snap.MAX_REL_SPREAD,
                "min_liquidity_max_volume_oi": snap.MIN_LIQUIDITY,
                "moneyness_band": list(snap.MONEYNESS_BAND),
                "iv_bounds": list(snap.IV_BOUNDS),
                "target_tenors_days": {
                    label: round(tte * 365) for label, tte in snap.TARGET_TTES.items()
                },
                "max_quotes_per_slice_for_heston": snap.MAX_QUOTES_PER_SLICE,
            },
        },
        **fits,
    }
    results_path = out_dir / "spx_calibration_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")

    # Upsert provenance entries into data/manifest.json (idempotent).
    provenance = "derived from SPX_snapshot as-of " + AS_OF
    register_processed(ROOT / "data" / "manifest.json", [
        make_entry(
            "processed/spx_calibration_slices",
            "Processed SPX calibration slices (committed, offline-reproducible)",
            slices_path, ROOT,
            source=provenance,
            source_release=raw_entry["source_release"],
        ),
        make_entry(
            "processed/spx_calibration_results",
            "Fitted SABR/Heston parameters + provenance (committed)",
            results_path, ROOT,
            source=provenance,
            source_release=raw_entry["source_release"],
        ),
    ])

    for expiry, f in fits["sabr_by_expiry"].items():
        print(f"SABR {expiry}: atm={f['atm_vol']:.4f} rmse={f['rmse']:.5f}")
    h = fits["heston"]
    print(f"Heston: iv_rmse={h['iv_rmse']:.5f} v0={h['v0']:.4f} kappa={h['kappa']:.2f} "
          f"theta={h['theta']:.4f} xi={h['xi']:.2f} rho={h['rho']:.2f}")


if __name__ == "__main__":
    main()
