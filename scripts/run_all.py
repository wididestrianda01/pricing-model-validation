"""One-command deterministic validation run.

Executes every numerical experiment offline from frozen data, writes the run
manifest and the evidence tables under data/processed/, and prints the
verdict summary:

    uv run python scripts/run_all.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest import make_entry, register_processed

from pricing.validation import evidence_table, run_all

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="reduced path counts for a quick check")
    args = parser.parse_args()

    result = run_all(fast=args.fast)
    result["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")

    table = evidence_table(result)
    table_path = out_dir / "evidence_tables.csv"
    table.to_csv(table_path, index=False)

    entry = make_entry(
        "processed/run_manifest", "Deterministic full-run manifest (env, checksums, metrics)",
        manifest_path, ROOT, "pricing.validation.run_all", "repo",
    )

    register_processed(ROOT / "data" / "manifest.json", [entry])

    n_pass = int((table["verdict"] == "pass").sum())
    n_fail = int((table["verdict"] == "FAIL").sum())
    print(f"{len(table)} evidence rows: {n_pass} pass, {n_fail} FAIL, "
          f"{int((table['verdict'] == 'info').sum())} info")
    print(f"manifest -> {manifest_path.relative_to(ROOT)}")
    print(f"tables   -> {table_path.relative_to(ROOT)}")
    if n_fail:
        print(table[table["verdict"] == "FAIL"].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
