# Runbook

Operating procedure for the numerical pricing and model-validation core.

## One-command run

```bash
uv sync                      # locked environment
uv run python scripts/run_all.py          # full run (~9 s)
uv run python scripts/run_all.py --fast   # reduced path counts for a quick check
```

The run is offline and deterministic: fixed seeds everywhere, no network calls,
and two consecutive runs produce identical metrics (enforced by
`tests/test_validation_pipeline.py`). It writes:

- `data/processed/run_manifest.json` — run mode, environment versions (Python,
  NumPy, pandas, Numba, JAX, QuantLib), SHA-256 checksums of every manifest
  dataset present locally, per-experiment runtime, and all result metrics.
- `data/processed/evidence_tables.csv` — one row per metric: component,
  measured value, acceptance bar, verdict (`pass` / `FAIL` / `info`). The
  script exits non-zero and prints the failing rows if any bar breaches.

The full-mode manifest is the committed artifact; regenerate it after any
engine change so the validation report's numbers stay reproducible.

## Data contracts

- Raw data lives in `data/raw/`, is gitignored (ADR-0003), and is fetched once
  via `scripts/fetch_data.py`. Contracts, sources, licenses: `docs/data-sources.md`.
- `data/manifest.json` records every dataset with source release, as-of date,
  SHA-256 checksum, and size. Each pipeline run re-verifies the checksums of
  all committed `processed/` artifacts; a mismatch sets
  `processed_checksums_ok = 0` and fails its evidence bar.
- Committed processed snapshots (`data/processed/`) are the only inputs the
  calibration snapshot experiment consumes; nothing after the initial fetch
  touches the network.

## Monitoring thresholds

Watch these evidence-table rows; they are the bars most sensitive to engine or
data drift:

| Metric | Bar | Component |
| --- | --- | --- |
| `pla_ks_statistic` | < 0.09 | anchor |
| `pla_spearman` | >= 0.80 | anchor |
| `bermudan_lsm_vs_pde_rel_diff` | <= 2% | anchor |
| `strong_order_euler` / `_milstein` | in [0.35,0.65] / [0.8,1.2] | monte_carlo |
| `spatial_order_vanilla_cn` | in [1.8, 2.9] | pde |
| `processed_checksums_ok` | == 1 | calibration |

A breach means an engine change altered behavior or an artifact was edited out
of band; do not widen the bar to go green.

## Rollback and re-run

1. Identify the offending commit: every engine and artifact change lands as its
   own commit; `git log --oneline` shows the change sequence.
2. Re-run from any prior state: `git checkout <sha>` then
   `uv run python scripts/run_all.py`. The manifest records which code produced
   it via the environment block and provenance entries.
3. Raw-data refresh (rare, breaks offline guarantees until re-frozen):
   `uv sync --group fetch && uv run --group fetch python scripts/fetch_data.py`,
   then re-run `scripts/calibrate_snapshot.py` and `run_spy_backtest.py` to
   regenerate committed processed snapshots, then `scripts/run_all.py`.

## Test suite

`uv run pytest -q` runs the full suite (165 tests, about 90 s). The fast
validation subset is `uv run pytest tests/test_validation_pipeline.py`.
