"""Validation pipeline: one-command run, determinism, evidence tables."""

import pytest

from pricing.validation import BARS, evidence_table, run_all


@pytest.fixture(scope="module")
def fast_run():
    return run_all(fast=True)


def test_fast_run_covers_every_experiment(fast_run):
    assert set(fast_run["experiments"]) == {
        "benchmarks_parity", "mc_convergence", "mc_variance_reduction",
        "greeks_cross_method", "pde_convergence", "anchor_pricing",
        "anchor_pla", "calibration_synthetic", "calibration_snapshot",
    }
    assert fast_run["environment"]["QuantLib"]
    assert fast_run["input_checksums"]  # frozen-data manifest present


def test_two_consecutive_runs_identical_metrics():
    a = run_all(fast=True)["experiments"]
    b = run_all(fast=True)["experiments"]
    assert {k: v["metrics"] for k, v in a.items()} == {k: v["metrics"] for k, v in b.items()}


def test_all_bars_pass_in_fast_mode(fast_run):
    """Every hard bar the memo will cite holds even at reduced path counts."""
    table = evidence_table(fast_run)
    failed = table[table["verdict"] == "FAIL"]
    assert failed.empty, failed.to_string()


def test_evidence_table_rows_match_bars_and_metrics(fast_run):
    table = evidence_table(fast_run)
    n_metrics = sum(len(e["metrics"]) for e in fast_run["experiments"].values())
    assert len(table) == n_metrics
    defined = {m for bars in BARS.values() for m in bars}
    recorded = set(table.loc[table["verdict"] == "info", "metric"])
    # every metric without an explicit bar must be info, never FAIL
    assert recorded <= set(table["metric"])
    for _, row in table.iterrows():
        if row["metric"] not in defined:
            assert row["verdict"] == "info"


def test_pla_frtb_bar_explicit(fast_run):
    metrics = fast_run["experiments"]["anchor_pla"]["metrics"]
    assert metrics["pla_spearman"] > 0.80
    assert metrics["pla_ks_statistic"] < 0.09
