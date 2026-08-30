"""Tests for the snapshot calibration pipeline.

Unit tests build a synthetic chain around a known SABR smile; the integration
test consumes only committed artifacts under data/processed/ (offline).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pricing.calibration.snapshot import (
    extract_slices,
    parity_forward,
    select_slice_expiries,
)


def _flat_discount(rate: float = 0.04):
    """Stub curve matching the synthetic chain's own pricing rate."""
    return lambda tte: float(np.exp(-rate * tte))

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _synthetic_chain(forward: float = 5000.0, tte: float = 0.1,
                     alpha=0.15, beta=1.0, rho=-0.5, nu=1.5) -> pd.DataFrame:
    """Chain of OTM quotes priced off a SABR smile at a 4% flat rate."""
    from pricing.benchmarks.black_scholes import black_scholes

    df = np.exp(-0.04 * tte)
    strikes = forward * np.concatenate([np.linspace(0.85, 0.98, 8),
                                        np.linspace(1.00, 1.15, 8)])
    ivs = np.array([_sabr_vol(forward, k, tte, alpha, beta, rho, nu) for k in strikes])
    rows = []
    for k, iv in zip(strikes, ivs):
        for otype in ("call", "put"):
            px = black_scholes(forward * df, k, tte, float(iv), 0.0, otype)
            spread = max(0.02 * px, 0.05)
            bid, ask = px - spread / 2, px + spread / 2
            rows.append({"strike": k, "expiry": "2026-09-21",
                         "option_type": otype, "bid": bid, "ask": ask,
                         "volume": 100.0, "openInterest": 50.0})
    return pd.DataFrame(rows)


def _sabr_vol(f, k, t, a, b, r, n):
    from pricing.calibration.sabr import sabr_vol
    return float(sabr_vol(f, np.array([k]), t, a, b, r, n)[0])


class TestParityForward:
    def test_recovers_known_forward(self):
        # Quotes constructed exactly on C - P = DF (F - K).
        df, fwd = 0.99726, 5000.0
        rng = np.random.default_rng(0)
        strikes = np.linspace(4500.0, 5500.0, 11)
        base = rng.uniform(10.0, 200.0, len(strikes))
        calls = base + df * (fwd - strikes)
        puts = base
        est = parity_forward(strikes, calls, puts, df)
        assert est == pytest.approx(fwd, abs=1e-6)

    def test_too_few_pairs_rejected(self):
        with pytest.raises(ValueError, match="parity pairs"):
            parity_forward(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                           np.array([1.0, 2.0]), 0.99)


class TestExtractSlices:
    def test_extracts_quotes_with_correct_ivs_and_forward(self):
        forward, tte, alpha, rho, nu = 5000.0, 30 / 365, 0.15, -0.5, 1.5
        chain = _synthetic_chain(forward, tte, alpha=alpha, rho=rho, nu=nu)
        discount_fn = _flat_discount()
        slices = extract_slices(chain, pd.Timestamp("2026-09-01"), discount_fn)
        assert not slices.empty
        assert slices["forward"].iloc[0] == pytest.approx(
            forward * discount_fn(tte), rel=5e-3)
        assert slices["iv"].min() > 0.03 and slices["iv"].max() < 3.0
        # Smile is downward sloping: low-strike IVs exceed high-strike IVs.
        lo = slices[slices.strike < forward]["iv"].mean()
        hi = slices[slices.strike > forward]["iv"].mean()
        assert lo > hi

    def test_stale_wide_quotes_are_dropped(self):
        chain = _synthetic_chain()
        junk = chain.iloc[[0]].copy()
        junk["bid"], junk["ask"] = 0.01, 900.0  # absurd spread
        filtered = pd.concat([chain, junk], ignore_index=True)
        discount_fn = _flat_discount()
        a = extract_slices(filtered, pd.Timestamp("2026-09-01"), discount_fn)
        b = extract_slices(chain, pd.Timestamp("2026-09-01"), discount_fn)
        assert len(a) == len(b)

    def test_illiquid_quotes_are_dropped(self):
        chain = _synthetic_chain()
        chain["volume"] = 1.0
        chain["openInterest"] = 1.0
        discount_fn = _flat_discount()
        assert extract_slices(chain, pd.Timestamp("2026-09-01"), discount_fn).empty


class TestSelectSliceExpiries:
    def test_picks_closest_tenor(self):
        slices = pd.DataFrame({
            "expiry": ["2026-09-05", "2026-12-31"],
            "tte": [12 / 365, 129 / 365],
        })
        chosen = select_slice_expiries(slices, {"a": 10 / 365, "b": 120 / 365})
        assert chosen == {"a": "2026-09-05", "b": "2026-12-31"}


@pytest.mark.integration
class TestCommittedSnapshotArtifacts:
    """Offline regression guard over the committed processed snapshot."""

    RESULTS = json.loads((PROCESSED / "spx_calibration_results.json").read_text())

    def test_provenance_recorded(self):
        src = self.RESULTS["source"]
        assert src["raw_sha256"] and len(src["raw_sha256"]) == 64
        assert self.RESULTS["as_of"] == "2026-08-24"

    def test_three_slices_present_with_quality_fits(self):
        fits = self.RESULTS["sabr_by_expiry"]
        assert len(fits) == 3
        for f in fits.values():
            assert f["converged"]
            assert f["rmse"] < 0.03  # within 3 vol pts on real quotes
            assert 0.05 < f["atm_vol"] < 0.60

    def test_heston_surface_fit_quality(self):
        h = self.RESULTS["heston"]
        assert h["iv_rmse"] < 0.04  # within 4 vol pts surface-wide
        assert 0.001 <= h["v0"] <= 1.0
        assert -1 < h["rho"] < 1
