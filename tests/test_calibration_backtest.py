"""Tests for the multi-date SABR backtest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pricing.calibration.backtest import _observation_dates, fit_backtest_date
from pricing.calibration.sabr import sabr_vol

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _synthetic_slices(forward=500.0, tte=30 / 365, alpha=0.12, rho=-0.3,
                      nu=2.0) -> pd.DataFrame:
    """One clean tenor slice in extract_slices output format."""
    strikes = forward * np.linspace(0.9, 1.1, 13)
    ivs = sabr_vol(forward, strikes, tte, alpha, 1.0, rho, nu)
    return pd.DataFrame({
        "expiry": "2024-03-01",
        "tte": tte,
        "strike": strikes,
        "forward": forward,
        "discount": np.exp(-0.05 * tte),
        "mid": np.full(len(strikes), 10.0),
        "option_type": np.where(strikes >= forward, "call", "put"),
        "iv": ivs,
    })


class TestObservationDates:
    def test_month_ends_only(self):
        u = pd.DataFrame({"date": pd.to_datetime([
            "2024-01-03", "2024-01-31", "2024-02-15", "2024-02-29", "2024-03-05",
        ])})
        dates = _observation_dates(u, "2024-01", "2024-03")
        assert [d.strftime("%Y-%m-%d") for d in dates] == ["2024-01-31", "2024-02-29"]

    def test_range_filtering(self):
        u = pd.DataFrame({"date": pd.to_datetime(["2023-12-29", "2024-06-28", "2025-02-28"])})
        dates = _observation_dates(u, "2024-01", "2024-12")
        assert [d.strftime("%Y-%m-%d") for d in dates] == ["2024-06-28"]


class TestFitBacktestDate:
    def test_recovers_synthetic_parameters(self):
        row = fit_backtest_date(_synthetic_slices(alpha=0.12, rho=-0.3, nu=2.0))
        assert row is not None
        assert row["converged"]
        assert row["alpha"] == pytest.approx(0.12, rel=0.05)
        assert row["rho"] == pytest.approx(-0.3, abs=0.05)
        assert row["nu"] == pytest.approx(2.0, rel=0.1)
        assert row["rmse"] < 1e-8

    def test_too_few_quotes_returns_none(self):
        thin = _synthetic_slices().iloc[:4]
        assert fit_backtest_date(thin) is None

    def test_empty_slice_returns_none(self):
        assert fit_backtest_date(pd.DataFrame()) is None


@pytest.mark.integration
class TestCommittedBacktestArtifact:
    """Offline guard over the committed backtest table."""

    TABLE = pd.read_csv(PROCESSED / "spy_sabr_backtest.csv")

    def test_covers_many_dates_with_quality_fits(self):
        assert len(self.TABLE) >= 60
        assert self.TABLE["obs_date"].is_monotonic_increasing
        assert (self.TABLE["rmse"] < 0.03).all()  # within 3 vol pts everywhere
        assert (self.TABLE["n_quotes"] >= 5).all()

    def test_parameters_in_sane_ranges(self):
        assert self.TABLE["atm_vol"].between(0.03, 2.0).all()
        assert self.TABLE["rho"].between(-0.999, 0.999).all()
        assert self.TABLE["nu"].ge(0.0).all()
