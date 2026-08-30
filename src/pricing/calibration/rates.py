"""Zero-discount curve from the frozen FRED rate CSVs.

Tenor set: SOFR (overnight), DGS3MO, DGS6MO, DGS1, DGS2 — continuously
compounded zero rates, linearly interpolated in T, flat-extrapolated beyond
the end points. Each series uses its latest observation on or before as_of.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

_TENOR_FILES: list[tuple[float, str, str]] = [
    (1 / 365, "SOFR.csv", "SOFR"),
    (0.25, "DGS3MO.csv", "DGS3MO"),
    (0.5, "DGS6MO.csv", "DGS6MO"),
    (1.0, "DGS1.csv", "DGS1"),
    (2.0, "DGS2.csv", "DGS2"),
]


def load_zero_curve(
    rates_dir: Path | str, as_of: pd.Timestamp
) -> Callable[[float], float]:
    """Return a discount_factor(tte) function."""
    rates_dir = Path(rates_dir)
    nodes: list[tuple[float, float]] = []
    for tenor, fname, col in _TENOR_FILES:
        df = pd.read_csv(rates_dir / fname)
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        obs = df[df["observation_date"] <= as_of]
        if obs.empty:
            raise ValueError(f"{fname}: no observation on or before {as_of.date()}")
        rate = float(obs[col].iloc[-1]) / 100.0
        if np.isnan(rate):
            raise ValueError(f"{fname}: latest usable rate is NaN")
        nodes.append((tenor, rate))

    nodes.sort()
    ts = np.array([t for t, _ in nodes])
    rs = np.array([r for _, r in nodes])

    def zero_rate(tte: float) -> float:
        return float(np.interp(tte, ts, rs))  # np.interp flats outside range

    def discount_factor(tte: float) -> float:
        return float(np.exp(-zero_rate(tte) * tte))

    return discount_factor
