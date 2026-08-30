"""Render the two figures used in the README from committed, deterministic runs.

Usage: ``uv run python scripts/make_figures.py``

Writes ``docs/figures/mc_convergence.png`` and
``docs/figures/variance_reduction.png``. Every plotted number comes from the
same seeded, offline experiments the validation pipeline runs, so the figures
reproduce exactly: two consecutive runs produce byte-identical images.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pricing.monte_carlo.convergence import strong_convergence  # noqa: E402
from pricing.monte_carlo.variance_reduction import (  # noqa: E402
    mc_european_antithetic,
    mc_european_control_variate,
    mc_european_importance_sampling,
)

OUT = Path("docs/figures")

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 0.5, 0.2, 0.03
OTM_STRIKE = 130.0


def _convergence_figure() -> None:
    """Strong convergence of the Euler and Milstein schemes, with reference slopes."""
    n_paths = 200_000
    steps = [2, 4, 8, 16, 32]

    order_e, dts_e, err_e = strong_convergence(
        SPOT, TTE, VOL, RATE, n_paths, steps, seed=0, scheme="euler"
    )
    order_m, dts_m, err_m = strong_convergence(
        SPOT, TTE, VOL, RATE, n_paths, steps, seed=0, scheme="milstein"
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.loglog(dts_e, err_e, "o-", color="#c0392b", label=f"Euler (measured order {order_e:.2f})")
    ax.loglog(dts_m, err_m, "s-", color="#2c6fbb", label=f"Milstein (measured order {order_m:.2f})")

    # Reference slope lines anchored to the coarsest measured error.
    ax.loglog(dts_e, err_e[0] * np.sqrt(dts_e / dts_e[0]), "--", color="#c0392b", alpha=0.5, lw=1, label="order 0.5")
    ax.loglog(dts_m, err_m[0] * (dts_m / dts_m[0]), "--", color="#2c6fbb", alpha=0.5, lw=1, label="order 1")

    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel("pathwise error vs exact terminal value")
    ax.set_title("Strong convergence: pathwise accuracy of the Monte Carlo schemes")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "mc_convergence.png", dpi=150)
    plt.close(fig)


def _variance_reduction_figure() -> None:
    """Variance reduction ratio of the three techniques on the exact scheme."""
    n_paths = 150_000
    anti = mc_european_antithetic(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=n_paths, seed=0, scheme="exact")
    cv = mc_european_control_variate(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=n_paths, seed=0, scheme="exact")
    ism = mc_european_importance_sampling(SPOT, OTM_STRIKE, TTE, VOL, RATE, "call", n_paths=n_paths, seed=0, scheme="exact")

    ratios = [anti.variance_ratio, cv.variance_ratio, ism.variance_ratio]
    labels = ["Antithetic\nvariates", "Control\nvariate", "Importance\nsampling"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(labels, ratios, color=["#7f8c8d", "#2c6fbb", "#c0392b"])
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, ratio, f"{ratio:.1f}x", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("variance reduction ratio (cost-normalized)")
    ax.set_title("Monte Carlo variance reduction, exact scheme")
    ax.axhline(1.0, color="#666", lw=0.8, ls=":")
    ax.text(2.4, 1.02, "break-even (1.0x)", ha="right", va="bottom", fontsize=8, color="#666")
    ax.set_ylim(0, max(ratios) * 1.25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "variance_reduction.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _convergence_figure()
    _variance_reduction_figure()
    print(f"wrote {OUT / 'mc_convergence.png'} and {OUT / 'variance_reduction.png'}")


if __name__ == "__main__":
    main()
