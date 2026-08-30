"""Evidence tables: turn pipeline metrics into the outcomes-analysis rows the memo cites.

Every metric gets one row per experiment: component, metric, measured value,
the bar it is judged against, and a pass/fail verdict. Metrics with no hard
bar are recorded as ``info`` — degradation results (digital/barrier/PSOR
orders) are findings, not failures.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pandas as pd


class Bar(NamedTuple):
    """Acceptance bar: human-readable text plus an optional pass predicate."""

    text: str
    check: Callable[[float], bool] | None = None


# component -> {metric: Bar}
BARS: dict[str, dict[str, Bar]] = {
    "benchmarks": {
        "bs_vs_quantlib_max_abs_diff": Bar("<= 1e-10", lambda v: v <= 1e-10),
    },
    "monte_carlo": {
        "weak_order_euler": Bar("in [0.6, 1.5]", lambda v: 0.6 <= v <= 1.5),
        "weak_order_milstein": Bar("in [0.6, 1.5]", lambda v: 0.6 <= v <= 1.5),
        "strong_order_euler": Bar("in [0.35, 0.65]", lambda v: 0.35 <= v <= 0.65),
        "strong_order_milstein": Bar("in [0.8, 1.2]", lambda v: 0.8 <= v <= 1.2),
        "se_scaling_slope": Bar("in [-0.6, -0.4]", lambda v: -0.6 <= v <= -0.4),
        "antithetic_variance_ratio": Bar("> 1", lambda v: v > 1.0),
        "control_variate_variance_ratio": Bar("> 1", lambda v: v > 1.0),
        "importance_sampling_variance_ratio": Bar("> 1", lambda v: v > 1.0),
        "antithetic_unbiased": Bar("== 1 (within 4 SE)", lambda v: v == 1.0),
        "control_variate_unbiased": Bar("== 1 (within 4 SE)", lambda v: v == 1.0),
        "importance_sampling_unbiased": Bar("== 1 (within 4 SE)", lambda v: v == 1.0),
    },
    "greeks": {
        "bump_delta_rel_err": Bar("<= 5% vs closed form", lambda v: v <= 0.05),
        "bump_vega_rel_err": Bar("<= 15% vs closed form", lambda v: v <= 0.15),
        "pathwise_delta_rel_err": Bar("<= 5% vs closed form", lambda v: v <= 0.05),
        "pathwise_vega_rel_err": Bar("<= 15% vs closed form", lambda v: v <= 0.15),
        "likelihood_ratio_delta_rel_err": Bar("<= 5% vs closed form", lambda v: v <= 0.05),
        "likelihood_ratio_vega_rel_err": Bar("<= 15% vs closed form", lambda v: v <= 0.15),
    },
    "pde": {
        "spatial_order_vanilla_cn": Bar("in [1.8, 2.9]", lambda v: 1.8 <= v <= 2.9),
        "temporal_order_crank_nicolson": Bar("in [1.7, 2.6]", lambda v: 1.7 <= v <= 2.6),
        "temporal_order_implicit": Bar("in [0.7, 1.3]", lambda v: 0.7 <= v <= 1.3),
        # Degradation cases: the reduced order IS the delivered result.
        "spatial_order_digital": Bar("recorded (kink degrades order)"),
        "spatial_order_barriers": Bar("recorded (barrier discontinuity)"),
        "early_exercise_order_psor": Bar("recorded (free-boundary kink)"),
    },
    "anchor": {
        "bermudan_lsm_vs_pde_rel_diff": Bar("<= 2% engine agreement", lambda v: v <= 0.02),
        "european_lsm_vs_jamshidian_rel_diff": Bar("<= 1% vs Jamshidian", lambda v: v <= 0.01),
        "pla_spearman": Bar(">= 0.80 FRTB bar", lambda v: v >= 0.80),
        "pla_ks_statistic": Bar("< 0.09 FRTB bar", lambda v: v < 0.09),
    },
    "calibration": {
        "sabr_alpha_rel_err": Bar("<= 2e-2 recovery", lambda v: v <= 2e-2),
        "sabr_rho_abs_err": Bar("<= 2e-2 recovery", lambda v: v <= 2e-2),
        "sabr_nu_rel_err": Bar("<= 2e-2 recovery", lambda v: v <= 2e-2),
        "heston_iv_rmse": Bar("<= 1e-6 on synthetic surface", lambda v: v <= 1e-6),
        "processed_checksums_ok": Bar("== 1 (all artifacts verified)", lambda v: v == 1.0),
    },
}

_EXPERIMENT_COMPONENT = {
    "benchmarks_parity": "benchmarks",
    "mc_convergence": "monte_carlo",
    "mc_variance_reduction": "monte_carlo",
    "greeks_cross_method": "greeks",
    "pde_convergence": "pde",
    "anchor_pricing": "anchor",
    "anchor_pla": "anchor",
    "calibration_synthetic": "calibration",
    "calibration_snapshot": "calibration",
}


def evidence_table(run_result: dict) -> pd.DataFrame:
    """One row per (experiment, metric): measured vs bar with a verdict."""
    rows = []
    for exp_name, exp in run_result["experiments"].items():
        component = _EXPERIMENT_COMPONENT[exp_name]
        bars = BARS.get(component, {})
        for metric, value in exp["metrics"].items():
            if metric in bars:
                bar = bars[metric]
                verdict = ("pass" if bar.check(value) else "FAIL") if bar.check else "info"
                bar_text = bar.text
            else:
                bar_text, verdict = "-", "info"
            rows.append({
                "component": component,
                "experiment": exp_name,
                "metric": metric,
                "measured": round(value, 6),
                "bar": bar_text,
                "verdict": verdict,
            })
    return pd.DataFrame(rows)
