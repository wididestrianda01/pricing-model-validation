"""Validation-report pipeline: deterministic one-command run of every experiment.

Each experiment returns a flat metrics dict; run_all executes them in dependency
order, records per-experiment runtime, and returns everything the evidence
tables and the SR 26-2 memo cite. ``fast=True`` shrinks path counts for the
offline pytest subset; bars and seeds are identical either way.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pricing.anchor import (
    BermudanSwaption,
    HullWhiteParams,
    lsm_bermudan_swaption,
    pde_bermudan_swaption,
)
from pricing.anchor.pla import pla_consistency
from pricing.benchmarks import black_76, black_scholes
from pricing.benchmarks.hull_white import hw_jamshidian_swaption
from pricing.benchmarks.quantlib_benchmarks import (
    ql_barrier,
    ql_black_76,
    ql_black_scholes,
)
from pricing.greeks.studies import bias_variance_sweep, cross_method_greeks
from pricing.monte_carlo import (
    mc_european_antithetic,
    mc_european_control_variate,
    mc_european_importance_sampling,
    se_scaling,
    strong_convergence,
    weak_convergence,
)
from pricing.pde import early_exercise_order, spatial_order, temporal_order

ROOT = Path(__file__).resolve().parents[3]

# Canonical market data used by every equity-option experiment.
SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 0.5, 0.2, 0.03
QL_TTE = 73 / 365  # whole Act/365 days so the QuantLib seam matches the closed form exactly

def _benchmarks() -> dict[str, float]:
    bs_mine = black_scholes(SPOT, STRIKE, QL_TTE, VOL, RATE, "call")
    bs_ql = ql_black_scholes(SPOT, STRIKE, QL_TTE, VOL, RATE, "call")
    fwd = 100.0 * float(np.exp(RATE * QL_TTE))
    b76_mine = black_76(fwd, STRIKE, QL_TTE, VOL, RATE, "call")
    b76_ql = ql_black_76(fwd, STRIKE, QL_TTE, VOL, RATE, "call")
    return {
        "bs_vs_quantlib_max_abs_diff": max(abs(bs_mine - bs_ql), abs(b76_mine - b76_ql)),
        "bs_call": bs_mine,
        "black76_call": b76_mine,
    }


def _mc_convergence(fast: bool) -> dict[str, float]:
    n_weak, n_strong = (200_000, 50_000) if fast else (1_000_000, 200_000)
    weak_e, _, _ = weak_convergence(SPOT, STRIKE, TTE, VOL, RATE, "call", n_weak, [1, 2, 4, 8, 16], seed=0, scheme="euler")
    weak_m, _, _ = weak_convergence(SPOT, STRIKE, TTE, VOL, RATE, "call", n_weak, [1, 2, 4, 8, 16], seed=0, scheme="milstein")
    strong_e, _, _ = strong_convergence(SPOT, TTE, VOL, RATE, n_strong, [2, 4, 8, 16, 32], seed=0, scheme="euler")
    strong_m, _, _ = strong_convergence(SPOT, TTE, VOL, RATE, n_strong, [2, 4, 8, 16, 32], seed=0, scheme="milstein")
    slope, _, _ = se_scaling(SPOT, STRIKE, TTE, VOL, RATE, "call", [10_000, 40_000, 160_000], seed=0, scheme="exact")
    return {
        "weak_order_euler": weak_e,
        "weak_order_milstein": weak_m,
        "strong_order_euler": strong_e,
        "strong_order_milstein": strong_m,
        "se_scaling_slope": slope,
    }


def _variance_reduction(fast: bool) -> dict[str, float]:
    n_small = 40_000 if fast else 100_000
    n_cv = 60_000 if fast else 200_000
    n_is = 80_000 if fast else 400_000
    bs_atm = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    strike_otm = 130.0
    bs_otm = black_scholes(SPOT, strike_otm, TTE, VOL, RATE, "call")
    anti = mc_european_antithetic(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=n_small, n_steps=1, seed=0, scheme="exact")
    cv = mc_european_control_variate(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=n_cv, n_steps=1, seed=0, scheme="exact")
    ism = mc_european_importance_sampling(SPOT, strike_otm, TTE, VOL, RATE, "call", n_paths=n_is, seed=0, scheme="exact")
    return {
        "antithetic_variance_ratio": anti.variance_ratio,
        "control_variate_variance_ratio": cv.variance_ratio,
        "importance_sampling_variance_ratio": ism.variance_ratio,
        "antithetic_unbiased": float(abs(anti.price - bs_atm) < 4 * anti.std_error),
        "control_variate_unbiased": float(abs(cv.price - bs_atm) < 4 * cv.std_error),
        "importance_sampling_unbiased": float(abs(ism.price - bs_otm) < 4 * ism.std_error),
    }


def _greeks(fast: bool) -> dict[str, float]:
    n_paths = 50_000 if fast else 200_000
    methods = cross_method_greeks(SPOT, STRIKE, 1.0, 0.2, 0.05, "call", n_paths=n_paths, seed=0)
    cf = methods["closed_form"]
    out: dict[str, float] = {}
    for name in ("bump", "pathwise", "likelihood_ratio"):
        for greek, tol in (("delta", 0.05), ("vega", 0.15)):
            rel = abs(methods[name][greek] - cf[greek]) / abs(cf[greek])
            out[f"{name}_{greek}_rel_err"] = rel
            out[f"{name}_{greek}_within_tol"] = float(rel < tol)
    hs, biases, variances = bias_variance_sweep(
        SPOT, STRIKE, 1.0, 0.2, 0.05, "call",
        h_list=[0.5, 1.0, 2.0, 4.0], n_paths=n_paths // 2, n_replications=8 if fast else 16, scheme="exact",
    )
    rmse = np.sqrt(np.asarray(biases) ** 2 + np.asarray(variances))
    best = int(np.argmin(rmse))
    out["optimal_bump_h"] = float(hs[best])
    out["optimal_bump_rmse"] = float(rmse[best])
    return out


def _pde(fast: bool) -> dict[str, float]:
    spaces: tuple[int, ...] = (25, 50, 100, 200) if fast else (50, 100, 200, 400)
    n_time = 200 if fast else 400
    sp_vanilla, errs_sp, _ = spatial_order(SPOT, STRIKE, TTE, VOL, RATE, "put", scheme="cn", n_space_list=spaces, n_time=n_time)
    ot_cn, _, _ = temporal_order(SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="cn", n_space=2000, n_time_list=(5, 10, 20, 40))
    ot_imp, _, _ = temporal_order(SPOT, STRIKE, TTE, VOL, RATE, "call", scheme="implicit", n_space=2000, n_time_list=(5, 10, 20, 40))
    sp_digital, _, _ = spatial_order(SPOT, STRIKE, TTE, VOL, RATE, "put", scheme="cn", payoff="digital", n_space_list=spaces, n_time=n_time)
    barrier_tte = 73 / 365
    ref_barrier = ql_barrier(SPOT, STRIKE, barrier_tte, VOL, RATE, 90.0, option_type="put")
    sp_barrier, _, _ = spatial_order(
        SPOT, STRIKE, barrier_tte, VOL, RATE, "put", scheme="cn", barrier=90.0,
        n_space_list=spaces, n_time=n_time, reference=ref_barrier,
    )
    ee_order, _, _ = early_exercise_order(SPOT, STRIKE, TTE, VOL, RATE, "put", n_space_list=spaces[:-1], n_time=n_time)
    richardson_gain = float(errs_sp[0] / errs_sp[-1]) if errs_sp[-1] > 0 else float("inf")
    return {
        "spatial_order_vanilla_cn": sp_vanilla,
        "temporal_order_crank_nicolson": ot_cn,
        "temporal_order_implicit": ot_imp,
        "spatial_order_digital": sp_digital,
        "spatial_order_barriers": sp_barrier,
        "early_exercise_order_psor": ee_order,
        "grid_refinement_error_reduction": richardson_gain,
        "barrier_reference_price": ref_barrier,
    }


_ANCHOR_PARAMS = HullWhiteParams(r0=0.03, a=0.05, sigma=0.01)
_ANCHOR_SWAPTION = BermudanSwaption(
    strike_rate=0.032,
    taus=np.full(10, 0.5),
    pay_times=5.0 + np.arange(1, 11) * 0.5,
    exercise_times=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
)


def _anchor_pricing(fast: bool) -> dict[str, float]:
    n_paths = 100_000 if fast else 300_000
    lsm = lsm_bermudan_swaption(_ANCHOR_SWAPTION, _ANCHOR_PARAMS, n_paths=n_paths)
    pde = pde_bermudan_swaption(_ANCHOR_SWAPTION, _ANCHOR_PARAMS)
    euro_lsm = lsm_bermudan_swaption(
        BermudanSwaption(strike_rate=0.032, taus=_ANCHOR_SWAPTION.taus,
                         pay_times=_ANCHOR_SWAPTION.pay_times, exercise_times=np.array([5.0])),
        _ANCHOR_PARAMS, n_paths=n_paths,
    )
    jam = hw_jamshidian_swaption(5.0, [float(t) for t in _ANCHOR_SWAPTION.pay_times], 0.032,
                                 _ANCHOR_PARAMS.a, _ANCHOR_PARAMS.sigma, _ANCHOR_PARAMS.r0)
    return {
        "bermudan_lsm_price": lsm.price,
        "bermudan_lsm_std_error": lsm.std_error,
        "bermudan_pde_price": pde,
        "bermudan_lsm_vs_pde_rel_diff": abs(lsm.price - pde) / abs(pde),
        "european_lsm_price": euro_lsm.price,
        "european_jamshidian_price": jam,
        "european_lsm_vs_jamshidian_rel_diff": abs(euro_lsm.price - jam) / abs(jam),
    }


def _anchor_pla(_: bool) -> dict[str, float]:
    res = pla_consistency(_ANCHOR_SWAPTION, _ANCHOR_PARAMS)
    return {
        "pla_spearman": res.spearman,
        "pla_ks_statistic": res.ks_statistic,
        "n_scenarios": float(len(res.hpl)),
    }


def _calibration_synthetic(fast: bool) -> dict[str, float]:
    del fast  # fits are already seconds-scale; same config both modes
    from pricing.benchmarks.heston import heston
    from pricing.calibration.heston_fit import calibrate_heston
    from pricing.calibration.sabr import calibrate_sabr, sabr_vol

    forward, tte, true = 5500.0, 0.4, {"alpha": 0.35, "rho": -0.65, "nu": 2.1}
    strikes = forward * np.linspace(0.75, 1.25, 15)
    iv = sabr_vol(forward, strikes, tte, true["alpha"], 1.0, true["rho"], true["nu"])
    fit = calibrate_sabr(strikes, iv, forward, tte, beta=1.0)

    spot, rate = 4500.0, 0.045
    true_h = {"v0": 0.035, "kappa": 2.0, "theta": 0.05, "xi": 0.55, "rho": -0.6}
    ks, ts, prices = [], [], []
    for m_tte in (0.25, 0.5):
        for m in (0.85, 0.95, 1.0, 1.05, 1.15):
            k = spot * m
            ks.append(k)
            ts.append(m_tte)
            prices.append(heston(spot, k, m_tte, true_h["v0"], true_h["kappa"], true_h["theta"], true_h["xi"], true_h["rho"], rate))
    hfit = calibrate_heston(spot, np.array(ks), np.array(ts), np.array(prices), rate)
    return {
        "sabr_alpha_rel_err": abs(fit.alpha - true["alpha"]) / true["alpha"],
        "sabr_rho_abs_err": abs(fit.rho - true["rho"]),
        "sabr_nu_rel_err": abs(fit.nu - true["nu"]) / true["nu"],
        "sabr_rmse_vol_pts": fit.rmse,
        "heston_iv_rmse": hfit.iv_rmse,
        "heston_rho_abs_err": abs(hfit.rho - true_h["rho"]),
        "heston_feller_flagged": float(hfit.feller_violation < 0),
    }


def _calibration_snapshot() -> dict[str, float]:
    """Carry the committed real-snapshot results; verify artifact checksums."""
    processed = ROOT / "data" / "processed"
    results = json.loads((processed / "spx_calibration_results.json").read_text())
    manifest = _manifest_datasets()
    ok = all(
        _sha256(ROOT / d["path"]) == d["sha256"]
        for d in manifest
        if d["name"].startswith("processed/")
    )
    sabr_rows = list(results.get("sabr_by_expiry", {}).values())
    rmses = [r["rmse"] for r in sabr_rows if isinstance(r, dict) and "rmse" in r]
    heston_res = results.get("heston", {})
    return {
        "processed_checksums_ok": float(ok),
        "sabr_mean_fit_rmse_vol_pts": float(np.mean(rmses)) if rmses else float("nan"),
        "heston_iv_rmse_snapshot": float(heston_res.get("iv_rmse", float("nan"))),
    }


EXPERIMENTS: list[tuple[str, object]] = [
    ("benchmarks_parity", lambda fast: _benchmarks()),
    ("mc_convergence", _mc_convergence),
    ("mc_variance_reduction", _variance_reduction),
    ("greeks_cross_method", _greeks),
    ("pde_convergence", _pde),
    ("anchor_pricing", _anchor_pricing),
    ("anchor_pla", _anchor_pla),
    ("calibration_synthetic", _calibration_synthetic),
    ("calibration_snapshot", lambda fast: _calibration_snapshot()),
]



def environment() -> dict[str, str]:
    import jax
    import numba
    import QuantLib as ql

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": numba.__version__,
        "jax": jax.__version__,
        "QuantLib": ql.__version__,
    }


def _manifest_datasets() -> list[dict]:
    p = ROOT / "data" / "manifest.json"
    return json.loads(p.read_text())["datasets"] if p.exists() else []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_checksums() -> dict[str, str]:
    out = {}
    for d in _manifest_datasets():
        p = ROOT / d["path"]
        if p.exists():
            out[d["path"]] = _sha256(p)
    return out


def run_all(*, fast: bool = False) -> dict:
    """Run every validation experiment; returns meta + per-experiment metrics."""
    experiments: dict[str, dict] = {}
    for name, fn in EXPERIMENTS:
        t0 = time.perf_counter()
        metrics = fn(fast)
        experiments[name] = {
            "runtime_s": round(time.perf_counter() - t0, 3),
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
    return {
        "mode": "fast" if fast else "full",
        "environment": environment(),
        "input_checksums": input_checksums(),
        "experiments": experiments,
    }
