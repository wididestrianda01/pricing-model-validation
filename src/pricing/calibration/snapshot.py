"""Real-snapshot calibration pipeline: chain filtering, parity forwards, fitting.

Turns the frozen SPX option-chain snapshot into clean calibration slices
(own Black-Scholes IVs from mid quotes — the vendor IV column is unreliable),
derives forwards/discounts from put-call parity rather than assuming them,
fits SABR per maturity slice and Heston surface-wide.

Model prices are expressed in discounted-forward terms: pricing with
spot = F * DF and rate = 0 is Black-76, so each quote keeps its own
parity-implied forward and discount inside one surface-wide Heston fit.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pricing.benchmarks.heston import heston
from pricing.calibration.heston_fit import (
    HESTON_LOWER,
    HESTON_UPPER,
    HestonFit,
)
from pricing.calibration.implied_vol import implied_vol
from pricing.calibration.rates import load_zero_curve
from pricing.calibration.sabr import SABRFit, calibrate_sabr

# Chain-filter defaults (documented in docs/data-sources.md provenance).
MAX_REL_SPREAD = 0.10  # (ask - bid) / mid ceiling
MIN_LIQUIDITY = 5  # max(volume, openInterest) floor — the snapshot's OI column
                   # is mostly zero, so volume carries the liquidity signal
MONEYNESS_BAND = (0.80, 1.20)  # strike / forward
IV_BOUNDS = (0.03, 3.0)
MIN_PARITY_PAIRS = 5  # call/put strike pairs needed for a parity forward

TARGET_TTES = {"30d": 30 / 365, "90d": 90 / 365, "1y": 365 / 365}


def parity_forward(
    strikes: np.ndarray,
    call_mids: np.ndarray,
    put_mids: np.ndarray,
    discount: float,
) -> float:
    """Robust forward from put-call parity at a curve-implied discount.

    C - P = DF * (F - K) rearranges to F = K + (C - P)/DF for every paired
    strike; the forward is the median of these estimates, which tolerates
    the stale/wide quotes in a yfinance snapshot far better than OLS.
    Raises ValueError with fewer than ``MIN_PARITY_PAIRS`` pairs.
    """
    x = np.asarray(strikes, dtype=float)
    y = np.asarray(call_mids, dtype=float) - np.asarray(put_mids, dtype=float)
    if len(x) < MIN_PARITY_PAIRS:
        raise ValueError(f"need >= {MIN_PARITY_PAIRS} parity pairs, got {len(x)}")
    estimates = x + y / discount
    return float(np.median(estimates))


def extract_slices(
    chain: pd.DataFrame,
    as_of: pd.Timestamp,
    discount_fn,
    *,
    max_rel_spread: float = MAX_REL_SPREAD,
    min_liquidity: int = MIN_LIQUIDITY,
    moneyness_band: tuple[float, float] = MONEYNESS_BAND,
    iv_bounds: tuple[float, float] = IV_BOUNDS,
) -> pd.DataFrame:
    """Filter a raw chain into clean per-expiry slices with own implied vols.

    ``chain`` columns: strike, expiry (YYYY-MM-DD str), option_type
    ('call'/'put'), bid, ask, volume, openInterest. ``discount_fn`` maps
    tte -> discount factor from the frozen rate curve; forwards come from
    parity at that discount. Expiries yielding too few liquid OTM quotes
    are dropped. Returns one row per surviving quote: expiry, tte, strike,
    forward, discount, mid, option_type, iv.
    """
    chain = chain.copy()
    chain["expiry_ts"] = pd.to_datetime(chain["expiry"])
    chain["tte"] = (chain["expiry_ts"] - as_of).dt.days / 365.0
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2.0
    keep = (
        (chain["tte"] > 7 / 365)
        & (chain["tte"] < 3.0)
        & (chain["bid"] > 0.0)
        & (chain["ask"] > chain["bid"])
        & (chain["mid"] > 0.05)
        & ((chain["ask"] - chain["bid"]) / chain["mid"] <= max_rel_spread)
        & (chain[["volume", "openInterest"]].fillna(0.0).max(axis=1) >= min_liquidity)
    )
    chain = chain[keep]

    rows = []
    for expiry, grp in chain.groupby("expiry_ts"):
        calls = grp[grp["option_type"] == "call"].set_index("strike")["mid"]
        puts = grp[grp["option_type"] == "put"].set_index("strike")["mid"]
        common = calls.index.intersection(puts.index)
        discount = discount_fn(float(grp["tte"].iloc[0]))
        try:
            forward = parity_forward(
                np.asarray(common), calls.loc[common].to_numpy(),
                puts.loc[common].to_numpy(), discount,
            )
        except ValueError:
            continue  # too few liquid pairs at this expiry — skip it
        lo, hi = moneyness_band[0] * forward, moneyness_band[1] * forward
        otm = grp[
            ((grp["option_type"] == "call") & (grp["strike"] >= forward))
            | ((grp["option_type"] == "put") & (grp["strike"] < forward))
        ]
        otm = otm[(otm["strike"] >= lo) & (otm["strike"] <= hi)]
        for _, q in otm.iterrows():
            try:
                iv = implied_vol(q["mid"], forward * discount, q["strike"], q["tte"],
                                 0.0, q["option_type"])
            except ValueError:
                continue
            if not iv_bounds[0] <= iv <= iv_bounds[1]:
                continue
            rows.append({
                "expiry": expiry.date().isoformat(),
                "tte": q["tte"],
                "strike": float(q["strike"]),
                "forward": forward,
                "discount": discount,
                "mid": float(q["mid"]),
                "option_type": q["option_type"],
                "iv": iv,
            })
    return pd.DataFrame(rows)


def select_slice_expiries(
    slices: pd.DataFrame, targets: dict[str, float] | None = None
) -> dict[str, str]:
    """Pick the observed expiry closest to each target tenor; returns {label: expiry}."""
    targets = targets or TARGET_TTES
    by_expiry = slices.groupby("expiry")["tte"].first()
    return {label: str((by_expiry - tte).abs().idxmin()) for label, tte in targets.items()}


MAX_QUOTES_PER_SLICE = 12


def _thin_slice(grp: pd.DataFrame, cap: int = MAX_QUOTES_PER_SLICE) -> pd.DataFrame:
    """Keep ``cap`` quotes evenly spaced in log-moneyness (wings + ATM kept)."""
    if len(grp) <= cap:
        return grp
    mny = np.log(grp["strike"] / grp["forward"])
    order = grp.iloc[np.argsort(mny.to_numpy())]
    idx = np.unique(np.linspace(0, len(order) - 1, cap).astype(int))
    return order.iloc[idx]


def fit_snapshot(slices: pd.DataFrame) -> dict:
    """Calibrate SABR per selected tenor slice + Heston across the same slices."""
    chosen = set(select_slice_expiries(slices).values())
    slices = slices[slices["expiry"].isin(chosen)]
    sabr_fits: dict[str, dict] = {}
    for expiry, grp in slices.groupby("expiry"):
        fit: SABRFit = calibrate_sabr(
            grp["strike"].to_numpy(), grp["iv"].to_numpy(),
            float(grp["forward"].iloc[0]), float(grp["tte"].iloc[0]),
        )
        sabr_fits[expiry] = {**asdict(fit), "n_quotes": len(grp)}

    heston_input = pd.concat([_thin_slice(slices[slices["expiry"] == e]) for e in sorted(chosen)])
    heston_fit = _fit_heston_surface(heston_input)
    return {
        "sabr_by_expiry": sabr_fits,
        "heston": {**asdict(heston_fit), "n_quotes": len(heston_input)},
    }


def _heston_df_price(forward: float, discount: float, strike: float, tte: float,
                     x: np.ndarray, option_type: str = "call") -> float:
    """Heston price in quoted index points: spot=F*DF, rate=0 (Black-76 form)."""
    v0, kappa, theta, xi, rho = x
    return heston(forward * discount, strike, tte, v0, kappa, theta, xi, rho,
                  rate=0.0, option_type=option_type)


def _fit_heston_surface(slices: pd.DataFrame) -> HestonFit:
    """IV-space least squares: a price-space objective lets long-dated premiums
    (hundreds of index points) swamp the short end, so fits are done on implied
    vols. Start parameters come from the slice ATMs, deterministically."""
    strikes = slices["strike"].to_numpy()
    ttes = slices["tte"].to_numpy()
    mids = slices["mid"].to_numpy()
    forwards = slices["forward"].to_numpy()
    discounts = slices["discount"].to_numpy()
    opt_types = np.where(strikes >= forwards, "call", "put")
    fwd_px = forwards * discounts

    target_ivs = np.array([
        implied_vol(mids[i], fwd_px[i], strikes[i], ttes[i], 0.0, opt_types[i])
        for i in range(len(strikes))
    ])

    def model_ivs(x: np.ndarray) -> np.ndarray:
        out = np.empty(len(strikes))
        for i in range(len(strikes)):
            price = _heston_df_price(forwards[i], discounts[i], strikes[i],
                                     ttes[i], x, opt_types[i])
            try:
                out[i] = implied_vol(price, fwd_px[i], strikes[i], ttes[i], 0.0,
                                     opt_types[i])
            except ValueError:  # extreme params can price outside BS bounds
                return target_ivs + 10.0  # heavy penalty, keeps optimizer in-bounds
        return out

    # Start from observed term structure: v0 = short ATM variance, theta = long.
    by_tte = slices.assign(iv=target_ivs).sort_values("tte")
    atm_short = float(by_tte.iloc[0]["iv"])
    atm_long = float(by_tte.iloc[-1]["iv"])
    x0 = np.array([atm_short**2, 1.5, max(atm_long**2, 1e-4), 0.5, -0.3])
    x0 = np.clip(x0, HESTON_LOWER + 1e-6, HESTON_UPPER - 1e-6)

    res = least_squares(
        lambda x: model_ivs(x) - target_ivs, x0,
        bounds=(HESTON_LOWER, HESTON_UPPER), xtol=1e-8, ftol=1e-8,
    )
    fitted_ivs = model_ivs(res.x)
    fitted_prices = np.array([
        _heston_df_price(forwards[i], discounts[i], strikes[i], ttes[i], res.x,
                         opt_types[i])
        for i in range(len(strikes))
    ])

    v0, kappa, theta, xi, rho = res.x
    return HestonFit(
        v0=float(v0), kappa=float(kappa), theta=float(theta),
        xi=float(xi), rho=float(rho),
        price_rmse=float(np.sqrt(np.mean((fitted_prices - mids) ** 2))),
        iv_rmse=float(np.sqrt(np.mean((fitted_ivs - target_ivs) ** 2))),
        feller_violation=2.0 * kappa * theta - xi**2,
        converged=bool(res.success),
    )


def run_snapshot_calibration(
    raw_csv: Path | str,
    rates_dir: Path | str,
    *,
    as_of: str,
) -> tuple[pd.DataFrame, dict]:
    """End-to-end: raw frozen SPX CSV + frozen rate CSVs -> (slices, fits)."""
    discount_fn = load_zero_curve(rates_dir, pd.Timestamp(as_of))
    chain = pd.read_csv(raw_csv)
    slices = extract_slices(chain, pd.Timestamp(as_of), discount_fn)
    if slices.empty:
        raise ValueError("no quotes survived filtering — check chain quality")
    return slices, fit_snapshot(slices)
