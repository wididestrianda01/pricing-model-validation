"""QuantLib-Python wrapper producing independent reference prices on the same seam.

Each function mirrors a closed form in this package so the from-scratch
benchmarks can be challenged by a bank-recognized library. Prices are computed
at a fixed evaluation date (2024-01-01) for determinism.
"""

from __future__ import annotations

import numpy as np
import QuantLib as ql

_TODAY = ql.Date(1, 1, 2024)


def _set_eval_date() -> None:
    ql.Settings.instance().evaluationDate = _TODAY


def _day_count(time: float, name: str) -> int:
    """Convert a float-year time to whole Act/365 days, refusing the unrepresentable.

    QuantLib dates/exercises resolve maturities in whole days under Actual/365,
    while the closed forms take continuous time. Rounding a non-integral day
    count silently prices a shifted maturity (up to half a day off — measured
    ~0.1% on a 3M ATM option), so reject it instead.
    """
    days = time * 365.0
    nearest = round(days)
    if abs(days - nearest) > 1e-6:
        raise ValueError(
            f"{name}={time!r} is not a whole number of Act/365 days; pass a "
            "multiple of 1/365 so the QuantLib seam matches the closed form exactly"
        )
    return int(nearest)


def _bs_option(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str,
) -> ql.EuropeanOption:
    """A priced QuantLib AnalyticEuropeanEngine option for the given parameters."""
    _set_eval_date()
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(_TODAY, rate, ql.Actual365Fixed()))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(_TODAY, 0.0, ql.Actual365Fixed()))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(_TODAY, ql.NullCalendar(), vol, ql.Actual365Fixed())
    )
    process = ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h)

    opt_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(opt_type, strike)
    exercise = ql.EuropeanExercise(_TODAY + _day_count(tte, "tte"))
    option = ql.EuropeanOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


def ql_black_scholes(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes via QuantLib AnalyticEuropeanEngine."""
    return float(_bs_option(spot, strike, tte, vol, rate, option_type).NPV())


def ql_black_scholes_greeks(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Black-Scholes delta/gamma/vega/theta/rho via QuantLib AnalyticEuropeanEngine.

    vega is dV/dσ per unit volatility (not per 1%), matching the closed form.
    theta is dV/dt per year (same convention as the closed form: -dV/dtte).
    """
    option = _bs_option(spot, strike, tte, vol, rate, option_type)
    return {
        "delta": float(option.delta()),
        "gamma": float(option.gamma()),
        "vega": float(option.vega()),
        "theta": float(option.theta()),
        "rho": float(option.rho()),
    }

def ql_black_76(
    forward: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "call",
) -> float:
    """Black-76 via QuantLib BlackCalculator."""
    opt_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(opt_type, strike)
    calc = ql.BlackCalculator(payoff, forward, vol * np.sqrt(tte), np.exp(-rate * tte))
    return float(calc.value())


def _hw_model(r0: float, a: float, sigma: float) -> ql.HullWhite:
    curve = ql.YieldTermStructureHandle(ql.FlatForward(_TODAY, r0, ql.Actual365Fixed()))
    return ql.HullWhite(curve, a, sigma)


def ql_hw_zcb_option(
    expiry: float,
    maturity: float,
    a: float,
    sigma: float,
    r0: float,
    strike: float,
    option_type: str = "call",
) -> float:
    """Hull-White zero-coupon bond option via QuantLib's analytic formula."""
    _set_eval_date()
    hw = _hw_model(r0, a, sigma)
    opt_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    return float(hw.discountBondOption(opt_type, strike, expiry, maturity))


def ql_hw_caplet(
    expiry: float,
    maturity: float,
    a: float,
    sigma: float,
    r0: float,
    strike_rate: float,
) -> float:
    """Hull-White caplet via the (1 + tau*K) * bond-put identity."""
    tau = maturity - expiry
    bond_strike = 1.0 / (1.0 + tau * strike_rate)
    put = ql_hw_zcb_option(expiry, maturity, a, sigma, r0, bond_strike, "put")
    return float((1.0 + tau * strike_rate) * put)


def ql_jamshidian_swaption(
    expiry: float,
    tenors: list[float],
    strike_rate: float,
    a: float,
    sigma: float,
    r0: float,
    option_type: str = "payer",
) -> float:
    """Hull-White swaption via QuantLib's JamshidianSwaptionEngine (annual tenor)."""
    _set_eval_date()
    curve = ql.YieldTermStructureHandle(ql.FlatForward(_TODAY, r0, ql.Actual365Fixed()))
    cal = ql.NullCalendar()
    index = ql.IborIndex(
        "USD1Y", ql.Period(1, ql.Years), 0, ql.USDCurrency(),
        cal, ql.Unadjusted, False, ql.Actual365Fixed(), curve,
    )

    prev = expiry
    for ti in tenors:
        _day_count(ti, "tenor")
        if abs((ti - prev) - 1.0) > 1e-6:
            raise ValueError(
                "tenors must be equally spaced at 1.0 years to match the annual swap schedule"
            )
        prev = ti
    start = _TODAY + _day_count(expiry, "expiry")
    end = _TODAY + _day_count(tenors[-1], "tenors[-1]")
    period = ql.Period(1, ql.Years)
    fix_sched = ql.Schedule(start, end, period, cal, ql.Unadjusted, ql.Unadjusted,
                            ql.DateGeneration.Forward, False)
    float_sched = ql.Schedule(start, end, period, cal, ql.Unadjusted, ql.Unadjusted,
                              ql.DateGeneration.Forward, False)

    swap_type = ql.VanillaSwap.Payer if option_type == "payer" else ql.VanillaSwap.Receiver
    swap = ql.VanillaSwap(swap_type, 1.0, fix_sched, strike_rate, ql.Actual365Fixed(),
                          float_sched, index, 0.0, ql.Actual365Fixed())
    swaption = ql.Swaption(swap, ql.EuropeanExercise(start))

    hw = ql.HullWhite(curve, a, sigma)
    swaption.setPricingEngine(ql.JamshidianSwaptionEngine(hw, curve))
    return float(swaption.NPV())

def _bs_process(
    spot: float,
    vol: float,
    rate: float,
) -> ql.BlackScholesMertonProcess:
    """Black-Scholes-Merton process with flat term structures."""
    _set_eval_date()
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(_TODAY, rate, ql.Actual365Fixed()))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(_TODAY, 0.0, ql.Actual365Fixed()))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(_TODAY, ql.NullCalendar(), vol, ql.Actual365Fixed())
    )
    return ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h)


def ql_barrier(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    barrier: float,
    rebate: float = 0.0,
    option_type: str = "call",
) -> float:
    """Continuous knock-out barrier option via AnalyticBarrierEngine.

    Direction inferred: barrier below spot is down-and-out, above is up-and-out.
    """
    process = _bs_process(spot, vol, rate)
    opt_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(opt_type, strike)
    exercise = ql.EuropeanExercise(_TODAY + _day_count(tte, "tte"))
    b_type = ql.Barrier.DownOut if barrier < spot else ql.Barrier.UpOut
    option = ql.BarrierOption(b_type, barrier, rebate, payoff, exercise)
    option.setPricingEngine(ql.AnalyticBarrierEngine(process))
    return float(option.NPV())


def ql_american(
    spot: float,
    strike: float,
    tte: float,
    vol: float,
    rate: float,
    option_type: str = "put",
    time_steps: int = 800,
) -> float:
    """American option via a Cox-Ross-Rubinstein binomial tree."""
    process = _bs_process(spot, vol, rate)
    opt_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(opt_type, strike)
    exercise = ql.AmericanExercise(_TODAY, _TODAY + _day_count(tte, "tte"))
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", time_steps))
    return float(option.NPV())
