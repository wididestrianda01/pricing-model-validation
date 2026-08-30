"""Scrambled Sobol QMC: benchmark agreement, determinism, faster convergence."""

from pricing.benchmarks import black_scholes
from pricing.monte_carlo import mc_european, mc_european_sobol

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 0.5, 0.2, 0.03


def test_sobol_matches_black_scholes():
    bs = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    r = mc_european_sobol(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2**16, n_steps=1, seed=0, scheme="exact")
    assert abs(r.price - bs) < 4 * r.std_error + 1e-4


def test_sobol_deterministic():
    a = mc_european_sobol(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2**14, seed=3, scheme="exact")
    b = mc_european_sobol(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=2**14, seed=3, scheme="exact")
    assert a.price == b.price

def test_sobol_converges_faster_than_pseudo_random():
    # At equal path counts, scrambled Sobol sits consistently closer to the
    # benchmark than a single pseudo-random draw, and the gap narrows with n.
    bs = black_scholes(SPOT, STRIKE, TTE, VOL, RATE, "call")
    for n in (2**12, 2**14, 2**16):
        qmc = mc_european_sobol(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=n, n_steps=1, seed=0, scheme="exact")
        mc = mc_european(SPOT, STRIKE, TTE, VOL, RATE, "call", n_paths=n, n_steps=1, seed=0, scheme="exact")
        assert abs(qmc.price - bs) < abs(mc.price - bs)
