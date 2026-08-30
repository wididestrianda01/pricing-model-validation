"""Convergence studies: weak/strong order and standard-error scaling."""

from pricing.monte_carlo import se_scaling, strong_convergence, weak_convergence

SPOT, STRIKE, TTE, VOL, RATE = 100.0, 105.0, 0.5, 0.2, 0.03


def test_weak_convergence_order_about_one_euler():
    # The call payoff's kink at the strike makes the weak-order estimate noisy
    # (measured ~1.3); the band still separates order ~1 from 0.5 or 2.
    order, _, _ = weak_convergence(SPOT, STRIKE, TTE, VOL, RATE, "call", 1_000_000, [1, 2, 4, 8, 16], seed=0, scheme="euler")
    assert 0.6 < order < 1.5


def test_weak_convergence_order_about_one_milstein():
    order, _, _ = weak_convergence(SPOT, STRIKE, TTE, VOL, RATE, "call", 1_000_000, [1, 2, 4, 8, 16], seed=0, scheme="milstein")
    assert 0.6 < order < 1.5


def test_strong_convergence_order_half_euler():
    order, _, _ = strong_convergence(SPOT, TTE, VOL, RATE, 200_000, [2, 4, 8, 16, 32], seed=0, scheme="euler")
    assert 0.35 < order < 0.65


def test_strong_convergence_order_one_milstein():
    order, _, _ = strong_convergence(SPOT, TTE, VOL, RATE, 200_000, [2, 4, 8, 16, 32], seed=0, scheme="milstein")
    assert 0.8 < order < 1.2


def test_standard_error_decays_inverse_sqrt_n():
    slope, _, _ = se_scaling(SPOT, STRIKE, TTE, VOL, RATE, "call", [10_000, 40_000, 160_000, 640_000], seed=0, scheme="exact")
    assert -0.6 < slope < -0.4
