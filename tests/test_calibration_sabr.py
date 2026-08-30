from typing import ClassVar

import numpy as np
import pytest

from pricing.calibration.sabr import SABRFit, calibrate_sabr, sabr_vol


class TestSabrVol:
    def test_flat_smile_when_nu_zero(self):
        # nu = 0 kills skew/smile: every strike prices at the ATM-adjusted vol.
        forward, tte, alpha, beta, rho = 100.0, 1.0, 0.2, 1.0, -0.5
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = sabr_vol(forward, strikes, tte, alpha, beta, rho, nu=0.0)
        correction = (
            ((1.0 - beta) ** 2 / 24.0) * alpha**2 / forward ** (2 * (1.0 - beta))
            + rho * beta * 0.0 * alpha / (4.0 * forward ** (1.0 - beta))
            + 0.0
        )
        expected = alpha / forward ** (1.0 - beta) * (1.0 + correction * tte)
        assert np.allclose(vols, expected)

    def test_atm_matches_closed_form_limit(self):
        # K = F: z -> 0, so vol must equal the closed-form ATM expression.
        forward, tte, alpha, beta, rho, nu = 4500.0, 0.25, 0.15, 1.0, -0.7, 1.8
        f_pow = forward ** (1.0 - beta)
        correction = (
            ((1.0 - beta) ** 2 / 24.0) * alpha**2 / f_pow**2
            + rho * beta * nu * alpha / (4.0 * f_pow)
            + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
        )
        atm = alpha / f_pow * (1.0 + correction * tte)
        assert sabr_vol(forward, np.array([forward]), tte, alpha, beta, rho, nu)[0] == pytest.approx(atm)

    @pytest.mark.parametrize("rho", [-0.6, +0.6])
    def test_smile_minimum_shifts_with_rho_sign(self, rho):
        # A skewed smile is not monotone (the nu term bends it back up). Negative
        # rho loads the low-strike wing hardest, so its minimum sits above the
        # forward; positive rho mirrors it below.
        forward, tte = 5000.0, 0.5
        strikes = np.linspace(4000.0, 6000.0, 41)
        vols = sabr_vol(forward, strikes, tte, 0.18, 1.0, rho, 1.2)
        argmin = strikes[int(np.argmin(vols))]
        assert np.sign(argmin - forward) == -np.sign(rho)
        wing_skew = vols[strikes < forward].mean() - vols[strikes > forward].mean()
        assert np.sign(wing_skew) == -np.sign(rho)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"forward": -1.0},
            {"tte": 0.0},
            {"beta": 1.5},
            {"rho": 1.0},
            {"alpha": 0.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs):
        base = {"forward": 100.0, "tte": 1.0, "alpha": 0.2, "beta": 1.0, "rho": -0.3, "nu": 1.0}
        base.update(kwargs)
        with pytest.raises(ValueError):
            sabr_vol(base["forward"], np.array([90.0, 100.0, 110.0]), base["tte"],
                     base["alpha"], base["beta"], base["rho"], base["nu"])

    def test_rejects_nonpositive_strike(self):
        with pytest.raises(ValueError, match="strikes"):
            sabr_vol(100.0, np.array([0.0, 100.0, 110.0]), 1.0, 0.2, 1.0, -0.3, 1.0)


class TestCalibrateSabr:
    TRUE_PARAMS: ClassVar[dict[str, float]] = {"alpha": 0.35, "beta": 1.0, "rho": -0.65, "nu": 2.1}

    def _synthetic(self, forward, tte, n=15):
        strikes = forward * np.linspace(0.75, 1.25, n)
        iv = sabr_vol(forward, strikes, tte, self.TRUE_PARAMS["alpha"], self.TRUE_PARAMS["beta"],
                      self.TRUE_PARAMS["rho"], self.TRUE_PARAMS["nu"])
        return strikes, iv

    def test_recovers_ground_truth_noiseless(self):
        forward, tte = 5500.0, 0.4
        strikes, iv = self._synthetic(forward, tte)
        fit: SABRFit = calibrate_sabr(strikes, iv, forward, tte, beta=self.TRUE_PARAMS["beta"])
        assert fit.converged
        assert fit.alpha == pytest.approx(self.TRUE_PARAMS["alpha"], rel=2e-2)
        assert fit.rho == pytest.approx(self.TRUE_PARAMS["rho"], abs=2e-2)
        assert fit.nu == pytest.approx(self.TRUE_PARAMS["nu"], rel=2e-2)
        assert fit.rmse < 1e-8

    def test_fit_is_deterministic(self):
        forward, tte = 5500.0, 0.4
        strikes, iv = self._synthetic(forward, tte)
        a = calibrate_sabr(strikes, iv, forward, tte)
        b = calibrate_sabr(strikes, iv, forward, tte)
        assert a == b

    def test_rmse_small_on_noisy_input(self):
        # With realistic quote noise the fit should still sit inside ~0.2 vol pts.
        rng = np.random.default_rng(42)
        forward, tte = 5500.0, 0.4
        strikes, iv = self._synthetic(forward, tte)
        noisy = iv + rng.normal(0.0, 0.002, size=iv.shape)
        fit = calibrate_sabr(strikes, noisy, forward, tte)
        assert fit.rmse < 0.003

    def test_beta_zero_low_strike_regime(self):
        # beta = 0 (lognormal stochastic vol on absolute strikes) still recovers.
        forward, tte = 100.0, 0.75
        strikes = np.linspace(85.0, 115.0, 13)
        iv = sabr_vol(forward, strikes, tte, 1.8, 0.0, -0.4, 0.9)
        fit = calibrate_sabr(strikes, iv, forward, tte, beta=0.0)
        assert fit.converged
        assert fit.rmse < 1e-7

    def test_weights_change_the_solution(self):
        forward, tte = 5500.0, 0.4
        strikes, iv = self._synthetic(forward, tte)
        unweighted = calibrate_sabr(strikes, iv, forward, tte)
        weights = np.where(np.abs(strikes / forward - 1.0) < 0.05, 10.0, 1.0)
        weighted = calibrate_sabr(strikes, iv, forward, tte, weights=weights)
        assert weighted.alpha != pytest.approx(unweighted.alpha, rel=1e-12)

    def test_too_few_points_rejected(self):
        with pytest.raises(ValueError, match="3 smile points"):
            calibrate_sabr(np.array([100.0, 110.0]), np.array([0.2, 0.19]), 100.0, 1.0)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same-length"):
            calibrate_sabr(np.array([100.0, 110.0, 120.0]), np.array([0.2, 0.19]), 100.0, 1.0)
