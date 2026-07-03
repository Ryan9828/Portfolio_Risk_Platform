"""GARCH fitting: parameter recovery on simulated data and the fallback chain."""

import numpy as np
import pandas as pd

from riskplatform.config import GarchConfig
from riskplatform.garch import fit_with_fallback

CFG = GarchConfig()


def _simulate_garch(n=2000, omega=0.02, alpha=0.08, beta=0.90, seed=11):
    """Simulate GARCH(1,1) in percent units, return decimal series."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    sigma2 = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(sigma2) * rng.standard_normal()
        sigma2 = omega + alpha * r[t] ** 2 + beta * sigma2
    return pd.Series(r / 100, index=pd.bdate_range("2019-01-01", periods=n))


def test_garch_recovers_persistence():
    fit = fit_with_fallback(_simulate_garch(), "SIM", CFG, min_obs=250)
    assert fit.model == "GARCH" and fit.converged
    persistence = fit.params["alpha[1]"] + fit.params["beta[1]"]
    assert 0.90 < persistence < 1.0
    assert np.all(fit.sigma_forecast > 0)


def test_fallback_to_ewma_on_degenerate_series():
    flat = pd.Series(np.zeros(400), index=pd.bdate_range("2024-01-01", periods=400))
    fit = fit_with_fallback(flat, "FLAT", CFG, min_obs=250)
    assert fit.model == "EWMA"
    assert not fit.converged


def test_fallback_to_ewma_on_short_series():
    short = pd.Series(
        np.random.default_rng(2).standard_normal(50) * 0.01,
        index=pd.bdate_range("2026-01-01", periods=50),
    )
    fit = fit_with_fallback(short, "SHORT", CFG, min_obs=250)
    assert fit.model == "EWMA"
