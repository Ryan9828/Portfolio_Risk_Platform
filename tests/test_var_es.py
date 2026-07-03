"""Known-answer tests for the VaR/ES engines."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskplatform.garch import MAX_HORIZON, VolFit
from riskplatform.var_es import (
    historical_var_es,
    monte_carlo_var_es,
    parametric_var_es,
)

SIGMA = 0.01
PATH = np.full(MAX_HORIZON, SIGMA)


def test_normal_var_95_known_answer():
    var, es = parametric_var_es(PATH, 0.95, 1, dist="normal")
    assert var == pytest.approx(0.0164485, abs=1e-6)
    assert es == pytest.approx(SIGMA * stats.norm.pdf(1.6448536) / 0.05, rel=1e-6)


def test_normal_es_exceeds_var():
    for alpha in (0.95, 0.99):
        var, es = parametric_var_es(PATH, alpha, 1, dist="normal")
        assert es > var > 0


def test_student_t_fatter_tail_than_normal_at_99():
    var_n, _ = parametric_var_es(PATH, 0.99, 1, dist="normal")
    var_t, _ = parametric_var_es(PATH, 0.99, 1, dist="t", nu=5.0)
    assert var_t > var_n


def test_term_structure_scaling():
    # flat sigma path: 10-day VaR must equal sqrt(10) x 1-day VaR exactly
    var1, _ = parametric_var_es(PATH, 0.99, 1, dist="normal")
    var10, _ = parametric_var_es(PATH, 0.99, 10, dist="normal")
    assert var10 == pytest.approx(var1 * np.sqrt(10), rel=1e-9)


def test_historical_var_known_quantile():
    # 1000 returns: uniform grid from -5% to +5% -> 99% loss quantile is known
    r = pd.Series(np.linspace(-0.05, 0.05, 1000))
    var, es = historical_var_es(r, 0.99, 1, window=1000)
    assert var == pytest.approx(np.quantile(-r.to_numpy(), 0.99), rel=1e-9)
    assert es >= var


def _fake_fit(sigma: float, n: int = 600, seed: int = 0) -> VolFit:
    rng = np.random.default_rng(seed)
    resid = pd.Series(rng.standard_normal(n), index=pd.bdate_range("2024-01-01", periods=n))
    return VolFit(
        ticker="X",
        model="GARCH",
        dist="normal",
        converged=True,
        cond_vol=resid * 0 + sigma,
        std_resid=resid,
        sigma_forecast=np.full(MAX_HORIZON, sigma),
    )


def test_monte_carlo_converges_to_parametric_normal_single_asset():
    fits = {"X": _fake_fit(SIGMA)}
    var_mc, es_mc = monte_carlo_var_es(fits, {"X": 1.0}, 0.95, 1, n_sims=200_000, seed=1)
    var_p, es_p = parametric_var_es(PATH, 0.95, 1, dist="normal")
    # MC is in simple-return space vs parametric log-return space; at sigma=1% the
    # difference is second-order, so 2% relative tolerance is meaningful
    assert var_mc == pytest.approx(var_p, rel=0.02)
    assert es_mc == pytest.approx(es_p, rel=0.03)


def test_monte_carlo_diversification_reduces_var():
    fits = {"A": _fake_fit(SIGMA, seed=1), "B": _fake_fit(SIGMA, seed=2)}
    var_div, _ = monte_carlo_var_es(fits, {"A": 0.5, "B": 0.5}, 0.99, 1, n_sims=50_000)
    var_single, _ = monte_carlo_var_es({"A": fits["A"]}, {"A": 1.0}, 0.99, 1, n_sims=50_000)
    assert var_div < var_single
