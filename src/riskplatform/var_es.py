"""Value-at-Risk and Expected Shortfall engines.

All functions return LOSSES AS POSITIVE NUMBERS in decimal return units.

Three methods, deliberately shown side by side:
- parametric: GARCH conditional sigma with the variance TERM STRUCTURE for multi-day
  horizons (sqrt of summed forecast variances — mean-reverting, not naive sqrt(h)),
  under normal and standardised Student-t innovations.
- historical: empirical quantile of the trailing window; multi-day scaled by sqrt(h)
  (a disclosed approximation).
- monte_carlo: correlated multi-asset simulation over full h-step sigma paths,
  aggregated with the actual portfolio weights.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from .config import PORTFOLIO_TICKER, Settings
from .garch import VolFit

RESID_CORR_WINDOW = 500
MIN_EIGENVALUE = 1e-8


def parametric_var_es(
    sigma_path: np.ndarray,
    alpha: float,
    horizon: int,
    dist: str = "normal",
    nu: float | None = None,
) -> tuple[float, float]:
    sigma_h = float(np.sqrt(np.sum(np.asarray(sigma_path[:horizon]) ** 2)))
    if dist == "t" and nu is not None and nu > 2:
        # standardised t: scale so the innovation has unit variance
        scale = sigma_h * np.sqrt((nu - 2) / nu)
        t_a = stats.t.ppf(alpha, nu)
        var = scale * t_a
        es = scale * stats.t.pdf(t_a, nu) * (nu + t_a**2) / ((nu - 1) * (1 - alpha))
    else:
        z = stats.norm.ppf(alpha)
        var = sigma_h * z
        es = sigma_h * stats.norm.pdf(z) / (1 - alpha)
    return float(var), float(es)


def historical_var_es(
    port_r: pd.Series, alpha: float, horizon: int, window: int = 500
) -> tuple[float, float]:
    losses = -port_r.dropna().tail(window).to_numpy()
    var1 = float(np.quantile(losses, alpha))
    tail = losses[losses >= var1]
    es1 = float(tail.mean()) if len(tail) else var1
    scale = np.sqrt(horizon)  # sqrt-time approximation, disclosed in methodology
    return var1 * scale, es1 * scale


def residual_correlation(fits: dict[str, VolFit], tickers: list[str]) -> np.ndarray:
    """Shrunk/PSD correlation of standardised residuals across portfolio assets."""
    resid = pd.concat(
        {t: fits[t].std_resid for t in tickers}, axis=1, join="inner"
    ).tail(RESID_CORR_WINDOW)
    corr = resid.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    # eigenvalue clipping keeps Cholesky feasible when short windows make it near-singular
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, MIN_EIGENVALUE, None)
    corr = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d)


def monte_carlo_var_es(
    fits: dict[str, VolFit],
    weights: dict[str, float],
    alpha: float,
    horizon: int,
    n_sims: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    tickers = list(weights)
    w = np.array([weights[t] for t in tickers])
    corr = residual_correlation(fits, tickers)
    chol = np.linalg.cholesky(corr)
    # sigma paths: (horizon, n_assets), decimal units
    sigma = np.column_stack([fits[t].sigma_forecast[:horizon] for t in tickers])

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, horizon, len(tickers)))
    shocks = z @ chol.T
    log_paths = (shocks * sigma[None, :, :]).sum(axis=1)     # (n_sims, n_assets)
    port_simple = np.expm1(log_paths) @ w
    losses = -port_simple
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(tail.mean()) if len(tail) else var
    return var, es


def compute_all_metrics(
    fits: dict[str, VolFit],
    port_r: pd.Series,
    settings: Settings,
    asof: date,
) -> pd.DataFrame:
    """Tidy risk-metric rows for every method x horizon x confidence combination."""
    pf = fits[PORTFOLIO_TICKER]
    rows = []
    for horizon in settings.var.horizons:
        for alpha in settings.var.confidences:
            combos = {
                "parametric_normal": parametric_var_es(pf.sigma_forecast, alpha, horizon),
                "parametric_t": parametric_var_es(
                    pf.sigma_forecast, alpha, horizon, dist="t", nu=pf.nu
                ),
                "historical": historical_var_es(
                    port_r, alpha, horizon, settings.var.hs_window
                ),
                "monte_carlo": monte_carlo_var_es(
                    fits,
                    settings.weights,
                    alpha,
                    horizon,
                    settings.var.mc_sims,
                    settings.var.mc_seed,
                ),
            }
            for method, (var, es) in combos.items():
                rows.append(
                    {
                        "date": pd.Timestamp(asof),
                        "method": method,
                        "horizon": horizon,
                        "confidence": alpha,
                        "var": var,
                        "es": es,
                        "portfolio_sigma": float(pf.sigma_forecast[0]),
                    }
                )
    return pd.DataFrame(rows)
