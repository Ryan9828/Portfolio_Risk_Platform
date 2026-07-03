"""Volatility model fitting (arch package) with a degradation-safe fallback chain.

Returns are rescaled x100 before fitting (arch's optimizer misbehaves at ~1e-2 scale)
and all outputs are de-rescaled back to decimal units HERE, exactly once — nothing
downstream ever sees rescaled units.

Fallback chain: configured model with Student-t -> GARCH(1,1) normal -> EWMA(0.94).
The model actually used is recorded on the VolFit so degraded fits stay visible.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arch import arch_model

from .config import GarchConfig

log = logging.getLogger(__name__)

EWMA_LAMBDA = 0.94
MAX_HORIZON = 10


@dataclass
class VolFit:
    ticker: str
    model: str                    # "GARCH" | "EGARCH" | "EWMA" (fallback)
    dist: str                     # "t" | "normal" | "empirical" (EWMA)
    converged: bool
    cond_vol: pd.Series           # daily conditional sigma, decimal units
    std_resid: pd.Series          # standardised residuals (unitless)
    sigma_forecast: np.ndarray    # sigma for h=1..MAX_HORIZON, decimal units
    nu: float | None = None       # Student-t dof when dist == "t"
    params: dict = field(default_factory=dict)


def _fit_arch(
    r: pd.Series, ticker: str, model: str, dist: str, rescale: float
) -> VolFit | None:
    o = 1 if model == "EGARCH" else 0
    am = arch_model(r * rescale, mean="Zero", vol=model, p=1, o=o, q=1, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = am.fit(disp="off", show_warning=False)
        except Exception as err:
            log.warning("%s: %s/%s fit raised %s", ticker, model, dist, err)
            return None
    if res.convergence_flag != 0:
        log.warning("%s: %s/%s did not converge", ticker, model, dist)
        return None

    if model == "EGARCH":
        # multi-step EGARCH variance has no analytic form
        fc = res.forecast(horizon=MAX_HORIZON, method="simulation", simulations=2000, reindex=False)
    else:
        fc = res.forecast(horizon=MAX_HORIZON, reindex=False)
    sigma_forecast = np.sqrt(fc.variance.to_numpy()[0]) / rescale

    if not np.all(np.isfinite(sigma_forecast)) or np.any(sigma_forecast <= 0):
        log.warning("%s: %s/%s produced degenerate forecast", ticker, model, dist)
        return None

    nu = float(res.params["nu"]) if dist == "t" and "nu" in res.params else None
    return VolFit(
        ticker=ticker,
        model=model,
        dist=dist,
        converged=True,
        cond_vol=res.conditional_volatility / rescale,
        std_resid=res.std_resid.dropna(),
        sigma_forecast=sigma_forecast,
        nu=nu,
        params={k: float(v) for k, v in res.params.items()},
    )


def _fit_ewma(r: pd.Series, ticker: str) -> VolFit:
    """RiskMetrics EWMA(0.94) — last-resort estimator that always succeeds."""
    x = r.to_numpy()
    var = np.empty(len(x))
    var[0] = np.var(x) if len(x) > 1 else 1e-8
    for t in range(1, len(x)):
        var[t] = EWMA_LAMBDA * var[t - 1] + (1 - EWMA_LAMBDA) * x[t - 1] ** 2
    sigma = pd.Series(np.sqrt(np.maximum(var, 1e-12)), index=r.index)
    return VolFit(
        ticker=ticker,
        model="EWMA",
        dist="empirical",
        converged=False,
        cond_vol=sigma,
        std_resid=(r / sigma).dropna(),
        # EWMA variance forecast is flat at the next-step value
        sigma_forecast=np.full(
            MAX_HORIZON,
            float(np.sqrt(EWMA_LAMBDA * var[-1] + (1 - EWMA_LAMBDA) * x[-1] ** 2)),
        ),
        params={"lambda": EWMA_LAMBDA},
    )


def fit_with_fallback(r: pd.Series, ticker: str, cfg: GarchConfig, min_obs: int) -> VolFit:
    r = r.dropna()
    if len(r) >= min_obs and float(r.std()) > 0:
        model = "EGARCH" if ticker in cfg.egarch_assets else cfg.default
        for m, d in [(model, cfg.dist), ("GARCH", "normal")]:
            fit = _fit_arch(r, ticker, m, d, cfg.rescale)
            if fit is not None:
                return fit
    else:
        log.warning("%s: insufficient/degenerate history (%d obs), using EWMA", ticker, len(r))
    return _fit_ewma(r, ticker)
