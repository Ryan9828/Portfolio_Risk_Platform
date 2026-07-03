"""Rolling out-of-sample VaR backtesting with Kupiec / Christoffersen tests.

Walk-forward design: the portfolio GARCH(1,1)-t is refit every `refit_every` days on
an expanding window; between refits the conditional variance is rolled forward daily
with the fitted parameters (sigma2_{t+1} = omega + alpha r_t^2 + beta sigma2_t), so
every day's VaR forecast uses only information available at that day's close.
Historical-simulation VaR is recomputed daily from the trailing window.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from .config import Settings

log = logging.getLogger(__name__)

_EPS = 1e-12


def _loglik_bernoulli(x: int, n: int, p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return (n - x) * np.log(1 - p) + x * np.log(p)


def kupiec_pof(n_breaches: int, n_obs: int, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR test. alpha is the VaR confidence (e.g. 0.99)."""
    p_expected = 1 - alpha
    p_observed = n_breaches / n_obs
    lr = -2 * (
        _loglik_bernoulli(n_breaches, n_obs, p_expected)
        - _loglik_bernoulli(n_breaches, n_obs, p_observed)
    )
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def christoffersen_independence(breaches: np.ndarray) -> tuple[float, float]:
    """LR test that breaches are serially independent (first-order Markov)."""
    b = np.asarray(breaches).astype(int)
    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_null = _loglik_bernoulli(n01 + n11, n00 + n01 + n10 + n11, pi)
    ll_alt = _loglik_bernoulli(n01, n00 + n01, pi01) + _loglik_bernoulli(n11, n10 + n11, pi11)
    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def conditional_coverage(breaches: np.ndarray, alpha: float) -> tuple[float, float]:
    """Christoffersen conditional coverage: joint correct-rate + independence, chi2(2)."""
    b = np.asarray(breaches).astype(int)
    lr_uc, _ = kupiec_pof(int(b.sum()), len(b), alpha)
    lr_ind, _ = christoffersen_independence(b)
    lr = lr_uc + lr_ind
    return float(lr), float(1 - stats.chi2.cdf(lr, df=2))


def basel_traffic_light(n_breaches: int, n_obs: int) -> str:
    """Basel traffic-light zones for 99% VaR, scaled to the standard 250-day window."""
    scaled = n_breaches * 250 / n_obs
    if scaled <= 4:
        return "green"
    if scaled <= 9:
        return "yellow"
    return "red"


def _garch_sigma2_step(omega: float, a: float, b: float, r: float, sigma2: float) -> float:
    return omega + a * r**2 + b * sigma2


def rolling_backtest(port_r: pd.Series, settings: Settings) -> pd.DataFrame:
    """Walk-forward 1-day VaR forecasts vs realised portfolio returns.

    Output rows: [date, method, confidence, var_forecast, realised, breach]
    Methods: parametric_t (GARCH), historical. Monte Carlo is excluded from the daily
    backtest for runtime reasons (disclosed in the methodology).
    """
    r = port_r.dropna()
    window = settings.backtest.window
    refit_every = settings.backtest.refit_every
    rescale = settings.garch.rescale
    hs_window = settings.var.hs_window
    start = len(r) - window
    if start < settings.min_obs_garch:
        raise ValueError(
            f"need >= {settings.min_obs_garch + window} portfolio returns for the backtest, have {len(r)}"
        )

    x = r.to_numpy() * rescale
    rows = []
    params: dict[str, float] = {}
    sigma2 = float(np.var(x[:start]))

    for i in range(start, len(r)):
        if (i - start) % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(x[:i], mean="Zero", vol="GARCH", p=1, q=1, dist="t")
                res = am.fit(disp="off", show_warning=False)
            if res.convergence_flag == 0:
                params = {k: float(v) for k, v in res.params.items()}
                sigma2 = float(res.conditional_volatility[-1]) ** 2
                # roll one step so sigma2 is the forecast FOR day i
                sigma2 = _garch_sigma2_step(
                    params["omega"], params["alpha[1]"], params["beta[1]"], x[i - 1], sigma2
                )
            elif not params:
                raise RuntimeError("initial backtest GARCH fit failed to converge")

        sigma_fc = np.sqrt(sigma2) / rescale  # decimal, 1-day ahead for day i
        nu = params.get("nu", 8.0)
        scale = sigma_fc * np.sqrt((nu - 2) / nu)
        losses_hist = -r.iloc[max(0, i - hs_window) : i].to_numpy()
        realised = float(r.iloc[i])

        for alpha in settings.var.confidences:
            var_param = float(scale * stats.t.ppf(alpha, nu))
            var_hist = float(np.quantile(losses_hist, alpha))
            for method, var_fc in (("parametric_t", var_param), ("historical", var_hist)):
                rows.append(
                    {
                        "date": r.index[i],
                        "method": method,
                        "confidence": alpha,
                        "var_forecast": var_fc,
                        "realised": realised,
                        "breach": bool(-realised > var_fc),
                    }
                )

        # roll conditional variance forward with today's realised return
        sigma2 = _garch_sigma2_step(
            params["omega"], params["alpha[1]"], params["beta[1]"], x[i], sigma2
        )

    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    """Per (method, confidence): breach stats, test p-values, Basel traffic light."""
    rows = []
    for (method, alpha), grp in bt.groupby(["method", "confidence"]):
        grp = grp.sort_values("date")
        breaches = grp["breach"].to_numpy()
        n, x = len(breaches), int(breaches.sum())
        lr_uc, p_uc = kupiec_pof(x, n, float(alpha))
        lr_ind, p_ind = christoffersen_independence(breaches)
        lr_cc, p_cc = conditional_coverage(breaches, float(alpha))
        rows.append(
            {
                "method": method,
                "confidence": float(alpha),
                "n_obs": n,
                "n_breaches": x,
                "breach_rate": x / n,
                "expected_rate": 1 - float(alpha),
                "kupiec_lr": lr_uc,
                "kupiec_p": p_uc,
                "christoffersen_p": p_ind,
                "conditional_coverage_p": p_cc,
                # Basel traffic-light zones are defined for 99% VaR only
                "traffic_light": basel_traffic_light(x, n) if abs(alpha - 0.99) < 1e-9 else "n/a",
            }
        )
    return pd.DataFrame(rows)
