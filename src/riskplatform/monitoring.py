"""Data-quality, drift, and VaR-breach monitoring.

Each check returns OK / WARN / ALERT with a human-readable detail string. The full
report is written to monitor_status.json (the dashboard's alert banner reads it) and
ALERT/WARN rows are appended to alerts_history.parquet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import PORTFOLIO_TICKER, MonitoringConfig, Settings

OK, WARN, ALERT = "OK", "WARN", "ALERT"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def check_missing_days(
    prices: pd.DataFrame, calendar: pd.DatetimeIndex, tickers: list[str]
) -> CheckResult:
    """Coverage of each ticker on the master (ASX) calendar over the last 30 sessions."""
    recent = calendar[-30:]
    gaps = {}
    for t in tickers:
        have = set(pd.to_datetime(prices.loc[prices["ticker"] == t, "date"]))
        n_missing = sum(1 for d in recent if d not in have)
        if n_missing:
            gaps[t] = n_missing
    # FX/crypto legitimately miss some ASX days pre-ffill; only sustained gaps alert
    worst = max(gaps.values(), default=0)
    status = ALERT if worst > 10 else WARN if worst > 5 else OK
    detail = f"missing sessions in last 30: {gaps}" if gaps else "all tickers present"
    return CheckResult("missing_days", status, detail)


def check_stale_prices(aligned: pd.DataFrame, cfg: MonitoringConfig) -> CheckResult:
    """Run-length of unchanged prices at the end of each series (ffill or dead feed)."""
    stale = {}
    for t in aligned.columns:
        s = aligned[t].dropna()
        if len(s) < 2:
            stale[t] = len(s)
            continue
        changed = s.ne(s.shift())
        last_change = changed[::-1].idxmax()
        stale_run = int((s.index > last_change).sum())
        if stale_run >= cfg.stale_days_alert:
            stale[t] = stale_run
    status = ALERT if stale else OK
    detail = f"stale price runs >= {cfg.stale_days_alert}d: {stale}" if stale else "no stale series"
    return CheckResult("stale_prices", status, detail)


def check_extreme_jumps(returns: pd.DataFrame, cfg: MonitoringConfig) -> CheckResult:
    """Latest |return| vs rolling vol — z > threshold flags a data error or true shock."""
    jumps = {}
    for t in returns.columns:
        s = returns[t].dropna()
        if len(s) < cfg.jump_vol_window + 1:
            continue
        vol = s.iloc[-(cfg.jump_vol_window + 1) : -1].std()
        if vol > 0:
            z = abs(float(s.iloc[-1])) / float(vol)
            if z > cfg.jump_zscore:
                jumps[t] = round(z, 1)
    status = ALERT if jumps else OK
    detail = f"|z| > {cfg.jump_zscore}: {jumps}" if jumps else "no extreme jumps"
    return CheckResult("extreme_jumps", status, detail)


def compute_psi(current: np.ndarray, reference: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the reference window."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_prop = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_prop = np.histogram(current, bins=edges)[0] / len(current)
    ref_prop = np.clip(ref_prop, 1e-6, None)
    cur_prop = np.clip(cur_prop, 1e-6, None)
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def check_return_drift(port_r: pd.Series, cfg: MonitoringConfig) -> CheckResult:
    r = port_r.dropna().to_numpy()
    far, near = cfg.psi_reference_window
    if len(r) < far + 10:
        return CheckResult("psi_drift", OK, f"insufficient history for PSI ({len(r)} obs)")
    reference = r[-far:-near]
    current = r[-cfg.psi_current_window :]
    psi = compute_psi(current, reference, cfg.psi_bins)
    status = ALERT if psi > cfg.psi_alert else WARN if psi > cfg.psi_warn else OK
    return CheckResult(
        "psi_drift", status, f"PSI={psi:.4f} (warn>{cfg.psi_warn}, alert>{cfg.psi_alert})"
    )


def check_var_breach(port_r: pd.Series, risk_metrics: pd.DataFrame | None) -> CheckResult:
    """Did today's realised portfolio loss exceed the previous run's 99% 1-day VaR?"""
    if risk_metrics is None or risk_metrics.empty or len(port_r.dropna()) == 0:
        return CheckResult("var_breach", OK, "no prior VaR to compare against")
    today = port_r.dropna().index[-1]
    prior = risk_metrics[
        (risk_metrics["method"] == "parametric_t")
        & (risk_metrics["horizon"] == 1)
        & (risk_metrics["confidence"] == 0.99)
        & (pd.to_datetime(risk_metrics["date"]) < today)
    ]
    if prior.empty:
        return CheckResult("var_breach", OK, "no prior VaR to compare against")
    var_row = prior.sort_values("date").iloc[-1]
    realised_loss = -float(port_r.dropna().iloc[-1])
    if realised_loss > float(var_row["var"]):
        return CheckResult(
            "var_breach",
            ALERT,
            f"{today.date()}: loss {realised_loss:.4%} breached prior 99% 1-day VaR "
            f"{float(var_row['var']):.4%} (forecast on {pd.Timestamp(var_row['date']).date()})",
        )
    return CheckResult(
        "var_breach",
        OK,
        f"{today.date()}: return {float(port_r.dropna().iloc[-1]):.4%} within prior 99% VaR "
        f"{float(var_row['var']):.4%}",
    )


def run_monitoring(
    prices: pd.DataFrame,
    aligned: pd.DataFrame,
    returns_wide: pd.DataFrame,
    port_r: pd.Series,
    risk_metrics: pd.DataFrame | None,
    calendar: pd.DatetimeIndex,
    settings: Settings,
) -> list[CheckResult]:
    cfg = settings.monitoring
    return [
        check_missing_days(prices, calendar, settings.all_tickers),
        check_stale_prices(aligned, cfg),
        check_extreme_jumps(returns_wide, cfg),
        check_return_drift(port_r, cfg),
        check_var_breach(port_r, risk_metrics),
    ]


def report_to_dict(checks: list[CheckResult]) -> dict:
    worst = ALERT if any(c.status == ALERT for c in checks) else (
        WARN if any(c.status == WARN for c in checks) else OK
    )
    return {"overall": worst, "checks": [asdict(c) for c in checks]}
