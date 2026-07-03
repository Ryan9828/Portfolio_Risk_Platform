"""Shared helpers for the Streamlit dashboard.

The dashboard is a pure READER of committed artifacts — no model fitting happens
here (Streamlit Cloud memory stays low and pages load instantly). Anything shown
that is not in an artifact (rolling/EWMA vol, correlations) is cheap arithmetic
on the returns table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import artifacts
from .config import DEFAULT_DATA_DIR, PORTFOLIO_TICKER

# Dark-mode categorical palette (validated), fixed slot order — never cycled.
SERIES = ["#3987e5", "#199e70", "#c98500", "#9085e9", "#e66767"]
STATUS = {"OK": "#0ca30c", "WARN": "#fab219", "ALERT": "#d03b3b"}
TRAFFIC = {"green": "#0ca30c", "yellow": "#fab219", "red": "#d03b3b", "n/a": "#898781"}
INK_MUTED = "#898781"
GRID = "#2c2c2a"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color="#c3c2b7"),
    xaxis=dict(gridcolor=GRID, zeroline=False),
    yaxis=dict(gridcolor=GRID, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)


def load_prices() -> pd.DataFrame | None:
    return artifacts.read(DEFAULT_DATA_DIR / artifacts.PRICES)


def load_returns() -> pd.DataFrame | None:
    return artifacts.read(DEFAULT_DATA_DIR / artifacts.RETURNS)


def load_risk_metrics() -> pd.DataFrame | None:
    return artifacts.read(DEFAULT_DATA_DIR / artifacts.RISK_METRICS)


def load_backtest() -> pd.DataFrame | None:
    return artifacts.read(DEFAULT_DATA_DIR / artifacts.BACKTEST)


def load_backtest_summary() -> pd.DataFrame | None:
    return artifacts.read(DEFAULT_DATA_DIR / "backtest_summary.parquet")


def load_alerts() -> pd.DataFrame | None:
    return artifacts.read(DEFAULT_DATA_DIR / artifacts.ALERTS)


def load_monitor_status() -> dict | None:
    return artifacts.read_json(DEFAULT_DATA_DIR / artifacts.MONITOR_STATUS)


def returns_wide(returns_table: pd.DataFrame) -> pd.DataFrame:
    wide = returns_table.pivot_table(index="date", columns="ticker", values="log_return")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def portfolio_series(returns_table: pd.DataFrame) -> pd.Series:
    return returns_wide(returns_table)[PORTFOLIO_TICKER].dropna()


def ewma_vol(r: pd.Series, lam: float = 0.94, annualise: bool = False) -> pd.Series:
    """RiskMetrics EWMA volatility — display-only arithmetic, not the model."""
    var = r.dropna().pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    vol = np.sqrt(var)
    return vol * np.sqrt(252) if annualise else vol
