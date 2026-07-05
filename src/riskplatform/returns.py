"""Calendar alignment and log-return construction.

The master calendar is the set of observed ^AXJO trading days. Series on other
calendars (e.g. the 24/5 AUD/USD benchmark) are forward-filled onto the ASX grid
(limit `max_ffill_days`). This convention — mark everything at ASX close — is
disclosed in docs/methodology.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PORTFOLIO_TICKER, Settings


def build_master_calendar(prices: pd.DataFrame, index_ticker: str) -> pd.DatetimeIndex:
    days = prices.loc[prices["ticker"] == index_ticker, "date"]
    if days.empty:
        raise ValueError(f"no rows for index ticker {index_ticker}; cannot build calendar")
    return pd.DatetimeIndex(sorted(pd.to_datetime(days).unique()))


def align_prices(
    prices: pd.DataFrame, calendar: pd.DatetimeIndex, max_ffill: int
) -> pd.DataFrame:
    """Wide (date x ticker) adjusted closes reindexed to the master calendar."""
    wide = prices.pivot_table(index="date", columns="ticker", values="adj_close")
    wide.index = pd.to_datetime(wide.index)
    return wide.reindex(calendar).ffill(limit=max_ffill)


def compute_log_returns(aligned: pd.DataFrame) -> pd.DataFrame:
    return np.log(aligned / aligned.shift(1)).iloc[1:]


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Daily portfolio log return under fixed weights (linear approximation in
    simple-return space; documented in the methodology)."""
    missing = [t for t in weights if t not in returns.columns]
    if missing:
        raise ValueError(f"return series missing for portfolio assets: {missing}")
    w = pd.Series(weights)
    simple = np.expm1(returns[list(weights)])
    port_simple = simple.mul(w, axis=1).sum(axis=1, min_count=len(weights))
    return np.log1p(port_simple).rename(PORTFOLIO_TICKER)


def build_returns_table(prices: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Long-format returns for every ticker plus the PORTFOLIO pseudo-ticker.

    Rebuilt in full from prices each run — cheap at this scale and guarantees
    returns are always consistent with the latest (possibly re-adjusted) prices.
    """
    calendar = build_master_calendar(prices, settings.index_ticker)
    aligned = align_prices(prices, calendar, settings.max_ffill_days)
    rets = compute_log_returns(aligned)
    port = portfolio_returns(rets, settings.weights)

    long = rets.reset_index(names="date").melt(
        id_vars="date", var_name="ticker", value_name="log_return"
    )
    port_long = port.reset_index()
    port_long.columns = ["date", "log_return"]
    port_long["ticker"] = PORTFOLIO_TICKER

    out = pd.concat([long, port_long[["date", "ticker", "log_return"]]], ignore_index=True)
    return out.dropna(subset=["log_return"]).sort_values(["ticker", "date"]).reset_index(drop=True)
