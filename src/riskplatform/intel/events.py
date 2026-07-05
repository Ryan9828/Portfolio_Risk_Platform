"""Event study: do LLM-flagged announcements precede returns and volatility moves?

For each extracted signal this computes, around the first trading session on or
after the announcement:

    abnormal_return   ticker log return minus index log return on the event day
    pre_vol/post_vol  realised daily vol over the N sessions before / after
    vol_ratio         post/pre — >1 means the stock entered a higher-vol regime
    max_abs_z         largest |return|/pre_vol over the reaction window
    reacted           max_abs_z >= threshold (a move GARCH would call a shock)

The table is rebuilt from scratch each run (cheap arithmetic on stored returns),
matching the overwrite pattern used for the backtest artifact. Rows without
enough return history around the event are dropped rather than half-filled.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import artifacts
from ..config import Settings

log = logging.getLogger(__name__)

STUDY_COLUMNS = [
    "doc_key",
    "ticker",
    "event_date",
    "abnormal_return",
    "pre_vol",
    "post_vol",
    "vol_ratio",
    "max_abs_z",
    "reacted",
]


def _study_one(
    ann_date: pd.Timestamp,
    ticker_r: pd.Series,
    index_r: pd.Series,
    window: int,
    reaction_days: int,
    jump_z: float,
) -> dict | None:
    """Event-window arithmetic for one announcement; None if history is insufficient."""
    future = ticker_r.index[ticker_r.index >= ann_date]
    if len(future) == 0:
        return None
    event_date = future[0]
    pos = ticker_r.index.get_loc(event_date)
    if pos < window or pos + window >= len(ticker_r):
        return None

    pre = ticker_r.iloc[pos - window : pos]
    post = ticker_r.iloc[pos + 1 : pos + 1 + window]
    pre_vol = float(pre.std(ddof=1))
    post_vol = float(post.std(ddof=1))
    if not np.isfinite(pre_vol) or pre_vol == 0.0:
        return None

    abnormal = float(ticker_r.iloc[pos] - index_r.reindex([event_date]).fillna(0.0).iloc[0])
    reaction = ticker_r.iloc[pos : pos + 1 + reaction_days]
    max_abs_z = float((reaction.abs() / pre_vol).max())

    return {
        "event_date": event_date,
        "abnormal_return": abnormal,
        "pre_vol": pre_vol,
        "post_vol": post_vol,
        "vol_ratio": post_vol / pre_vol,
        "max_abs_z": max_abs_z,
        "reacted": max_abs_z >= jump_z,
    }


def run_event_study(settings: Settings, data_dir: Path) -> pd.DataFrame:
    """Join signals with returns and rebuild the event-study artifact."""
    ann = artifacts.read(data_dir / artifacts.ANNOUNCEMENTS)
    signals = artifacts.read(data_dir / artifacts.ANNOUNCEMENT_SIGNALS)
    returns = artifacts.read(data_dir / artifacts.RETURNS)
    empty = pd.DataFrame(columns=STUDY_COLUMNS)
    if ann is None or signals is None or returns is None or signals.empty:
        artifacts.overwrite(data_dir / artifacts.EVENT_STUDY, empty)
        return empty

    returns["date"] = pd.to_datetime(returns["date"])
    wide = returns.pivot_table(index="date", columns="ticker", values="log_return").sort_index()
    index_r = wide.get(settings.index_ticker, pd.Series(dtype=float))

    cfg = settings.intel
    events = ann.merge(signals[["doc_key"]], on="doc_key")
    events["date"] = pd.to_datetime(events["date"])
    rows: list[dict] = []
    for _, ev in events.iterrows():
        if ev["ticker"] not in wide.columns:
            continue
        result = _study_one(
            ev["date"],
            wide[ev["ticker"]].dropna(),
            index_r,
            cfg.event_vol_window,
            cfg.event_reaction_days,
            cfg.event_jump_zscore,
        )
        if result is not None:
            rows.append({"doc_key": ev["doc_key"], "ticker": ev["ticker"], **result})

    study = pd.DataFrame(rows, columns=STUDY_COLUMNS) if rows else empty
    log.info("event study: %d of %d signals had sufficient return history", len(study), len(events))
    artifacts.overwrite(data_dir / artifacts.EVENT_STUDY, study)
    return study
