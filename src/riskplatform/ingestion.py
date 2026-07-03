"""Daily price ingestion from Yahoo Finance.

Prices are stored long-format (date, ticker, adj_close) and upserted keyed on
(date, ticker): incremental runs re-fetch a small overlap window so late dividend
adjustments self-heal, and re-runs never duplicate rows.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import artifacts
from .config import Settings

log = logging.getLogger(__name__)

RETRY_DELAYS = [5, 15, 45]


def fetch_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Batched download of adjusted closes, long format, tz-naive dates.

    Raises RuntimeError if no ticker returns any data after all retries — a hard
    ingestion failure must surface as a red pipeline run, not a silent no-op.
    """
    last_err: Exception | None = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            log.warning("yfinance retry %d in %ds", attempt, delay)
            time.sleep(delay)
        try:
            raw = yf.download(
                tickers,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                threads=False,
                group_by="ticker",
                progress=False,
            )
        except Exception as err:  # network / API flakiness
            last_err = err
            continue
        frames = []
        for ticker in tickers:
            try:
                closes = raw[ticker]["Close"] if len(tickers) > 1 else raw["Close"]
            except KeyError:
                log.warning("no data returned for %s", ticker)
                continue
            closes = closes.dropna()
            if closes.empty:
                log.warning("empty series for %s", ticker)
                continue
            frames.append(
                pd.DataFrame(
                    {
                        # tz-naive normalised dates at the boundary; nothing
                        # downstream ever sees a timezone
                        "date": pd.to_datetime(closes.index).tz_localize(None).normalize(),
                        "ticker": ticker,
                        "adj_close": closes.to_numpy(dtype=float),
                    }
                )
            )
        if frames:
            out = pd.concat(frames, ignore_index=True)
            missing = set(tickers) - set(out["ticker"].unique())
            if missing:
                log.warning("tickers with no data this fetch: %s", sorted(missing))
            return out
        last_err = RuntimeError("yfinance returned no data for any ticker")
    raise RuntimeError(f"price fetch failed after {len(RETRY_DELAYS) + 1} attempts: {last_err}")


def run_ingestion(settings: Settings, data_dir: Path) -> pd.DataFrame:
    """Cold-start backfill or incremental fetch; returns the full upserted price table."""
    path = data_dir / artifacts.PRICES
    existing = artifacts.read(path)
    today = date.today()

    if existing is None or existing.empty:
        start = today - timedelta(days=int(settings.backfill_years * 365.25))
        log.info("cold start: backfilling from %s", start)
    else:
        last = pd.to_datetime(existing["date"]).max().date()
        start = last - timedelta(days=settings.ingest_overlap_days)
        log.info("incremental fetch from %s (last stored %s)", start, last)

    new_rows = fetch_prices(settings.all_tickers, start, today)
    return artifacts.upsert(path, new_rows, keys=["date", "ticker"])
