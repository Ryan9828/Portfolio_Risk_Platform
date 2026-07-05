"""ASX announcement ingestion from the public Markit Digital feed.

Announcements are stored long-format keyed on `doc_key` (the exchange's stable
document identifier), so re-runs upsert in place and never duplicate rows. Only
`.AX` portfolio tickers are covered — FX/crypto/index series have no announcements.

Note: the unauthenticated feed silently caps the response at the 5 most recent
announcements per ticker regardless of `itemsPerPage`/`page` (verified 2026-07).
Coverage therefore builds up through the daily runs rather than one backfill.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .. import artifacts
from ..config import Settings

log = logging.getLogger(__name__)

FEED_URL = "https://asx.api.markitdigital.com/asx-research/1.0/companies/{code}/announcements"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
RETRY_DELAYS = [5, 15]

COLUMNS = ["doc_key", "date", "ticker", "headline", "ann_type", "price_sensitive"]


def asx_codes(settings: Settings) -> dict[str, str]:
    """Map ASX code (e.g. 'VGS') -> portfolio ticker (e.g. 'VGS.AX')."""
    return {t[:-3]: t for t in settings.portfolio_tickers if t.endswith(".AX")}


def parse_feed(payload: dict, ticker: str) -> pd.DataFrame:
    """Normalise one feed response into the announcements schema (tz-naive dates)."""
    items = payload.get("data", {}).get("items", [])
    rows = [
        {
            "doc_key": item["documentKey"],
            "date": pd.Timestamp(item["date"]).tz_convert("Australia/Sydney").tz_localize(None).normalize(),
            "ticker": ticker,
            "headline": str(item.get("headline", "")).strip(),
            "ann_type": str(item.get("announcementType", "")).strip(),
            "price_sensitive": bool(item.get("isPriceSensitive", False)),
        }
        for item in items
        if item.get("documentKey") and item.get("headline")
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_announcements(code: str, ticker: str, count: int) -> pd.DataFrame:
    """Fetch recent announcements for one ASX code; empty frame on persistent failure.

    A dead announcements feed must not fail the whole run — prices and risk
    metrics are still worth computing — so this logs and degrades instead of raising.
    """
    last_err: Exception | None = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            log.warning("announcement feed retry %d for %s in %ds", attempt, code, delay)
            time.sleep(delay)
        try:
            resp = requests.get(
                FEED_URL.format(code=code.lower()),
                headers=HEADERS,
                params={"page": 0, "itemsPerPage": count},
                timeout=30,
            )
            resp.raise_for_status()
            return parse_feed(resp.json(), ticker)
        except Exception as err:  # network / schema flakiness
            last_err = err
    log.error("announcement fetch failed for %s: %s", code, last_err)
    return pd.DataFrame(columns=COLUMNS)


def run_intel_ingestion(settings: Settings, data_dir: Path) -> pd.DataFrame:
    """Fetch all portfolio tickers and upsert into the announcements artifact."""
    frames = [
        fetch_announcements(code, ticker, settings.intel.per_ticker_fetch)
        for code, ticker in sorted(asx_codes(settings).items())
    ]
    new_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    log.info("fetched %d announcements across %d tickers", len(new_rows), len(frames))
    return artifacts.upsert(data_dir / artifacts.ANNOUNCEMENTS, new_rows, keys=["doc_key"])
