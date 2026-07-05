"""LLM extraction: one announcement headline -> one typed risk signal.

Uses the Claude API with a strict JSON schema (`output_config.format`), so every
response parses; the system prompt is cached across calls in a run. Each signal
row records the model, token usage, cost and latency — the extraction step is
audited the same way the risk models are.

Signals upsert keyed on `doc_key`: an announcement is only ever paid for once,
and re-runs skip everything already extracted.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from .. import artifacts
from ..config import IntelConfig

log = logging.getLogger(__name__)

EVENT_TYPES = [
    "earnings_result",
    "guidance_update",
    "capital_raising",
    "dividend_or_buyback",
    "acquisition_or_divestment",
    "management_change",
    "operational_update",
    "legal_or_regulatory",
    "admin_or_compliance",
    "other",
]
MATERIALITY = ["high", "medium", "low"]

SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "materiality": {
            "type": "string",
            "enum": MATERIALITY,
            "description": "Expected impact on the stock's risk profile: high = likely "
            "volatility-moving (downgrades, raises, major M&A), medium = worth watching, "
            "low = routine filings with no market impact.",
        },
        "sentiment": {
            "type": "number",
            "description": "Expected market reception from -1.0 (clearly negative) through "
            "0.0 (neutral/unclear) to 1.0 (clearly positive).",
        },
        "rationale": {"type": "string", "description": "One short sentence."},
    },
    "required": ["event_type", "materiality", "sentiment", "rationale"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You classify ASX company announcements into risk signals for a market-risk \
platform. You see only the headline and feed metadata, not the document body — classify from \
what the headline states, and when it is genuinely ambiguous choose materiality "low" and \
sentiment 0.0 rather than guessing.

Routine administrative filings (Appendix 3Y/3B, change of director's interest, cessation of \
securities, becoming/ceasing substantial holder, unquoted security notices) are \
"admin_or_compliance" with low materiality. Earnings, guidance changes, capital raisings, \
significant M&A and regulator actions are the announcements that move volatility — reserve \
"high" materiality for those. The exchange's price-sensitive flag is a strong hint but is \
sometimes missing on genuinely material announcements."""

SIGNAL_COLUMNS = [
    "doc_key",
    "event_type",
    "materiality",
    "sentiment",
    "rationale",
    "model",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "latency_s",
    "extracted_utc",
]


def pending_announcements(data_dir: Path) -> pd.DataFrame:
    """Announcements with no signal yet, newest first."""
    ann = artifacts.read(data_dir / artifacts.ANNOUNCEMENTS)
    if ann is None or ann.empty:
        return pd.DataFrame()
    signals = artifacts.read(data_dir / artifacts.ANNOUNCEMENT_SIGNALS)
    done = set() if signals is None else set(signals["doc_key"])
    return ann[~ann["doc_key"].isin(done)].sort_values("date", ascending=False)


def _announcement_prompt(row: pd.Series) -> str:
    return (
        f"Ticker: {row['ticker']}\n"
        f"Headline: {row['headline']}\n"
        f"Feed category: {row['ann_type']}\n"
        f"Exchange price-sensitive flag: {bool(row['price_sensitive'])}"
    )


def extract_one(client, row: pd.Series, cfg: IntelConfig) -> dict:
    """One API call -> one signal row (raises on API failure; caller decides policy)."""
    start = time.perf_counter()
    response = client.messages.create(
        model=cfg.model,
        max_tokens=1024,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        output_config={"format": {"type": "json_schema", "schema": SIGNAL_SCHEMA}},
        messages=[{"role": "user", "content": _announcement_prompt(row)}],
    )
    latency = time.perf_counter() - start
    payload = json.loads(next(b.text for b in response.content if b.type == "text"))
    usage = response.usage
    in_tokens = usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
    cost = (
        in_tokens * cfg.input_usd_per_mtok + usage.output_tokens * cfg.output_usd_per_mtok
    ) / 1_000_000
    return {
        "doc_key": row["doc_key"],
        "event_type": payload["event_type"],
        "materiality": payload["materiality"],
        "sentiment": max(-1.0, min(1.0, float(payload["sentiment"]))),
        "rationale": payload["rationale"],
        "model": cfg.model,
        "input_tokens": in_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost,
        "latency_s": latency,
        "extracted_utc": pd.Timestamp.now("UTC").isoformat(),
    }


def run_extraction(cfg: IntelConfig, data_dir: Path, client=None) -> pd.DataFrame:
    """Extract signals for pending announcements, capped at cfg.max_new_per_run.

    Returns the full upserted signals table. Skips cleanly (with a log line) when
    no API credentials are available, so the offline pipeline stays green.
    """
    pending = pending_announcements(data_dir)
    if pending.empty:
        log.info("no new announcements to extract")
        return artifacts.read(data_dir / artifacts.ANNOUNCEMENT_SIGNALS)

    if client is None:
        try:
            import anthropic

            client = anthropic.Anthropic()
        except Exception as err:
            log.warning("Claude client unavailable (%s) — skipping extraction", err)
            return artifacts.read(data_dir / artifacts.ANNOUNCEMENT_SIGNALS)
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            log.warning("no ANTHROPIC_API_KEY/AUTH_TOKEN — skipping extraction")
            return artifacts.read(data_dir / artifacts.ANNOUNCEMENT_SIGNALS)

    batch = pending.head(cfg.max_new_per_run)
    rows: list[dict] = []
    for _, ann in batch.iterrows():
        try:
            rows.append(extract_one(client, ann, cfg))
        except Exception as err:  # one bad call must not lose the batch
            log.error("extraction failed for %s (%s): %s", ann["doc_key"], ann["headline"], err)

    if not rows:
        return artifacts.read(data_dir / artifacts.ANNOUNCEMENT_SIGNALS)
    new_signals = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    log.info(
        "extracted %d signals (%d pending, cap %d) — $%.4f total",
        len(new_signals),
        len(pending),
        cfg.max_new_per_run,
        new_signals["cost_usd"].sum(),
    )
    return artifacts.upsert(data_dir / artifacts.ANNOUNCEMENT_SIGNALS, new_signals, keys=["doc_key"])
