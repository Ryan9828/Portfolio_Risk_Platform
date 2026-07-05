"""Announcement-intelligence tests — fully offline (fake feed payloads, stub LLM client)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from riskplatform import artifacts
from riskplatform.intel import evals, extract
from riskplatform.intel.events import run_event_study
from riskplatform.intel.ingest import asx_codes, parse_feed

FEED_PAYLOAD = {
    "data": {
        "items": [
            {
                "documentKey": "2924-001",
                "date": "2026-06-19T07:35:11.000Z",
                "announcementType": "PROGRESS REPORT",
                "headline": "FY26 guidance downgrade",
                "isPriceSensitive": True,
            },
            {
                "documentKey": "2924-002",
                "date": "2026-06-18T04:55:36.000Z",
                "announcementType": "ISSUED CAPITAL",
                "headline": "Notification regarding unquoted securities",
                "isPriceSensitive": False,
            },
            {"documentKey": "", "date": "2026-06-17T00:00:00.000Z", "headline": "dropped"},
        ]
    }
}


class StubClient:
    """Mimics anthropic.Anthropic().messages.create for structured-output extraction."""

    def __init__(self, payload: dict | None = None, fail_keys: set[str] | None = None):
        self.payload = payload or {
            "event_type": "guidance_update",
            "materiality": "high",
            "sentiment": -0.8,
            "rationale": "Guidance downgrade is volatility-moving.",
        }
        self.fail_keys = fail_keys or set()
        self.calls: list[str] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        doc = kwargs["messages"][0]["content"]
        self.calls.append(doc)
        if any(key in doc for key in self.fail_keys):
            raise RuntimeError("boom")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_creation_input_tokens=300,
                cache_read_input_tokens=0,
            ),
        )


def _seed_announcements(data_dir, n=2):
    ann = parse_feed(FEED_PAYLOAD, "WOW.AX").head(n)
    artifacts.upsert(data_dir / artifacts.ANNOUNCEMENTS, ann, keys=["doc_key"])
    return ann


def test_asx_codes_only_ax_tickers(settings):
    codes = asx_codes(settings)
    assert all(ticker.endswith(".AX") for ticker in codes.values())
    assert set(codes) == {t[:-3] for t in settings.portfolio_tickers if t.endswith(".AX")}


def test_parse_feed_schema_and_drops_keyless():
    df = parse_feed(FEED_PAYLOAD, "WOW.AX")
    assert list(df["doc_key"]) == ["2924-001", "2924-002"]
    assert df["price_sensitive"].tolist() == [True, False]
    assert df["date"].dt.tz is None  # tz-naive at the boundary, like prices


def test_extraction_idempotent_and_costed(tmp_path, settings):
    _seed_announcements(tmp_path)
    client = StubClient()
    signals = extract.run_extraction(settings.intel, tmp_path, client=client)
    assert len(signals) == 2 and len(client.calls) == 2
    row = signals.iloc[0]
    assert row["event_type"] == "guidance_update" and row["materiality"] == "high"
    # (100 + 300) uncached+cache-write input, 50 output at configured $/MTok
    assert row["cost_usd"] == pytest.approx((400 * 5.0 + 50 * 25.0) / 1e6)

    # second run: nothing pending, no new API calls
    extract.run_extraction(settings.intel, tmp_path, client=client)
    assert len(client.calls) == 2


def test_extraction_survives_single_failure(tmp_path, settings):
    _seed_announcements(tmp_path)
    client = StubClient(fail_keys={"guidance downgrade"})
    signals = extract.run_extraction(settings.intel, tmp_path, client=client)
    assert len(signals) == 1  # the failing announcement is skipped, not fatal


def test_extraction_respects_cost_cap(tmp_path, settings):
    _seed_announcements(tmp_path)
    cfg = settings.intel.__class__(max_new_per_run=1)
    signals = extract.run_extraction(cfg, tmp_path, client=StubClient())
    assert len(signals) == 1


def test_event_study_detects_vol_regime_change(tmp_path, settings):
    rng = np.random.default_rng(3)
    dates = pd.bdate_range(end="2026-06-30", periods=200)
    event_pos = 150
    r = rng.standard_normal(200) * 0.01
    r[event_pos] = -0.08  # announcement-day shock
    r[event_pos + 1 :] = rng.standard_normal(200 - event_pos - 1) * 0.03  # elevated regime
    returns = pd.concat(
        [
            pd.DataFrame({"date": dates, "ticker": "WOW.AX", "log_return": r}),
            pd.DataFrame({"date": dates, "ticker": settings.index_ticker, "log_return": rng.standard_normal(200) * 0.008}),
        ]
    )
    artifacts.overwrite(tmp_path / artifacts.RETURNS, returns)

    ann = pd.DataFrame(
        [
            {"doc_key": "k1", "date": dates[event_pos], "ticker": "WOW.AX",
             "headline": "downgrade", "ann_type": "X", "price_sensitive": True},
            {"doc_key": "k2", "date": dates[2], "ticker": "WOW.AX",  # too early: dropped
             "headline": "old", "ann_type": "X", "price_sensitive": False},
        ]
    )
    artifacts.upsert(tmp_path / artifacts.ANNOUNCEMENTS, ann, keys=["doc_key"])
    signals = pd.DataFrame([{"doc_key": "k1"}, {"doc_key": "k2"}])
    artifacts.upsert(tmp_path / artifacts.ANNOUNCEMENT_SIGNALS, signals, keys=["doc_key"])

    study = run_event_study(settings, tmp_path)
    assert list(study["doc_key"]) == ["k1"]
    row = study.iloc[0]
    assert row["vol_ratio"] > 1.5 and bool(row["reacted"]) and abs(row["abnormal_return"]) > 0.05


def test_event_study_empty_without_signals(tmp_path, settings):
    study = run_event_study(settings, tmp_path)
    assert study.empty and (tmp_path / artifacts.EVENT_STUDY).exists()


def test_eval_scoring(tmp_path, monkeypatch):
    signals = pd.DataFrame(
        [
            {"doc_key": "a", "event_type": "guidance_update", "materiality": "high", "model": "m"},
            {"doc_key": "b", "event_type": "admin_or_compliance", "materiality": "low", "model": "m"},
            {"doc_key": "c", "event_type": "earnings_result", "materiality": "high", "model": "m"},
        ]
    )
    artifacts.upsert(tmp_path / artifacts.ANNOUNCEMENT_SIGNALS, signals, keys=["doc_key"])
    golden = tmp_path / "golden_set.csv"
    pd.DataFrame(
        [
            {"doc_key": "a", "label_event_type": "guidance_update", "label_materiality": "high"},
            {"doc_key": "b", "label_event_type": "admin_or_compliance", "label_materiality": "low"},
            {"doc_key": "c", "label_event_type": "earnings_result", "label_materiality": "medium"},
        ]
    ).to_csv(golden, index=False)

    metrics = evals.score(tmp_path, golden_path=golden)
    assert metrics["n_labelled"] == 3
    assert metrics["event_type_accuracy"] == 1.0
    assert metrics["materiality_accuracy"] == pytest.approx(2 / 3)
    assert metrics["high_materiality_precision"] == pytest.approx(0.5)  # 1 of 2 predicted highs
    assert metrics["high_materiality_recall"] == 1.0  # the only labelled high was found
    assert (tmp_path / artifacts.INTEL_EVAL).exists()
