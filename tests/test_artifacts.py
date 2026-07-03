"""Idempotence guarantees of the artifact upsert."""

import pandas as pd

from riskplatform.artifacts import upsert


def _rows(dates, values):
    return pd.DataFrame({"date": pd.to_datetime(dates), "ticker": "X", "adj_close": values})


def test_upsert_is_idempotent(tmp_path):
    path = tmp_path / "prices.parquet"
    rows = _rows(["2026-01-01", "2026-01-02"], [1.0, 2.0])
    first = upsert(path, rows, keys=["date", "ticker"])
    second = upsert(path, rows, keys=["date", "ticker"])
    pd.testing.assert_frame_equal(first, second)
    assert len(second) == 2


def test_upsert_overlap_overwrites_not_duplicates(tmp_path):
    path = tmp_path / "prices.parquet"
    upsert(path, _rows(["2026-01-01", "2026-01-02"], [1.0, 2.0]), keys=["date", "ticker"])
    # overlap window re-fetches 01-02 with a revised (dividend-adjusted) value
    out = upsert(path, _rows(["2026-01-02", "2026-01-03"], [2.5, 3.0]), keys=["date", "ticker"])
    assert len(out) == 3
    assert out.loc[out["date"] == pd.Timestamp("2026-01-02"), "adj_close"].item() == 2.5
