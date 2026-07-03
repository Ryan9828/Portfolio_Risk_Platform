"""Offline end-to-end smoke test: full pipeline on synthetic fixture prices.

This is exactly what CI runs — zero network access.
"""

import json

import pandas as pd

from riskplatform import artifacts, pipeline


def test_full_pipeline_offline(fixture_prices, tmp_path, settings):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixture_prices.to_parquet(data_dir / artifacts.PRICES, index=False)

    exit_code = pipeline.main(
        ["run", "--offline", "--dry-run-alerts", "--data-dir", str(data_dir)]
    )
    assert exit_code == 0

    # every artifact exists with a sane schema and no NaNs in the numbers
    metrics = pd.read_parquet(data_dir / artifacts.RISK_METRICS)
    assert set(metrics.columns) >= {"date", "method", "horizon", "confidence", "var", "es"}
    assert len(metrics) == 4 * len(settings.var.horizons) * len(settings.var.confidences)
    assert not metrics[["var", "es"]].isna().any().any()
    assert (metrics["es"] >= metrics["var"]).all()

    bt = pd.read_parquet(data_dir / artifacts.BACKTEST)
    assert len(bt) == settings.backtest.window * 2 * len(settings.var.confidences)

    summary = pd.read_parquet(data_dir / pipeline.BACKTEST_SUMMARY)
    assert set(summary["method"]) == {"parametric_t", "historical"}

    status = json.loads((data_dir / artifacts.MONITOR_STATUS).read_text())
    assert status["overall"] in {"OK", "WARN", "ALERT"}
    assert len(status["checks"]) == 5

    returns = pd.read_parquet(data_dir / artifacts.RETURNS)
    assert "PORTFOLIO" in set(returns["ticker"])

    # holiday-safe: second run on the same data is a no-op success
    assert (
        pipeline.main(["run", "--offline", "--dry-run-alerts", "--data-dir", str(data_dir)])
        == 0
    )
