"""Pipeline CLI — the single entry point run by GitHub Actions and locally.

Commands:
    python -m riskplatform.pipeline backfill              first run (3y history)
    python -m riskplatform.pipeline run [--offline] [--dry-run-alerts] [--data-dir D]
    python -m riskplatform.pipeline backtest               walk-forward backtest only

`run` is holiday-safe: if no new ASX session has appeared since the last stored
risk-metric date, it logs and exits 0 without touching artifacts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import artifacts, ingestion
from .alerting import send_alerts
from .backtest import rolling_backtest, summarize_backtest
from .config import DEFAULT_DATA_DIR, PORTFOLIO_TICKER, Settings, load_settings
from .garch import fit_with_fallback
from .monitoring import report_to_dict, run_monitoring
from .returns import align_prices, build_master_calendar, build_returns_table, compute_log_returns
from .var_es import compute_all_metrics

log = logging.getLogger("riskplatform")

BACKTEST_SUMMARY = "backtest_summary.parquet"


def _load_port_returns(returns_table: pd.DataFrame) -> pd.Series:
    port = returns_table[returns_table["ticker"] == PORTFOLIO_TICKER]
    return pd.Series(
        port["log_return"].to_numpy(),
        index=pd.to_datetime(port["date"]),
        name=PORTFOLIO_TICKER,
    ).sort_index()


def cmd_run(settings: Settings, data_dir: Path, offline: bool, dry_run_alerts: bool) -> int:
    if not offline:
        ingestion.run_ingestion(settings, data_dir)

    prices = artifacts.read(data_dir / artifacts.PRICES)
    if prices is None or prices.empty:
        log.error("no price data available in %s", data_dir)
        return 1
    prices["date"] = pd.to_datetime(prices["date"])

    returns_table = build_returns_table(prices, settings)
    artifacts.overwrite(data_dir / artifacts.RETURNS, returns_table)

    port_r = _load_port_returns(returns_table)
    asof = port_r.index[-1]
    prior_metrics = artifacts.read(data_dir / artifacts.RISK_METRICS)

    if (
        prior_metrics is not None
        and not prior_metrics.empty
        and pd.to_datetime(prior_metrics["date"]).max() >= asof
    ):
        log.info("no new ASX session since %s — nothing to do (holiday-safe exit)", asof.date())
        # still honour FORCE_TEST_ALERT so the alert path can be verified any day
        send_alerts([], str(asof.date()), dry_run=dry_run_alerts)
        return 0

    calendar = build_master_calendar(prices, settings.index_ticker)
    aligned = align_prices(prices, calendar, settings.max_ffill_days)
    returns_wide = compute_log_returns(aligned)

    log.info("fitting volatility models for %d series", len(settings.all_tickers) + 1)
    fits = {
        t: fit_with_fallback(returns_wide[t], t, settings.garch, settings.min_obs_garch)
        for t in settings.all_tickers
    }
    fits[PORTFOLIO_TICKER] = fit_with_fallback(
        port_r, PORTFOLIO_TICKER, settings.garch, settings.min_obs_garch
    )

    metrics = compute_all_metrics(fits, port_r, settings, asof)
    artifacts.upsert(data_dir / artifacts.RISK_METRICS, metrics, keys=["date", "method", "horizon", "confidence"])
    log.info("risk metrics computed for %s (%d rows)", asof.date(), len(metrics))

    log.info("running %d-day walk-forward backtest", settings.backtest.window)
    bt = rolling_backtest(port_r, settings)
    artifacts.overwrite(data_dir / artifacts.BACKTEST, bt)
    artifacts.overwrite(data_dir / BACKTEST_SUMMARY, summarize_backtest(bt))

    checks = run_monitoring(prices, aligned, returns_wide, port_r, prior_metrics, calendar, settings)
    report = report_to_dict(checks)
    report.update(
        {
            "asof": str(asof.date()),
            "generated_utc": pd.Timestamp.now("UTC").isoformat(),
            "fits": {
                t: {
                    "model": f.model,
                    "dist": f.dist,
                    "converged": f.converged,
                    "nu": f.nu,
                    "sigma_1d": float(f.sigma_forecast[0]),
                }
                for t, f in fits.items()
            },
        }
    )
    artifacts.write_json(data_dir / artifacts.MONITOR_STATUS, report)

    alert_rows = pd.DataFrame(
        [
            {"date": asof, "check": c.name, "status": c.status, "detail": c.detail}
            for c in checks
            if c.status != "OK"
        ]
    )
    if not alert_rows.empty:
        artifacts.upsert(data_dir / artifacts.ALERTS, alert_rows, keys=["date", "check"])

    send_alerts(checks, str(asof.date()), dry_run=dry_run_alerts)
    log.info("pipeline complete — overall monitor status: %s", report["overall"])
    return 0


def cmd_backtest(settings: Settings, data_dir: Path) -> int:
    returns_table = artifacts.read(data_dir / artifacts.RETURNS)
    if returns_table is None:
        log.error("no returns artifact; run the pipeline first")
        return 1
    bt = rolling_backtest(_load_port_returns(returns_table), settings)
    artifacts.overwrite(data_dir / artifacts.BACKTEST, bt)
    summary = summarize_backtest(bt)
    artifacts.overwrite(data_dir / BACKTEST_SUMMARY, summary)
    print(summary.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="riskplatform")
    parser.add_argument("command", choices=["backfill", "run", "backtest"])
    parser.add_argument("--offline", action="store_true", help="skip ingestion (use stored prices)")
    parser.add_argument("--dry-run-alerts", action="store_true", help="log alerts instead of filing issues")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config) if args.config else load_settings()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.command in ("backfill", "run"):
        return cmd_run(settings, args.data_dir, offline=args.offline, dry_run_alerts=args.dry_run_alerts)
    return cmd_backtest(settings, args.data_dir)


if __name__ == "__main__":
    sys.exit(main())
