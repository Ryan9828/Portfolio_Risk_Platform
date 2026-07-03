# Portfolio Market-Risk Platform

[![ci](https://github.com/Ryan9828/Portfolio_Risk_Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryan9828/Portfolio_Risk_Platform/actions/workflows/ci.yml)
[![daily-risk-pipeline](https://github.com/Ryan9828/Portfolio_Risk_Platform/actions/workflows/daily_pipeline.yml/badge.svg)](https://github.com/Ryan9828/Portfolio_Risk_Platform/actions/workflows/daily_pipeline.yml)

A **live, self-monitoring market-risk system** — not a notebook. Every weekday after ASX
close it ingests prices, refits GARCH volatility models, computes portfolio VaR and
Expected Shortfall three ways, backtests itself Basel-style, runs data-quality and drift
checks, files GitHub issues on alerts, and publishes the results to a public dashboard.

**Live dashboard**: https://ryan9828-portfolio-risk-platform.streamlit.app

## What it does

```
GitHub Actions (cron, weekdays 07:30 UTC ≈ ASX close + 90min)
 └─ python -m riskplatform.pipeline run
     ├─ ingest      Yahoo Finance adjusted closes, 11 tickers, idempotent upsert
     ├─ returns     ASX master calendar, ffill FX/crypto, portfolio log returns
     ├─ model       GARCH(1,1)-t per series (EGARCH for BTC), fallback chain → EWMA
     ├─ risk        VaR + ES, 95%/99% × 1d/10d × {parametric-n, parametric-t,
     │              historical, Monte Carlo (10k correlated paths)}
     ├─ backtest    500-day walk-forward; Kupiec, Christoffersen, Basel traffic light
     ├─ monitor     missing data · stale feeds · extreme jumps · PSI drift · VaR breaches
     ├─ alert       GitHub issue per ALERT (deduped, labelled risk-alert)
     └─ commit      parquet artifacts → repo → Streamlit Cloud auto-redeploys
```

The Streamlit app is a **pure reader** of the committed artifacts — no model code runs in
the dashboard, so what you see is exactly what the audited pipeline produced.

## Portfolio

8 ASX blue chips (CBA, BHP, CSL, WES, MQG, WBC, TLS, WOW) equal-weight 11.25% + 10%
BTC-USD, benchmarked against the S&P/ASX 200 and AUD/USD. Configured in
[config/portfolio.yaml](config/portfolio.yaml).

## Documentation

| Document | Audience |
|---|---|
| [docs/how_it_works.md](docs/how_it_works.md) | Theory from zero — returns, GARCH, VaR/ES, backtesting, monitoring, and the nightly cycle, assuming no prior risk knowledge |
| [docs/how_it_was_built.md](docs/how_it_was_built.md) | Engineering — architecture, design decisions and their reasons, testing, operations, extension points |
| [docs/methodology.md](docs/methodology.md) | Formal audit-style model specification (also rendered on the dashboard's Methodology page) |

Methodology highlights:

- Multi-day parametric VaR uses the GARCH **variance term structure**, not naive √h scaling.
- Monte Carlo simulates correlated multi-asset GARCH paths (eigenvalue-clipped residual
  correlation, fixed seed → reproducible committed artifacts).
- The walk-forward backtest refits every 5 sessions and rolls conditional variance forward
  daily, so every VaR forecast is genuinely out-of-sample.
- PSI drift bands (0.10 / 0.25) mirror standard model-risk governance thresholds.

## Run it locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m riskplatform.pipeline backfill   # 3 years of history, ~15s
.venv/bin/streamlit run app/Home.py
```

## Tests

```bash
pytest   # 24 tests, fully offline (synthetic fixtures; known-answer tests for VaR math)
```

CI runs the suite — including an end-to-end pipeline run on synthetic data — on every push.
