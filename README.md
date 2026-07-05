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
     ├─ ingest      Yahoo Finance adjusted closes, 5 tickers, idempotent upsert
     ├─ returns     ASX master calendar, ffill the FX benchmark, portfolio log returns
     ├─ model       GARCH(1,1)-t per series, fallback chain → EWMA
     ├─ risk        VaR + ES, 95%/99% × 1d/10d × {parametric-n, parametric-t,
     │              historical, Monte Carlo (10k correlated paths)}
     ├─ backtest    500-day walk-forward; Kupiec, Christoffersen, Basel traffic light
     ├─ monitor     missing data · stale feeds · extreme jumps · PSI drift · VaR breaches
     ├─ alert       GitHub issue per ALERT (deduped, labelled risk-alert)
     ├─ intel       ASX announcements → Claude → typed risk signals (event type,
     │              materiality, sentiment) + event study vs abnormal returns and vol
     └─ commit      parquet artifacts → repo → Streamlit Cloud auto-redeploys
```

The Streamlit app is a **pure reader** of the committed artifacts — no model code runs in
the dashboard, so what you see is exactly what the audited pipeline produced.

## Portfolio

Three ASX-listed ETFs — 40% VGS (international shares), 30% VAS (Australian shares),
30% NDQ (Nasdaq 100) — benchmarked against the S&P/ASX 200 and AUD/USD (VGS and NDQ are
unhedged, so the currency matters). Configured in
[config/portfolio.yaml](config/portfolio.yaml).

## Announcement intelligence (NLP layer)

GARCH only learns about a shock after it shows up in returns. The intel stage adds a
**leading indicator**: every run pulls the latest ASX announcements for the portfolio
names and has Claude convert each headline into a typed risk signal under a strict JSON
schema — event type (guidance update, capital raising, M&A, …), materiality, sentiment.
Every signal row records the model, token usage, dollar cost and latency, and signals
are extracted exactly once per announcement (upsert keyed on the exchange's document ID,
with a per-run call cap as a cost guard).

Two things make this more than "calling an API":

- **Evaluation** — a blind-labelled golden set (`riskplatform.intel.evals`) scores the
  extraction with per-class precision/recall and, for the decision that matters
  downstream, precision/recall on *high-materiality* classification. Metrics render on
  the dashboard next to the signals they audit.
- **Event study** — for each signal the platform measures the event-day abnormal return
  (vs the ASX 200) and the realised-vol regime change (20 sessions post / pre), testing
  the hypothesis that flagged announcements precede volatility the GARCH layer hasn't
  seen yet.

Run it with `python -m riskplatform.pipeline intel` (needs `ANTHROPIC_API_KEY`; without
one, ingestion and the event study still run and extraction skips cleanly).

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
