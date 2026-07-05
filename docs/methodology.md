# Model Methodology

This document specifies the models, assumptions, and known limitations of the platform to
the standard expected of production risk-model documentation: a third-party reviewer should
be able to reproduce every number on the dashboard from this description plus the code.

## 1. Scope and purpose

The platform estimates **market risk** — Value-at-Risk (VaR) and Expected Shortfall (ES) —
for a fixed-weight multi-asset portfolio, validates those estimates out-of-sample, and
monitors its own inputs and outputs. It is a demonstration system: no real positions,
no intraday risk, and simple fixed weights (no rebalancing drift, no cash flows).

**Portfolio**: three ASX-listed ETFs — VGS.AX (international shares) 40%, VAS.AX
(Australian shares) 30%, NDQ.AX (Nasdaq 100) 30%. VGS and NDQ are unhedged, so AUD/USD
is a material driver of their AUD returns.
**Benchmarks** (tracked, not in portfolio VaR): S&P/ASX 200 (^AXJO), AUD/USD.

## 2. Data

- **Source**: Yahoo Finance adjusted closes (`auto_adjust=True` — distribution and split
  adjusted, essential for income-paying ETFs).
- **Master calendar**: observed ^AXJO trading days. All three ETFs trade on that calendar
  natively; the AUD/USD benchmark trades 24/5 and is forward-filled onto the ASX grid
  (limit 5 sessions). Everything is marked at ASX close.
- **Returns**: daily log returns. The portfolio return is the weighted sum of simple
  returns (converted back to log), i.e. fixed daily rebalancing to target weights.
- **Ingestion** re-fetches a 7-day overlap window daily; artifacts are upserted keyed on
  (date, ticker) so late adjustments self-heal and re-runs are idempotent.

## 3. Volatility models

Each series (every asset, each benchmark, and the portfolio itself) gets a daily fit:

- **Default**: GARCH(1,1) with Student-t innovations, zero mean, on returns ×100.
- EGARCH(1,1,1)-t is available per asset via `models.garch.egarch_assets` for series with
  asymmetric volatility response (none currently configured).
- **Fallback chain** (recorded and displayed whenever used):
  GARCH-t → GARCH-normal → RiskMetrics EWMA (λ = 0.94).
  A fit "fails" on non-convergence, a degenerate forecast, or < 250 observations.
- Multi-step variance forecasts use the analytic GARCH recursion; EGARCH has no analytic
  multi-step form, so 2,000-path simulation is used.

## 4. VaR and Expected Shortfall

All figures are losses as positive fractions of portfolio value, at 95%/99% confidence,
1-day and 10-day horizons. Three methods are computed side by side — divergence between
them is informative about tail shape, not an error:

| Method | Description |
|---|---|
| **Parametric (normal / Student-t)** | Portfolio-level GARCH forecast. Multi-day horizon uses the **variance term structure** `σ_H = sqrt(Σ_h σ²_h)` — the mean-reverting forecast path, *not* naive `σ√h`. Student-t uses the fitted degrees of freedom with unit-variance scaling; ES uses the closed forms for both distributions. |
| **Historical simulation** | Empirical quantile / tail mean of the trailing 500-day portfolio return window. Multi-day horizon uses √h scaling — a disclosed approximation (empirical h-day resampling is a planned refinement). |
| **Monte Carlo** | 10,000 correlated multi-asset paths: standardised GARCH residuals give the correlation matrix (eigenvalue-clipped to stay positive-definite), shocks are drawn multivariate normal, each asset follows its own GARCH σ-path over the horizon, and paths aggregate with actual portfolio weights. Fixed seed for reproducible committed artifacts. |

**Known limitation**: MC shocks are Gaussian (correlation from residuals, but no tail
dependence); a Student-t copula is the natural extension. The parametric-t column is the
fat-tailed reference point.

## 5. Backtesting

Walk-forward over the last 500 trading days:

- The portfolio GARCH(1,1)-t is refit every 5 sessions on an expanding window; between
  refits, conditional variance rolls forward daily with fixed parameters
  (σ²ₜ₊₁ = ω + αr²ₜ + βσ²ₜ). Every VaR forecast therefore uses only information available
  at that day's close.
- Methods backtested daily: parametric-t and historical simulation. (Monte Carlo is
  excluded from the daily walk-forward for runtime reasons; its current-day estimates are
  benchmarked against the other methods on the dashboard instead.)
- **Kupiec proportion-of-failures**: LR test that the breach rate equals 1 − confidence.
- **Christoffersen independence**: LR test against first-order breach clustering.
- **Conditional coverage**: joint test (χ², 2 df).
- **Basel traffic light**: breach count scaled to 250 days, 99% VaR only —
  green ≤ 4, yellow 5–9, red ≥ 10.

## 6. Monitoring and alerting

Five checks run daily after metrics are computed:

| Check | Rule |
|---|---|
| Missing sessions | per-ticker coverage of the last 30 ASX sessions (WARN > 5, ALERT > 10) |
| Stale prices | ALERT if a series is unchanged ≥ 3 consecutive sessions |
| Extreme jumps | ALERT if today's \|return\| > 6× its trailing 60-day volatility |
| PSI drift | returns[-60:] vs reference window returns[-750:-250], 10 quantile bins; WARN > 0.10, ALERT > 0.25 |
| VaR breach | ALERT if today's loss exceeds yesterday's 99% 1-day parametric-t VaR |

ALERTs automatically file a GitHub issue (deduplicated by date-stamped title, labelled
`risk-alert`). The dashboard banner reflects the worst current status. A hard ingestion
failure exits non-zero, turning the GitHub Actions run red — pipeline health is itself
monitored by the scheduler.

## 7. Operational architecture

GitHub Actions (cron, weekdays 07:30 UTC ≈ after ASX close) runs the full pipeline and
commits refreshed artifacts to the repository; Streamlit Cloud redeploys on commit. The
dashboard **never fits models** — it reads committed parquet artifacts only, so what you
see is exactly what the audited pipeline produced. All environments (CI, scheduler,
dashboard) install identical pinned dependencies.

## 8. Reproducibility

- Deterministic seeds for all simulation.
- Idempotent artifact writes (natural-key upserts, atomic file replacement).
- CI runs the full test suite — including an offline end-to-end pipeline run on synthetic
  fixtures — on every push; known-answer tests pin the VaR/ES math and the backtest
  statistics to hand-computed values.
