# How It Was Built — Engineering Documentation

The companion to [how_it_works.md](how_it_works.md) (theory) and
[methodology.md](methodology.md) (formal model spec). This document covers the
architecture, the code, the design decisions and their reasons, and how to operate,
verify, and extend the system.

---

## 1. Design goals

1. **A production system, not a notebook** — scheduled, self-healing, self-monitoring,
   publicly visible, zero manual steps.
2. **Free-tier only** — no servers to pay for or maintain. GitHub Actions supplies the
   compute; Streamlit Community Cloud supplies the hosting; the git repository itself
   supplies the database.
3. **Auditable** — deterministic outputs, pinned environments, documented methodology,
   test-pinned mathematics. A reviewer should reproduce any number on the dashboard.

## 2. The core architectural decision: compute/serve split

```
┌────────────────────────────┐      commits       ┌──────────────────────────┐
│  GitHub Actions (nightly)  │ ─── parquet ──────▶│  git repository (main)   │
│  ALL modelling runs here   │     artifacts      │  data/ = the "database"  │
└────────────────────────────┘                    └────────────┬─────────────┘
                                                               │ redeploy on commit
                                                               ▼
                                                  ┌──────────────────────────┐
                                                  │  Streamlit Cloud         │
                                                  │  pure READER of data/    │
                                                  └──────────────────────────┘
```

The dashboard never fits a model. Reasons:

- **Memory** — Streamlit Cloud's free tier has ~1 GB; GARCH/Monte Carlo comfortably fits
  in Actions' runners instead.
- **Speed** — pages load instantly from small parquet files.
- **Auditability** — what you see is exactly what the scheduled pipeline produced; there
  is no second code path that could disagree.

**Git as the database** works because the state is tiny (< 2 MB of parquet). Every daily
run commits refreshed artifacts; history is versioned for free; the dashboard, CI, and
any laptop clone all read identical state.

## 3. Repository layout

```
├── config/portfolio.yaml        single source of truth: tickers, weights, all thresholds
├── data/                        committed artifacts (the "database")
│   ├── prices.parquet             long format (date, ticker, adj_close)
│   ├── returns.parquet            per-ticker + PORTFOLIO pseudo-ticker log returns
│   ├── risk_metrics.parquet       one row per (date, method, horizon, confidence)
│   ├── backtest_results.parquet   daily walk-forward forecasts vs realised
│   ├── backtest_summary.parquet   Kupiec/Christoffersen/traffic-light verdicts
│   ├── monitor_status.json        latest run's check results + per-asset fit status
│   ├── alerts_history.parquet     every WARN/ALERT ever raised
│   ├── announcements.parquet      ASX announcement headlines (keyed on doc_key)
│   ├── announcement_signals.parquet   LLM-extracted signals + per-call cost/latency
│   ├── announcement_event_study.parquet  abnormal returns + vol regimes per event
│   └── intel_eval_metrics.json    golden-set precision/recall for the extraction
├── src/riskplatform/            the installable package (src layout, pyproject.toml)
│   ├── config.py                  YAML → frozen typed dataclasses
│   ├── ingestion.py               yfinance fetch, retries, tz-naive normalisation
│   ├── returns.py                 ASX master calendar, alignment, portfolio returns
│   ├── garch.py                   arch-package fits + fallback chain (→ EWMA)
│   ├── var_es.py                  the four VaR/ES engines
│   ├── backtest.py                walk-forward loop + coverage tests
│   ├── monitoring.py              the five checks, incl. PSI
│   ├── alerting.py                GitHub-issue filing with dedupe
│   ├── artifacts.py               idempotent upsert — the ONLY write path
│   ├── dashboard.py               shared loaders + chart palette for the app
│   ├── pipeline.py                CLI entry point (backfill | run | backtest | intel)
│   └── intel/                     announcement intelligence layer (see §11)
│       ├── ingest.py                ASX announcement feed → announcements.parquet
│       ├── extract.py               Claude structured-output extraction + cost audit
│       ├── events.py                event study vs abnormal returns and realised vol
│       └── evals.py                 golden-set labelling + scoring CLI
├── app/                         Streamlit pages (Home + 5 subpages), thin readers
├── evals/                       hand-labelled golden set for the extraction step
├── docs/                        methodology + this documentation
├── tests/                       32 offline tests (see §7)
└── .github/workflows/
    ├── daily_pipeline.yml         the nightly cron job
    └── ci.yml                     pytest on every push
```

Everything tunable — tickers, weights, GARCH settings, VaR confidences/horizons,
backtest window, every monitoring threshold — lives in `config/portfolio.yaml`, not in
code.

## 4. Data engineering decisions

**Idempotent upserts.** All writes go through one function
(`artifacts.upsert(path, rows, keys)`): concat with existing rows, drop duplicate keys
keeping the newest, atomic write (temp file + `os.replace`). Consequences:

- Re-running a day can never duplicate rows.
- Ingestion deliberately re-fetches a 7-day overlap window every run, so late dividend
  adjustments in Yahoo's data self-heal by overwriting the affected rows.

**The calendar problem.** The portfolio's ETFs all trade on the ASX calendar, but the
AUD/USD benchmark trades 24/5. The rule: **the master calendar is the observed ^AXJO
trading days**; anything on another calendar is forward-filled onto it (limit 5
sessions), marking everything at ASX close — disclosed in the methodology.

**Timezones** are annihilated at the boundary: `ingestion.py` normalises every timestamp
to a tz-naive date immediately; nothing downstream ever sees a timezone.

**yfinance resilience**: pinned version, single batched request, `threads=False`,
3 retries with exponential backoff (5/15/45 s), per-ticker empty-result validation. If
all retries fail the pipeline exits non-zero → the Actions run shows red → that *is* the
alerting for feed death.

## 5. Modelling decisions worth knowing

- **Returns are rescaled ×100 before fitting** — `arch`'s optimiser is numerically
  unhappy at raw daily-return scale (~1e-2). De-rescaling happens exactly once, inside
  `garch.py`; nothing downstream ever sees rescaled units.
- **Zero-mean models** — standard for daily-frequency risk (the daily mean is
  statistically indistinguishable from zero and estimating it adds noise).
- **Fallback chain** GARCH-t → GARCH-normal → EWMA(0.94), triggered by non-convergence,
  degenerate forecasts, or < 250 observations. The model actually used is recorded per
  asset per run and shown on the dashboard — degradation is visible, never silent.
- **10-day parametric VaR uses the variance term structure** `σ_H = √(Σ σ²_h)` from the
  GARCH forecast path, not √10 scaling. Historical simulation *does* use √10, disclosed
  as an approximation — the contrast between methods is a deliberate feature.
- **Monte Carlo correlation matrix** comes from standardised GARCH residuals
  (raw-return correlation would double-count volatility). Short windows can make it
  numerically non-positive-definite, so eigenvalues are clipped before the Cholesky
  factorisation. A fixed RNG seed makes committed artifacts reproducible.
- **Walk-forward backtest efficiency**: refitting GARCH 500 times would be slow, so the
  model refits every 5 sessions and, between refits, rolls conditional variance forward
  daily with the fitted parameters (σ²ₜ₊₁ = ω + αr²ₜ + βσ²ₜ). Every forecast still uses
  only information available at that day's close — the no-lookahead property is what
  makes the backtest honest.

## 6. Automation and operations

**The nightly job** (`daily_pipeline.yml`): cron `30 7 * * 1-5` (07:30 UTC = 17:30 AEST /
18:30 AEDT — after ASX close in both daylight-saving regimes), plus `workflow_dispatch`
for manual runs with an optional `force_test_alert` input for alert-path drills.

Three operational subtleties:

1. **No infinite loops**: the job commits data back to the repo. Commits made with the
   default `GITHUB_TOKEN` don't trigger new workflow runs (GitHub's own loop protection),
   and the commit message carries `[skip ci]` as belt-and-braces. Streamlit Cloud watches
   the repo directly, so it still redeploys.
2. **Holidays**: cron fires, ingestion finds no new ^AXJO session, the pipeline logs it
   and exits 0 — green run, no commit, no redeploy.
3. **Alerting needs no secrets**: the workflow grants `issues: write` to the built-in
   token. Issues are date-stamped and deduplicated against open `risk-alert` issues, so
   a re-run never files duplicates.

**Environment pinning**: `requirements.txt` starts with `-e .` and pins every direct
dependency. CI, the nightly job, and Streamlit Cloud all install from the same file —
training/serving skew (the classic silent killer) is structurally excluded.

**Known operational quirk**: GitHub pauses cron workflows on repos with no *human*
commits for 60 days (bot commits don't count). GitHub emails a warning; one click
re-enables. Any small commit resets the clock.

## 7. Testing strategy

32 tests, **fully offline** — CI never touches Yahoo Finance or the Claude API. Three layers:

1. **Known-answer tests** pin the mathematics to hand-computed values: normal 95% VaR on
   σ = 1% must equal 1.64485%; the Kupiec statistic for 5 breaches in 250 days must be
   1.9568; ES > VaR always; t-VaR > normal-VaR at 99%; Basel boundaries (4 → green,
   5 → yellow, 10 → red). A refactor that changes any formula fails loudly.
2. **Behavioural tests**: Monte Carlo with 200k paths converges to the parametric answer
   when shocks are exactly normal (validates the whole simulation chain);
   diversification reduces VaR; Christoffersen flags clustered breaches but passes iid
   ones; PSI(x, x) = 0; upsert is idempotent; the GARCH fallback chain triggers on
   degenerate series.
3. **End-to-end integration**: the full pipeline runs against synthetic GBM price
   fixtures in a temp directory — every artifact must appear, schemas validate, no NaNs,
   and a second run must be a clean no-op.

The intel layer follows the same discipline (`tests/test_intel.py`): the feed parser
runs against a captured payload, the LLM client is a stub (so tests are free and
deterministic), extraction is proven idempotent and cost-capped, a single failing call
is proven non-fatal, and the event study must detect a synthetic vol-regime change and
drop events with insufficient history.

## 8. Dashboard engineering

Five pages, all thin readers via `st.cache_data(ttl=3600)`. Charts follow a validated
colourblind-safe palette (fixed slot order, checked programmatically for CVD separation
and contrast on the dark surface). Two details:

- Anything displayed that isn't in an artifact (EWMA vol curves, the correlation
  heatmap) is cheap arithmetic on the returns table — never model fitting.
- The VaR-history chart renders as a method-comparison bar chart until ≥ 2 days of
  history exist (a one-point time series axis is meaningless), then switches to lines.

## 9. Verification performed before go-live

- Real 3-year backfill: 8,700+ price rows, all 12 GARCH fits converged, metric sanity
  (99% > 95%, ES > VaR, 10-day > 1-day, methods within ~0.3 pp of each other).
- Idempotence proven live: pipeline run twice → zero duplicate keys.
- All 24 tests green locally and in CI.
- Every dashboard page rendered and visually inspected (screenshots via headless
  Chromium).
- Live drills after deployment: manual workflow dispatch (green), bot data commit landed
  without retriggering workflows, `force_test_alert=1` filed a real GitHub issue
  (then closed), unattended cron confirmed on the next trading day.

## 10. Extending the system

| Change | Where | Effort |
|---|---|---|
| Different portfolio / weights | `config/portfolio.yaml` | edit + push |
| New monitoring threshold | `config/portfolio.yaml` | edit + push |
| Student-t copula for Monte Carlo | `var_es.monte_carlo_var_es` | small — swap the shock draw |
| Empirical 10-day historical VaR | `var_es.historical_var_es` | small — overlapping-window resample |
| DCC (time-varying correlations) | new module + `var_es` | the natural "phase 2" |
| Intraday frequency | ingestion + calendar | large — different data source needed |

Known limitations (also stated in the methodology, § "deliberately disclosed"):
Gaussian Monte Carlo shocks (no tail dependence), √10 historical scaling, fixed
portfolio weights, a single external data vendor. Each is a conscious scope decision
with the upgrade path documented above.

## 11. The announcement-intelligence layer

### 11.1 What it is and why it exists

GARCH is reactive: it learns about a shock only after the shock appears in returns.
Company announcements are the most common *cause* of single-name volatility shocks and
are published before or as the move happens — but they arrive as unstructured text a
quantitative pipeline can't consume. The intel layer closes that gap: an LLM (Claude)
converts each announcement headline into a **typed risk signal** the platform can store,
chart, and test.

The framing is deliberate: the signal is a **volatility-regime leading indicator**, not
a return predictor. "This announcement means the stock is entering a higher-volatility
regime, so today's VaR is probably understated" is a testable, defensible claim;
"the stock will go down" is not.

### 11.2 How it was made — design decisions and their reasons

| Decision | Reason |
|---|---|
| Classify from the **headline + feed metadata only** (not the PDF body) | Keeps cost near zero, keeps the labelling task well-defined for evals, and ASX headlines are standardised enough to carry the classification. Body extraction is the documented next iteration, not scope creep in v1. |
| **Strict JSON schema** on the API (`output_config.format`) with enums for event type and materiality | Every response is guaranteed to parse — no regex-scraping model prose, no malformed rows. The schema *is* the contract. |
| One API call per announcement, **system prompt cached** (`cache_control`) | The instruction block is paid for once per run, not per announcement. |
| Signals **upsert keyed on the exchange's `documentKey`** | Same idempotence guarantee as prices: an announcement is extracted (and paid for) exactly once, re-runs are no-ops. |
| `max_new_per_run` **cost cap** in config | A feed anomaly can never trigger an unbounded API bill. Backlogs clear across runs instead. |
| Every signal row records **model, tokens, cost (USD), latency** | The extraction step is audited exactly like the risk models — the dashboard shows what the layer costs and how it behaves, not just what it says. |
| Extraction **self-skips without credentials** (log line, exit 0) | The nightly pipeline stays green whether or not the API key is configured; ingestion and the event study still run. The API key is the on/off switch. |
| **Blind golden-set evals** (`intel/evals.py`) | The labelling template deliberately hides the model's predictions, so human labels aren't anchored. Scoring reports per-class P/R/F1 plus P/R on the one decision that matters downstream: is this high-materiality? |
| **Event study** (`intel/events.py`) rebuilt from scratch each run | Event-day abnormal return (vs ASX 200) and post/pre realised-vol ratio per flagged announcement — the layer's claim ("flagged news precedes volatility") is tested with data, not asserted. |
| Stub LLM client in tests | CI never spends money and never flakes on network. |

**Operational quirk discovered during build**: the public ASX announcements feed
(`asx.api.markitdigital.com`) silently caps every response at the **5 most recent
announcements per ticker** — `itemsPerPage` and `page` are ignored (verified 2026-07).
Coverage therefore accumulates through the daily runs; there is no one-shot backfill.
Consequence: the event study starts empty (each event also needs 20 post-event sessions
of returns) and fills in over the first weeks of operation.

### 11.3 How to use it

**Prerequisite — an Anthropic API key** (platform.claude.com → API Keys; billed per
token, not covered by a Claude.ai subscription). At current volumes the spend is
roughly 1–5 US cents/day on `claude-opus-4-8`, ~10× less on `claude-haiku-4-5`.

**Run locally:**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."        # full string, including the sk-ant prefix
python -m riskplatform.pipeline intel        # ingest → extract → event study
python -m riskplatform.pipeline intel        # run again if a backlog exceeds the cap
streamlit run app/Home.py                    # → "Announcements" page
```

Without the key, the same command still refreshes announcements and the event study —
extraction logs a skip and the run stays green.

**Run nightly (GitHub Actions):** add `ANTHROPIC_API_KEY` as a repository secret
(Settings → Secrets and variables → Actions). The daily workflow already contains the
intel step; it extracts when the secret exists and self-skips when it doesn't.

**Turn it off / pause spending:** delete the repository secret (extraction skips,
ingestion keeps accumulating headlines for free), or set `max_new_per_run: 0` in
`config/portfolio.yaml` to keep the key but stop the calls. Turning it back on clears
the accumulated backlog at the configured cap per run.

**Evaluate the extraction (do this once enough announcements accumulate):**

```bash
python -m riskplatform.intel.evals template   # writes evals/golden_template.csv (blind)
# hand-fill label_event_type + label_materiality, save as evals/golden_set.csv
python -m riskplatform.intel.evals score      # writes data/intel_eval_metrics.json
```

The metrics render on the Announcements page next to the signals they audit. Re-run
`template` periodically — it only offers rows not already in the golden set, so the
set grows over time. Commit `evals/golden_set.csv`; it is the layer's test fixture.

**Tune it:** everything lives in the `intel:` section of `config/portfolio.yaml` —
model, per-run call cap, the $/MTok rates used for the cost column, and the event-study
windows/thresholds.
