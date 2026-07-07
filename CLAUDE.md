# Portfolio Risk Platform

Scheduled market-risk pipeline for a 3-ETF ASX portfolio (VGS.AX 40% / VAS.AX 30% / NDQ.AX 30%): GARCH(1,1)-t volatility → VaR/ES four ways → walk-forward backtest (Kupiec / Christoffersen / Basel traffic light) → data-quality monitoring → GitHub-issue alerts → parquet artifacts committed to the repo ("git as database") → Streamlit dashboard that only *reads*. An intel layer uses the Claude API to classify ASX announcement headlines into typed risk signals.

Public repo: https://github.com/Ryan9828/Portfolio_Risk_Platform · Live dashboard on Streamlit Cloud.

## Commands

```bash
pip install -r requirements.txt          # installs the package itself via -e .
python -m riskplatform.pipeline backfill # first run — builds history
python -m riskplatform.pipeline run      # daily incremental run (what the cron does)
python -m riskplatform.pipeline backtest # walk-forward backtest, prints summary table
python -m riskplatform.pipeline intel    # announcement ingestion + LLM extraction
streamlit run app/Home.py                # dashboard (reads data/, never fits models)
pytest                                   # 32 tests, ~2.5s, zero network calls
```

Python ≥3.12. All versions pinned in `requirements.txt`.

## Architecture rules (do not break these)

- **Compute/serve split**: `app/` pages are thin readers over `src/riskplatform/dashboard.py` loaders. Never put model-fitting, data-fetching, or pipeline logic in `app/`.
- **`config/portfolio.yaml` is the single source of truth** for tickers, weights, GARCH/VaR/backtest/monitoring/intel settings. `config.py` loads it into frozen dataclasses. Don't hardcode tickers or parameters anywhere else.
- **All paths derive from `PROJECT_ROOT`** (`config.py`) or CLI args — no absolute paths anywhere.
- **`data/` parquet artifacts are committed on purpose** (git-as-database, currently ~172 KB; keep under ~2 MB). The daily GitHub Actions workflow (`daily_pipeline.yml`, cron 30 7 * * 1-5) commits them back with `[skip ci]`. Artifact writes go through the idempotent upsert in `artifacts.py` — re-running a day must be a no-op.
- **Backtest is genuinely walk-forward**: GARCH refit every 5 sessions on an expanding window, variance rolled forward between refits, no look-ahead (`backtest.py`). Preserve this property in any change.
- **Tests are known-answer + behavioural** (exact Kupiec LR values, MC-converges-to-parametric, clustered-vs-iid Christoffersen). New numerical code should come with the same style of test. The integration test runs the full CLI offline against synthetic fixtures — keep it network-free.
- **Intel layer**: Claude calls go through `intel/extract.py` with cost accounting and a cost cap; tests stub the client (`StubClient` in `tests/test_intel.py`) — never mock library internals. The public ASX feed silently caps at 5 announcements/ticker (`intel/ingest.py`) — this is a known data limitation, not a bug.

## Known issues / doc drift (verified 2026-07-07 — fix before showing to reviewers)

1. **`docs/how_it_works.md:19,69` describes the OLD portfolio** ("8 ASX blue-chips + 10% Bitcoin, EGARCH for Bitcoin"). Actual: 3 ETFs, `egarch_assets: []` (EGARCH coded in `garch.py` but unused, and untested on the EGARCH branch).
2. **`docs/how_it_works.md:139` claims the parametric-t backtest is "currently green"** — the committed `data/backtest_summary.parquet` says **yellow** at 99%: 10/500 breaches, Kupiec p=0.048, Christoffersen p=0.012 (significant breach clustering). Either update the doc to report the real (more interesting) result, or make the doc reference live data instead of a snapshot claim.
3. **README says "24 tests"; actual is 32.** `app/Home.py:21` says "three methods"; actual is 4 (parametric_normal, parametric_t, historical, monte_carlo). The portfolio-website project card also says "24 self-checks" — update together.
4. **The golden-set LLM eval has never been run for real**: README/docs describe precision-recall eval metrics, but no `evals/golden_set.csv` or `data/intel_eval_metrics.json` exists. Code + unit tests exist (`intel/evals.py`); either run it against real announcements and commit the metrics, or soften the README claim.
5. **No lint/type-check gate in CI** (`ci.yml` runs pytest only). Adding ruff + mypy would close the most visible engineering-rigor gap; the code already uses type hints throughout.
6. Single data vendor (yfinance) — disclosed in `docs/methodology.md`, but no documented fallback plan.

## Layout

```
config/portfolio.yaml      source of truth
src/riskplatform/          config, ingestion, returns, garch, var_es, backtest,
                           monitoring, alerting, artifacts, dashboard, pipeline (CLI)
src/riskplatform/intel/    ingest, extract (Claude), events, evals
app/                       Home.py + 5 numbered Streamlit pages (readers only)
data/                      committed parquet/json artifacts
tests/                     8 files, 32 tests
docs/                      methodology.md, how_it_works.md (see drift above), how_it_was_built.md
.github/workflows/         ci.yml (pytest), daily_pipeline.yml (cron + commit data/)
```
