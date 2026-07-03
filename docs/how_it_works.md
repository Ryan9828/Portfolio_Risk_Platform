# How It Works — Theory and System Guide

This guide assumes no prior knowledge of financial risk management. It builds up the
theory from scratch, then shows how the platform turns that theory into a running system.
(The companion documents: [methodology.md](methodology.md) is the formal audit-style model
specification; [how_it_was_built.md](how_it_was_built.md) covers the engineering.)

---

## Part 1 — The problem this system solves

You hold a portfolio of investments. The single most practical question in risk
management is:

> **"How much could I lose tomorrow, on a bad day?"**

Not the average day — bad days are what destroy portfolios, and what regulators require
banks to measure. This platform answers that question every night, for a demonstration
portfolio of 8 ASX blue-chip stocks (90%, equally weighted) plus Bitcoin (10%), and then
— crucially — **checks whether its own answers have been right**.

## Part 2 — The building blocks

### 2.1 Returns, not prices

Prices themselves aren't comparable (a $2 move on a $20 stock ≠ a $2 move on a $200
stock), so everything works with **daily returns** — today's percentage change. The
platform uses *log returns*, `r_t = ln(P_t / P_{t-1})`, which are mathematically
convenient: they add up across days, so a 10-day return is just the sum of ten daily ones.

The portfolio return each day is the weighted average of the assets' returns — 11.25%
of CBA's return, 11.25% of BHP's, ..., 10% of Bitcoin's.

### 2.2 Volatility — the size of typical moves

**Volatility (σ)** is the standard deviation of returns: a measure of how large moves
tend to be. A stock with 1% daily volatility usually moves ±1%-ish; on rare days ±3%.

The crucial empirical fact — visible on the dashboard's volatility chart — is that
volatility is not constant. It **clusters**: markets have calm months and panicked
months, and a violent day is likely to be followed by more violent days. This means
yesterday's data should influence today's risk estimate more than data from a year ago.

### 2.3 GARCH — forecasting tomorrow's volatility

**GARCH(1,1)** (Generalised Autoregressive Conditional Heteroskedasticity — the name is
worse than the idea) is the standard model for clustering volatility. It says tomorrow's
variance is a blend of three ingredients:

```
σ²_{t+1} = ω + α·r²_t + β·σ²_t
            │    │        └── persistence: yesterday's variance carries over (β ≈ 0.9)
            │    └── reaction: a big move today raises tomorrow's estimate (α ≈ 0.08)
            └── a floor: the long-run baseline level
```

Because α + β is close to (but below) 1, shocks fade gradually — exactly the "panic
decays over weeks" pattern real markets show. The model is fitted to three years of
returns by maximum likelihood, separately for every asset and for the portfolio itself,
every night.

Two refinements used here:

- **Student-t innovations**: real returns have "fat tails" — extreme days happen far more
  often than a bell curve predicts. The Student-t distribution has a tail-thickness
  parameter (ν, "degrees of freedom") fitted from the data; small ν = fat tails. Your
  fitted values (ν ≈ 3–8 across assets) confirm markets are decidedly non-normal.
- **EGARCH for Bitcoin**: an asymmetric variant, because volatility often reacts
  differently to crashes than to rallies.

If a fit fails (rare — insufficient data or the optimiser not converging), the system
falls back to simpler models (GARCH-normal, then EWMA — a simple weighted average of
recent squared returns) and **displays that degradation on the dashboard** rather than
hiding it.

### 2.4 VaR — the headline risk number

**Value-at-Risk (VaR)** at 99% confidence over 1 day is the loss threshold such that:

> "There is only a 1% chance tomorrow's loss exceeds this number."

If the 99% 1-day VaR is 1.95%, then on 99 days out of 100 you should lose less than
1.95% of the portfolio's value. It's a percentile of the forecast loss distribution —
the standard risk currency of every bank's trading floor and the number regulators
anchor capital requirements to.

**Expected Shortfall (ES)** answers the follow-up question VaR ignores: *"and if we DO
land in that worst 1%, how bad is it on average?"* ES is always larger than VaR, and it's
the better measure of tail catastrophe (post-2008 regulation shifted toward it for
exactly that reason).

### 2.5 Four ways to compute VaR — and why the platform does all of them

Each method makes different assumptions. Computing all four side by side turns their
*disagreement* into information:

1. **Parametric (normal)** — assume returns follow a bell curve with tomorrow's GARCH
   volatility. VaR is then just σ × 2.33 (the 99th percentile of the normal). Simple,
   fast, but underestimates fat tails.
2. **Parametric (Student-t)** — same, but with the fitted fat-tailed distribution.
   Typically the most defensible single number; it's the one on the dashboard's
   headline tiles.
3. **Historical simulation** — no distributional assumption at all: take the last 500
   actual daily returns and read off the worst 1% directly. "What would the recent past
   have done to today's portfolio?"
4. **Monte Carlo** — simulate 10,000 possible tomorrows: draw random shocks that respect
   the estimated **correlations** between the assets (bank stocks move together; Bitcoin
   mostly doesn't), push each asset along its own GARCH volatility path, and aggregate
   with the portfolio weights. This is the only method that natively prices the
   diversification benefit of the BTC sleeve.

For 10-day horizons, the parametric methods use the GARCH **term structure** — variance
forecasts for each of the next 10 days, summed — rather than the lazy "multiply by √10"
shortcut (which ignores that volatility mean-reverts). The difference between those two
answers is a favourite risk-interview topic.

## Part 3 — Checking the model's homework: backtesting

A risk model that never gets audited is just an opinion. The scientific question is:

> "If I had used this model every day for the last two years, would its 99% claims
> have actually been right 99% of the time?"

The platform runs a **walk-forward backtest** over the last 500 trading days. For each
day it refits the model *using only data available before that day* (no peeking), makes
the VaR forecast, and records whether the next day's actual loss **breached** it. At 99%
confidence, 500 days should produce about 5 breaches.

Three formal statistical tests then judge the breach record:

- **Kupiec test** — is the *number* of breaches consistent with the promised rate?
  (6 observed vs 5 expected → p = 0.66 → no evidence of a problem.)
- **Christoffersen test** — do breaches *cluster*? A model can have the right count but
  fail catastrophically by producing all its breaches in one crisis week, which is when
  it matters most.
- **Basel traffic light** — the actual regulatory rule: per 250 days at 99%, ≤4 breaches
  = green zone, 5–9 = yellow (capital penalty), ≥10 = red (model rejected). This
  platform's parametric-t model is currently **green**; historical simulation is yellow.

## Part 4 — The system watching itself: monitoring

Models fail quietly in production — usually because the *data* changed, not the code.
After every run, five checks execute:

| Check | Failure it catches | Analogy |
|---|---|---|
| Missing days | a ticker stopped returning data | dead sensor |
| Stale prices | same price repeating for days | frozen sensor |
| Extreme jumps | a return >6 standard deviations | corrupted reading (or a genuine crash — either deserves attention) |
| **PSI drift** | the return distribution has shifted away from what the model learned on | the world changed under the model |
| VaR breach | yesterday's actual loss blew through the forecast | the model was wrong today |

**PSI (Population Stability Index)** deserves a note: it compares the histogram of recent
returns against a reference window and produces one number. Under 0.10 = stable, 0.10–0.25
= watch, over 0.25 = investigate. These are the same governance bands used for production
credit-risk models in industry.

Any ALERT automatically **files a GitHub issue** — the system pages its operator. A hard
data-feed failure makes the whole run fail red, so even the pipeline's own health is
monitored by the scheduler.

## Part 5 — The nightly cycle, end to end

```
17:30 Sydney, every weekday (GitHub's servers — your laptop can be off)
│
├─ 1. Download latest prices (Yahoo Finance, 11 tickers)
├─ 2. Rebuild returns on the ASX trading calendar
│      (BTC trades weekends; its weekend move lands in Monday's return)
├─ 3. Fit GARCH per asset + portfolio  → tomorrow's volatility forecasts
├─ 4. Compute VaR + ES  (4 methods × 95%/99% × 1-day/10-day)
├─ 5. Walk-forward backtest + Kupiec/Christoffersen/Basel verdicts
├─ 6. Run the 5 monitoring checks  → file GitHub issues on ALERT
└─ 7. Commit results to the repository
        └─ Streamlit Cloud sees the commit → dashboard redeploys itself
```

The dashboard **never computes anything** — it only displays the committed results. That
separation (heavy, audited computation in one place; a thin display layer in another) is
deliberate and mirrors how regulated risk systems are architected.

## Part 6 — Reading the dashboard

- **Home** — today's headline numbers, the four methods side by side, and per-asset model
  fit status (if anything says EWMA, a fallback fired).
- **Volatility & VaR** — volatility history (see the clustering!), VaR history as it
  accrues daily, and the asset correlation heatmap that explains the diversification.
- **Backtesting** — the breach chart (grey = daily returns, blue = the moving VaR floor,
  red ✕ = breaches) and the formal test table.
- **Monitoring & Alerts** — current check statuses and the full alert history.
- **Methodology** — the formal model documentation.

## Glossary

| Term | Plain meaning |
|---|---|
| Log return | today's % change, in a form that adds across days |
| Volatility (σ) | typical size of daily moves |
| GARCH | model that forecasts tomorrow's volatility from recent turbulence |
| Student-t / ν | fat-tailed distribution; lower ν = wilder tails |
| VaR 99% | loss level exceeded only 1 day in 100 |
| Expected Shortfall | average loss on the days VaR is exceeded |
| Breach | a day whose actual loss exceeded the VaR forecast |
| Kupiec / Christoffersen | statistical tests of the breach record (count / clustering) |
| Basel traffic light | the regulatory green/yellow/red verdict on a VaR model |
| PSI | one-number measure of distribution drift |
| Walk-forward | testing each day using only information available at the time |
