"""Monitoring checks and alert history."""

import pandas as pd
import streamlit as st

from riskplatform import dashboard as dash

st.set_page_config(page_title="Monitoring & Alerts", page_icon="📉", layout="wide")
st.title("Monitoring & Alerts")
st.caption(
    "Every daily run executes five checks — data quality (missing sessions, stale feeds, "
    "extreme jumps), distribution drift (PSI), and VaR breaches. Any ALERT files a GitHub "
    "issue automatically and shows here."
)


@st.cache_data(ttl=3600)
def _load():
    return dash.load_monitor_status(), dash.load_alerts()


status, alerts = _load()
if status is None:
    st.warning("No monitor status yet — run the pipeline first.")
    st.stop()

badge = {"OK": "🟢", "WARN": "🟡", "ALERT": "🔴"}
st.subheader(f"Latest run — {badge.get(status['overall'], '')} {status['overall']} "
             f"(as of {status.get('asof', '?')})")

checks = pd.DataFrame(status["checks"])
checks["status"] = checks["status"].map(lambda s: f"{badge.get(s, '')} {s}")
st.dataframe(checks, width="stretch", hide_index=True)

st.subheader("Check definitions")
st.markdown(
    """
| Check | What it catches | Threshold |
|---|---|---|
| `missing_days` | tickers absent from recent ASX sessions (dead source) | WARN > 5 / ALERT > 10 of last 30 |
| `stale_prices` | a feed returning the same price repeatedly | ALERT ≥ 3 sessions unchanged |
| `extreme_jumps` | data errors or genuine shocks | ALERT \\|z\\| > 6 vs 60-day vol |
| `psi_drift` | the return distribution shifting away from the reference window | WARN > 0.10 / ALERT > 0.25 |
| `var_breach` | realised loss exceeding the prior day's 99% 1-day VaR | ALERT on breach |
"""
)
st.caption(
    "The PSI thresholds mirror standard model-risk governance bands: below 0.10 stable, "
    "0.10–0.25 watch, above 0.25 investigate/refit."
)

st.subheader("Alert history")
if alerts is not None and not alerts.empty:
    alerts = alerts.sort_values("date", ascending=False)
    alerts["date"] = pd.to_datetime(alerts["date"]).dt.date
    st.dataframe(alerts, width="stretch", hide_index=True)
else:
    st.info("No WARN/ALERT events recorded yet.")

with st.expander("Run metadata"):
    st.json({k: v for k, v in status.items() if k not in ("checks", "fits")})
