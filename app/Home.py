"""Home — current portfolio risk metrics and monitor banner."""

import pandas as pd
import streamlit as st

from riskplatform import dashboard as dash
from riskplatform.config import load_settings

st.set_page_config(page_title="Portfolio Risk Platform", page_icon="📉", layout="wide")


@st.cache_data(ttl=3600)
def _load():
    return dash.load_risk_metrics(), dash.load_monitor_status(), dash.load_returns()


metrics, status, returns_table = _load()

st.title("Portfolio Market-Risk Platform")
st.caption(
    "Self-updating risk engine — daily GARCH volatility, VaR/ES across four methods, "
    "Basel-style backtesting and drift monitoring. Updated automatically by GitHub Actions "
    "after each ASX close."
)

if metrics is None or status is None:
    st.warning("No data artifacts yet — the daily pipeline has not run. Check back after the first run.")
    st.stop()

overall = status.get("overall", "OK")
if overall == "ALERT":
    details = "; ".join(c["detail"] for c in status["checks"] if c["status"] == "ALERT")
    st.error(f"🔴 **Monitoring ALERT** — {details}", icon="🚨")
elif overall == "WARN":
    details = "; ".join(c["detail"] for c in status["checks"] if c["status"] == "WARN")
    st.warning(f"**Monitoring WARN** — {details}")
else:
    st.success("All monitoring checks passing.")

latest_date = pd.to_datetime(metrics["date"]).max()
latest = metrics[pd.to_datetime(metrics["date"]) == latest_date]
st.subheader(f"Risk as of {latest_date.date()} (ASX close)")


def _metric(method: str, horizon: int, conf: float, col: str = "var") -> float:
    row = latest[
        (latest["method"] == method)
        & (latest["horizon"] == horizon)
        & (latest["confidence"] == conf)
    ]
    return float(row[col].iloc[0]) if not row.empty else float("nan")


c1, c2, c3, c4 = st.columns(4)
c1.metric("1-day VaR (95%)", f"{_metric('parametric_t', 1, 0.95):.2%}")
c2.metric("1-day VaR (99%)", f"{_metric('parametric_t', 1, 0.99):.2%}")
c3.metric("1-day ES (99%)", f"{_metric('parametric_t', 1, 0.99, 'es'):.2%}")
c4.metric(
    "Portfolio vol (annualised)",
    f"{float(latest['portfolio_sigma'].iloc[0]) * (252 ** 0.5):.1%}",
)
st.caption(
    "Parametric GARCH(1,1) with Student-t innovations, as a share of portfolio value. "
    "VaR = loss level exceeded with probability 1 − confidence; ES = average loss beyond VaR."
)

st.subheader("All methods, side by side")
pivot = (
    latest.assign(metric=lambda d: d["confidence"].map(lambda c: f"{c:.0%}"))
    .pivot_table(index="method", columns=["horizon", "metric"], values=["var", "es"])
    .round(4)
)
st.dataframe((pivot * 100).style.format("{:.2f}%"), width="stretch")
st.caption(
    "Methods deliberately differ: parametric assumes a distribution, historical replays the "
    "trailing window, Monte Carlo simulates correlated multi-asset paths. Divergence between "
    "them is itself information about tail shape."
)

st.subheader("Model fit status")
fits = status.get("fits", {})
if fits:
    fit_df = pd.DataFrame(fits).T.reset_index(names="series")
    fit_df["sigma_1d"] = fit_df["sigma_1d"].map(lambda v: f"{float(v):.3%}")
    st.dataframe(fit_df, width="stretch", hide_index=True)
    st.caption(
        "Degraded fits stay visible: a series showing EWMA means the GARCH fallback chain "
        "was triggered for it on the latest run."
    )
