"""VaR backtesting — walk-forward breaches and coverage tests."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from riskplatform import dashboard as dash

st.set_page_config(page_title="Backtesting", page_icon="📉", layout="wide")
st.title("VaR Backtesting")
st.caption(
    "Walk-forward validation over the last 500 trading days: the GARCH model is refit every "
    "5 days on data available at the time, its 1-day VaR forecast is compared against the "
    "next day's realised return — genuinely out-of-sample, the way a bank validates a "
    "production risk model."
)


@st.cache_data(ttl=3600)
def _load():
    return dash.load_backtest(), dash.load_backtest_summary()


bt, summary = _load()
if bt is None or summary is None:
    st.warning("No backtest artifacts yet — run the pipeline first.")
    st.stop()

bt["date"] = pd.to_datetime(bt["date"])

# ---- Coverage test summary ---------------------------------------------------
st.subheader("Coverage tests")
disp = summary.copy()
disp["confidence"] = disp["confidence"].map(lambda c: f"{c:.0%}")
for col in ("breach_rate", "expected_rate"):
    disp[col] = disp[col].map(lambda v: f"{v:.2%}")
for col in ("kupiec_p", "christoffersen_p", "conditional_coverage_p"):
    disp[col] = disp[col].map(lambda v: f"{v:.3f}")
disp["traffic_light"] = disp["traffic_light"].map(
    {"green": "🟢 green", "yellow": "🟡 yellow", "red": "🔴 red", "n/a": "— n/a"}
)
st.dataframe(
    disp[
        [
            "method", "confidence", "n_obs", "n_breaches", "breach_rate", "expected_rate",
            "kupiec_p", "christoffersen_p", "conditional_coverage_p", "traffic_light",
        ]
    ],
    width="stretch",
    hide_index=True,
)
st.caption(
    "Kupiec tests whether the breach *rate* matches the confidence level; Christoffersen "
    "tests whether breaches *cluster* (a model can have the right count but fail in crises); "
    "conditional coverage combines both. p < 0.05 rejects the model. The Basel traffic light "
    "applies to 99% VaR scaled to a 250-day window."
)

# ---- Breach chart --------------------------------------------------------------
st.subheader("Realised returns vs 1-day VaR forecasts")
c1, c2 = st.columns(2)
method = c1.selectbox("Method", sorted(bt["method"].unique()))
conf = c2.selectbox("Confidence", sorted(bt["confidence"].unique(), reverse=True),
                    format_func=lambda c: f"{c:.0%}")

view = bt[(bt["method"] == method) & (bt["confidence"] == conf)].sort_values("date")
breaches = view[view["breach"]]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=view["date"], y=view["realised"], name="realised return",
        line=dict(width=1, color=dash.INK_MUTED), hovertemplate="%{y:.2%}",
    )
)
fig.add_trace(
    go.Scatter(
        x=view["date"], y=-view["var_forecast"], name=f"-VaR ({conf:.0%})",
        line=dict(width=2, color=dash.SERIES[0]), hovertemplate="%{y:.2%}",
    )
)
fig.add_trace(
    go.Scatter(
        x=breaches["date"], y=breaches["realised"], name="breach", mode="markers",
        marker=dict(size=9, color=dash.STATUS["ALERT"], symbol="x"),
        hovertemplate="breach: %{y:.2%}",
    )
)
fig.update_layout(**dash.PLOTLY_LAYOUT, yaxis_tickformat=".1%", height=420)
st.plotly_chart(fig, width="stretch")
st.caption(
    f"{len(breaches)} breaches out of {len(view)} days "
    f"(expected ≈ {(1 - conf) * len(view):.0f} at {conf:.0%} confidence). "
    "A breach is a day whose loss exceeded the VaR forecast made the evening before."
)
