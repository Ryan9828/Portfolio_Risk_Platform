"""Volatility and VaR time series, plus the asset correlation structure."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from riskplatform import dashboard as dash
from riskplatform.config import PORTFOLIO_TICKER, load_settings

st.set_page_config(page_title="Volatility & VaR", page_icon="📉", layout="wide")
st.title("Volatility & VaR")


@st.cache_data(ttl=3600)
def _load():
    return dash.load_returns(), dash.load_risk_metrics()


returns_table, metrics = _load()
if returns_table is None:
    st.warning("No data yet — run the pipeline first.")
    st.stop()

settings = load_settings()
wide = dash.returns_wide(returns_table)

# ---- EWMA volatility (annualised) ------------------------------------------
st.subheader("Annualised volatility (EWMA λ=0.94)")
sel = st.multiselect(
    "Series",
    [PORTFOLIO_TICKER, settings.index_ticker, "BTC-USD", "AUDUSD=X"],
    default=[PORTFOLIO_TICKER, settings.index_ticker],
    max_selections=4,
)
fig = go.Figure()
for i, ticker in enumerate(sel):
    vol = dash.ewma_vol(wide[ticker], annualise=True)
    fig.add_trace(
        go.Scatter(
            x=vol.index,
            y=vol,
            name=ticker,
            line=dict(width=2, color=dash.SERIES[i % len(dash.SERIES)]),
            hovertemplate="%{y:.1%}",
        )
    )
fig.update_layout(**dash.PLOTLY_LAYOUT, yaxis_tickformat=".0%", height=380)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Display-level EWMA estimate; the risk metrics themselves use full GARCH fits from the "
    "daily pipeline. Volatility clustering — calm regimes and stress regimes — is exactly "
    "what GARCH models capture."
)

# ---- VaR history (accrues one point per pipeline run) -----------------------
st.subheader("1-day 99% VaR history by method")
METHODS = ["parametric_t", "parametric_normal", "historical", "monte_carlo"]
if metrics is not None:
    hist = metrics[(metrics["horizon"] == 1) & (metrics["confidence"] == 0.99)].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    n_days = hist["date"].nunique()
    fig2 = go.Figure()
    if n_days < 2:
        # a single day has no time axis — compare methods as bars instead
        latest = hist[hist["date"] == hist["date"].max()].set_index("method").loc[METHODS]
        fig2.add_trace(
            go.Bar(
                x=METHODS,
                y=latest["var"],
                marker=dict(color=dash.SERIES[: len(METHODS)]),
                text=[f"{v:.2%}" for v in latest["var"]],
                textposition="outside",
                hovertemplate="%{x}: %{y:.2%}<extra></extra>",
            )
        )
        layout2 = {**dash.PLOTLY_LAYOUT, "hovermode": "closest", "showlegend": False}
    else:
        for i, method in enumerate(METHODS):
            m = hist[hist["method"] == method].sort_values("date")
            fig2.add_trace(
                go.Scatter(
                    x=m["date"],
                    y=m["var"],
                    name=method,
                    mode="lines+markers" if n_days < 30 else "lines",
                    line=dict(width=2, color=dash.SERIES[i % len(dash.SERIES)]),
                    marker=dict(size=8),
                    hovertemplate="%{y:.2%}",
                )
            )
        layout2 = dash.PLOTLY_LAYOUT
    fig2.update_layout(**layout2, yaxis_tickformat=".1%", height=380)
    st.plotly_chart(fig2, width="stretch")
    if n_days < 5:
        st.info(
            f"History accrues one point per trading day — the platform has run on {n_days} "
            f"day(s) so far. This becomes a time series as the daily pipeline keeps running."
        )

# ---- Correlation heatmap -----------------------------------------------------
st.subheader("Return correlations (portfolio assets, full sample)")
assets = [t for t in settings.weights]
corr = wide[assets].corr()
# diverging blue <-> red with a neutral gray midpoint at zero
diverging = [
    (0.0, "#e66767"), (0.35, "#8a5555"), (0.5, "#383835"), (0.65, "#2f5687"), (1.0, "#3987e5"),
]
fig3 = go.Figure(
    go.Heatmap(
        z=corr.to_numpy(),
        x=corr.columns,
        y=corr.index,
        zmin=-1,
        zmax=1,
        colorscale=diverging,
        colorbar=dict(title="ρ", tickformat=".1f"),
        hovertemplate="%{y} × %{x}: ρ=%{z:.2f}<extra></extra>",
    )
)
layout3 = {**dash.PLOTLY_LAYOUT, "hovermode": "closest"}
fig3.update_layout(**layout3, height=460)
st.plotly_chart(fig3, width="stretch")
st.caption(
    "The bank pairs (CBA–WBC) and diversified miners vs banks structure drives how much "
    "diversification the Monte Carlo VaR credits relative to a single-asset view. BTC's low "
    "correlation to ASX names is why a 10% sleeve adds less portfolio VaR than its "
    "standalone volatility suggests."
)
