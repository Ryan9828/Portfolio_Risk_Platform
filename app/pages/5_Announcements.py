"""Announcement intelligence — LLM-extracted signals, event study, and extraction evals."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from riskplatform import dashboard as dash

st.set_page_config(page_title="Announcement Intelligence", page_icon="📉", layout="wide")
st.title("Announcement Intelligence")
st.caption(
    "Claude reads every ASX announcement for the portfolio names and converts each headline "
    "into a typed risk signal — event type, materiality, sentiment — before the price data "
    "can show it. The extraction step is evaluated against a hand-labelled golden set, and "
    "the event study below tests whether flagged announcements actually precede volatility."
)


@st.cache_data(ttl=3600)
def _load():
    return (
        dash.load_announcements(),
        dash.load_announcement_signals(),
        dash.load_event_study(),
        dash.load_intel_eval(),
    )


ann, signals, study, evals = _load()
if ann is None or signals is None or signals.empty:
    st.warning("No announcement signals yet — run `python -m riskplatform.pipeline intel` first.")
    st.stop()

feed = ann.merge(signals, on="doc_key").sort_values("date", ascending=False)
feed["date"] = pd.to_datetime(feed["date"]).dt.date

# --- Extraction audit -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Signals extracted", f"{len(signals):,}")
c2.metric("High materiality", f"{(signals['materiality'] == 'high').sum():,}")
c3.metric("Total LLM cost", f"${signals['cost_usd'].sum():.2f}")
c4.metric("Median latency", f"{signals['latency_s'].median():.1f}s")

# --- Feed -------------------------------------------------------------------
st.subheader("Signal feed")
mat_filter = st.multiselect("Materiality", ["high", "medium", "low"], default=["high", "medium", "low"])
shown = feed[feed["materiality"].isin(mat_filter)]
st.dataframe(
    shown[["date", "ticker", "headline", "event_type", "materiality", "sentiment", "rationale"]],
    width="stretch",
    hide_index=True,
    column_config={"sentiment": st.column_config.NumberColumn(format="%.2f")},
)

# --- Signal mix -------------------------------------------------------------
st.subheader("What the model is seeing")
mix = signals["event_type"].value_counts().sort_values()
fig = go.Figure(
    go.Bar(
        x=mix.values,
        y=mix.index,
        orientation="h",
        marker=dict(color=dash.SERIES[0], cornerradius=4),
        width=0.55,
        text=mix.values,
        textposition="outside",
        textfont=dict(color=dash.INK_MUTED),
        hovertemplate="%{y}: %{x} announcements<extra></extra>",
    )
)
fig.update_layout(**dash.PLOTLY_LAYOUT, height=360, showlegend=False, title="Announcements by event type")
fig.update_xaxes(title="count")
st.plotly_chart(fig, width="stretch")

# --- Event study ------------------------------------------------------------
st.subheader("Do flagged announcements precede risk?")
if study is None or study.empty:
    st.info("Event study needs more return history around the stored announcements.")
else:
    joined = study.merge(signals[["doc_key", "materiality"]], on="doc_key")
    order = [m for m in ["high", "medium", "low"] if m in set(joined["materiality"])]
    agg = joined.groupby("materiality").agg(
        abs_abnormal=("abnormal_return", lambda s: s.abs().mean()),
        vol_ratio=("vol_ratio", "mean"),
        reacted=("reacted", "mean"),
        n=("doc_key", "size"),
    ).reindex(order)

    left, right = st.columns(2)
    with left:
        fig = go.Figure(
            go.Bar(
                x=order,
                y=(agg["abs_abnormal"] * 100).round(2),
                marker=dict(color=dash.SERIES[0], cornerradius=4),
                width=0.5,
                text=[f"{v:.2f}%" for v in agg["abs_abnormal"] * 100],
                textposition="outside",
                textfont=dict(color=dash.INK_MUTED),
                hovertemplate="%{x}: %{y:.2f}% (n=%{customdata})<extra></extra>",
                customdata=agg["n"],
            )
        )
        fig.update_layout(**dash.PLOTLY_LAYOUT, height=320, showlegend=False,
                          title="Mean |abnormal return| on event day, by model materiality")
        fig.update_yaxes(title="%")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure(
            go.Bar(
                x=order,
                y=agg["vol_ratio"].round(2),
                marker=dict(color=dash.SERIES[0], cornerradius=4),
                width=0.5,
                text=[f"{v:.2f}×" for v in agg["vol_ratio"]],
                textposition="outside",
                textfont=dict(color=dash.INK_MUTED),
                hovertemplate="%{x}: %{y:.2f}× (n=%{customdata})<extra></extra>",
                customdata=agg["n"],
            )
        )
        fig.update_layout(**dash.PLOTLY_LAYOUT, height=320, showlegend=False,
                          title="Realised vol after ÷ before (20 sessions each side)")
        fig.update_yaxes(title="post/pre vol ratio")
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "If the extraction is informative, high-materiality announcements should show larger "
        "event-day abnormal returns and a vol ratio above 1 — volatility the GARCH layer only "
        "learns about after the move. Small samples early on; read direction, not decimals."
    )

# --- Evals ------------------------------------------------------------------
st.subheader("Extraction quality (golden set)")
if evals is None:
    st.info(
        "No eval run yet. Build a blind labelling template with "
        "`python -m riskplatform.intel.evals template`, hand-label it into "
        "`evals/golden_set.csv`, then score with `python -m riskplatform.intel.evals score`."
    )
else:
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Labelled examples", evals["n_labelled"])
    e2.metric("Event-type accuracy", f"{evals['event_type_accuracy']:.0%}")
    e3.metric("High-materiality precision", f"{evals['high_materiality_precision']:.0%}")
    e4.metric("High-materiality recall", f"{evals['high_materiality_recall']:.0%}")
    per_class = pd.DataFrame(evals["event_type_per_class"]).T.reset_index(names="event_type")
    st.dataframe(
        per_class,
        width="stretch",
        hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("precision", "recall", "f1")},
    )
    st.caption(f"Scored {evals['scored_utc'][:10]} against model `{evals['model']}`.")

with st.expander("Method notes"):
    st.markdown(
        """
- Signals are extracted from the **headline and feed metadata only** (event type, the
  exchange's price-sensitive flag) — a deliberate scope cut that keeps cost near zero and
  makes the golden-set labelling task well-defined. Document-body extraction is the obvious
  next iteration.
- Extraction uses a **strict JSON schema** (`output_config.format`), so responses always
  parse; every signal row records model, tokens, cost and latency.
- The event study measures **event-day abnormal return** (ticker minus ASX 200 log return)
  and the **realised-vol regime change** (post/pre ratio). "Reacted" means a return within
  3 sessions exceeded 2× the pre-event daily vol.
"""
    )
