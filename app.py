"""CausalCoach — Streamlit app for Bayesian causal impact analysis.

Pure-Python implementation using `tfcausalimpact` (a maintained successor to
`pycausalimpact`). Deployable to Streamlit Cloud with no R dependency.
"""
from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from causalimpact import CausalImpact  # provided by tfcausalimpact

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CausalCoach — Causal Impact Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("📈 CausalCoach")
st.caption(
    "Upload a time series, mark when your campaign launched, and get a "
    "Bayesian causal impact estimate of the lift."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "date" not in df.columns or "y" not in df.columns:
        raise ValueError("CSV must contain columns named `date` and `y`.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some `date` values could not be parsed (use YYYY-MM-DD).")
    df = df.sort_values("date").reset_index(drop=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if df["y"].isna().any():
        raise ValueError("Column `y` must be fully numeric.")
    return df


def run_causal_impact(
    df: pd.DataFrame, covariates: list[str], campaign_start: pd.Timestamp
) -> tuple[CausalImpact, pd.DataFrame]:
    data = df.copy()
    cols = ["y"] + covariates
    data = data.set_index("date")[cols]

    pre_end_idx = data.index.searchsorted(campaign_start) - 1
    pre_period = [data.index[0], data.index[pre_end_idx]]
    post_period = [data.index[pre_end_idx + 1], data.index[-1]]

    ci = CausalImpact(data, pre_period, post_period)
    return ci, data


def summary_metrics(ci: CausalImpact) -> dict:
    s = ci.summary_data  # pandas DataFrame with columns ['average', 'cumulative']
    avg_effect = float(s.loc["abs_effect", "average"])
    ci_lower = float(s.loc["abs_effect_lower", "average"])
    ci_upper = float(s.loc["abs_effect_upper", "average"])
    rel_lift = float(s.loc["rel_effect", "average"])
    prob = float(1 - ci.p_value)
    return {
        "avg_effect": avg_effect,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "rel_lift": rel_lift,
        "prob_positive": prob,
    }


def build_plot(
    ci: CausalImpact, data: pd.DataFrame, campaign_start: pd.Timestamp
) -> go.Figure:
    inf = ci.inferences.copy()
    inf.index = data.index  # align dates

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=inf.index,
            y=inf["preds_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=inf.index,
            y=inf["preds_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.20)",
            name="95% credible interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=inf.index,
            y=inf["preds"],
            mode="lines",
            line=dict(color="#3b82f6", dash="dash", width=2),
            name="Counterfactual",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["y"],
            mode="lines",
            line=dict(color="#ef4444", width=2.5),
            name="Actual",
        )
    )
    fig.add_vline(
        x=campaign_start,
        line=dict(color="#111827", dash="dot"),
        annotation_text="Campaign start",
        annotation_position="top",
    )
    fig.update_layout(
        title="Actual vs. Counterfactual",
        xaxis_title="Date",
        yaxis_title="y",
        hovermode="x unified",
        template="plotly_white",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use sample dataset", value=not bool(uploaded))

    df: pd.DataFrame | None = None
    error: str | None = None
    try:
        if uploaded is not None:
            df = load_csv(uploaded.getvalue())
        elif use_sample:
            with open("sample_data.csv", "rb") as f:
                df = load_csv(f.read())
    except Exception as e:  # noqa: BLE001
        error = str(e)

    campaign_start: date | None = None
    covariates: list[str] = []

    if df is not None:
        st.success(f"Loaded {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
        st.header("2. Campaign start")
        min_d, max_d = df["date"].min().date(), df["date"].max().date()
        default = df["date"].iloc[len(df) // 2].date()
        campaign_start = st.date_input(
            "Date the campaign began",
            value=default,
            min_value=min_d,
            max_value=max_d,
        )

        extra_cols = [c for c in df.columns if c not in ("date", "y")]
        if extra_cols:
            st.header("3. Covariates (optional)")
            covariates = st.multiselect(
                "Control series used to model the counterfactual",
                options=extra_cols,
                default=extra_cols,
            )

        run = st.button("Run causal impact", type="primary", use_container_width=True)
    else:
        run = False
        if error:
            st.error(error)
        else:
            st.info("Upload a CSV or enable the sample dataset to get started.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if df is None:
    st.markdown(
        """
        ### How to use
        1. Upload a CSV with columns **`date`** (YYYY-MM-DD) and **`y`** (numeric).
        2. Optionally include extra numeric columns as covariates.
        3. Pick the campaign start date in the sidebar.
        4. Click **Run causal impact**.
        """
    )
    st.stop()

assert campaign_start is not None
campaign_ts = pd.Timestamp(campaign_start)

pre_n = int((df["date"] < campaign_ts).sum())
post_n = int((df["date"] >= campaign_ts).sum())

c1, c2 = st.columns(2)
c1.metric("Pre-campaign points", pre_n)
c2.metric("Post-campaign points", post_n)

if pre_n < 8:
    st.error("Need at least 8 pre-campaign observations.")
    st.stop()
if post_n < 3:
    st.error("Need at least 3 post-campaign observations.")
    st.stop()

if not run:
    st.info("Click **Run causal impact** in the sidebar when you're ready.")
    st.stop()

with st.spinner("Fitting Bayesian structural time series model…"):
    try:
        ci, data = run_causal_impact(df, covariates, campaign_ts)
        m = summary_metrics(ci)
    except Exception as e:  # noqa: BLE001
        st.error(f"Model failed: {e}")
        st.stop()

# Metric cards
m1, m2, m3 = st.columns(3)
m1.metric(
    "Average effect",
    f"{m['avg_effect']:+,.2f}",
    delta=f"{m['rel_lift'] * 100:+.1f}% lift",
)
m2.metric("95% CI lower", f"{m['ci_lower']:+,.2f}")
m3.metric("95% CI upper", f"{m['ci_upper']:+,.2f}")

if m["ci_lower"] > 0:
    st.success(
        f"Statistically significant positive impact "
        f"(P(effect > 0) ≈ {m['prob_positive']:.1%})."
    )
elif m["ci_upper"] < 0:
    st.warning(
        f"Statistically significant negative impact "
        f"(P(effect > 0) ≈ {m['prob_positive']:.1%})."
    )
else:
    st.info(
        f"Effect not distinguishable from zero at the 95% level "
        f"(P(effect > 0) ≈ {m['prob_positive']:.1%})."
    )

st.plotly_chart(build_plot(ci, data, campaign_ts), use_container_width=True)

with st.expander("Full model summary"):
    st.text(ci.summary())
    st.text(ci.summary(output="report"))
