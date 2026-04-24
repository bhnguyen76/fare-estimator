"""US Fare Atlas — Streamlit App
Main entrypoint. Run with:
    streamlit run streamlit_app.py

Landing page = Estimator.
Other pages (Explore, About) are in pages/ and auto-discovered by Streamlit.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from components.styling import apply_base_style
from components.artifacts import load_artifacts, CARRIER_NAMES, QUARTER_LABELS

# ---- Page config (must be first st.* call) ----
st.set_page_config(
    page_title="US Fare Atlas",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_base_style()

# ---- Load artifacts (cached) ----
art = load_artifacts()

# ---- Header ----
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("✈️ US Fare Atlas")
    st.caption("Estimate historical US domestic airfares using a tuned "
               "Random Forest model trained on DOT + EIA + BLS data.")
with col_badge:
    st.markdown(
        "<div style='text-align:right; padding-top:1.2rem;'>"
        "<span style='background:#fef3c7; color:#92400e; "
        "padding:0.3rem 0.75rem; border-radius:999px; font-size:0.75rem; "
        "font-weight:600; letter-spacing:0.05em;'>"
        "ESTIMATOR · NOT A FORECASTER"
        "</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ============================================================
# ESTIMATOR FORM
# ============================================================
left, right = st.columns([1, 1.3])

with left:
    st.subheader("Query")

    city1 = st.selectbox(
        "Origin City",
        options=art['cities'],
        index=art['cities'].index("Chicago, IL") if "Chicago, IL" in art['cities'] else 0,
    )
    city2 = st.selectbox(
        "Destination City",
        options=art['cities'],
        index=art['cities'].index(
            "New York City, NY (Metropolitan Area)"
        ) if "New York City, NY (Metropolitan Area)" in art['cities'] else 1,
    )

    carrier_choices = [f"{CARRIER_NAMES.get(c, c)} ({c})" for c in art['carriers']]
    carrier_label = st.selectbox(
        "Airline",
        options=carrier_choices,
        index=next((i for i, c in enumerate(carrier_choices) if 'United' in c), 0),
    )

    col_y, col_q = st.columns(2)
    with col_y:
        year = st.slider("Year", 2010, 2025, 2024, step=1)
    with col_q:
        quarter_label = st.selectbox("Quarter", options=list(QUARTER_LABELS.keys()), index=1)

    run = st.button("Estimate Fare", type="primary", use_container_width=True)

with right:
    st.subheader("Result")

    if city1 == city2:
        st.warning("⚠️ Origin and destination must be different cities.")
    else:
        # Always compute on first render OR when button clicked
        from components.prediction import predict_fare, get_route_history

        result = predict_fare(art, city1, city2, carrier_label, year, quarter_label)

        # Hero metric
        st.markdown(
            f"<div style='padding:1.5rem; background:#f1f5f9; border-radius:0.75rem; "
            f"border:1px solid #e2e8f0;'>"
            f"<div style='font-size:0.8rem; color:#64748b; "
            f"text-transform:uppercase; letter-spacing:0.05em; font-weight:600;'>"
            f"Estimated Average Fare</div>"
            f"<div style='font-family:ui-monospace, monospace; font-size:3rem; "
            f"font-weight:700; color:#0c4a6e; line-height:1.2;'>"
            f"${result['estimate']:,.2f}</div>"
            f"<div style='font-size:0.9rem; color:#64748b;'>"
            f"± ${result['residual_std']:,.2f} (1σ band)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Route summary
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Distance", f"{result['distance']:,} mi")
        c2.metric("Passengers (quarterly avg)", f"{result['passengers']:,}")
        c3.metric("Largest-carrier share", f"{result['large_ms']*100:.0f}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("Jet fuel price", f"${result['macro']['jet_fuel_usd_per_gal']:.2f}/gal")
        c5.metric("CPI index", f"{result['macro']['cpi']:.1f}")
        c6.metric("YoY inflation", f"{result['macro']['cpi_yoy_change']:+.1f}%")

        if not result['route_known']:
            st.warning("⚠️ This route is not in the DOT Top-1000 dataset. "
                       "Distance and passenger defaults used; estimate is less reliable.")

# ============================================================
# HISTORICAL FARE CHART
# ============================================================
st.markdown("---")
st.subheader("Historical fare on this route")
st.caption("The observed average quarterly fare for your selected city pair over the training period. "
           "The highlighted point is your query.")

if city1 != city2:
    from components.prediction import get_route_history
    import altair as alt

    history = get_route_history(art, city1, city2)

    if history.empty:
        st.info("No historical DOT observations for this exact route. The model fell back "
                "to similar-route averages for its estimate.")
    else:
        quarter_num = QUARTER_LABELS[quarter_label]
        history['is_selected'] = (history['Year'] == year) & (history['quarter'] == quarter_num)
        history['period_label'] = history['Year'].astype(str) + ' Q' + history['quarter'].astype(str)
        history['period_num'] = history['Year'] + (history['quarter'] - 1) / 4

        base = alt.Chart(history).encode(
            x=alt.X('period_num:Q', title='Year',
                    scale=alt.Scale(domain=[2010, 2026]),
                    axis=alt.Axis(format='d', tickCount=8)),
            y=alt.Y('fare:Q', title='Average Fare ($)',
                    scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('period_label:N', title='Period'),
                     alt.Tooltip('fare:Q', title='Fare', format='$,.2f')],
        )
        line = base.mark_line(color='#0c4a6e', strokeWidth=2, opacity=0.7)
        dots = base.mark_circle(size=30, color='#0c4a6e', opacity=0.6)
        selected = alt.Chart(history[history['is_selected']]).mark_circle(
            size=250, color='#dc2626', opacity=1).encode(
            x='period_num:Q', y='fare:Q',
            tooltip=[alt.Tooltip('period_label:N', title='Your query'),
                     alt.Tooltip('fare:Q', title='Actual fare', format='$,.2f')],
        )

        st.altair_chart((line + dots + selected).properties(height=300), use_container_width=True)

# ============================================================
# DISCLOSURE
# ============================================================
st.markdown("---")
with st.expander("ℹ️ Why this is an estimator, not a forecaster", expanded=False):
    st.markdown("""
This model performs **cross-sectional estimation** — it fills in cells of a
route × carrier × time grid based on historical patterns. It is **not** a
forecaster.

Tree-based models cannot extrapolate beyond the year range they were trained on.
When asked about a year after 2025, the model would plateau at its 2025 behavior
regardless of underlying trend. We capped the year slider at 2025 to prevent
this misuse.

A temporal holdout experiment in the notebook (§10.5) trained the same model on
2010–2023 and tested on 2024–2025. Performance degrades substantially on the
out-of-sample years — see the **About the Model** page for the full analysis.
""")

st.caption(
    "Data: DOT Consumer Airfare Report (primary) · EIA Jet Fuel Spot Prices · BLS CPI · "
    "Model: RandomForestRegressor (scikit-learn) · "
    "Built for CS 451 Final Project."
)
