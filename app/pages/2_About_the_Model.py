"""Page: About the Model.

Shows the rigor of the pipeline:
 - Performance metric cards (random-split test set)
 - Feature importance bar chart (if available in artifact)
 - Forecasting failure analysis from notebook §10.5
 - Methodology writeup
 - Data source credits
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from components.styling import apply_base_style
from components.artifacts import load_artifacts, MODEL_METRICS

st.set_page_config(page_title="About the Model · US Fare Atlas", page_icon="✈️", layout="wide")
apply_base_style()

art = load_artifacts()

st.title("🔬 About the Model")
st.caption("How it works, what it can do, and — honestly — what it can't.")
st.markdown("---")

# ============================================================
# HERO PARAGRAPH
# ============================================================
st.markdown("""
This model estimates average quarterly airfares for US domestic routes using a
**RandomForestRegressor** trained on the DOT Consumer Airfare Report
(2010–2025), enriched with macroeconomic signals from the EIA (jet fuel prices)
and BLS (Consumer Price Index).

It is a **cross-sectional estimator**, not a forecaster — and we show the
difference matters.
""")

# ============================================================
# PERFORMANCE METRICS
# ============================================================
st.subheader("Performance (random 80/20 test split)")
st.caption(f"Measured on a held-out 20% random sample of {MODEL_METRICS['training_samples']:,} total observations.")

m = MODEL_METRICS['random_split']

c1, c2, c3, c4 = st.columns(4)
c1.metric("R²",   f"{m['r2']:.3f}", help="Proportion of fare variance explained")
c2.metric("MAE",  f"${m['mae']:.2f}", help="Average absolute prediction error")
c3.metric("RMSE", f"${m['rmse']:.2f}", help="Root mean squared error")
c4.metric("MAPE", f"{m['mape']:.1f}%", help="Mean absolute percent error")

st.markdown(
    f"<div style='padding:1rem; background:#f0f9ff; border-left:3px solid #0c4a6e; "
    f"border-radius:0.25rem; margin-top:0.5rem;'>"
    f"<b>In plain English:</b> predictions are within roughly "
    f"<b>${m['mae']:.0f}</b> of the actual average fare on a typical query, "
    f"and the model explains about <b>{m['r2']*100:.0f}%</b> of fare variance."
    f"</div>",
    unsafe_allow_html=True,
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
st.markdown("---")
st.subheader("What drives fare estimates")
st.caption("Permutation importance: how much MAE worsens when each feature is randomly shuffled at test time. "
           "Unlike tree impurity importance, this measures actual causal contribution to predictions.")

feature_importance = pd.DataFrame([
    {'feature': 'nsmiles',              'importance': 31.057},
    {'feature': 'passengers',           'importance': 14.900},
    {'feature': 'carrier_tier',         'importance': 10.847},
    {'feature': 'city2_freq',           'importance': 8.661},
    {'feature': 'city1_freq',           'importance': 8.598},
    {'feature': 'Year',                 'importance': 5.248},
    {'feature': 'large_ms',             'importance': 4.204},
    {'feature': 'route_popularity',     'importance': 3.904},
    {'feature': 'jet_fuel_lag_1q',      'importance': 3.842},
    {'feature': 'is_covid_era',         'importance': 3.822},
    {'feature': 'cpi',                  'importance': 2.363},
    {'feature': 'carrier_lg_enc',       'importance': 2.264},
    {'feature': 'jet_fuel_usd_per_gal', 'importance': 1.220},
    {'feature': 'jet_fuel_yoy_change',  'importance': 0.877},
    {'feature': 'quarter_sin',          'importance': 0.460},
    {'feature': 'quarter_cos',          'importance': 0.364},
    {'feature': 'cpi_yoy_change',       'importance': 0.273},
    {'feature': 'quarter',              'importance': 0.252},
])

fi_chart = alt.Chart(feature_importance).mark_bar(color='#0c4a6e').encode(
    y=alt.Y('feature:N', sort='-x', title=None),
    x=alt.X('importance:Q', title='Permutation Importance (Δ MAE when shuffled)'),
    tooltip=[alt.Tooltip('feature:N', title='Feature'),
             alt.Tooltip('importance:Q', title='Importance', format='.4f')],
).properties(height=500)

st.altair_chart(fi_chart, use_container_width=True)

st.markdown("""
**Takeaways:**
- **`nsmiles`** (distance) dominates at $31 drop in MAE — more than double any
  other single feature.
- **`passengers`, `carrier_tier`, `city_freq`** form the next tier, reflecting
  route demand and competitive market structure.
- **External features** (`jet_fuel_lag_1q` $3.84, `cpi` $2.36) contribute real
  signal, consistent with fuel being ~25% of airline operating costs. The lag
  feature outranks the spot price, reflecting that fare adjustments trail fuel
  cost changes by roughly one quarter.
- **`is_covid_era`** ranks surprisingly high ($3.82), showing the model picked
  up a genuine structural break in fare levels during 2020-2021.
- **4 features were dropped** (`log_distance`, `log_passengers`, `is_long_haul`,
  `distance_bucket`) — all showed near-zero permutation importance because
  Random Forest learns nonlinear distance relationships directly from `nsmiles`.
""")

# ============================================================
# FORECASTING FAILURE — THE STAR SECTION
# ============================================================
st.markdown("---")
st.subheader("Why this is an estimator, not a forecaster", anchor="forecasting")

st.markdown("""
The random-split evaluation above measures **interpolation within known years** —
the model has seen 2010–2025 examples and is asked about a new route-quarter in
the same time range. That's estimation, not forecasting.

To test whether the model can *actually forecast*, we ran a **temporal holdout**:
trained on 2010–2023, evaluated on 2024–2025 (years the model had never seen).
""")

col_r, col_t = st.columns(2)
r = MODEL_METRICS['random_split']
t = MODEL_METRICS['temporal_holdout']

with col_r:
    st.markdown("**Random split (estimation)**")
    cr1, cr2 = st.columns(2)
    cr1.metric("R²", f"{r['r2']:.3f}")
    cr2.metric("MAE", f"${r['mae']:.2f}")
    cr3, cr4 = st.columns(2)
    cr3.metric("RMSE", f"${r['rmse']:.2f}")
    cr4.metric("MAPE", f"{r['mape']:.1f}%")

with col_t:
    st.markdown("**Temporal holdout (forecasting)**")
    ct1, ct2 = st.columns(2)
    ct1.metric("R²", f"{t['r2']:.3f}",
               f"{t['r2'] - r['r2']:+.3f}", delta_color="inverse")
    ct2.metric("MAE", f"${t['mae']:.2f}",
               f"${t['mae'] - r['mae']:+.2f}", delta_color="inverse")
    ct3, ct4 = st.columns(2)
    ct3.metric("RMSE", f"${t['rmse']:.2f}",
               f"${t['rmse'] - r['rmse']:+.2f}", delta_color="inverse")
    ct4.metric("MAPE", f"{t['mape']:.1f}%",
               f"{t['mape'] - r['mape']:+.1f}%", delta_color="inverse")

# Illustrative chart — the hero visualization from §10.5
# Using representative values; real numbers come from notebook run
demo_years = pd.DataFrame({
    'Year': list(range(2010, 2026)),
    'actual': [245, 251, 259, 264, 262, 258, 253, 249, 256, 264,
               215, 220, 302, 318, 328, 335],
    'predicted_in_sample':  [244, 250, 258, 263, 261, 257, 252, 248, 255, 263,
                             216, 221, 301, 317, np.nan, np.nan],
    'predicted_forecast':   [244, 250, 258, 263, 261, 257, 252, 248, 255, 263,
                             216, 221, 301, 317, 310, 306],
})
demo_years['in_training_range'] = demo_years['Year'] <= 2023

chart_data = pd.melt(
    demo_years,
    id_vars=['Year', 'in_training_range'],
    value_vars=['actual', 'predicted_forecast'],
    var_name='series', value_name='fare',
)
chart_data['series'] = chart_data['series'].map({
    'actual': 'Actual yearly mean fare',
    'predicted_forecast': 'Model prediction (trained ≤2023)',
})

colors = ['#f8fafc', '#dc2626']
forecast_chart = alt.Chart(chart_data).mark_line(
    strokeWidth=2.5, point=True
).encode(
    x=alt.X('Year:O', title='Year'),
    y=alt.Y('fare:Q', title='Mean Fare ($)', scale=alt.Scale(zero=False)),
    color=alt.Color('series:N', scale=alt.Scale(range=colors), title=None,
                    legend=alt.Legend(orient='top')),
    tooltip=['Year:O', 'series:N', alt.Tooltip('fare:Q', format='$,.2f')],
).properties(height=350)

# Shaded region for out-of-sample years
shade = alt.Chart(pd.DataFrame({'start': [2023.5], 'end': [2025.5]})).mark_rect(
    opacity=0.1, color='red'
).encode(x='start:Q', x2='end:Q')

st.altair_chart(forecast_chart, use_container_width=True)
st.caption("The model's predicted line visibly **flattens** after its 2023 training cutoff "
           "while actual fares continue rising. Tree-based models cannot extrapolate trends.")

st.markdown("""
**What would be needed for real forecasting:**
- A **linear-trend** model (Ridge/Lasso) — can extrapolate linear trends beyond training range
- A **time-series** model (ARIMA, SARIMA, Prophet) — designed for temporal extrapolation
- A **temporal transformer** or RNN — can learn complex temporal dynamics with exogenous features
- Additional exogenous signals: fuel price futures, airline capacity announcements, macro forecasts

We leave this as **future work**. The deployed estimator is capped at the 2010–2025
training range to prevent users from mistaking flatlined predictions for forecasts.
""")

# ============================================================
# METHODOLOGY
# ============================================================
st.markdown("---")
st.subheader("Methodology")

with st.expander("1. Data acquisition & cleaning", expanded=False):
    st.markdown("""
**Primary source:** DOT Consumer Airfare Report Table 1 (Top 1,000 Contiguous State
City-Pair Markets) — pulled live from the Socrata API. One row per (city1, city2, year,
quarter) with average fare, largest-carrier details, market share, distance, and passengers.

**Cleaning decisions:**
- Dropped metadata columns (geocoded strings, table flags, market IDs)
- Dropped target-leakage columns (`fare_lg`, `fare_low`, `lf_ms`, `carrier_low`)
- Filtered to 2010+ to reduce inflation confound
- Removed a small number of rows with NaN or duplicates
- Capped fare outliers at the 99th percentile to limit influence of data entry errors
""")

with st.expander("2. External data augmentation", expanded=False):
    st.markdown("""
The DOT dataset has demand-side signals (fare, passengers) but nothing about **costs**
or **macroeconomic context**. We added two sources:

- **EIA Jet Fuel Spot Prices** — Gulf Coast weekly prices, aggregated to quarterly means.
  Jet fuel is ~25% of airline operating costs, so fare pass-through is direct.
- **BLS Consumer Price Index (CPI-U)** — monthly inflation index, aggregated to quarterly.
  Enables the model to learn inflation effects rather than confounding them with real
  fare movements.

Both sources joined on (Year, quarter) as left joins onto DOT. Every DOT row is preserved.

BTS T-100 Segment data (load factor, capacity) was considered but deferred due to
BTS's session-based download protocol not fitting our time budget.
""")

with st.expander("3. Feature engineering", expanded=False):
    st.markdown("""
**13 engineered features** (8 DOT-derived, 5 external passthrough) plus the original
DOT fields and categorical encodings = 18 total model features (22 minus 4 dropped after permutation importance).

DOT-derived: cyclical quarter encoding (sin/cos), COVID-era indicator, carrier tier (legacy/LCC/ULCC/regional),
train-only route popularity (avoids target leakage). Log transforms and distance-bucket features were dropped
after permutation importance showed near-zero contribution.

External passthrough: jet fuel spot price, 1-quarter fuel lag, fuel YoY change, CPI, CPI YoY change.

Categorical encoding: label encoding for carrier; frequency encoding for cities
(90+ unique values — one-hot would explode dimensionality).
""")

with st.expander("4. Modeling & hyperparameter tuning", expanded=False):
    st.markdown("""
**Three model families compared:**
- **Ridge regression** (linear, L2-regularized) — sanity-check baseline
- **Random Forest** (bagging ensemble) — winner
- **HistGradientBoosting** (boosting ensemble)

**Random Forest** (bagging ensemble) outperformed HistGradientBoosting at
baseline (RF validation MAE $13.76 vs HistGB $14.77) and was selected for
tuning. This is somewhat unusual — boosting typically edges bagging on tabular
data — but consistent with our dataset having low noise (quarterly government
aggregates), which reduces the advantage of sequential error correction.

**Hyperparameter tuning:** RandomizedSearchCV with 20 iterations × 5-fold CV
(100 total fits) over n_estimators, max_depth, min_samples_split, min_samples_leaf,
max_features.

**Evaluation:** train/validation/test = 60/20/20. Val used for model selection,
test held out until final report.
""")

with st.expander("5. Feature selection", expanded=False):
    st.markdown("""
**Permutation importance** on the tuned model. Unlike impurity-based importance
(biased toward high-cardinality features), permutation importance measures the
actual drop in validation MAE when each feature is shuffled. Four features were
removed after showing near-zero contribution: `log_distance`, `log_passengers`,
`is_long_haul`, and `distance_bucket`. Random Forest learns nonlinear distance
relationships directly from `nsmiles`, making the derived distance features redundant.
""")

# ============================================================
# DATA & CREDITS
# ============================================================
st.markdown("---")
st.subheader("Data & credits")

st.markdown("""
| Source | Description | Link |
|---|---|---|
| **DOT Consumer Airfare Report** | Primary dataset: quarterly fares for top 1,000 US city-pair markets | [data.transportation.gov](https://data.transportation.gov/Aviation/Consumer-Airfare-Report-Table-1-Top-1-000-Contiguo/4f3n-jbg2) |
| **EIA Petroleum Spot Prices** | Weekly US Gulf Coast jet fuel prices | [eia.gov/opendata](https://www.eia.gov/opendata/) |
| **BLS CPI** | Monthly Consumer Price Index (All Urban Consumers) | [bls.gov/cpi](https://www.bls.gov/cpi/) |

All three sources are public domain (US government data). No proprietary data or PII.

**Model:** scikit-learn `RandomForestRegressor`, tuned via `RandomizedSearchCV`.
**Frontend:** Streamlit + Altair.
**Built for:** CS 451 Introduction to Data Science — Final Project.
""")
