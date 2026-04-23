"""Page: Explore the Data.

Tabs:
  1. Top Routes — ranked by volume, with distance/fare context
  2. Compare Routes — side-by-side historical fare chart for two routes
  3. Carrier Profile — per-carrier fare distribution + trend
  4. Seasonality — quarterly heatmap for a chosen route
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from components.styling import apply_base_style
from components.artifacts import load_artifacts, load_route_history_csv, CARRIER_NAMES
from components.prediction import get_top_routes, get_route_history

st.set_page_config(page_title="Explore · US Fare Atlas", page_icon="✈️", layout="wide")
apply_base_style()

art = load_artifacts()
history_df = load_route_history_csv()

st.title("📊 Explore")
st.caption("Interactive views of the underlying DOT dataset and model patterns.")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Top Routes", "Compare Routes", "Carrier Profile", "Seasonality"
])

# ============================================================
# TAB 1 — TOP ROUTES
# ============================================================
with tab1:
    st.subheader("Busiest routes by quarterly passenger volume")
    st.caption("Ranked by average quarterly passengers across the training period (2010–2025). "
               "Bar color encodes average fare — greener = cheaper.")

    n = st.slider("Show top N routes", 10, 100, 30, step=5, key="top_n")
    top = get_top_routes(art, limit=n).copy()

    # Enrich with avg fare from route_history if available
    if not history_df.empty:
        route_fares = (history_df.groupby(['city1', 'city2'])['fare']
                                .mean().reset_index()
                                .rename(columns={'fare': 'avg_fare'}))
        # Try both directions for the merge
        top = top.merge(route_fares, on=['city1', 'city2'], how='left')
        missing_mask = top['avg_fare'].isna()
        if missing_mask.any():
            reverse = top.loc[missing_mask, ['city1', 'city2']].rename(
                columns={'city1': 'city2', 'city2': 'city1'})
            rev_fares = reverse.merge(route_fares, on=['city1', 'city2'], how='left')
            top.loc[missing_mask, 'avg_fare'] = rev_fares['avg_fare'].values
    else:
        top['avg_fare'] = np.nan

    chart = alt.Chart(top).mark_bar().encode(
        y=alt.Y('route:N', sort='-x', title=None),
        x=alt.X('passengers:Q', title='Avg Quarterly Passengers'),
        color=alt.Color('avg_fare:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True),
                        title='Avg Fare ($)'),
        tooltip=[alt.Tooltip('route:N', title='Route'),
                 alt.Tooltip('distance:Q', title='Distance (mi)', format=',.0f'),
                 alt.Tooltip('passengers:Q', title='Avg passengers', format=',.0f'),
                 alt.Tooltip('avg_fare:Q', title='Avg fare', format='$,.2f')],
    ).properties(height=max(400, 22 * len(top)))
    st.altair_chart(chart, use_container_width=True)

# ============================================================
# TAB 2 — COMPARE ROUTES
# ============================================================
with tab2:
    st.subheader("Side-by-side fare history for two routes")
    st.caption("Pick two routes to see how their fares have evolved. "
               "Useful for competitive-market comparisons.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Route A**")
        a1 = st.selectbox("Origin A", art['cities'], key="a_orig",
                          index=art['cities'].index("Chicago, IL") if "Chicago, IL" in art['cities'] else 0)
        a2 = st.selectbox("Destination A", art['cities'], key="a_dest",
                          index=art['cities'].index(
                              "New York City, NY (Metropolitan Area)"
                          ) if "New York City, NY (Metropolitan Area)" in art['cities'] else 1)
    with col_b:
        st.markdown("**Route B**")
        b1 = st.selectbox("Origin B", art['cities'], key="b_orig",
                          index=art['cities'].index(
                              "Los Angeles, CA (Metropolitan Area)"
                          ) if "Los Angeles, CA (Metropolitan Area)" in art['cities'] else 2)
        b2 = st.selectbox("Destination B", art['cities'], key="b_dest",
                          index=art['cities'].index(
                              "San Francisco, CA (Metropolitan Area)"
                          ) if "San Francisco, CA (Metropolitan Area)" in art['cities'] else 3)

    if a1 == a2 or b1 == b2:
        st.warning("Origin and destination must differ for each route.")
    else:
        hist_a = get_route_history(art, a1, a2)
        hist_b = get_route_history(art, b1, b2)

        if hist_a.empty and hist_b.empty:
            st.info("No historical data found for either route. "
                    "This happens when route_history.csv isn't present — "
                    "see the README for how to generate it from the notebook.")
        else:
            def prep(frame, label):
                if frame.empty:
                    return frame
                f = frame.copy()
                f['route'] = label
                f['period_num'] = f['Year'] + (f['quarter'] - 1) / 4
                return f

            route_a_label = f"{a1} → {a2}"
            route_b_label = f"{b1} → {b2}"
            combined = pd.concat([prep(hist_a, route_a_label),
                                   prep(hist_b, route_b_label)], ignore_index=True)

            chart = alt.Chart(combined).mark_line(strokeWidth=2.5).encode(
                x=alt.X('period_num:Q', title='Year', axis=alt.Axis(format='d', tickCount=8)),
                y=alt.Y('fare:Q', title='Avg Fare ($)', scale=alt.Scale(zero=False)),
                color=alt.Color('route:N', scale=alt.Scale(range=['#0c4a6e', '#dc2626'])),
                tooltip=[alt.Tooltip('route:N', title='Route'),
                         alt.Tooltip('Year:O', title='Year'),
                         alt.Tooltip('quarter:O', title='Quarter'),
                         alt.Tooltip('fare:Q', title='Fare', format='$,.2f')],
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)

            # Summary table
            if not hist_a.empty and not hist_b.empty:
                sum_a = hist_a['fare'].agg(['mean', 'std', 'min', 'max']).to_dict()
                sum_b = hist_b['fare'].agg(['mean', 'std', 'min', 'max']).to_dict()
                summary = pd.DataFrame({
                    route_a_label: [f"${sum_a['mean']:.2f}", f"${sum_a['std']:.2f}",
                                     f"${sum_a['min']:.2f}", f"${sum_a['max']:.2f}"],
                    route_b_label: [f"${sum_b['mean']:.2f}", f"${sum_b['std']:.2f}",
                                     f"${sum_b['min']:.2f}", f"${sum_b['max']:.2f}"],
                }, index=['Mean fare', 'Std dev', 'Min', 'Max'])
                st.markdown("**Summary statistics**")
                st.table(summary)

# ============================================================
# TAB 3 — CARRIER PROFILE
# ============================================================
with tab3:
    st.subheader("Carrier fare profile")
    st.caption("Fare distribution and yearly trend for a selected carrier. "
               "Aggregated across all routes where this carrier was the largest carrier.")

    carrier_options = [f"{CARRIER_NAMES.get(c, c)} ({c})" for c in art['carriers']]
    default_idx = next((i for i, c in enumerate(carrier_options) if 'United' in c), 0)
    carrier_label = st.selectbox("Carrier", carrier_options, index=default_idx, key="carrier_profile")
    carrier_code = carrier_label.split('(')[-1].replace(')', '').strip()

    # Tier badge
    if carrier_code in art['legacy_set']:
        tier = "Legacy"
        tier_color = "#0c4a6e"
    elif carrier_code in art['lcc_set']:
        tier = "Low-Cost Carrier (LCC)"
        tier_color = "#059669"
    elif carrier_code in art['ulcc_set']:
        tier = "Ultra Low-Cost (ULCC)"
        tier_color = "#d97706"
    else:
        tier = "Regional / Other"
        tier_color = "#6b7280"

    st.markdown(
        f"<div style='margin-bottom:1rem;'>"
        f"<span style='background:{tier_color}; color:white; padding:0.25rem 0.75rem; "
        f"border-radius:999px; font-size:0.75rem; font-weight:600; "
        f"letter-spacing:0.05em; text-transform:uppercase;'>{tier}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if history_df.empty:
        st.info("Carrier profile requires route_history.csv with per-carrier data. "
                "This export currently only aggregates fares by route. For full "
                "carrier-level analytics, re-export with the carrier_lg column included.")
    else:
        # Approximate: use route-level fare history as proxy for carrier performance
        # (since route_history.csv doesn't have carrier in its simple form)
        st.info("Note: the simple route_history.csv export does not include per-carrier "
                "breakdowns. The chart below shows overall fare patterns across all routes "
                "as a baseline. Re-export with carrier_lg to unlock carrier-specific stats.")

        fare_dist = history_df['fare']
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean fare (all routes)", f"${fare_dist.mean():.2f}")
        c2.metric("Median fare", f"${fare_dist.median():.2f}")
        c3.metric("Observations", f"{len(fare_dist):,}")

        # Histogram
        hist_chart = alt.Chart(history_df).mark_bar(color='#0c4a6e').encode(
            x=alt.X('fare:Q', bin=alt.Bin(maxbins=40), title='Fare ($)'),
            y=alt.Y('count():Q', title='Count'),
        ).properties(height=300, title='Fare Distribution')
        st.altair_chart(hist_chart, use_container_width=True)

        # Yearly trend
        yearly = history_df.groupby('Year')['fare'].mean().reset_index()
        trend_chart = alt.Chart(yearly).mark_line(
            color='#0c4a6e', strokeWidth=2.5
        ).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('fare:Q', title='Mean Fare ($)', scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('Year:O'), alt.Tooltip('fare:Q', format='$,.2f')],
        ).properties(height=300, title='Yearly Mean Fare Trend')
        st.altair_chart(trend_chart, use_container_width=True)

# ============================================================
# TAB 4 — SEASONALITY HEATMAP
# ============================================================
with tab4:
    st.subheader("Quarterly seasonality on a single route")
    st.caption("Year × quarter heatmap of average fare. Helps answer 'when should I fly?'")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s1 = st.selectbox("Origin", art['cities'], key="seasonal_orig",
                          index=art['cities'].index("Chicago, IL") if "Chicago, IL" in art['cities'] else 0)
    with col_s2:
        s2 = st.selectbox("Destination", art['cities'], key="seasonal_dest",
                          index=art['cities'].index(
                              "New York City, NY (Metropolitan Area)"
                          ) if "New York City, NY (Metropolitan Area)" in art['cities'] else 1)

    if s1 == s2:
        st.warning("Origin and destination must differ.")
    else:
        seasonal_hist = get_route_history(art, s1, s2)
        if seasonal_hist.empty:
            st.info(f"No historical observations for {s1} → {s2}. "
                    "Try a more popular route or ensure route_history.csv is present.")
        else:
            heatmap_chart = alt.Chart(seasonal_hist).mark_rect().encode(
                x=alt.X('quarter:O', title='Quarter', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Year:O', title='Year', sort='descending'),
                color=alt.Color('fare:Q',
                                scale=alt.Scale(scheme='redyellowgreen', reverse=True),
                                title='Fare ($)'),
                tooltip=[alt.Tooltip('Year:O'), alt.Tooltip('quarter:O'),
                         alt.Tooltip('fare:Q', format='$,.2f')],
            ).properties(height=500, width=400)

            # Overlay with fare values as text
            text_chart = alt.Chart(seasonal_hist).mark_text(
                fontSize=10, fontWeight=500
            ).encode(
                x='quarter:O', y='Year:O',
                text=alt.Text('fare:Q', format='$.0f'),
                color=alt.condition(
                    alt.datum.fare > seasonal_hist['fare'].median(),
                    alt.value('white'), alt.value('black'),
                ),
            )
            st.altair_chart(heatmap_chart + text_chart, use_container_width=True)

            # Insights
            quarterly_means = seasonal_hist.groupby('quarter')['fare'].mean()
            cheapest_q = quarterly_means.idxmin()
            priciest_q = quarterly_means.idxmax()
            st.markdown(
                f"**Insight:** on this route, **Q{cheapest_q}** is typically cheapest "
                f"(\\${quarterly_means[cheapest_q]:.0f} avg) and **Q{priciest_q}** is "
                f"priciest (\\${quarterly_means[priciest_q]:.0f} avg)."
            )

st.markdown("---")
st.caption("Data: DOT Consumer Airfare Report · EIA · BLS · CS 451 Final Project")
