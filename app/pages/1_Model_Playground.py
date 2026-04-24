"""Page: Model Playground.

Tabs:
  1. Route Residuals     — where does the model agree with reality?
  2. Actual vs Predicted — compare model accuracy on two routes over time
  3. Carrier Response    — learned response curves per carrier
  4. Seasonality         — actual vs predicted seasonal patterns
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from components.styling import apply_base_style
from components.artifacts import load_artifacts, load_route_history_csv, CARRIER_NAMES
from components.prediction import (
    get_top_routes_with_predictions,
    get_route_with_predictions,
    batch_predict,
    _get_reference_carrier,
    get_feature_contributions,
    lookup_both_directions,
)

st.set_page_config(page_title="Model Playground · US Fare Atlas",
                   page_icon="✈️", layout="wide")
apply_base_style()

art = load_artifacts()
history_df = load_route_history_csv()

st.title("🧪 Model Playground")
st.caption("Probe what the model learned. Compare its predictions against reality "
           "on real routes, carriers, and time periods.")
st.markdown("---")

if history_df.empty:
    st.warning(
        "⚠️ `route_history.csv` not found. Tabs that compare actual vs predicted fares "
        "will show model predictions only. Generate the file by running notebook §12.1."
    )

# Color conventions (consistent across all tabs per spec)
C_ACTUAL    = '#f8fafc'
C_PREDICTED = '#dc2626'
C_AMBER     = '#f59e0b'
C_BLUE      = '#2563eb'
C_GRAY      = '#9ca3af'

tab1, tab2, tab3, tab4 = st.tabs([
    "Route Residuals", "Actual vs Predicted", "Carrier Response", "Seasonality"
])


# ── Shared heatmap helper ─────────────────────────────────────────────────────

def _heatmap_with_text(df, value_col, title, color_scheme='redyellowgreen',
                       reverse=True, domain_mid=None, fmt='$.0f'):
    """Year × quarter heatmap with text overlay."""
    color_scale = (
        alt.Scale(scheme=color_scheme, reverse=reverse, domainMid=domain_mid)
        if domain_mid is not None
        else alt.Scale(scheme=color_scheme, reverse=reverse)
    )
    hm = alt.Chart(df).mark_rect().encode(
        x=alt.X('quarter:O', title='Quarter', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Year:O', title='Year', sort='descending'),
        color=alt.Color(f'{value_col}:Q', scale=color_scale, title='($)'),
        tooltip=[
            alt.Tooltip('Year:O'),
            alt.Tooltip('quarter:O', title='Quarter'),
            alt.Tooltip(f'{value_col}:Q', title=title, format=fmt),
        ],
    ).properties(height=400, title=title)

    threshold = df[value_col].median() if domain_mid is None else 0
    txt = alt.Chart(df).mark_text(fontSize=10, fontWeight=500).encode(
        x='quarter:O',
        y=alt.Y('Year:O', sort='descending'),
        text=alt.Text(f'{value_col}:Q', format=fmt),
        color=alt.condition(
            alt.datum[value_col] > threshold,
            alt.value('white'), alt.value('black'),
        ),
    )
    return hm + txt


# ============================================================
# TAB 1 — ROUTE RESIDUALS
# ============================================================
with tab1:
    st.subheader("Where does the model agree with reality?")
    st.caption(
        "Top routes by passenger volume. Bar length = mean actual fare. "
        "Color = model residual (actual − predicted): blue = model under-predicts, "
        "red = model over-predicts."
    )

    col_n, col_sort = st.columns([1, 2])
    with col_n:
        n = st.slider("Show top N routes", 10, 100, 30, step=5, key="t1_n")
    with col_sort:
        sort_by = st.radio(
            "Sort by",
            ["By passenger volume",
             "By |residual| (hardest first)",
             "By |residual| (easiest first)"],
            horizontal=True,
            key="t1_sort",
        )

    with st.spinner("Computing model predictions..."):
        top = get_top_routes_with_predictions(art, limit=n)

    if top['actual_fare'].isna().all():
        st.info(
            "Actual fare data requires `route_history.csv` — generate it from notebook §12.1. "
            "Showing model predictions only."
        )
        top_pred = top.dropna(subset=['predicted_fare'])
        if not top_pred.empty:
            chart = alt.Chart(top_pred).mark_bar(color='#0c4a6e').encode(
                y=alt.Y('route:N', sort='-x', title=None),
                x=alt.X('predicted_fare:Q', title='Predicted Fare ($)'),
                tooltip=[
                    alt.Tooltip('route:N', title='Route'),
                    alt.Tooltip('distance:Q', title='Distance (mi)', format=',.0f'),
                    alt.Tooltip('predicted_fare:Q', title='Predicted fare', format='$,.2f'),
                ],
            ).properties(height=max(400, 22 * len(top_pred)))
            st.altair_chart(chart, use_container_width=True)
    else:
        top_plot = top.dropna(subset=['actual_fare', 'residual']).copy()

        if "hardest" in sort_by:
            top_plot = top_plot.iloc[top_plot['residual'].abs().argsort()[::-1].values]
        elif "easiest" in sort_by:
            top_plot = top_plot.iloc[top_plot['residual'].abs().argsort().values]

        chart = alt.Chart(top_plot).mark_bar().encode(
            y=alt.Y('route:N', sort=None, title=None),
            x=alt.X('actual_fare:Q', title='Mean Actual Fare ($)'),
            color=alt.Color(
                'residual:Q',
                scale=alt.Scale(scheme='redblue', domainMid=0),
                title='Residual ($)',
            ),
            tooltip=[
                alt.Tooltip('route:N', title='Route'),
                alt.Tooltip('distance:Q', title='Distance (mi)', format=',.0f'),
                alt.Tooltip('passengers:Q', title='Avg passengers', format=',.0f'),
                alt.Tooltip('actual_fare:Q', title='Actual fare', format='$,.2f'),
                alt.Tooltip('predicted_fare:Q', title='Predicted fare', format='$,.2f'),
                alt.Tooltip('residual:Q', title='Residual', format='+,.2f'),
            ],
        ).properties(height=max(400, 22 * len(top_plot)))
        st.altair_chart(chart, use_container_width=True)

        excels    = int((top_plot['residual'].abs() < 10).sum())
        struggles = int((top_plot['residual'].abs() > 30).sum())
        worst_routes = (top_plot.assign(_a=top_plot['residual'].abs())
                                .nlargest(3, '_a')['route'].tolist())

        m1, m2 = st.columns(2)
        m1.metric("Routes model excels on (|residual| < \$10)", excels)
        m2.metric("Routes model struggles on (|residual| > \$30)", struggles)
        if worst_routes:
            st.caption("Hardest routes: " + " · ".join(worst_routes))

    st.markdown("---")
    st.page_link("streamlit_app.py", label="Open Estimator ✈️",
                 help="Re-enter the route manually in the Estimator.")


# ============================================================
# TAB 2 — ACTUAL vs PREDICTED ROUTE HISTORY
# ============================================================
with tab2:
    st.subheader("Actual vs predicted fare history")
    st.caption(
        "How well does the model track fare trends on specific routes over time? "
        "White solid = actual, red dashed = model prediction."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Route A**")
        a1 = st.selectbox(
            "Origin A", art['cities'], key="t2_a_orig",
            index=art['cities'].index("Chicago, IL") if "Chicago, IL" in art['cities'] else 0)
        a2 = st.selectbox(
            "Destination A", art['cities'], key="t2_a_dest",
            index=art['cities'].index("New York City, NY (Metropolitan Area)")
            if "New York City, NY (Metropolitan Area)" in art['cities'] else 1)
    with col_b:
        st.markdown("**Route B**")
        b1 = st.selectbox(
            "Origin B", art['cities'], key="t2_b_orig",
            index=art['cities'].index("Los Angeles, CA (Metropolitan Area)")
            if "Los Angeles, CA (Metropolitan Area)" in art['cities'] else 2)
        b2 = st.selectbox(
            "Destination B", art['cities'], key="t2_b_dest",
            index=art['cities'].index("San Francisco, CA (Metropolitan Area)")
            if "San Francisco, CA (Metropolitan Area)" in art['cities'] else 3)

    carrier_opts = ["Auto (dominant carrier)"] + [
        f"{CARRIER_NAMES.get(c, c)} ({c})" for c in art['carriers']
    ]
    carrier_sel = st.selectbox("Carrier (both routes)", carrier_opts, key="t2_carrier")
    carrier_for_pred = (None if carrier_sel == "Auto (dominant carrier)"
                        else carrier_sel.split('(')[-1].replace(')', '').strip())

    if a1 == a2 or b1 == b2:
        st.warning("Origin and destination must differ for each route.")
    else:
        with st.spinner("Computing model predictions..."):
            hist_a = get_route_with_predictions(art, a1, a2, carrier=carrier_for_pred)
            hist_b = get_route_with_predictions(art, b1, b2, carrier=carrier_for_pred)

        label_a = f"{a1} → {a2}"
        label_b = f"{b1} → {b2}"

        if history_df.empty:
            st.info("No historical data — generate `route_history.csv` from notebook §12.1.")
        elif hist_a.empty and hist_b.empty:
            st.info("No historical observations found for either route.")
        else:
            def _render_route_chart(df, label):
                if df.empty:
                    st.info(f"No historical data for {label}.")
                    return float('nan')

                df = df.copy()
                df['period_num']   = df['Year'] + (df['quarter'] - 1) / 4
                df['period_label'] = df['Year'].astype(str) + ' Q' + df['quarter'].astype(str)

                has_resid = ('residual' in df.columns and not df['residual'].isna().all())
                mae = df['residual'].abs().mean() if has_resid else float('nan')

                if len(df) < 8:
                    st.info(f"Route {label} has limited history ({len(df)} observations); "
                            "accuracy estimates may be unreliable.")

                mae_str = f"  —  MAE: **\\${mae:.2f}**" if not np.isnan(mae) else ""
                st.markdown(f"**{label}**{mae_str}")

                enc_x = alt.X('period_num:Q', title='Year',
                               scale=alt.Scale(domain=[2010, 2026]),
                               axis=alt.Axis(format='d', tickCount=8))
                tooltip = [
                    alt.Tooltip('period_label:N', title='Period'),
                    alt.Tooltip('fare:Q', title='Actual', format='$,.2f'),
                    alt.Tooltip('predicted:Q', title='Predicted', format='$,.2f'),
                    alt.Tooltip('residual:Q', title='Residual', format='+$,.2f'),
                ]

                actual_line = alt.Chart(df).mark_line(
                    color=C_ACTUAL, strokeWidth=2.5
                ).encode(
                    x=enc_x,
                    y=alt.Y('fare:Q', title='Avg Fare ($)', scale=alt.Scale(zero=False)),
                    tooltip=tooltip,
                )
                pred_line = alt.Chart(df).mark_line(
                    color=C_PREDICTED, strokeWidth=2, strokeDash=[6, 3]
                ).encode(
                    x=enc_x,
                    y=alt.Y('predicted:Q', scale=alt.Scale(zero=False)),
                    tooltip=tooltip,
                )
                st.altair_chart((actual_line + pred_line).properties(height=300),
                                use_container_width=True)
                return mae

            _render_route_chart(hist_a, label_a)
            _render_route_chart(hist_b, label_b)

            if not hist_a.empty and not hist_b.empty:
                def _stats(df):
                    mae      = df['residual'].abs().mean()
                    best_yr  = int(df.loc[df['residual'].abs().idxmin(), 'Year'])
                    worst_yr = int(df.loc[df['residual'].abs().idxmax(), 'Year'])
                    return {
                        'Mean actual fare':                 f"${df['fare'].mean():.2f}",
                        'Mean predicted fare':              f"${df['predicted'].mean():.2f}",
                        'MAE':                              f"${mae:.2f}",
                        'Best year (smallest |residual|)':  str(best_yr),
                        'Worst year (largest |residual|)':  str(worst_yr),
                    }

                sa = _stats(hist_a)
                sb = _stats(hist_b)
                winner = label_a if hist_a['residual'].abs().mean() < hist_b['residual'].abs().mean() else label_b

                summary = pd.DataFrame({
                    label_a: list(sa.values()),
                    label_b: list(sb.values()),
                    'Lower MAE': ['', '', f"📈 {winner}", '', ''],
                }, index=list(sa.keys()))
                st.markdown("**Summary**")
                st.table(summary)

        st.markdown("---")
        st.page_link("streamlit_app.py", label="Open Estimator ✈️",
                     help="Re-enter the route manually in the Estimator.")


# ============================================================
# TAB 3 — CARRIER RESPONSE CURVES
# ============================================================
with tab3:
    st.subheader("Carrier response curves")
    st.caption(
        "What the model learned about each carrier's pricing — "
        "all other features held fixed, one input varied at a time."
    )

    t3_opts    = [f"{CARRIER_NAMES.get(c, c)} ({c})" for c in art['carriers']]
    t3_default = next((i for i, c in enumerate(t3_opts) if 'United' in c), 0)
    t3_label   = st.selectbox("Carrier", t3_opts, index=t3_default, key="t3_carrier")
    t3_code    = t3_label.split('(')[-1].replace(')', '').strip()

    if t3_code in art['legacy_set']:
        tier, tier_color = "Legacy", "#0c4a6e"
    elif t3_code in art['lcc_set']:
        tier, tier_color = "Low-Cost (LCC)", "#059669"
    elif t3_code in art['ulcc_set']:
        tier, tier_color = "Ultra Low-Cost (ULCC)", "#d97706"
    else:
        tier, tier_color = "Regional / Other", "#6b7280"

    st.markdown(
        f"<div style='margin-bottom:0.6rem;'>"
        f"<span style='background:{tier_color}; color:white; padding:0.25rem 0.75rem; "
        f"border-radius:999px; font-size:0.75rem; font-weight:600; letter-spacing:0.05em; "
        f"text-transform:uppercase;'>{tier}</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Each chart holds all inputs constant except one, showing how the model's "
               "fare prediction responds to that single variable.")

    ref_code = _get_reference_carrier(art)
    ref_name  = CARRIER_NAMES.get(ref_code, ref_code)
    sel_name  = CARRIER_NAMES.get(t3_code, t3_code)

    # Fixed cities for response curves — keep city-frequency contributions constant
    rc1, rc2 = None, None
    for c in ["Chicago, IL", "New York City, NY (Metropolitan Area)"]:
        if c in art['cities']:
            if rc1 is None:
                rc1 = c
            elif rc2 is None:
                rc2 = c
    if rc1 is None:
        rc1 = art['cities'][0]
    if rc2 is None:
        rc2 = art['cities'][1] if len(art['cities']) > 1 else art['cities'][0]

    if t3_code not in art['carriers']:
        st.error(f"Carrier '{t3_code}' is not in the training data.")
    else:
        def _response_chart(x_vals, x_col, x_title, p_sel, p_ref,
                            x_fmt=',.0f', x_axis_fmt=None):
            n = len(x_vals)
            sel_df = pd.DataFrame({'x': x_vals, 'predicted_fare': p_sel})
            ref_df = pd.DataFrame({'x': x_vals, 'predicted_fare': p_ref})

            x_enc = alt.X('x:Q', title=x_title,
                           axis=alt.Axis(format=x_axis_fmt) if x_axis_fmt else alt.Axis())
            y_enc = alt.Y('predicted_fare:Q', title='Predicted Fare ($)',
                          scale=alt.Scale(zero=False))
            tt = [
                alt.Tooltip('x:Q', title=x_title, format=x_fmt),
                alt.Tooltip('predicted_fare:Q', title='Predicted fare', format='$,.2f'),
            ]
            line_sel = (alt.Chart(sel_df).mark_line(color=C_PREDICTED, strokeWidth=2.5)
                        .encode(x=x_enc, y=y_enc, tooltip=tt))
            line_ref = (alt.Chart(ref_df).mark_line(color=C_GRAY, strokeWidth=2,
                                                     strokeDash=[6, 3])
                        .encode(x=x_enc, y=y_enc, tooltip=tt))
            return (line_sel + line_ref).properties(height=350)

        sub1, sub2, sub3 = st.tabs(["vs Distance", "vs Year", "vs Market Share"])

        with sub1:
            distances = np.linspace(100, 3000, 50)
            df_sel = pd.DataFrame(dict(city1=rc1, city2=rc2, carrier_lg=t3_code,
                                       Year=2024, quarter=2, nsmiles=distances,
                                       passengers=500, large_ms=0.5))
            df_ref = df_sel.copy(); df_ref['carrier_lg'] = ref_code

            with st.spinner("Computing curves..."):
                p_sel = batch_predict(art, f"resp_dist_{t3_code}", df_sel)
                p_ref = batch_predict(art, f"resp_dist_{ref_code}", df_ref)

            st.altair_chart(
                _response_chart(distances, 'x', 'Distance (mi)', p_sel, p_ref),
                use_container_width=True)

            mid  = int(np.argmin(np.abs(distances - 1500)))
            diff = p_sel[mid] - p_ref[mid]
            if abs(diff) < 5:
                st.markdown(f"**Model predicts that {sel_name} prices similarly to {ref_name}.**")
            else:
                st.markdown(
                    f"**Model predicts that {sel_name} charges \${abs(diff):.0f} "
                    f"{'more' if diff > 0 else 'less'} than {ref_name} "
                    f"at 1,500-mile distance.**")

        with sub2:
            years = np.arange(2010, 2026, dtype=float)
            df_sel = pd.DataFrame(dict(city1=rc1, city2=rc2, carrier_lg=t3_code,
                                       Year=years, quarter=2, nsmiles=1000,
                                       passengers=500, large_ms=0.5))
            df_ref = df_sel.copy(); df_ref['carrier_lg'] = ref_code

            with st.spinner("Computing curves..."):
                p_sel = batch_predict(art, f"resp_year_{t3_code}", df_sel)
                p_ref = batch_predict(art, f"resp_year_{ref_code}", df_ref)

            st.altair_chart(
                _response_chart(years, 'x', 'Year', p_sel, p_ref,
                                x_fmt='d', x_axis_fmt='d'),
                use_container_width=True)

            idx = np.where(years == 2024)[0]
            if len(idx):
                diff = p_sel[idx[0]] - p_ref[idx[0]]
                if abs(diff) < 5:
                    st.markdown(f"**Model predicts that {sel_name} prices similarly to {ref_name} in 2024.**")
                else:
                    st.markdown(
                        f"**Model predicts that {sel_name} charges \${abs(diff):.0f} "
                        f"{'more' if diff > 0 else 'less'} than {ref_name} in 2024.**")

        with sub3:
            ms_vals = np.linspace(0.1, 1.0, 50)
            df_sel = pd.DataFrame(dict(city1=rc1, city2=rc2, carrier_lg=t3_code,
                                       Year=2024, quarter=2, nsmiles=1000,
                                       passengers=500, large_ms=ms_vals))
            df_ref = df_sel.copy(); df_ref['carrier_lg'] = ref_code

            with st.spinner("Computing curves..."):
                p_sel = batch_predict(art, f"resp_ms_{t3_code}", df_sel)
                p_ref = batch_predict(art, f"resp_ms_{ref_code}", df_ref)

            st.altair_chart(
                _response_chart(ms_vals, 'x', 'Market Share', p_sel, p_ref,
                                x_fmt='.0%', x_axis_fmt='%'),
                use_container_width=True)

            mid  = int(np.argmin(np.abs(ms_vals - 0.5)))
            diff = p_sel[mid] - p_ref[mid]
            if abs(diff) < 5:
                st.markdown(f"**Model predicts that {sel_name} prices similarly to {ref_name}.**")
            else:
                st.markdown(
                    f"**Model predicts that {sel_name} charges \${abs(diff):.0f} "
                    f"{'more' if diff > 0 else 'less'} than {ref_name} "
                    f"at 50% market share.**")


# ============================================================
# TAB 4 — SEASONALITY: ACTUAL vs LEARNED
# ============================================================
with tab4:
    st.subheader("Seasonality: actual vs model-learned patterns")
    st.caption(
        "Did the model capture seasonal fare patterns? "
        "Left = observed, Right = predicted, Bottom = residual (actual − predicted)."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s1 = st.selectbox(
            "Origin", art['cities'], key="t4_orig",
            index=art['cities'].index("Chicago, IL") if "Chicago, IL" in art['cities'] else 0)
    with col_s2:
        s2 = st.selectbox(
            "Destination", art['cities'], key="t4_dest",
            index=art['cities'].index("New York City, NY (Metropolitan Area)")
            if "New York City, NY (Metropolitan Area)" in art['cities'] else 1)

    if s1 == s2:
        st.warning("Origin and destination must differ.")
    else:
        with st.spinner("Computing model predictions..."):
            seasonal_df = get_route_with_predictions(art, s1, s2)

        if seasonal_df.empty:
            st.info(
                f"No historical observations for {s1} → {s2}. "
                "Try a more popular route or ensure route_history.csv is present."
            )
        else:
            if len(seasonal_df) < 8:
                st.info("Route has limited history; model accuracy estimates may be unreliable.")

            has_preds = ('predicted' in seasonal_df.columns and
                         not seasonal_df['predicted'].isna().all())

            if has_preds:
                col_left, col_right = st.columns(2)
                with col_left:
                    st.altair_chart(
                        _heatmap_with_text(seasonal_df, 'fare', 'Actual Fare'),
                        use_container_width=True)
                with col_right:
                    st.altair_chart(
                        _heatmap_with_text(seasonal_df, 'predicted', 'Model Prediction'),
                        use_container_width=True)

                st.altair_chart(
                    _heatmap_with_text(
                        seasonal_df, 'residual', 'Residual (Actual − Predicted)',
                        color_scheme='redblue', reverse=False, domain_mid=0, fmt='+$.0f'),
                    use_container_width=True)
            else:
                st.altair_chart(
                    _heatmap_with_text(seasonal_df, 'fare', 'Actual Fare'),
                    use_container_width=True)

            # Auto-generated insights
            q_actual   = seasonal_df.groupby('quarter')['fare'].mean()
            cheapest_q = int(q_actual.idxmin())
            priciest_q = int(q_actual.idxmax())

            insights = [
                f"The **actual** data shows Q{priciest_q} as the most expensive quarter "
                f"(\\${q_actual[priciest_q]:.0f} avg) and Q{cheapest_q} as cheapest "
                f"(\\${q_actual[cheapest_q]:.0f} avg)."
            ]

            if has_preds:
                q_pred         = seasonal_df.groupby('quarter')['predicted'].mean()
                priciest_q_pred = int(q_pred.idxmax())
                if priciest_q_pred == priciest_q:
                    insights.append(
                        f"The **model** successfully captures this — "
                        f"it also peaks at Q{priciest_q_pred}.")
                else:
                    insights.append(
                        f"The **model** peaks at Q{priciest_q_pred} rather than "
                        f"Q{priciest_q} — a partial mismatch in the learned seasonal pattern.")

                recent = seasonal_df[seasonal_df['Year'] >= 2023]
                if not recent.empty:
                    q3_resid = recent[recent['quarter'] == 3]['residual']
                    if len(q3_resid) > 0 and q3_resid.mean() > 10:
                        insights.append(
                            f"The model tends to under-predict Q3 fares in 2023–present "
                            f"(avg residual: \\${q3_resid.mean():.0f}) — a known limitation "
                            "discussed in §10.5.")

            for line in insights:
                st.markdown(f"- {line}")

        st.markdown("---")
        st.page_link("streamlit_app.py", label="Open Estimator ✈️",
                     help="Re-enter the route manually in the Estimator.")


st.markdown("---")
st.caption("Data: DOT Consumer Airfare Report · EIA · BLS · CS 451 Final Project")
