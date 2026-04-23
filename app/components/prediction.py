"""Prediction and data helpers used across pages."""
import streamlit as st
import pandas as pd
import numpy as np
from components.artifacts import (
    CARRIER_NAMES,
    QUARTER_LABELS,
    MODEL_METRICS,
    load_route_history_csv,
)


def carrier_tier_fn(code: str, art) -> int:
    if code in art['legacy_set']: return 0
    if code in art['lcc_set']:    return 1
    if code in art['ulcc_set']:   return 2
    return 3


def lookup_both_directions(d, c1, c2):
    """Route lookups are direction-agnostic. Try (c1,c2) then (c2,c1)."""
    return d.get((c1, c2)) or d.get((c2, c1))


def _parse_carrier_code(carrier_label: str) -> str:
    """'United Airlines (UA)' -> 'UA'."""
    return carrier_label.split('(')[-1].replace(')', '').strip()


def predict_fare(art, city1: str, city2: str, carrier_label: str,
                 year: int, quarter_label: str) -> dict:
    """Build the feature row and run the model. Returns a dict with the
    prediction, route metadata, and macro context."""
    carrier_code = _parse_carrier_code(carrier_label)
    quarter = QUARTER_LABELS[quarter_label]

    # Route-level lookups
    distance   = lookup_both_directions(art['distance_lookup'],   city1, city2)
    passengers = lookup_both_directions(art['passengers_lookup'], city1, city2)
    large_ms   = lookup_both_directions(art['ms_lookup'],         city1, city2)

    route_known = distance is not None
    if not route_known:
        distance, passengers, large_ms = 800, 500, 0.5

    distance   = int(round(distance))
    passengers = int(round(passengers))

    try:
        carrier_enc = art['carrier_encoder'].transform([carrier_code])[0]
    except ValueError:
        return {'error': f"Carrier {carrier_code} not in training data."}

    # Macro (fuel + CPI) lookup for this quarter
    macro = art['macro_lookup'].get((int(year), int(quarter)),
                                     art.get('macro_defaults', {}))

    # Build row matching FEATURE_COLS order exactly
    row = pd.DataFrame([{
        'Year': year, 'quarter': quarter,
        'nsmiles': distance, 'passengers': passengers, 'large_ms': large_ms,
        'log_distance':    np.log1p(distance),
        'log_passengers':  np.log1p(passengers),
        'is_long_haul':    int(distance > 1500),
        'distance_bucket': int(pd.cut([distance], bins=[0, 500, 1000, 1500, np.inf],
                                      labels=[0, 1, 2, 3])[0]),
        'quarter_sin':     np.sin(2 * np.pi * quarter / 4),
        'quarter_cos':     np.cos(2 * np.pi * quarter / 4),
        'is_covid_era':    int(year in (2020, 2021)),
        'carrier_tier':    carrier_tier_fn(carrier_code, art),
        'carrier_lg_enc':  carrier_enc,
        'city1_freq':      art['city1_freq'].get(city1, 0.001),
        'city2_freq':      art['city2_freq'].get(city2, 0.001),
        'route_popularity': art['route_pop_map'].get((city1, city2),
                             art['route_pop_map'].get((city2, city1),
                             art['global_mean_pop'])),
        'jet_fuel_usd_per_gal': macro.get('jet_fuel_usd_per_gal', 2.5),
        'jet_fuel_lag_1q':      macro.get('jet_fuel_lag_1q', 2.5),
        'jet_fuel_yoy_change':  macro.get('jet_fuel_yoy_change', 0.0),
        'cpi':                  macro.get('cpi', 300.0),
        'cpi_yoy_change':       macro.get('cpi_yoy_change', 3.0),
    }])[art['feature_cols']]

    pred = float(art['model'].predict(row)[0])
    resid_std = MODEL_METRICS['residual_std']

    return {
        'estimate': pred,
        'residual_std': resid_std,
        'distance': distance,
        'passengers': passengers,
        'large_ms': large_ms,
        'route_known': route_known,
        'carrier_code': carrier_code,
        'carrier_name': CARRIER_NAMES.get(carrier_code, carrier_code),
        'year': year,
        'quarter': quarter,
        'macro': {
            'jet_fuel_usd_per_gal': macro.get('jet_fuel_usd_per_gal', 2.5),
            'jet_fuel_lag_1q':      macro.get('jet_fuel_lag_1q', 2.5),
            'jet_fuel_yoy_change':  macro.get('jet_fuel_yoy_change', 0.0),
            'cpi':                  macro.get('cpi', 300.0),
            'cpi_yoy_change':       macro.get('cpi_yoy_change', 3.0),
        },
    }


@st.cache_data
def get_route_history(_art, city1: str, city2: str) -> pd.DataFrame:
    """Return historical fare observations for this route from route_history.csv.
    Tries both directions (c1,c2) and (c2,c1)."""
    df = load_route_history_csv()
    if df.empty:
        return df
    mask = ((df['city1'] == city1) & (df['city2'] == city2)) | \
           ((df['city1'] == city2) & (df['city2'] == city1))
    return df[mask].sort_values(['Year', 'quarter']).reset_index(drop=True)


@st.cache_data
def get_top_routes(_art, limit: int = 100) -> pd.DataFrame:
    """Return the top-N routes by observation count (a proxy for passenger volume,
    using the route_pop_map which is proportional to training-set popularity)."""
    pop = _art['route_pop_map']
    rows = []
    for (c1, c2), v in pop.items():
        dist = _art['distance_lookup'].get((c1, c2)) or \
               _art['distance_lookup'].get((c2, c1))
        pax = _art['passengers_lookup'].get((c1, c2)) or \
              _art['passengers_lookup'].get((c2, c1))
        if dist is None or pax is None:
            continue
        rows.append({
            'city1': c1, 'city2': c2,
            'distance': dist, 'passengers': pax,
            'popularity': v,
        })
    top = (pd.DataFrame(rows)
             .sort_values('passengers', ascending=False)
             .head(limit)
             .reset_index(drop=True))
    top['route'] = top['city1'] + ' → ' + top['city2']
    return top


@st.cache_data
def get_carrier_stats(_art, carrier_code: str) -> dict:
    """Aggregate route history for a given carrier (from route_history.csv
    enriched implicitly) — we approximate using the route_history file if present."""
    history = load_route_history_csv()
    if history.empty:
        return {}

    # We don't have carrier per row in the simple export; so these stats are
    # route-level averages. This is a known limitation — note it in the UI.
    return {
        'routes_served': len(history[['city1', 'city2']].drop_duplicates()),
    }


# ── New helpers for Model Playground ─────────────────────────────────────────

def _get_reference_carrier(art) -> str:
    """Most common legacy carrier available in the artifact (used as baseline)."""
    for code in ['AA', 'DL', 'UA']:
        if code in art['carriers']:
            return code
    return art['carriers'][0]


@st.cache_data
def batch_predict(_art, df_hash_key: str, df: pd.DataFrame) -> np.ndarray:
    """Predict fares for a DataFrame of route rows.

    Required columns: city1, city2, carrier_lg, Year, quarter, nsmiles, passengers, large_ms.
    df_hash_key is a stable string that uniquely identifies the df content for caching.
    """
    rows = df.copy()
    rows['log_distance']    = np.log1p(rows['nsmiles'])
    rows['log_passengers']  = np.log1p(rows['passengers'])
    rows['is_long_haul']    = (rows['nsmiles'] > 1500).astype(int)
    rows['distance_bucket'] = pd.cut(
        rows['nsmiles'], bins=[0, 500, 1000, 1500, np.inf],
        labels=[0, 1, 2, 3]).astype(int)
    rows['quarter_sin'] = np.sin(2 * np.pi * rows['quarter'] / 4)
    rows['quarter_cos'] = np.cos(2 * np.pi * rows['quarter'] / 4)
    rows['is_covid_era'] = rows['Year'].isin([2020, 2021]).astype(int)
    rows['carrier_tier'] = rows['carrier_lg'].apply(lambda c: carrier_tier_fn(c, _art))

    known    = set(_art['carrier_encoder'].classes_)
    fallback = 'AA' if 'AA' in known else _art['carrier_encoder'].classes_[0]
    rows['carrier_lg_enc'] = _art['carrier_encoder'].transform(
        rows['carrier_lg'].apply(lambda c: c if c in known else fallback))

    rows['city1_freq'] = rows['city1'].map(_art['city1_freq']).fillna(0.001)
    rows['city2_freq'] = rows['city2'].map(_art['city2_freq']).fillna(0.001)
    rows['route_popularity'] = rows.apply(
        lambda r: _art['route_pop_map'].get(
            (r['city1'], r['city2']),
            _art['route_pop_map'].get((r['city2'], r['city1']),
                                      _art['global_mean_pop'])),
        axis=1)

    defaults = _art.get('macro_defaults', {
        'jet_fuel_usd_per_gal': 2.5, 'jet_fuel_lag_1q': 2.5,
        'jet_fuel_yoy_change': 0.0,  'cpi': 300.0, 'cpi_yoy_change': 3.0,
    })
    for col in ['jet_fuel_usd_per_gal', 'jet_fuel_lag_1q', 'jet_fuel_yoy_change',
                'cpi', 'cpi_yoy_change']:
        rows[col] = rows.apply(
            lambda r, c=col: _art['macro_lookup'].get(
                (int(r['Year']), int(r['quarter'])), defaults).get(c, defaults[c]),
            axis=1)

    return _art['model'].predict(rows[_art['feature_cols']])


@st.cache_data
def get_route_with_predictions(_art, city1: str, city2: str,
                                carrier: str = None) -> pd.DataFrame:
    """Historical fare data for a route with added `predicted` and `residual` columns.

    If carrier is None, uses the most common carrier in history (or the reference
    legacy carrier if carrier_lg column is absent from route_history.csv).
    """
    hist = get_route_history(_art, city1, city2)
    if hist.empty:
        return hist

    if carrier is None:
        if 'carrier_lg' in hist.columns and not hist['carrier_lg'].isna().all():
            carrier = hist['carrier_lg'].mode().iloc[0]
        else:
            carrier = _get_reference_carrier(_art)

    dist = lookup_both_directions(_art['distance_lookup'], city1, city2) or 800
    pax  = lookup_both_directions(_art['passengers_lookup'], city1, city2) or 500
    ms   = lookup_both_directions(_art['ms_lookup'],         city1, city2) or 0.5

    pred_df = hist[['Year', 'quarter']].copy()
    pred_df['city1']      = city1
    pred_df['city2']      = city2
    pred_df['carrier_lg'] = carrier
    pred_df['nsmiles']    = dist
    pred_df['passengers'] = pax
    pred_df['large_ms']   = ms

    try:
        preds      = batch_predict(_art, f"route_preds_{city1}_{city2}_{carrier}", pred_df)
        out        = hist.copy()
        out['predicted'] = preds
        out['residual']  = out['fare'] - out['predicted']
    except Exception:
        out = hist.copy()
        out['predicted'] = np.nan
        out['residual']  = np.nan

    return out


@st.cache_data
def get_top_routes_with_predictions(_art, limit: int = 100) -> pd.DataFrame:
    """Top routes by passenger volume with actual_fare, predicted_fare, and residual columns.

    predicted_fare uses 2025 Q2 and the reference legacy carrier as a consistent baseline.
    actual_fare is the mean observed fare across all historical quarters.
    """
    top = get_top_routes(_art, limit=limit).copy()

    history_df = load_route_history_csv()
    if not history_df.empty:
        route_fares = (history_df.groupby(['city1', 'city2'])['fare']
                       .mean().reset_index()
                       .rename(columns={'fare': 'actual_fare'}))
        top = top.merge(route_fares, on=['city1', 'city2'], how='left')
        missing = top['actual_fare'].isna()
        if missing.any():
            rev = (top.loc[missing, ['city1', 'city2']]
                      .rename(columns={'city1': 'city2', 'city2': 'city1'}))
            rev_fares = rev.merge(route_fares, on=['city1', 'city2'], how='left')
            top.loc[missing, 'actual_fare'] = rev_fares['actual_fare'].values
    else:
        top['actual_fare'] = np.nan

    ref     = _get_reference_carrier(_art)
    pred_df = top[['city1', 'city2', 'passengers']].copy()
    pred_df['carrier_lg'] = ref
    pred_df['Year']       = 2025
    pred_df['quarter']    = 2
    pred_df['nsmiles']    = top['distance'].values
    pred_df['large_ms']   = top.apply(
        lambda r: lookup_both_directions(_art['ms_lookup'], r['city1'], r['city2']) or 0.5,
        axis=1).values

    try:
        top['predicted_fare'] = batch_predict(_art, f"top_routes_{limit}_preds", pred_df)
    except Exception:
        top['predicted_fare'] = np.nan

    top['residual'] = top['actual_fare'] - top['predicted_fare']
    return top


def get_feature_contributions(_art, city1: str, city2: str, carrier: str,
                               year: int, quarter: int) -> list:
    """Top-5 feature contributions to a single prediction via lightweight counterfactual.

    For each key input, swaps it to a "dataset average" value and measures the change in
    prediction. Returns a ranked list of dicts with feature, contribution, abs_contribution.
    """
    dist = lookup_both_directions(_art['distance_lookup'], city1, city2) or 800
    pax  = lookup_both_directions(_art['passengers_lookup'], city1, city2) or 500
    ms   = lookup_both_directions(_art['ms_lookup'], city1, city2) or 0.5
    ref  = _get_reference_carrier(_art)

    base_row = {
        'city1': city1, 'city2': city2, 'carrier_lg': carrier,
        'nsmiles': dist, 'passengers': pax, 'large_ms': ms,
        'Year': year, 'quarter': quarter,
    }

    def _pred(key_suffix, **overrides):
        row = {**base_row, **overrides}
        return float(batch_predict(_art, f"contrib_{key_suffix}", pd.DataFrame([row]))[0])

    try:
        base = _pred('base')
    except Exception:
        return []

    sweeps = [
        ('Distance',       f'd{dist}y{year}q{quarter}',   {'nsmiles': 1000}),
        ('Market share',   f'ms{ms}y{year}q{quarter}',    {'large_ms': 0.5}),
        ('Year trend',     f'y{year}d{dist}',             {'Year': 2017}),
        ('Seasonality',    f'q{quarter}y{year}d{dist}',   {'quarter': 2}),
        ('Carrier choice', f'c{carrier}y{year}q{quarter}', {'carrier_lg': ref}),
    ]

    results = []
    for name, key_sfx, override in sweeps:
        try:
            mean_pred = _pred(key_sfx, **override)
            contrib   = base - mean_pred
            results.append({'feature': name, 'contribution': contrib,
                             'abs_contribution': abs(contrib)})
        except Exception:
            pass

    results.sort(key=lambda x: x['abs_contribution'], reverse=True)
    return results[:5]
