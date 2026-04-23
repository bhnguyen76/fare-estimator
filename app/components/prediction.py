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
