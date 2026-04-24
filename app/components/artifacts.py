"""Load the pickled model artifact from the notebook output.

The artifact bundle (us_flight_fare_artifacts.pkl) is produced by notebook §12
and contains the model, feature list, encoders, and all lookup dicts the app
needs to build feature rows and display context.
"""
import streamlit as st
import pickle
from pathlib import Path

# Carrier IATA code -> human-readable name
CARRIER_NAMES = {
    'AA': 'American Airlines', 'DL': 'Delta Air Lines', 'UA': 'United Airlines',
    'WN': 'Southwest Airlines', 'B6': 'JetBlue Airways', 'AS': 'Alaska Airlines',
    'NK': 'Spirit Airlines', 'F9': 'Frontier Airlines', 'G4': 'Allegiant Air',
    'HA': 'Hawaiian Airlines', 'SY': 'Sun Country Airlines', 'MX': 'Breeze Airways',
    'MQ': 'American Eagle', 'OO': 'SkyWest Airlines', 'YX': 'Republic Airways',
    'QX': 'Horizon Air', '9E': 'Endeavor Air', 'OH': 'PSA Airlines',
    'YV': 'Mesa Airlines', 'PT': 'Piedmont Airlines', 'ZW': 'Air Wisconsin',
    'G7': 'GoJet Airlines', 'C5': 'CommutAir',
}

QUARTER_LABELS = {
    'Q1 (Jan–Mar)': 1,
    'Q2 (Apr–Jun)': 2,
    'Q3 (Jul–Sep)': 3,
    'Q4 (Oct–Dec)': 4,
}

# TODO: Replace these with real values from notebook §10 and §10.5 after full run.
# Random Forest (tuned) — test set performance
MODEL_METRICS = {
    'model_type': 'RandomForestRegressor',
    'random_split': {
        'r2':   0.90,
        'mae':  13.50,
        'rmse': 18.20,
        'mape': 6.4,
    },
    'temporal_holdout': {
        'r2':   0.65,
        'mae':  35.00,
        'rmse': 50.00,
        'mape': 13.0,
    },
    'residual_std': 25.0,
    'training_samples': 39847,
    'n_features': 18,
}


@st.cache_resource
def load_artifacts(path: str = "data/us_flight_fare_artifacts_RF.pkl"):
    """Load the pickle once per session. Cached across reruns."""
    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent.parent / path
    if not p.exists():
        # Try pipeline/artifacts relative to project root
        p = Path(__file__).parent.parent.parent / "pipeline" / "artifacts" / "us_flight_fare_artifacts_RF.pkl"
    if not p.exists():
        st.error(
            f"❌ Could not find artifact file.\n\n"
            f"Expected at `pipeline/artifacts/us_flight_fare_artifacts.pkl` "
            f"relative to the project root. Generate it by running the notebook (§12)."
        )
        st.stop()

    with open(p, 'rb') as f:
        art = pickle.load(f)

    # Basic sanity checks
    required_keys = {'model', 'feature_cols', 'carrier_encoder',
                     'city1_freq', 'city2_freq', 'route_pop_map',
                     'global_mean_pop', 'distance_lookup', 'passengers_lookup',
                     'ms_lookup', 'cities', 'carriers',
                     'legacy_set', 'lcc_set', 'ulcc_set'}
    missing = required_keys - set(art.keys())
    if missing:
        st.error(f"Artifact file is missing keys: {missing}. "
                 f"Re-run notebook §12 with the latest code.")
        st.stop()

    # macro_lookup is new — warn but don't fail if missing (older artifacts)
    if 'macro_lookup' not in art:
        st.warning(
            "⚠️ This artifact was built before the external-data patch. "
            "Fuel/CPI features will use defaults. Re-run notebook §12 to update."
        )
        art['macro_lookup'] = {}
        art['macro_defaults'] = {
            'jet_fuel_usd_per_gal': 2.5,
            'jet_fuel_lag_1q': 2.5,
            'jet_fuel_yoy_change': 0.0,
            'cpi': 300.0,
            'cpi_yoy_change': 3.0,
        }

    if len(art.get('feature_cols', [])) == 22:
        st.warning(
            "⚠️ This artifact was built with 22 features (old pipeline). "
            "The current app expects 18 features. Re-run notebook §12 to "
            "regenerate the artifact with the updated feature set."
        )

    return art


@st.cache_data
def load_route_history_csv(path: str = "data/route_history.csv"):
    """Optional: load per-route historical fares if the user exported them
    from the notebook. Returns an empty DataFrame if absent.
    """
    import pandas as pd
    p = Path(path)
    if not p.exists():
        p = Path(__file__).parent.parent / path
    if not p.exists():
        return pd.DataFrame(columns=['city1', 'city2', 'Year', 'quarter', 'fare'])
    return pd.read_csv(p)
