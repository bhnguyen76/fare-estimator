# US Fare Atlas — Streamlit App

Interactive dashboard for estimating US domestic airfares using a
RandomForest model trained on DOT + EIA + BLS data.

**This is an estimator, not a forecaster.** The model performs cross-sectional
estimation — filling in historical route-carrier-quarter fare patterns. It
cannot extrapolate to future years; see the **About the Model** page for the
full temporal-holdout analysis.

## Pages

1. **Estimator** (landing) — interactive fare estimation with confidence band,
   route context, historical fare chart.
2. **📊 Model Playground** — top routes, route comparator, carrier profile, seasonality heatmap.
3. **🔬 About the Model** — performance metrics, feature importance, forecasting
   failure analysis, methodology writeup.

## Setup

### 1. Generate artifacts from the notebook

Run the notebook (`Final_Project_USA_RF.ipynb`) end-to-end. The last deployment
cell (§12) saves `us_flight_fare_artifacts_RF.pkl`. Copy it into the app's data dir (should also work stragith from artifacts):

```bash
cp us_flight_fare_artifacts.pkl streamlit_app/data/
```

### 2. Optional — export route history

For the Explore page's comparison + seasonality tabs and the Estimator's
historical fare chart to work, add this cell to your notebook after §2.5 and run it:

```python
df[['city1', 'city2', 'Year', 'quarter', 'fare']].to_csv(
    '/content/drive/MyDrive/Colab Notebooks/Final project/route_history.csv',
    index=False
)
```

Then copy it to the app:

```bash
cp route_history.csv streamlit_app/data/
```

If this file is missing, the app runs fine but those features show a helpful
"no data" message.

### 3. Install dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 4. Run

```bash
streamlit run streamlit_app.py
```

App opens at `http://localhost:8501`.

## Updating metrics

The About page shows hardcoded metric values (R², MAE, RMSE, MAPE) that should
match the notebook's §10 and §10.5 output. After running the notebook, update:

- `components/artifacts.py` → `MODEL_METRICS` dict
- `pages/2_🔬_About_the_Model.py` → `feature_importance` DataFrame
  (copy the permutation-importance numbers from notebook §9)
- `pages/2_🔬_About_the_Model.py` → `demo_years` DataFrame
  (copy the yearly actual/predicted values from notebook §10.5 output)

## File structure

```
streamlit_app/
├── streamlit_app.py          # Main entrypoint (Estimator page)
├── pages/
│   ├── 1_📊_Explore.py
│   └── 2_🔬_About_the_Model.py
├── components/
│   ├── __init__.py
│   ├── artifacts.py          # Pickle loader + carrier name map + metrics
│   ├── prediction.py         # Feature row builder, route history
│   └── styling.py            # Shared CSS
├── data/
│   ├── us_flight_fare_artifacts.pkl   # (you provide)
│   └── route_history.csv              # (you provide, optional)
├── requirements.txt
└── README.md
```

## Credits

- Data: DOT Consumer Airfare Report, EIA Petroleum Spot Prices, BLS CPI
- Model: scikit-learn HistGradientBoostingRegressor
- Built for CS 451 Introduction to Data Science — Final Project
