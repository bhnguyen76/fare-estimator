# US Fare Atlas

A Streamlit app that estimates historical US domestic airfares using a
HistGradientBoostingRegressor trained on DOT, EIA, and BLS data.
Select an origin, destination, airline, year, and quarter to get a fare
estimate with a confidence band and historical fare chart. Built for CS 451.

**Make sure your computer is in dark mode. You will not be able to see some graph lines in light mode.**
## Pages

| Page | Description |
|---|---|
| **Estimator** | Fare estimate with route context and historical chart |
| **Explore** | Top routes, route comparator, carrier profile, seasonality heatmap |
| **About the Model** | Performance metrics, feature importance, methodology |

## Quickstart

```bash
make setup   # create .venv and install dependencies (first time only)
make run     # start the app at http://localhost:8501
```

## Make Commands

| Command | Description |
|---|---|
| `make run` | Start the Streamlit app |
| `make install` | Install/update dependencies from `app/requirements.txt` |
| `make setup` | Create `.venv` from scratch and install dependencies |

## Model Artifact Setup

The trained model artifact (`us_flight_fare_artifacts.pkl`) is too large for GitHub and must be downloaded separately.

1. Download the file from Google Drive:
   [us_flight_fare_artifacts.pkl](https://drive.google.com/file/d/1QupFF9y3VRlEOfmaOC-TtXGWO_z9EE3k/view?usp=drive_link)

2. Place the file at:
   ```
   pipeline/artifacts/us_flight_fare_artifacts.pkl
   ```

3. The app will not run without this file — it must be in that exact location.

## Project Structure

```
fare-estimator/
├── app/
│   ├── streamlit_app.py       # landing page (Estimator)
│   ├── pages/
│   │   ├── 1_📊_Explore.py
│   │   └── 2_🔬_About_the_Model.py
│   ├── components/
│   │   ├── artifacts.py       # pickle loader, carrier map, metrics
│   │   ├── prediction.py      # feature builder, route history helpers
│   │   └── styling.py         # shared CSS
│   └── requirements.txt
├── pipeline/
│   ├── Final_Project_USA_v2.ipynb   # data pipeline + model training
│   └── artifacts/
│       └── us_flight_fare_artifacts.pkl
├── Makefile
└── .gitignore
```

## Data Sources

- DOT Consumer Airfare Report
- EIA Petroleum Spot Prices (jet fuel)
- BLS Consumer Price Index
