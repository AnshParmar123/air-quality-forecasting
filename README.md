# Air Quality Forecasting

Machine learning project for analyzing and forecasting air quality using historical pollutant measurements and time-based feature engineering.

This repository includes both a notebook workflow and a standalone Python script for exploring the dataset, building features, training tree-based regressors, and saving trained model artifacts.

## Project Scope

- Forecasts the next-hour target from historical air-quality observations.
- Compares `RandomForest`, `LightGBM`, and `XGBoost`.
- Includes saved model artifacts in [`models/`](/Users/anshparmar/Desktop/PROJECTS/AQI/models).
- Uses time-based train/test splitting instead of a random split.

## Important Note

The provided dataset [`delhi_aqi.csv`](/Users/anshparmar/Desktop/PROJECTS/AQI/delhi_aqi.csv) does not include an explicit `AQI` column. The notebook therefore falls back to `pm2_5` as the target proxy when AQI is unavailable. If you later replace the dataset with one that includes an AQI field, the notebook is already written to use it automatically.

## Dataset

Current dataset columns:

- `date`
- `co`
- `no`
- `no2`
- `o3`
- `so2`
- `pm2_5`
- `pm10`
- `nh3`

## Repository Structure

- [`aqi.ipynb`](/Users/anshparmar/Desktop/PROJECTS/AQI/aqi.ipynb): end-to-end analysis, feature engineering, training, evaluation, and model export
- [`air_quality_forecasting.py`](/Users/anshparmar/Desktop/PROJECTS/AQI/air_quality_forecasting.py): standalone training script for terminal-based runs
- [`delhi_aqi.csv`](/Users/anshparmar/Desktop/PROJECTS/AQI/delhi_aqi.csv): source dataset used by the notebook
- [`models/lightgbm_best.joblib`](/Users/anshparmar/Desktop/PROJECTS/AQI/models/lightgbm_best.joblib): saved LightGBM model
- [`models/xgb_model.joblib`](/Users/anshparmar/Desktop/PROJECTS/AQI/models/xgb_model.joblib): saved XGBoost model
- [`models/model_metadata.joblib`](/Users/anshparmar/Desktop/PROJECTS/AQI/models/model_metadata.joblib): saved feature metadata
- [`requirements.txt`](/Users/anshparmar/Desktop/PROJECTS/AQI/requirements.txt): Python dependencies

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run the standalone Python script:

```bash
python air_quality_forecasting.py
```

Or open the notebook locally:

```bash
jupyter notebook aqi.ipynb
```

## Modeling Workflow

1. Load and clean the source air-quality dataset.
2. Detect the datetime column and normalize timestamps to hourly values.
3. Fill missing numeric values with interpolation plus forward/backward fill.
4. Build time, seasonal, lag, rolling, and ratio features.
5. Split the dataset chronologically into training and test sets.
6. Train and compare `RandomForest`, `LightGBM`, and `XGBoost`.
7. Save the best-performing model and feature metadata.

## Current Limitations

- The project is script- and notebook-based rather than packaged as a Python module.
- Reproducing exact metrics may depend on local library versions.
- The current dataset supports pollutant forecasting more directly than official AQI forecasting.
