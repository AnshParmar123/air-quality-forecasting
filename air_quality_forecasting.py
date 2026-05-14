from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.style.use("classic")


DEFAULT_LGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train air-quality forecasting models from a time-series dataset."
    )
    parser.add_argument(
        "--data-path",
        default="delhi_aqi.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory where trained models and metadata will be saved.",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of rows to reserve for the time-based test split.",
    )
    return parser.parse_args()


def find_datetime_column(columns: list[str]) -> str:
    possible = [
        column
        for column in columns
        if any(token in column.lower() for token in ["date", "time", "datetime"])
    ]
    return possible[0] if possible else columns[0]


def season_from_month(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "monsoon"
    return "autumn"


def load_and_clean_data(data_path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(data_path)
    df.columns = [column.strip() for column in df.columns]

    datetime_column = find_datetime_column(df.columns.tolist())
    if datetime_column != "datetime":
        df = df.rename(columns={datetime_column: "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df.columns = [column.strip().replace(" ", "_").lower() for column in df.columns]
    df["datetime"] = df["datetime"].dt.floor("h")
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    has_aqi = any("aqi" in column for column in df.columns)
    if has_aqi:
        target_column = [column for column in df.columns if "aqi" in column][0]
    else:
        pm25_candidates = [column for column in df.columns if "pm2" in column]
        if not pm25_candidates:
            raise ValueError("No AQI or PM2.5 column found in the dataset.")
        target_column = pm25_candidates[0]

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    fill_columns = [column for column in numeric_columns if "target" not in column]
    df[fill_columns] = df[fill_columns].interpolate().ffill().bfill()

    return df, target_column


def engineer_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()
    df["target_next_hour"] = df[target_column].shift(-1)
    df["target_next_24h_avg"] = (
        df[target_column].shift(-24).rolling(window=24, min_periods=1).mean().shift(-23)
    )
    df["target"] = df["target_next_hour"]

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["season"] = df["month"].apply(season_from_month)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_crop_burning_season"] = df["month"].isin([10, 11]).astype(int)
    df["is_holiday"] = 0

    wind_deg_columns = [
        column
        for column in df.columns
        if any(token in column for token in ["wind_direction", "wind_deg", "winddirection", "wind_dir"])
    ]
    if wind_deg_columns:
        wind_column = wind_deg_columns[0]
        df[wind_column] = df[wind_column].fillna(0)
        wind_radians = np.deg2rad(df[wind_column])
        df["wind_sin"] = np.sin(wind_radians)
        df["wind_cos"] = np.cos(wind_radians)

    pollutant_columns = [
        column
        for column in df.columns
        if any(token in column for token in ["pm2", "pm10", "no2", "so2", "o3", "co"])
    ]
    pollutant_columns = [column for column in pollutant_columns if column in df.columns]

    for column in pollutant_columns:
        for lag in [1, 3, 6, 12, 24]:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
        df[f"{column}_roll_3"] = df[column].rolling(window=3, min_periods=1).mean()
        df[f"{column}_roll_24"] = df[column].rolling(window=24, min_periods=1).mean()
        df[f"{column}_diff_1"] = df[column] - df[column].shift(1)

    pm25_candidates = [column for column in pollutant_columns if "pm2" in column]
    pm10_candidates = [column for column in pollutant_columns if "pm10" in column]
    if pm25_candidates and pm10_candidates:
        df["pm25_pm10_ratio"] = df[pm25_candidates[0]] / df[pm10_candidates[0]].replace(0, np.nan)

    df = pd.get_dummies(df, columns=["season"], drop_first=True)
    return df


def prepare_model_data(
    df: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    df_model = df.dropna(subset=["target"]).copy()
    drop_columns = ["datetime", "target_next_hour", "target_next_24h_avg", "target"]
    drop_columns += [column for column in df_model.columns if df_model[column].dtype == object]

    x = df_model.drop(columns=[column for column in drop_columns if column in df_model.columns])
    y = df_model["target"].copy()
    x = x.select_dtypes(include=[np.number]).fillna(0)

    split_index = int(len(x) * (1 - test_size))
    x_train = x.iloc[:split_index]
    x_test = x.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return x_train, x_test, y_train, y_test, df_model


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def save_feature_importance_plot(
    model: lgb.LGBMRegressor, features: list[str], output_path: Path
) -> None:
    importance = (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    plt.figure(figsize=(9, 10))
    plt.barh(importance["feature"], importance["importance"], color="steelblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top 20 LightGBM Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_prediction_plot(
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    plt.figure(figsize=(16, 6))
    n_plot = min(200, len(y_test))
    plt.plot(y_test.values[:n_plot], label="Actual", linewidth=2, color="black")
    for model_name, model_predictions in predictions.items():
        plt.plot(model_predictions[:n_plot], label=model_name, linewidth=1.5, alpha=0.8)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Target")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    models_dir = Path(args.models_dir)
    plots_dir = Path(args.plots_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df, target_column = load_and_clean_data(data_path)
    df = engineer_features(df, target_column)
    x_train, x_test, y_train, y_test, df_model = prepare_model_data(df, args.test_size)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(x_train, y_train)
    rf_predictions = rf_model.predict(x_test)
    rf_metrics = evaluate_predictions(y_test, rf_predictions)

    lgb_model = lgb.LGBMRegressor(
        **DEFAULT_LGB_PARAMS,
        n_estimators=500,
        objective="regression",
        random_state=42,
        n_jobs=-1,
    )
    lgb_model.fit(x_train, y_train)
    lgb_predictions = lgb_model.predict(x_test)
    lgb_metrics = evaluate_predictions(y_test, lgb_predictions)

    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(x_train, y_train)
    xgb_predictions = xgb_model.predict(x_test)
    xgb_metrics = evaluate_predictions(y_test, xgb_predictions)

    results = pd.DataFrame(
        [
            {"model": "RandomForest", **rf_metrics},
            {"model": "LightGBM", **lgb_metrics},
            {"model": "XGBoost", **xgb_metrics},
        ]
    )

    joblib.dump(lgb_model, models_dir / "lightgbm_best.joblib")
    joblib.dump(xgb_model, models_dir / "xgb_model.joblib")
    joblib.dump(
        {
            "model": "lightgbm",
            "model_path": str(models_dir / "lightgbm_best.joblib"),
            "features": x_train.columns.tolist(),
            "target_column": target_column,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        models_dir / "model_metadata.joblib",
    )

    results.to_csv(models_dir / "model_comparison.csv", index=False)
    with open(models_dir / "metrics_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "target_column": target_column,
                "dataset_rows": int(len(df_model)),
                "results": results.to_dict(orient="records"),
            },
            file,
            indent=2,
        )

    save_feature_importance_plot(
        lgb_model,
        x_train.columns.tolist(),
        plots_dir / "lightgbm_feature_importance.png",
    )
    save_prediction_plot(
        y_test,
        {
            "LightGBM": lgb_predictions,
            "XGBoost": xgb_predictions,
            "RandomForest": rf_predictions,
        },
        plots_dir / "model_predictions.png",
    )

    print(f"Target column: {target_column}")
    print(f"Train rows: {len(x_train)}, Test rows: {len(x_test)}")
    print("\nModel comparison:")
    print(results.to_string(index=False))
    print(f"\nSaved models to: {models_dir.resolve()}")
    print(f"Saved plots to: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
