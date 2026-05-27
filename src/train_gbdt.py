import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train(
    data_path: Path,
    target_col: str,
    model_path: Path,
    metrics_path: Path,
    pred_path: Path,
    n_jobs: int,
) -> None:
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(X_train)

    param_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 1.0],
        "model__min_samples_leaf": [1, 3, 5],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=n_jobs,
        verbose=1,
    )

    try:
        search.fit(X_train, y_train)
    except PermissionError:
        print("Parallel training tidak diizinkan. Fallback ke n_jobs=1.")
        search.set_params(n_jobs=1)
        search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100

    metrics = {
        "best_params": search.best_params_,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_percent": mape,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_out = X_test.copy()
    pred_out[target_col] = y_test.values
    pred_out["prediction"] = y_pred
    pred_out.to_csv(pred_path, index=False)

    print("Training selesai.")
    print(f"Model disimpan di: {model_path}")
    print(f"Metrik disimpan di: {metrics_path}")
    print(f"Sample prediksi disimpan di: {pred_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model GDBT untuk data housing.")
    parser.add_argument("--data", type=Path, default=Path("data/housing.csv"))
    parser.add_argument("--target", type=str, default="median_house_value")
    parser.add_argument("--model-out", type=Path, default=Path("models/gbdt_housing.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("reports/metrics_gbdt.json"))
    parser.add_argument("--pred-out", type=Path, default=Path("reports/predictions_holdout.csv"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    train(
        data_path=args.data,
        target_col=args.target,
        model_path=args.model_out,
        metrics_path=args.metrics_out,
        pred_path=args.pred_out,
        n_jobs=args.n_jobs,
    )
