#!/usr/bin/env python
# coding: utf-8
"""
Car Price Prediction — Production Training Pipeline
===================================================
Orchestrates data loading, cleaning, feature engineering, evaluation,
and serialisation of the final Random Forest model.

Dataset : Car Features and MSRP
Target  : MSRP (Manufacturer Suggested Retail Price)
Model   : Random Forest Regressor
"""

# ──────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 2. Data Ingestion
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Read the raw CSV dataset from *filepath* and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {filepath}")
    return df


# ──────────────────────────────────────────────
# 3. Data Cleaning
# ──────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, lowercase strings, drop duplicates,
    drop market_category, and impute missing values.

    Returns a cleaned copy of the DataFrame.
    """
    df = df.copy()

    # --- Column & text normalisation ---
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.lower().str.replace(" ", "_")

    # --- Duplicates ---
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Dropped {n_dupes} duplicate rows  ->  {df.shape[0]} rows remain")

    # --- Drop high-cardinality / noisy column ---
    df.drop(columns="market_category", inplace=True)

    # --- Impute missing values ---
    df["engine_hp"]        = df["engine_hp"].fillna(df["engine_hp"].median())
    df["engine_cylinders"] = df["engine_cylinders"].fillna(df["engine_cylinders"].median())
    df["number_of_doors"]  = df["number_of_doors"].fillna(df["number_of_doors"].mode()[0])
    df["engine_fuel_type"] = df["engine_fuel_type"].fillna(df["engine_fuel_type"].mode()[0])

    print(f"Missing values after imputation: {df.isnull().sum().sum()}")
    return df


# ──────────────────────────────────────────────
# 4. Global Shuffle
# ──────────────────────────────────────────────

def shuffle_dataframe(df: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    """Reproduce the exact global shuffle used during development."""
    idx = np.arange(len(df))
    np.random.seed(seed)
    np.random.shuffle(idx)
    return df.iloc[idx].reset_index(drop=True)


# ──────────────────────────────────────────────
# 5. Feature Engineering
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features and drop unnecessary / collinear columns.

    New features:
        age             = 2017 − year (Dataset collected in 2017)
        hp_per_cylinder = engine_hp / max(engine_cylinders, 1)

    Dropped columns:
        year, popularity, highway_mpg
    """
    df = df.copy()

    df["age"] = 2017 - df["year"]
    df["hp_per_cylinder"] = df["engine_hp"] / df["engine_cylinders"].replace(0, 1)
    df.drop(columns=["year", "popularity", "highway_mpg"], inplace=True)

    print(f"Features after engineering: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────
# 6. Train / Test Split
# ──────────────────────────────────────────────

def split_data(df: pd.DataFrame,
               test_size: float = 0.20,
               random_state: int = 42):
    """Split into train / test features and targets.

    y_train and y_test are returned in raw dollars.
    """
    X = df.drop("msrp", axis=1)
    y = df["msrp"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    print(f"Train set : {X_train.shape[0]} rows")
    print(f"Test set  : {X_test.shape[0]} rows")
    print(f"Categorical: {cat_cols.tolist()}")
    print(f"Numerical:   {num_cols.tolist()}")

    return X, y, X_train, X_test, y_train, y_test, cat_cols, num_cols


# ──────────────────────────────────────────────
# 7. Model Building
# ──────────────────────────────────────────────

def build_rf_pipeline(num_cols, cat_cols) -> Pipeline:
    """Return a Pipeline containing the preprocessor and the Random Forest model."""
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])
    return pipeline


# ──────────────────────────────────────────────
# 8. Training & Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model: Pipeline,
                   X_train, X_test,
                   y_train, y_test) -> dict:
    """Train the model, make predictions on the test set, and print metrics."""
    print("Training Random Forest model on split training data for evaluation...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2   = r2_score(y_test, y_pred)

    metrics = {
        "MAE ($)":  round(mae, 2),
        "RMSE ($)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R²":       round(r2, 4),
    }
    
    print("\nModel Evaluation Metrics (Test Set):")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}\n")
    
    return metrics


# ──────────────────────────────────────────────
# 9. Model Serialisation
# ──────────────────────────────────────────────

def save_model(model: Pipeline, X, y, output_dir: str = "models",
               filename: str = "car_price_rf_model.pkl") -> str:
    """Retrain the model on the full dataset and persist it to disk."""
    print("Training final Random Forest model on the full dataset...")
    model.fit(X, y)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    joblib.dump(model, model_path)
    print(f"Model successfully saved to: {model_path}")
    return model_path


# ──────────────────────────────────────────────
# 10. Main Entrypoint
# ──────────────────────────────────────────────

def main() -> None:
    """Execute the full training pipeline end-to-end."""
    # --- Data ingestion & cleaning ---
    df = load_data(os.path.join("data", "car_price_prediction.csv"))
    df = clean_data(df)

    # --- Global shuffle (reproduces notebook ordering) ---
    df = shuffle_dataframe(df, seed=2)

    # --- Feature engineering ---
    df = engineer_features(df)

    # --- Train / test split ---
    X, y, X_train, X_test, y_train, y_test, cat_cols, num_cols = split_data(df)

    # --- Build & evaluate model ---
    pipeline = build_rf_pipeline(num_cols, cat_cols)
    evaluate_model(pipeline, X_train, X_test, y_train, y_test)

    # --- Save the final model (retrained on full data) ---
    save_model(pipeline, X, y)


if __name__ == "__main__":
    main()
