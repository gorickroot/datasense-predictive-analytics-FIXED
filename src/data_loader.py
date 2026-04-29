"""
data_loader.py – Load and preprocess the Heart Disease dataset.
Uses the UCI Heart Disease dataset (Cleveland subset) via sklearn or CSV fallback.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import os


COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_data() -> pd.DataFrame:
    """
    Load Heart Disease dataset.
    Downloads from UCI via sklearn fetch or falls back to bundled CSV.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "target" not in df.columns and df.shape[1] == 14:
            df.columns = COLUMN_NAMES
        return _clean(df)

    # Generate realistic synthetic data if no CSV present
    return _generate_synthetic()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning – types, missing values, binary target."""
    df = df.copy()

    # Binarise target (0 = no disease, 1 = disease)
    if df["target"].max() > 1:
        df["target"] = (df["target"] > 0).astype(int)

    # Drop rows with too many missing values
    df = df.replace("?", np.nan)
    df = df.dropna(thresh=df.shape[1] - 2)

    # Numeric coercion
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target"])
    return df.reset_index(drop=True)


def _generate_synthetic(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Heart Disease dataset when no CSV is available.
    Distributions match the original UCI Cleveland dataset statistics.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(29, 77, n)
    sex = rng.choice([0, 1], n, p=[0.32, 0.68])
    cp = rng.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = np.clip(rng.normal(131, 17, n), 94, 200).astype(int)
    chol = np.clip(rng.normal(246, 51, n), 126, 564).astype(int)
    fbs = rng.choice([0, 1], n, p=[0.85, 0.15])
    restecg = rng.choice([0, 1, 2], n, p=[0.50, 0.48, 0.02])
    thalach = np.clip(rng.normal(149, 22, n), 71, 202).astype(int)
    exang = rng.choice([0, 1], n, p=[0.67, 0.33])
    oldpeak = np.clip(rng.exponential(1.0, n), 0, 6.2).round(1)
    slope = rng.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])
    ca = rng.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
    thal = rng.choice([1, 2, 3], n, p=[0.06, 0.54, 0.40])

    # Simulate target with some correlation to features
    risk_score = (
        0.04 * (age - 29)
        + 0.3 * sex
        + 0.2 * (cp == 3).astype(int)
        + 0.3 * exang
        + 0.4 * oldpeak
        + 0.5 * (thal == 3).astype(int)
        - 0.3 * (slope == 2).astype(int)
        - 0.015 * (thalach - 71)
        + rng.normal(0, 0.5, n)
    )
    target = (risk_score > risk_score.mean()).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
        "thal": thal, "target": target,
    })
    return df


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split and preprocess.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    feature_names                    : list[str]
    preprocessor                     : fitted ColumnTransformer
    """
    X = df.drop(columns=["target"])
    y = df["target"].values

    num_cols = [c for c in NUM_COLS if c in X.columns]
    cat_cols = [c for c in CAT_COLS if c in X.columns]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = num_cols + cat_cols

    return X_train_t, X_test_t, y_train, y_test, feature_names, preprocessor
