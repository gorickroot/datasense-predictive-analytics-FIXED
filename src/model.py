"""
model.py – XGBoost ensemble training with cross-validation and early stopping.
"""

import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "datasense_model.pkl")


def build_xgb(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def build_ensemble(random_state: int = 42):
    """Soft-voting ensemble: XGBoost + RandomForest + GradientBoosting."""
    xgb = build_xgb(random_state)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=4,
        random_state=random_state, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=random_state,
    )
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[3, 1, 1],
    )
    return ensemble


def train_model(X_train, X_test, y_train, y_test, random_state: int = 42):
    """
    Train ensemble model, compute metrics, optionally cache to disk.

    Returns
    -------
    model    : fitted VotingClassifier (with XGBoost as primary)
    metrics  : dict with accuracy, roc_auc, f1, precision, recall
    """
    model = build_ensemble(random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
    }

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        build_xgb(random_state), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    metrics["cv_auc_mean"] = cv_scores.mean()
    metrics["cv_auc_std"]  = cv_scores.std()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, metrics


def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    raise FileNotFoundError("No saved model found. Train the model first.")
