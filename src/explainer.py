"""
explainer.py – SHAP-based model explainability.
Handles both VotingClassifier (uses XGBoost sub-model) and plain XGBClassifier.
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List


def _get_xgb_from_model(model):
    """Extract the XGBoost estimator from a VotingClassifier or return as-is."""
    if hasattr(model, "estimators_"):
        for item in model.estimators_:
            est = item[1] if isinstance(item, (list, tuple)) else item
            if hasattr(est, "feature_importances_"):
                return est
        item = model.estimators_[0]
        return item[1] if isinstance(item, (list, tuple)) else item
    return model


def get_shap_values(model, X: np.ndarray) -> Tuple[np.ndarray, shap.TreeExplainer]:
    """
    Compute SHAP values using TreeExplainer on the XGBoost sub-model.

    Returns
    -------
    shap_values : np.ndarray, shape (n_samples, n_features)
    explainer   : shap.TreeExplainer
    """
    xgb_model = _get_xgb_from_model(model)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    # For binary classifiers, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, explainer


def shap_summary_plot(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 13,
) -> go.Figure:
    """
    Beeswarm-style SHAP summary as a Plotly scatter plot.
    Each point is one test sample; x = SHAP value, y = feature (jittered).
    """
    n_features = min(max_display, len(feature_names))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:n_features]

    rows = []
    rng = np.random.default_rng(0)
    for rank, fi in enumerate(reversed(top_idx)):
        sv = shap_values[:, fi]
        fv = X[:, fi]
        fv_norm = (fv - fv.min()) / (fv.ptp() + 1e-9)
        jitter = rng.uniform(-0.3, 0.3, len(sv))
        for s, fn, jit in zip(sv, fv_norm, jitter):
            rows.append({"feature": feature_names[fi], "shap": s, "fval_norm": fn, "rank": rank + jit})

    df = pd.DataFrame(rows)

    fig = px.scatter(
        df, x="shap", y="rank", color="fval_norm",
        color_continuous_scale=[[0, "#3b82f6"], [0.5, "#8b5cf6"], [1, "#f5b800"]],
        template="plotly_dark",
        labels={"shap": "SHAP Value (impact on prediction)", "rank": ""},
        title="Global Feature Impact (SHAP Beeswarm)",
    )

    tick_labels = [feature_names[fi] for fi in reversed(top_idx)]
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(1, n_features + 1)),
        ticktext=tick_labels,
    )
    fig.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#13161f",
        coloraxis_colorbar=dict(title="Feature<br>Value", tickvals=[0, 1], ticktext=["Low", "High"]),
        height=480,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#4a5168")
    return fig


def shap_waterfall_plot(
    shap_values: np.ndarray,
    sample_idx: int,
    feature_names: List[str],
    max_display: int = 12,
) -> go.Figure:
    """
    Waterfall chart showing how each feature pushes the prediction up/down
    for a single sample.
    """
    sv = shap_values[sample_idx]
    order = np.argsort(np.abs(sv))[::-1][:max_display]

    feats = [feature_names[i] for i in order]
    vals = sv[order]

    colors = ["#ef4444" if v > 0 else "#22c55e" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=feats,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
    ))

    fig.add_vline(x=0, line_color="#4a5168", line_width=1)
    fig.update_layout(
        title="Local Explanation – Feature Contributions for This Prediction",
        xaxis_title="SHAP Value",
        template="plotly_dark",
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#13161f",
        height=420,
        yaxis=dict(autorange="reversed"),
    )
    return fig
