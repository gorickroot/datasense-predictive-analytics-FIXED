"""
visualizations.py – Plotly chart builders for the DataSense dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List

DARK_BG   = "#0d0f14"
CARD_BG   = "#13161f"
GOLD      = "#f5b800"
BLUE      = "#3b82f6"
GREEN     = "#22c55e"
RED       = "#ef4444"
GRID_COL  = "#1e2230"
TEXT_COL  = "#8a93a8"


def _base_layout(title: str = "", height: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(color="#ffffff", size=15)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COL),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
    )


def plot_feature_importance(model, feature_names: List[str], top_n: int = 13) -> go.Figure:
    """Horizontal bar chart of XGBoost feature importances (gain)."""
    # Extract XGB from ensemble safely
    xgb = model
    if hasattr(model, "estimators_"):
        for item in model.estimators_:
            est = item[1] if isinstance(item, (list, tuple)) else item
            if hasattr(est, "feature_importances_"):
                xgb = est
                break

    scores = xgb.feature_importances_
    idx = np.argsort(scores)[::-1][:top_n]
    feats = [feature_names[i] for i in reversed(idx)]
    vals  = scores[list(reversed(idx))]

    colors = [GOLD if v == vals.max() else BLUE for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(**_base_layout("Feature Importance (Gain)", height=420))
    fig.update_xaxes(title="Importance Score")
    return fig


def plot_confusion_matrix(model, X_test, y_test) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    labels = ["No Disease", "Disease"]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        colorscale=[[0, CARD_BG], [1, GOLD]],
        showscale=False,
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=22, color="#ffffff"),
    ))
    fig.update_layout(**_base_layout("Confusion Matrix", height=340))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return fig


def plot_roc_curve(model, X_test, y_test) -> go.Figure:
    """ROC curve with AUC annotation."""
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"AUC = {roc_auc:.3f}",
        line=dict(color=GOLD, width=2.5),
        fill="tozeroy", fillcolor="rgba(245,184,0,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color=GRID_COL, dash="dash"),
    ))
    fig.update_layout(**_base_layout("ROC Curve", height=360))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return fig


def plot_prediction_distribution(model, X_test) -> go.Figure:
    """Histogram of predicted probabilities."""
    probs = model.predict_proba(X_test)[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=probs[probs < 0.5], nbinsx=25,
        name="No Disease", marker_color=GREEN, opacity=0.75,
    ))
    fig.add_trace(go.Histogram(
        x=probs[probs >= 0.5], nbinsx=25,
        name="Disease", marker_color=RED, opacity=0.75,
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color=GOLD,
                  annotation_text="Decision Boundary", annotation_font_color=GOLD)
    fig.update_layout(
        **_base_layout("Prediction Probability Distribution"),
        barmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title="Predicted Probability (Disease)")
    fig.update_yaxes(title="Count")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Full correlation heatmap for numeric features."""
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0, BLUE], [0.5, CARD_BG], [1, GOLD]],
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(**_base_layout("Feature Correlation Heatmap", height=500))
    return fig
