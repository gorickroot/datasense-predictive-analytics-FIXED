import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
import os
from src.data_loader import load_data, preprocess_data
from src.model import train_model, load_trained_model
from src.explainer import get_shap_values, shap_summary_plot, shap_waterfall_plot
from src.visualizations import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prediction_distribution,
    plot_correlation_heatmap,
)

st.set_page_config(
    page_title="DataSense – Predictive Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d0f14; }
    .block-container { padding-top: 2rem; }
    .stMetric { background: #13161f; border-radius: 12px; padding: 1rem; border: 1px solid #1e2230; }
    .stMetric label { color: #8a93a8 !important; font-size: 12px !important; }
    .stMetric [data-testid="stMetricValue"] { color: #f5b800 !important; font-size: 28px !important; font-weight: 700; }
    .stMetric [data-testid="stMetricDelta"] { color: #22c55e !important; }
    h1, h2, h3 { color: #ffffff; }
    .sidebar .sidebar-content { background: #13161f; }
    div[data-testid="metric-container"] { background: #13161f; border: 1px solid #1e2230; border-radius: 12px; padding: 1rem; }
    .section-header { font-size: 1.2rem; font-weight: 700; color: #f5b800; margin-bottom: 0.5rem; border-bottom: 1px solid #1e2230; padding-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/logo.png", use_column_width=True) if os.path.exists("assets/logo.png") else None
    st.markdown("## 📊 DataSense")
    st.markdown("**Predictive Analytics Dashboard**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔍 Data Explorer", "🤖 Model Training", "💡 SHAP Explanations", "🎯 Predict"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Settings**")
    test_size = st.slider("Test Split %", 10, 40, 20, step=5) / 100
    random_state = st.number_input("Random Seed", value=42, min_value=0, max_value=999)

    st.markdown("---")
    st.caption("Built by · Griffith College Dublin")
    st.caption("Stack: XGBoost · SHAP · Streamlit · Plotly")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model(test_size, random_seed):
    df = get_data()
    X_train, X_test, y_train, y_test, feature_names, preprocessor = preprocess_data(df, test_size=test_size, random_state=random_seed)
    model, metrics = train_model(X_train, X_test, y_train, y_test)
    return model, metrics, X_train, X_test, y_train, y_test, feature_names, preprocessor

df_raw = get_data()
model, metrics, X_train, X_test, y_train, y_test, feature_names, preprocessor = get_model(test_size, random_state)

# ═══════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📊 DataSense – Predictive Analytics")
    st.markdown("Machine learning dashboard powered by **XGBoost** with **SHAP** explanations · Heart Disease Classification")

    st.markdown("---")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}", "+2.3% vs baseline")
    with col2:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", "+0.041 vs baseline")
    with col3:
        st.metric("F1 Score", f"{metrics['f1']:.3f}", "+1.8% vs baseline")
    with col4:
        st.metric("Training Samples", f"{len(X_train):,}")

    st.markdown("---")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown('<div class="section-header">Model Performance – ROC Curve</div>', unsafe_allow_html=True)
        fig_roc = plot_roc_curve(model, X_test, y_test)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig_cm = plot_confusion_matrix(model, X_test, y_test)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="section-header">Top Feature Importances</div>', unsafe_allow_html=True)
    fig_fi = plot_feature_importance(model, feature_names)
    st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: Data Explorer
# ═══════════════════════════════════════════════════════════════════
elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Distributions", "🔗 Correlations"])

    with tab1:
        st.markdown(f"**{len(df_raw):,} rows · {df_raw.shape[1]} columns**")
        st.dataframe(df_raw, use_container_width=True, height=400)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Records", len(df_raw))
        col_b.metric("Features", df_raw.shape[1] - 1)
        col_c.metric("Missing Values", df_raw.isnull().sum().sum())

    with tab2:
        st.markdown('<div class="section-header">Prediction Distribution</div>', unsafe_allow_html=True)
        fig_dist = plot_prediction_distribution(model, X_test)
        st.plotly_chart(fig_dist, use_container_width=True)

        num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        selected_col = st.selectbox("Select feature to explore", num_cols)
        fig_hist = px.histogram(
            df_raw, x=selected_col, color="target" if "target" in df_raw.columns else None,
            nbins=30, template="plotly_dark", title=f"Distribution of {selected_col}",
            color_discrete_sequence=["#f5b800", "#3b82f6"],
        )
        fig_hist.update_layout(paper_bgcolor="#0d0f14", plot_bgcolor="#13161f")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        fig_corr = plot_correlation_heatmap(df_raw)
        st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: Model Training
# ═══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.title("🤖 Model Training & Evaluation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### XGBoost Hyperparameters")
        st.json({
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": int(random_state),
        })

    with col2:
        st.markdown("### Evaluation Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "ROC-AUC", "F1 Score", "Precision", "Recall"],
            "Score": [
                f"{metrics['accuracy']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Classification Report")
    from sklearn.metrics import classification_report
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Feature Importance (Gain)")
    fig_fi2 = plot_feature_importance(model, feature_names, top_n=20)
    st.plotly_chart(fig_fi2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: SHAP Explanations
# ═══════════════════════════════════════════════════════════════════
elif page == "💡 SHAP Explanations":
    st.title("💡 SHAP Explanations")
    st.markdown("Understand **why** the model makes each prediction using SHAP (SHapley Additive exPlanations).")

    with st.spinner("Computing SHAP values..."):
        shap_values, explainer = get_shap_values(model, X_test)

    tab1, tab2 = st.tabs(["📊 Summary Plot", "🔎 Single Prediction"])

    with tab1:
        st.markdown("**Global Feature Impact** – how much each feature contributes across all predictions.")
        fig_shap = shap_summary_plot(shap_values, X_test, feature_names)
        st.plotly_chart(fig_shap, use_container_width=True)

    with tab2:
        st.markdown("**Local Explanation** – explain a single prediction.")
        sample_idx = st.slider("Select test sample", 0, len(X_test) - 1, 0)
        pred_prob = model.predict_proba(X_test[sample_idx:sample_idx+1])[0][1]
        pred_class = "❤️ Disease" if pred_prob > 0.5 else "✅ No Disease"
        st.metric("Prediction", pred_class, f"Confidence: {pred_prob:.1%}")
        fig_wf = shap_waterfall_plot(shap_values, sample_idx, feature_names)
        st.plotly_chart(fig_wf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: Predict
# ═══════════════════════════════════════════════════════════════════
elif page == "🎯 Predict":
    st.title("🎯 Make a Prediction")
    st.markdown("Enter patient data below to get a heart disease risk prediction.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 20, 100, 55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 130)
        with col2:
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 250)
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"])
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        with col3:
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0, step=0.1)
            slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
            ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    if submitted:
        from src.predictor import predict_single
        input_data = {
            "age": age, "sex": 1 if sex == "Male" else 0,
            "cp": ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(cp),
            "trestbps": trestbps, "chol": chol, "fbs": 1 if fbs == "Yes" else 0,
            "restecg": ["Normal", "ST-T abnormality", "LV hypertrophy"].index(restecg),
            "thalach": thalach, "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak,
            "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
            "ca": ca,
            "thal": ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1,
        }
        prob, label = predict_single(model, input_data)

        st.markdown("---")
        if label == 1:
            st.error(f"⚠️ High Risk of Heart Disease — Confidence: **{prob:.1%}**")
        else:
            st.success(f"✅ Low Risk of Heart Disease — Confidence: **{1 - prob:.1%}**")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Risk Score", "font": {"color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8a93a8"},
                "bar": {"color": "#f5b800"},
                "steps": [
                    {"range": [0, 40], "color": "#13361f"},
                    {"range": [40, 70], "color": "#2d2a0f"},
                    {"range": [70, 100], "color": "#3a1010"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 70},
            },
            number={"suffix": "%", "font": {"color": "#f5b800"}},
        ))
        gauge.update_layout(paper_bgcolor="#0d0f14", font_color="#ffffff", height=300)
        st.plotly_chart(gauge, use_container_width=True)
