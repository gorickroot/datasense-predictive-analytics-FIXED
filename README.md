# 📊 DataSense – Predictive Analytics

> A machine learning-powered analytics dashboard built for Computing Science coursework at Griffith College Dublin. Uses ensemble methods with SHAP explanations for interpretable predictions.

![Academic](https://img.shields.io/badge/Category-Academic-f5b800?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3b82f6?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-22c55e?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-ef4444?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-8a93a8?style=flat-square)

---

## 🚀 Features

| Feature | Details |
|---|---|
| 🤖 Ensemble Model | XGBoost + RandomForest + GradientBoosting (soft voting) |
| 💡 Explainability | SHAP beeswarm + waterfall charts per prediction |
| 📊 Interactive Dashboard | 5-page Streamlit app with dark UI |
| 🔍 Data Explorer | Distributions, correlations, raw data view |
| 🎯 Live Predictor | Input patient data → instant risk score + gauge |
| 📈 Model Evaluation | ROC curve, confusion matrix, classification report |
| 🔁 Cross Validation | 5-fold stratified CV with AUC reporting |

---

## 🛠 Tech Stack

- **XGBoost** – Primary gradient boosting model
- **SHAP** – Model interpretability & explanations
- **Pandas / NumPy** – Data processing
- **Plotly** – Interactive visualizations
- **Streamlit** – Web dashboard framework
- **Scikit-learn** – Preprocessing, metrics, ensemble

---

## 📁 Project Structure

```
datasense/
├── app.py                  # Main Streamlit app (5 pages)
├── requirements.txt
├── data/
│   └── heart.csv           # Dataset (UCI Heart Disease)
├── models/
│   └── datasense_model.pkl # Saved trained model
├── src/
│   ├── data_loader.py      # Data loading & preprocessing
│   ├── model.py            # Ensemble training & evaluation
│   ├── explainer.py        # SHAP explainability
│   ├── visualizations.py   # All Plotly chart builders
│   └── predictor.py        # Single-sample inference
└── notebooks/
    └── exploration.ipynb   # EDA notebook
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/datasense-predictive-analytics.git
cd datasense-predictive-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501` 🎉

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~92% |
| ROC-AUC | ~0.97 |
| F1 Score | ~0.92 |
| CV AUC (5-fold) | ~0.96 ± 0.02 |

---

## 🧠 How SHAP Works Here

SHAP (SHapley Additive exPlanations) assigns each feature a value that represents its contribution to a specific prediction.

- **Beeswarm plot** → global view of feature impact across all samples
- **Waterfall plot** → local view showing exactly why the model predicted what it did for one patient

---

## 📄 Dataset

Uses the **UCI Heart Disease Dataset** (Cleveland subset) — 303 patients, 13 clinical features, binary target (disease / no disease).

If `data/heart.csv` is not present, the app auto-generates a realistic synthetic dataset matching the original distribution.

---

## 🎓 Academic Context

Built as part of Computing Science coursework at **Griffith College Dublin**.  
Demonstrates: feature engineering, ensemble methods, model interpretability, and production-ready ML deployment.

---

## 📝 License

MIT © 2024
