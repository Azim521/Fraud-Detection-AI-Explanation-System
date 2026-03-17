# 🔍 Fraud Detection + AI Explanation System

> Detect fraudulent credit card transactions using XGBoost — and explain *why* each transaction was flagged, in plain English.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-detection-ai-explanation-system-azim.streamlit.app)

![Demo](demo.gif)

---

## 🚀 Live Demo

👉 [fraud-detection-ai-explanation-system-azim.streamlit.app](https://fraud-detection-ai-explanation-system-azim.streamlit.app)

---

## 📌 Overview

Most fraud detection projects stop at "fraud or not fraud." This one goes further — it tells you **why**.

Built on the real-world [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) with 284,807 transactions (only 0.17% fraud), this system:

- Detects fraud with **97.76% ROC-AUC**
- Uses **SHAP** to explain which features drove each decision
- Generates a **natural language explanation** via OpenAI GPT
- Serves everything through an interactive **Streamlit dashboard**

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat&logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-412991?style=flat&logo=openai)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-blue?style=flat&logo=scikit-learn)

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.9776 |
| Average Precision | 0.8663 |
| Fraud Recall | 90% |
| Fraud Missed (FN) | 10 out of 98 |
| Legit Wrongly Flagged (FP) | 86 out of 56,864 |

---

## 🏗️ How It Works

```
Credit Card Transaction
        ↓
  XGBoost Model  →  Fraud Probability Score
        ↓
   SHAP Explainer  →  Feature-level explanation
        ↓
  OpenAI GPT-3.5  →  Plain English explanation
        ↓
   Streamlit App  →  Interactive dashboard
```

**Key design decisions:**
- Used **SMOTE** (not downsampling) to handle class imbalance — preserves all 284k legitimate transactions
- SHAP applied only on test set — no data leakage
- Explainability layer built on top of a production-grade XGBoost pipeline

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Azim521/Fraud-Detection-AI-Explanation-System.git
cd Fraud-Detection-AI-Explanation-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Add your OpenAI key to a `.streamlit/secrets.toml` file:
```toml
OPENAI_API_KEY = "sk-your-key-here"
```

---

## 📁 Project Structure

```
Fraud-Detection-AI-Explanation-System/
├── app.py                  ← Streamlit dashboard
├── requirements.txt        ← Dependencies
└── model/
    ├── xgb_model.pkl       ← Trained XGBoost model
    ├── shap_explainer.pkl  ← SHAP TreeExplainer
    ├── scaler.pkl          ← StandardScaler for Amount & Time
    └── test_samples.csv    ← Demo transactions (500 legit + 98 fraud)
```

---

## 📬 Contact

Built by **Azim** · [GitHub](https://github.com/Azim521)
