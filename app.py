# app.py — Fraud Detection + AI Explanation System

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🔍",
    layout="wide"
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load("model/xgb_model.pkl")
    explainer = joblib.load("model/shap_explainer.pkl")
    scaler    = joblib.load("model/scaler.pkl")
    test_df   = pd.read_csv("model/test_samples.csv")
    return model, explainer, scaler, test_df

model, explainer, scaler, test_df = load_artifacts()
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()
    filter_type = st.selectbox("Pick transaction type", ["Any", "Fraud", "Legitimate"])

    if filter_type == "Fraud":
        pool = test_df[test_df["Class"] == 1]
    elif filter_type == "Legitimate":
        pool = test_df[test_df["Class"] == 0]
    else:
        pool = test_df

    idx = st.slider("Transaction index", 0, len(pool) - 1, 0)
    selected = pool.iloc[idx]
    actual_label = "🔴 Fraud" if selected["Class"] == 1 else "🟢 Legitimate"
    st.markdown(f"**Actual Label:** {actual_label}")
    st.divider()
    openai_key = st.text_input("OpenAI API Key (optional)", type="password",
                               help="For AI natural language explanation")
    if not openai_key:
        openai_key = st.secrets.get("OPENAI_API_KEY", "")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔍 Fraud Detection + AI Explanation System")
st.caption("XGBoost · SHAP · Streamlit")
st.divider()

# Prepare input
X_input = pd.DataFrame([selected[FEATURES].values], columns=FEATURES)

# Predict
fraud_prob = model.predict_proba(X_input)[0][1]
fraud_pct  = round(fraud_prob * 100, 2)
verdict    = "🔴 FRAUD" if fraud_prob >= 0.5 else "🟢 LEGITIMATE"

# ── Row 1: Score + Gauge ──────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    st.markdown(f"### {verdict}")
    st.metric("Fraud Probability", f"{fraud_pct}%")
    st.metric("Actual Label", actual_label)

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_pct,
        number={"suffix": "%"},
        title={"text": "Fraud Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "crimson" if fraud_prob >= 0.5 else "steelblue"},
            "steps": [
                {"range": [0, 40],  "color": "#d4edda"},
                {"range": [40, 70], "color": "#fff3cd"},
                {"range": [70, 100],"color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Row 2: SHAP Waterfall ─────────────────────────────────────────────────────
st.subheader("🔎 Why did the model make this decision?")

shap_values = explainer.shap_values(X_input)

fig2, ax = plt.subplots(figsize=(10, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_input.values[0],
        feature_names=FEATURES
    ),
    show=False
)
st.pyplot(plt.gcf())
plt.clf()

st.divider()

# ── Row 3: Top features driving this decision ─────────────────────────────────
st.subheader("📊 Top Features Driving This Decision")

shap_df = pd.DataFrame({
    "Feature": FEATURES,
    "SHAP Value": shap_values[0],
    "Impact": np.abs(shap_values[0])
}).sort_values("Impact", ascending=False).head(10)

shap_df["Direction"] = shap_df["SHAP Value"].apply(
    lambda x: "↑ Increases Fraud Risk" if x > 0 else "↓ Decreases Fraud Risk"
)

st.dataframe(
    shap_df[["Feature", "SHAP Value", "Direction"]].reset_index(drop=True),
    use_container_width=True
)

st.divider()

# ── Row 4: AI Explanation (OpenAI) ────────────────────────────────────────────
st.subheader("🤖 AI Natural Language Explanation")

if openai_key:
    if st.button("Generate AI Explanation"):
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        top_features = shap_df.head(5)[["Feature", "SHAP Value", "Direction"]].to_string(index=False)

        prompt = f"""
You are a fraud analyst AI. A machine learning model analyzed a credit card transaction 
and gave it a fraud probability of {fraud_pct}%.

The top features driving this decision were:
{top_features}

In 3-4 sentences, explain in simple English why this transaction was flagged 
(or cleared) as fraud. Avoid technical jargon. Be specific about which features 
mattered most and what direction they pushed the decision.
"""
        with st.spinner("Generating explanation..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            explanation = response.choices[0].message.content

        st.info(explanation)
else:
    st.caption("Add your OpenAI API key in the sidebar to enable AI explanations.")
    st.info(
        f"This transaction has a **{fraud_pct}% fraud probability**. "
        f"The top driver is **{shap_df.iloc[0]['Feature']}** "
        f"which {shap_df.iloc[0]['Direction'].lower()}. "
        f"Add an OpenAI key for a full natural language explanation."
    )
