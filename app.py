# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="💳 Bank Fraud Detection System",
    page_icon="💰",
    layout="wide"
)

# ========== LOAD MODEL ARTIFACTS ==========
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_artifacts/fraud_detection_model.pkl")
    scaler = joblib.load("model_artifacts/scaler.pkl")
    with open("model_artifacts/features.json") as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

# ========== SIDEBAR ==========
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Single Transaction", "Batch Upload", "About"])

st.sidebar.markdown("---")
st.sidebar.markdown("💡 *Built using Streamlit & Scikit-learn*")

# ========== HELPER FUNCTION ==========
def preprocess_data(df: pd.DataFrame):
    """Converts all categorical columns to numeric codes"""
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].astype('category').cat.codes
    return df_copy

def predict_fraud(input_data: pd.DataFrame):
    """Scale, predict, and return probability"""
    scaled = scaler.transform(input_data)
    prob = model.predict_proba(scaled)[:, 1][0]
    pred = "Fraudulent" if prob >= 0.5 else "Legitimate"
    return pred, prob

# ========== PAGE 1: SINGLE TRANSACTION ==========
# ========== PAGE 1: SINGLE TRANSACTION ==========
if page == "Single Transaction":
    st.title("💳 Single Transaction Fraud Prediction")
    st.write("Enter key transaction details below to predict if it's **fraudulent or legitimate.**")

    # --- Only 5 main numeric inputs ---
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        with col1:
            transaction_amount = st.number_input("💰 Transaction Amount", min_value=0.0, step=0.01)
            account_balance = st.number_input("🏦 Account Balance", min_value=0.0, step=0.01)
            transaction_hour = st.number_input("⏰ Transaction Hour (0–23)", min_value=0, max_value=23, step=1)
        with col2:
            customer_age = st.number_input("👤 Customer Age", min_value=18, max_value=100, step=1)
            transaction_location_risk = st.number_input("📍 Transaction Location Risk Score (0–10)", min_value=0.0, max_value=10.0, step=0.1)

        submitted = st.form_submit_button("🔎 Predict Fraud")

    if submitted:
        try:
            # --- Prepare input data ---
            user_input = {
                "Transaction_Amount": transaction_amount,
                "Account_Balance": account_balance,
                "Transaction_Hour": transaction_hour,
                "Age": customer_age,
                "Location_Risk": transaction_location_risk
            }

            # Create dataframe from inputs
            input_df = pd.DataFrame([user_input])

            # --- Align with model features ---
            aligned_input = pd.DataFrame()
            for col in feature_names:
                aligned_input[col] = input_df[col] if col in input_df.columns else 0

            # --- Scale + predict ---
            scaled = scaler.transform(aligned_input)
            prob = model.predict_proba(scaled)[:, 1][0]
            pred = "Fraudulent" if prob >= 0.5 else "Legitimate"

            st.markdown("---")

            # === COLOR-CODED OUTPUT ===
            if pred == "Fraudulent":
                st.error(f"🚨 **Transaction is Fraudulent!** Probability: {prob:.2%}")
            else:
                st.success(f"✅ **Transaction is Legitimate.** Fraud probability: {prob:.2%}")

            # === GAUGE METER ===
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Fraud Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if prob >= 0.5 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightgreen'},
                        {'range': [50, 100], 'color': 'salmon'}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")


# ========== PAGE 2: BATCH UPLOAD ==========
# ========== PAGE 2: BATCH UPLOAD ==========
elif page == "Batch Upload":
    st.title("📂 Batch Fraud Prediction from CSV")
    st.write("Upload any CSV file — the system will adapt automatically and predict fraud risk.")

    uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {df.shape[0]} rows, {df.shape[1]} columns detected.")

            # === AUTO PREPROCESSING ===
            df_processed = preprocess_data(df)

            # Find overlap between CSV columns and model features
            common_cols = [c for c in df_processed.columns if c in feature_names]
            missing_cols = [c for c in feature_names if c not in df_processed.columns]
            extra_cols = [c for c in df_processed.columns if c not in feature_names]

            st.markdown("### 🧹 Data Alignment Summary")
            st.write(f"✅ **Used Columns:** {len(common_cols)} | ❌ **Missing:** {len(missing_cols)} | ➕ **Extra:** {len(extra_cols)}")

            # === SAFETY CHECK ===
            if len(common_cols) == 0:
                st.warning("⚠️ No matching columns found between your file and model features. "
                           "Predictions will be made using default values (may be less accurate).")

                # Create an all-zero DataFrame to match the model input shape
                aligned_df = pd.DataFrame(np.zeros((len(df), len(feature_names))), columns=feature_names)

            else:
                # Align columns dynamically
                aligned_df = pd.DataFrame()
                for col in feature_names:
                    aligned_df[col] = df_processed[col] if col in df_processed.columns else 0

            # === SCALING & PREDICTION ===
            if aligned_df.shape[0] == 0:
                st.error("❌ No data rows found in the file. Please upload a non-empty CSV.")
            else:
                scaled = scaler.transform(aligned_df)
                probs = model.predict_proba(scaled)[:, 1]
                df["Fraud_Probability"] = probs
                df["Prediction"] = np.where(probs >= 0.5, "Fraudulent", "Legitimate")

                # === SUMMARY ===
                fraud_rate = (df["Prediction"] == "Fraudulent").mean()
                total_txns = len(df)
                total_fraud = int((df["Prediction"] == "Fraudulent").sum())

                st.markdown("---")
                st.subheader("📊 Batch Fraud Summary")

                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.metric("💳 Total Transactions", f"{total_txns:,}")
                    st.metric("🚨 Fraudulent Transactions", f"{total_fraud:,}")
                    st.metric("📈 Fraudulent Percentage", f"{fraud_rate*100:.2f}%")

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

                with col2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fraud_rate * 100,
                        title={'text': "Overall Fraud Rate (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red" if fraud_rate >= 0.5 else "green"},
                            'steps': [
                                {'range': [0, 50], 'color': 'lightgreen'},
                                {'range': [50, 100], 'color': 'salmon'}
                            ],
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # === PIE CHART ===
                st.markdown("### 🥧 Fraud vs Legitimate Distribution")
                fraud_counts = df["Prediction"].value_counts().reset_index()
                fraud_counts.columns = ["Status", "Count"]

                fig_pie = px.pie(
                    fraud_counts,
                    names="Status",
                    values="Count",
                    color="Status",
                    color_discrete_map={"Fraudulent": "red", "Legitimate": "green"},
                    title="Fraudulent vs Legitimate Transactions"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # === SAMPLE OUTPUT ===
                st.markdown("### 🔍 Sample Predictions")
                st.dataframe(df.head(10))

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


# ========== PAGE 3: ABOUT ==========
elif page == "About":
    st.title("ℹ️ About the App")
    st.write("""
    This **Bank Transaction Fraud Detection System** uses a trained Random Forest model
    to detect potentially fraudulent transactions in real time.

    **Features:**
    - Random Forest + StandardScaler  
    - Handles text columns automatically (e.g., Gender, Transaction Type)  
    - Single or batch prediction  
    - Interactive gauge & pie chart visualizations  
    - Clean, modern UI built with Streamlit & Plotly  

    **Created by:** Aadi  
    """)
    st.info("For educational demonstration only — not for real-world banking use.")
