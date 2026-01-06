import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("Credit Card Fraud Detection System")
st.write("Upload a CSV file containing transaction data to detect fraudulent transactions.")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("credit_card_model.pkl", "rb"))

model = load_model()

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (same format as training data)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        # -----------------------------
        # FIX: Drop target column if present
        # -----------------------------
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        # -----------------------------
        # FIX: Match training feature order
        # -----------------------------
        df = df[model.feature_names_in_]

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df["Prediction"] = predictions
        df["Fraud_Probability"] = probabilities

        # -----------------------------
        # Summary
        # -----------------------------
        fraud_count = (df["Prediction"] == 1).sum()
        legit_count = (df["Prediction"] == 0).sum()

        st.subheader("Prediction Summary")
        col1, col2 = st.columns(2)
        col1.metric("Fraudulent Transactions", fraud_count)
        col2.metric("Legitimate Transactions", legit_count)

        st.subheader("Prediction Results")
        st.dataframe(df.head(10))

        # -----------------------------
        # Download results
        # -----------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions as CSV",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")

else:
    st.info("Please upload a CSV file to start prediction.")
