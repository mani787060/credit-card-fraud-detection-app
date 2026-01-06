import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("credit_card_model.pkl", "rb"))

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "Upload a CSV file containing transaction data to detect fraudulent transactions."
)

uploaded_file = st.file_uploader(
    "Upload CSV file (same format as training data)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Preview of Uploaded Data")
        st.dataframe(data.head())

        # Drop target column if present
        if "Class" in data.columns:
            X = data.drop("Class", axis=1)
        else:
            X = data.copy()

        predictions = model.predict(X)

        fraud_count = (predictions == 1).sum()
        legit_count = (predictions == 0).sum()

        st.subheader("Prediction Summary")
        st.success(f"Legitimate Transactions: {legit_count}")
        st.error(f"Fraudulent Transactions: {fraud_count}")

        # Add predictions to dataframe
        result_df = data.copy()
        result_df["Prediction"] = predictions
        result_df["Prediction"] = result_df["Prediction"].map(
            {0: "Legit", 1: "Fraud"}
        )

        st.subheader("Prediction Results")
        st.dataframe(result_df.head())

    except Exception as e:
        st.error("Error processing file. Please upload a valid CSV file.")
