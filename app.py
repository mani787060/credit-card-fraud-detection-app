import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("credit_card_model.pkl", "rb"))

model = load_model()

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.title("Credit Card Fraud Detection System")
st.write("Detect fraudulent credit card transactions using a trained machine learning model.")

# --------------------------------------------------
# Mode selection
# --------------------------------------------------
mode = st.radio(
    "Select Prediction Mode",
    ["Single Transaction", "Batch CSV Analysis"]
)

# ==================================================
# MODE 1: SINGLE TRANSACTION
# ==================================================
if mode == "Single Transaction":
    st.subheader("Single Transaction Fraud Check")

    st.info(
        "Upload a CSV file containing **exactly one transaction** "
        "with the same feature columns used during training."
    )

    single_file = st.file_uploader(
        "Upload 1-row CSV file",
        type=["csv"],
        key="single_csv"
    )

    if single_file is not None:
        try:
            data = pd.read_csv(single_file)

            if data.shape[0] != 1:
                st.error("Please upload a CSV file with exactly ONE row.")
            else:
                st.write("Uploaded Transaction:")
                st.dataframe(data)

                prediction = model.predict(data)[0]
                probability = model.predict_proba(data)[0][1]

                st.subheader("Prediction Result")

                if prediction == 1:
                    st.error(
                        f"High Risk Fraudulent Transaction\n\n"
                        f"Fraud Probability: {probability * 100:.2f}%"
                    )
                else:
                    st.success(
                        f"Transaction Appears Legitimate\n\n"
                        f"Legit Probability: {(1 - probability) * 100:.2f}%"
                    )

        except Exception as e:
            st.error("Invalid CSV format or column mismatch.")
            st.code(str(e))

# ==================================================
# MODE 2: BATCH CSV ANALYSIS
# ==================================================
else:
    st.subheader("Batch Fraud Detection (CSV Upload)")

    st.info(
        "Upload a CSV file containing multiple transactions. "
        "Predictions will be generated for all rows."
    )

    batch_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        key="batch_csv"
    )

    if batch_file is not None:
        try:
            df = pd.read_csv(batch_file)

            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head())

            with st.spinner("Running fraud detection..."):
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)[:, 1]

            df["Prediction"] = predictions
            df["Fraud_Probability"] = probabilities

            fraud_count = (predictions == 1).sum()
            legit_count = (predictions == 0).sum()
            total = len(df)

            st.subheader("Prediction Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", total)
            col2.metric("Fraudulent", fraud_count)
            col3.metric("Legitimate", legit_count)

            st.write("Prediction Results:")
            st.dataframe(df.head(10))

            st.download_button(
                label="Download Predictions as CSV",
                data=df.to_csv(index=False),
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Error processing CSV file.")
            st.code(str(e))
