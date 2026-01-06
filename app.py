import streamlit as st
import numpy as np
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)

# ---------------- Load Model ----------------
model = pickle.load(open("credit_card_model.pkl", "rb"))

# ---------------- Title & Description ----------------
st.title("Credit Card Fraud Detection System")
st.write(
    "This application predicts whether a credit card transaction is **Legitimate** "
    "or **Fraudulent** using a trained Machine Learning model."
)

st.markdown("---")

# ---------------- Sample Inputs ----------------
# NOTE: These are representative examples (28 features)
sample_legit = np.zeros((1, 28))   # Example of a normal transaction
sample_fraud = np.ones((1, 28))    # Example of a fraudulent transaction

# ---------------- User Selection ----------------
transaction_type = st.radio(
    "Select transaction type to simulate:",
    ("Legitimate Transaction", "Fraudulent Transaction")
)

# ---------------- Prediction ----------------
if st.button("Predict Transaction"):
    if transaction_type == "Legitimate Transaction":
        input_data = sample_legit
    else:
        input_data = sample_fraud

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(
            f"⚠️ **Fraudulent Transaction Detected**\n\n"
            f"Confidence: **{probability * 100:.2f}%**"
        )
    else:
        st.success(
            f"✅ **Legitimate Transaction**\n\n"
            f"Confidence: **{(1 - probability) * 100:.2f}%**"
        )

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Note: This demo uses sample feature representations to provide a clean and "
    "user-friendly prediction experience."
)
