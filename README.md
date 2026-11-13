## Live Demo
Streamlit App: https://credit-card-fraud-detection-app-5uimzmg2but5qy3sqyfijq.streamlit.app/

# Credit Card Fraud Detection System

An end-to-end **Machine Learning-based web application** that identifies potentially fraudulent credit card transactions using a trained classification model.

Built with Python, Scikit-learn, and Streamlit, this project demonstrates how machine learning can be applied to financial transaction analysis through an interactive web interface.

## Project Overview

Credit card fraud is a major challenge in the financial industry. Detecting suspicious transactions accurately is important for reducing financial losses and improving transaction security.

This project focuses on developing a machine learning-based fraud detection system that classifies transactions into two categories:

- **Legitimate:** Transactions identified as normal.
- **Fraudulent:** Transactions identified as potentially suspicious.

The trained model is integrated into a Streamlit application, allowing users to interact with the prediction system through a web interface.

## Key Features

- **Machine Learning-Based Classification:** Uses a trained supervised learning model to classify transactions.
- **Transaction Prediction:** Predicts whether a transaction is legitimate or potentially fraudulent.
- **Interactive Web Interface:** Provides a simple interface built using Streamlit.
- **Confidence Score:** Displays the model's prediction confidence, where supported by the implementation.
- **Real-Time Prediction:** Processes submitted transaction information and returns a prediction.
- **Model Serialization:** Uses Pickle to save and load the trained machine learning model.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical operations |
| Pandas | Data handling and preprocessing |
| Scikit-learn | Machine learning and prediction |
| Streamlit | Interactive web application |
| Pickle | Model serialization |

## Machine Learning Workflow

The project follows a typical machine learning workflow:

```text
Transaction Dataset
        |
        v
Data Preprocessing
        |
        v
Feature Scaling
        |
        v
Model Training
        |
        v
Model Evaluation
        |
        v
Save Trained Model
        |
        v
Streamlit Application
        |
        v
Transaction Prediction
        |
        v
Legitimate / Fraudulent
```

### 1. Data Preprocessing

Transaction data is prepared for machine learning by performing the necessary preprocessing operations.

Depending on the dataset and implementation, this may include handling missing values, selecting relevant features, and preparing numerical inputs.

### 2. Feature Scaling

Feature scaling is used to bring numerical features into an appropriate range when required by the selected machine learning algorithm.

The same preprocessing steps used during training must also be applied during prediction.

### 3. Model Training

A supervised classification model is trained using transaction data to learn patterns associated with legitimate and fraudulent transactions.

### 4. Model Evaluation

The trained model can be evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

For fraud detection, precision and recall are particularly important because accuracy alone may not adequately represent performance when fraudulent transactions are rare.

### 5. Model Serialization

The trained model is saved using Pickle so that it can be loaded into the Streamlit application without retraining every time the application starts.

### 6. Web Application

Streamlit provides an interactive interface for submitting transaction information and displaying the model's prediction.

## Web Application

The application provides a simple way to interact with the fraud detection model.

Depending on the implemented input workflow, users can submit transaction information or upload transaction data and receive predictions.

The application displays the predicted transaction category and a confidence score where supported.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mani787060/credit-card-fraud-detection-app.git

cd credit-card-fraud-detection-app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas scikit-learn streamlit
```

If a `requirements.txt` file is available, install the dependencies using:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

*Note: Replace `app.py` if your actual Streamlit entry-point filename is different.*

## Deployment

The application is deployed using **Streamlit Community Cloud**, making it accessible through a web browser.

**Live Application:** Add or verify your deployed Streamlit application URL here.

## Project Applications

The techniques demonstrated in this project are relevant to:

- Financial transaction monitoring
- Suspicious transaction identification
- Payment fraud detection systems
- Machine learning-based risk classification
- Financial data analysis

## Future Improvements

Potential improvements include:

- Experimenting with ensemble algorithms such as Random Forest, XGBoost, and Gradient Boosting.
- Handling class imbalance using appropriate sampling techniques.
- Improving fraud detection using precision-recall optimization.
- Adding model explainability using SHAP or feature importance.
- Supporting batch prediction for multiple transactions.
- Adding transaction-level prediction reports.
- Monitoring model performance as transaction patterns change.

## Important Note

This project is an educational machine learning application. Its predictions should not be treated as verified financial fraud decisions or used as the sole basis for real-world transaction approval or rejection.

## Repository

GitHub: https://github.com/mani787060/credit-card-fraud-detection-app
