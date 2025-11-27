"""
Model Evaluation Script for MedAI-Cardiac
-----------------------------------------

This module evaluates the trained neural network model on the
dedicated test dataset. It performs the following steps:

1. Load the test dataset (previously untouched by training).
2. Apply the exact same preprocessing used during training.
3. Load the trained neural network model.
4. Generate predictions.
5. Output evaluation metrics:
       - Accuracy score
       - Full classification report (precision, recall, F1-score)

This script is intended to provide a final, unbiased estimate of
model performance after the training process is complete.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_input


# ============================================================
# PATH SETUP
# ============================================================

# File: ml/src/evaluate.py → ml/ → project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
TEST_PATH = os.path.join(BASE_DIR, "data", "splits", "test.csv")


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate():
    """
    Loads the trained model and evaluates it on the test dataset.
    Produces:
      - Accuracy score
      - Classification report (precision, recall, F1-score)
    """
    print(f"Loading test dataset from: {TEST_PATH}")

    # Verify dataset exists
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test dataset not found at: {TEST_PATH}")

    # Load CSV file
    df = pd.read_csv(TEST_PATH)

    # Validate required label column
    if "heart_disease" not in df.columns:
        raise KeyError("The test dataset must contain a 'heart_disease' label column.")

    y_true = df["heart_disease"]
    X_raw = df.drop("heart_disease", axis=1)

    # Preprocess using stored scalers
    X = preprocess_input(X_raw)

    if X is None:
        raise RuntimeError("preprocess_input returned None — preprocessing failed.")

    # Check model existence
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Please run train.py first."
        )

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    # Run predictions
    print("Running predictions...")
    y_pred = model.predict(X)

    # ============================================================
    # OUTPUT RESULTS
    # ============================================================

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    evaluate()