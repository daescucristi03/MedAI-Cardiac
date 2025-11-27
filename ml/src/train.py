"""
Training script for MedAI-Cardiac
---------------------------------

This module:
 - Loads the training dataset
 - Applies preprocessing (fit + transform)
 - Creates the neural network model
 - Trains (fits) the model on the processed data
 - Saves the trained model to disk

Training must always be executed before evaluation or inference,
as it generates both:
 - best_model.joblib       (trained MLP)
 - scalers.pkl             (StandardScaler parameters)
"""

import os
import joblib
import pandas as pd
from preprocessing import fit_preprocess
from model_def import create_model


# ============================================================
# PATH HANDLING
# ============================================================

# ml/src/train.py → ml/ → project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "splits", "train.csv")


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train():
    """
    Loads training data, preprocesses it, trains the neural network,
    and saves the trained model.
    """
    print(f"Loading training dataset from: {TRAIN_PATH}")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training dataset not found at: {TRAIN_PATH}")

    # Load dataset
    df = pd.read_csv(TRAIN_PATH)

    if "heart_disease" not in df.columns:
        raise KeyError("Training dataset must contain a 'heart_disease' label column.")

    # Separate features and labels
    y = df["heart_disease"]
    X, scalers = fit_preprocess(df.drop("heart_disease", axis=1))

    # Create and train model
    model = create_model()
    print("Training model...")
    model.fit(X, y)

    # Save trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved at: {MODEL_PATH}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    train()