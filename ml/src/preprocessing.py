import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# PATH HANDLING
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SCALERS_PATH = os.path.join(BASE_DIR, "scalers.pkl")

# DEFINE FEATURES
NUMERIC_FEATURES = ["age", "resting_bp", "chol", "max_hr", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "fasting_glucose", "rest_ecg", "exercise_angina", "st_slope"]


# SCALER FUNCTIONS
def load_scalers():
    """Încarcam scaler-ele daca exista, altfel returnam None."""
    if os.path.exists(SCALERS_PATH):
        return joblib.load(SCALERS_PATH)
    return None

def save_scalers(scalers):
    """Salvam scaler-ele în ml/scalers.pkl."""
    joblib.dump(scalers, SCALERS_PATH)

# PRE-PROCESSING ( TRAINING )
def fit_preprocess(df: pd.DataFrame):
    """
    Preproceseaza datele (training):
    - potriveste scalerele
    - returneaza X procesat + salveaza scalerele
    """
    scalers = {}

    # 1. NUMERIC
    df_num = df[NUMERIC_FEATURES]
    num_scaler = StandardScaler()
    df_num_scaled = num_scaler.fit_transform(df_num)
    scalers["numeric"] = num_scaler

    # 2. CATEGORICAL
    df_cat = df[CATEGORICAL_FEATURES].astype(float)

    # 3. CONCATENATE NUMERIC + CATEGORICAL
    df_processed = np.concatenate([df_num_scaled, df_cat.to_numpy()], axis=1)

    # Save scalers
    save_scalers(scalers)

    return df_processed, scalers

# PREPROCESS INPUT
def preprocess_input(df: pd.DataFrame):
    scalers = load_scalers()
    if scalers is None:
        raise ValueError("Scalers not found! Ruleaza train.py pentru a le genera.")

    # Validate columns
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Numeric
    df_num = df[NUMERIC_FEATURES]
    df_num_scaled = scalers["numeric"].transform(df_num)

    # Categorical
    df_cat = df[CATEGORICAL_FEATURES].astype(float)

    result = np.concatenate([df_num_scaled, df_cat.to_numpy()], axis=1)

    return result
